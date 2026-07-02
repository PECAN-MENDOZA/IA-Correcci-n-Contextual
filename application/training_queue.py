"""
application/training_queue.py  v4 → v5 (desacoplado de inferencia)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATRÓN PRODUCTOR-CONSUMIDOR  ─  Online Learning sin bloquear inferencia
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                    ┌──────────────┐
    POST /feedback  │              │  queue.put(user_id)   ┌──────────┐
    ───────────────►│  Flask API   ├──────────────────────►│  Queue   │
    204 (instant)   │  (Producer)  │                       │ thread-  │
    ◄───────────────│              │                       │  safe    │
                    └──────────────┘                       └────┬─────┘
                                                               │ get()
                                                    ┌──────────▼──────────┐
                                                    │   TrainingWorker     │
                                                    │   (único Consumer)   │
                                                    │                      │
                                                    │  1. Espera tarea     │
                                                    │  2. Fine-tuning LoRA │
                                                    │     (modelo propio,  │
                                                    │      NO gpu_lock)    │
                                                    │  3. Invalida cache   │
                                                    └──────────────────────┘

CAMBIO CLAVE (v5): el training YA NO adquiere gpu_lock.
Antes, _process() envolvía todo el fine-tuning (~9s) dentro de
`with gpu_lock:`, y como generate_corrections()/generate_with_lora()
piden ese mismo lock para inferir, cualquier corrección que llegara
durante esos 9s quedaba bloqueada esperando.

Ahora TrainUserModelUseCase carga su PROPIA copia del modelo base
en memoria (ver application/use_cases/train_model.py:_get_base_model),
separada de la instancia que usa el pipeline de inferencia. Al ser
objetos distintos, entrenamiento e inferencia pueden correr en
paralelo en la misma GPU sin pisarse — las GPUs modernas soportan
ambas cargas simultáneas para un modelo de este tamaño (T5-base).

gpu_lock se mantiene, pero ahora vive solo del lado de inferencia
(infrastructure/ml/t5_model.py), protegiendo el acceso concurrente
al modelo COMPARTIDO cuando varios usuarios corrigen a la vez y/o
se attachea/desattachea un adaptador LoRA en generate_with_lora().
Ya no sirve para separar training de inferencia (para eso ahora
usamos instancias de modelo separadas), solo para serializar
inferencias concurrentes entre sí.

Tras un crash de CUDA (contexto "envenenado"), el Worker hace reset:
vacía la VRAM con empty_cache() y deja que el siguiente intento
reutilice la instancia de entrenamiento ya cargada (lazy singleton
dentro de TrainUserModelUseCase).

CAMBIO (v5.1): MÚLTIPLES PROCESOS EN LA MISMA VM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
`queue.Queue()` vive en la memoria de UN proceso. Si corres varios
procesos (ej. `gunicorn --workers 4`), cada uno tiene su propia cola
en memoria y su propio hilo TrainingWorker → N entrenamientos
descoordinados compitiendo por la misma GPU, y cada proceso además
carga su propia copia del modelo base en VRAM.

Para correr N réplicas en una sola VM sin ese problema, el patrón es:
  - N procesos "api" (rol=api): solo sirven Flask/inferencia. NO cargan
    un TrainingWorker local; solo empujan tareas a una cola compartida.
  - 1 único proceso "worker" (rol=worker): consume esa cola compartida
    y es el ÚNICO que entrena (así los entrenamientos siguen
    serializados entre sí, que es lo que queremos, sin competir con
    la inferencia de los procesos api).

La cola compartida entre procesos usa Redis (RedisTrainingQueue abajo)
en vez de queue.Queue en memoria. Ver main.py para el wiring por ROLE.
"""

import os
import queue
import threading
import time
from typing import Optional, Callable

try:
    import redis
except ImportError:
    redis = None

TRAINING_QUEUE_KEY   = "t5lora:training_queue"
TRAINING_PENDING_KEY = "t5lora:training_pending"

# Lock de inferencia. Protege generate_corrections() y generate_with_lora()
# en infrastructure/ml/t5_model.py cuando varias correcciones/adaptadores
# LoRA concurrentes tocan el mismo modelo compartido.
# Ya NO se usa para bloquear el entrenamiento (ver docstring arriba).
gpu_lock: threading.Lock = threading.Lock()


class RedisTrainingQueue:
    """
    Cola de entrenamiento respaldada por Redis. Sustituye a queue.Queue
    cuando hay más de un proceso (ROLE=api ×N + ROLE=worker ×1) en la
    misma VM (o incluso en VMs distintas, si REDIS_URL apunta a un
    Redis compartido).

    Misma idea que la cola en memoria: una LISTA para el orden FIFO y
    un SET aparte solo para descartar duplicados pendientes en O(1),
    igual que antes hacía `list(self._queue.queue)`.
    """

    def __init__(self, url: Optional[str] = None):
        if redis is None:
            raise ImportError(
                "Falta el paquete 'redis'. Ejecuta: pip install redis"
            )
        url = url or os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        self._client = redis.Redis.from_url(url)
        self._client.ping()  # falla rápido si Redis no está disponible

    def enqueue(self, user_id: str) -> None:
        # SADD devuelve 1 solo si el user_id no estaba ya en el set → dedup barato,
        # equivalente al chequeo "if user_id not in pending" de antes.
        is_new = self._client.sadd(TRAINING_PENDING_KEY, user_id)
        if is_new:
            self._client.lpush(TRAINING_QUEUE_KEY, user_id)
            print(f"[Queue-Redis] Tarea encolada para '{user_id}'. "
                  f"Pendientes: {self.pending_count}")
        else:
            print(f"[Queue-Redis] '{user_id}' ya en cola — duplicado descartado.")

    def blocking_pop(self, timeout: float = 1.0) -> Optional[str]:
        """Espera hasta `timeout` segundos por una tarea. None si no llegó ninguna."""
        result = self._client.brpop(TRAINING_QUEUE_KEY, timeout=timeout)
        if result is None:
            return None
        _, raw_user_id = result
        user_id = raw_user_id.decode("utf-8")
        self._client.srem(TRAINING_PENDING_KEY, user_id)
        return user_id

    @property
    def pending_count(self) -> int:
        return int(self._client.llen(TRAINING_QUEUE_KEY))


class TrainingWorker:
    """
    Hilo de fondo que consume tareas de entrenamiento, una a la vez.

    Parámetros
    ----------
    train_fn : Callable[[str], None]
        Función user_id → fine-tuning. Típicamente TrainUserModelUseCase.execute.
    on_train_end : Callable[[str], None] | None
        Callback invocado tras fine-tuning exitoso (invalida cache de pipelines).
    idle_sleep : float
        Segundos de espera cuando la cola está vacía (default 0.5).
    queue_backend : RedisTrainingQueue | None
        Si se pasa, el worker consume de ahí (multi-proceso, ver main.py ROLE=worker).
        Si es None (default), usa una queue.Queue en memoria — modo de un solo
        proceso, igual que antes.
    """

    def __init__(
        self,
        train_fn: Callable[[str], None],
        on_train_end: Optional[Callable[[str], None]] = None,
        idle_sleep: float = 0.5,
        queue_backend: Optional[RedisTrainingQueue] = None,
    ):
        self._train_fn     = train_fn
        self._on_train_end = on_train_end
        self._idle_sleep   = idle_sleep
        self._backend       = queue_backend
        self._queue: queue.Queue[str] = queue.Queue()  # solo se usa si _backend es None

        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="TrainingWorker"
        )
        self._thread.start()
        mode = "Redis (multi-proceso)" if self._backend else "en memoria (un solo proceso)"
        print(f"[TrainingWorker] Hilo de entrenamiento iniciado. Cola: {mode}.")

    # ── API pública ───────────────────────────────────────────────────────────

    def enqueue(self, user_id: str) -> None:
        """
        Encola una tarea. No bloquea. Descarta duplicados pendientes.
        Si el usuario ya tiene un entrenamiento en cola, el nuevo feedback
        igualmente se habrá guardado en CSV, por lo que el próximo run
        entrenará con todos los pares acumulados.
        """
        if self._backend is not None:
            self._backend.enqueue(user_id)
            return

        pending = list(self._queue.queue)
        if user_id not in pending:
            self._queue.put(user_id)
            print(f"[Queue] Tarea encolada para '{user_id}'. "
                  f"Pendientes: {self._queue.qsize()}")
        else:
            print(f"[Queue] '{user_id}' ya en cola — duplicado descartado.")

    @property
    def pending_count(self) -> int:
        if self._backend is not None:
            return self._backend.pending_count
        return self._queue.qsize()

    # ── Loop interno ──────────────────────────────────────────────────────────

    def _loop(self) -> None:
        while True:
            if self._backend is not None:
                user_id = self._backend.blocking_pop(timeout=self._idle_sleep)
                if user_id is None:
                    continue
                self._process(user_id)
                continue

            try:
                user_id = self._queue.get(timeout=self._idle_sleep)
            except queue.Empty:
                continue
            self._process(user_id)
            self._queue.task_done()

    def _process(self, user_id: str) -> None:
        """
        Ejecuta fine-tuning SIN gpu_lock: usa una instancia de modelo
        propia (cargada dentro de TrainUserModelUseCase), distinta de la
        que usa el pipeline de inferencia, así que corre en paralelo con
        las correcciones que sigan llegando por la API mientras entrena.

        La cola sigue siendo un único consumidor (este hilo), por lo que
        los entrenamientos entre sí quedan naturalmente serializados
        (uno a la vez); lo que se eliminó es el bloqueo cruzado con la
        inferencia.

        Si CUDA queda envenenado tras el error, vacía la VRAM para
        que el siguiente intento arranque con contexto limpio.
        """
        print(f"\n[🧠 FINE-TUNING] Iniciando en paralelo (sin bloquear inferencia) para '{user_id}'...")

        start = time.time()
        try:
            self._train_fn(user_id)
            elapsed = time.time() - start
            print(f"[✅ FINE-TUNING] Completado para '{user_id}' en {elapsed:.1f}s.")

            if self._on_train_end:
                self._on_train_end(user_id)
                print(f"[🔄 CACHE] Pipeline de '{user_id}' invalidado.")

        except Exception as exc:
            print(f"[❌ FINE-TUNING ERROR] '{user_id}': {exc}")
            # Limpiar VRAM para no dejar el contexto CUDA "envenenado"
            # que afectaría las siguientes llamadas de inferencia.
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    print("[🔧 CUDA RESET] VRAM liberada tras error.")
            except Exception as cuda_exc:
                print(f"[WARN] No se pudo limpiar VRAM: {cuda_exc}")
