"""
application/training_queue.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATRÓN PRODUCTOR-CONSUMIDOR  ─  Online Learning seguro en GPU
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
                                                    │  2. Adquiere lock    │
                                                    │  3. Fine-tuning GPU  │
                                                    │  4. Libera lock      │
                                                    │  5. Invalida cache   │
                                                    └──────────────────────┘

Tras un crash de CUDA (contexto "envenenado"), el Worker hace reset:
vacía la VRAM con empty_cache() y deja que el siguiente intento
instancie un modelo fresco desde disco (T5CorrectionModel se crea
dentro de TrainUserModelUseCase.execute(), no se reutiliza).
"""

import queue
import threading
import time
from typing import Optional, Callable

# Lock global compartido entre TrainingWorker (escritura) y
# T5CorrectionModel.generate_corrections() (lectura).
# Garantiza exclusión mutua: nunca inferencia + training al mismo tiempo.
gpu_lock: threading.Lock = threading.Lock()


class TrainingWorker:
    """
    Hilo de fondo que serializa el acceso a GPU para fine-tuning.

    Parámetros
    ----------
    train_fn : Callable[[str], None]
        Función user_id → fine-tuning. Típicamente TrainUserModelUseCase.execute.
    on_train_end : Callable[[str], None] | None
        Callback invocado tras fine-tuning exitoso (invalida cache de pipelines).
    idle_sleep : float
        Segundos de espera cuando la cola está vacía (default 0.5).
    """

    def __init__(
        self,
        train_fn: Callable[[str], None],
        on_train_end: Optional[Callable[[str], None]] = None,
        idle_sleep: float = 0.5,
    ):
        self._train_fn     = train_fn
        self._on_train_end = on_train_end
        self._idle_sleep   = idle_sleep
        self._queue: queue.Queue[str] = queue.Queue()

        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="TrainingWorker"
        )
        self._thread.start()
        print("[TrainingWorker] Hilo de entrenamiento iniciado.")

    # ── API pública ───────────────────────────────────────────────────────────

    def enqueue(self, user_id: str) -> None:
        """
        Encola una tarea. No bloquea. Descarta duplicados pendientes.
        Si el usuario ya tiene un entrenamiento en cola, el nuevo feedback
        igualmente se habrá guardado en CSV, por lo que el próximo run
        entrenará con todos los pares acumulados.
        """
        pending = list(self._queue.queue)
        if user_id not in pending:
            self._queue.put(user_id)
            print(f"[Queue] Tarea encolada para '{user_id}'. "
                  f"Pendientes: {self._queue.qsize()}")
        else:
            print(f"[Queue] '{user_id}' ya en cola — duplicado descartado.")

    @property
    def pending_count(self) -> int:
        return self._queue.qsize()

    # ── Loop interno ──────────────────────────────────────────────────────────

    def _loop(self) -> None:
        while True:
            try:
                user_id = self._queue.get(timeout=self._idle_sleep)
            except queue.Empty:
                continue
            self._process(user_id)
            self._queue.task_done()

    def _process(self, user_id: str) -> None:
        """
        Ejecuta fine-tuning con exclusión mutua en GPU.
        Si CUDA queda envenenado tras el error, vacía la VRAM para
        que el siguiente intento arranque con contexto limpio.
        """
        print(f"\n[🧠 FINE-TUNING] Esperando acceso exclusivo a GPU para '{user_id}'...")

        with gpu_lock:
            print(f"[🧠 FINE-TUNING] GPU adquirida. Iniciando para '{user_id}'...")
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
