"""
main.py - Composition Root  v3 → v5 (LoRA + multi-proceso)

Cambios:
  - TrainUserModelUseCase recibe loras_dir="./models/loras"
  - correct_uc.set_base_model() inyecta el modelo base para LoRA
  - USER_MODELS_DIR se mantiene solo por compatibilidad legacy
  - ROLE=api|worker|all controla qué arma este proceso (ver abajo)

MODOS (variable de entorno ROLE, default "all"):

  ROLE=all (default, comportamiento de siempre)
    Un solo proceso: sirve Flask/inferencia Y corre el TrainingWorker
    con cola en memoria. Simple, sirve para desarrollo o para un único
    proceso en producción (gunicorn --workers 1, como está hoy el
    Dockerfile). No sirve si corres varios procesos a la vez: cada uno
    tendría su propia cola en memoria y su propia copia del modelo en
    VRAM, sin coordinarse entre sí.

  ROLE=api
    Solo sirve Flask/inferencia. NO levanta un TrainingWorker local:
    cuando llega feedback, empuja el user_id a una cola Redis
    compartida (REDIS_URL) en vez de encolarlo en memoria. Pensado
    para correr N réplicas de este rol detrás de un load balancer.
    Requiere REDIS_URL.

  ROLE=worker
    NO sirve HTTP. Solo consume la cola Redis compartida y entrena
    (un único proceso worker → los entrenamientos siguen serializados
    entre sí, que es lo que queremos). Requiere REDIS_URL.
    Entry point: worker.py (o `ROLE=worker python main.py`).

Ejemplo local con Redis ya corriendo en localhost:6379:
    REDIS_URL=redis://localhost:6379/0 ROLE=worker python worker.py
    REDIS_URL=redis://localhost:6379/0 ROLE=api PORT=8080 python main.py
    REDIS_URL=redis://localhost:6379/0 ROLE=api PORT=8081 python main.py
    # + nginx/haproxy balanceando 8080 y 8081

Nota de VRAM: cada proceso ROLE=api carga el modelo base completo en
GPU para inferir. El número de réplicas que puedes correr en una sola
GPU está limitado por VRAM (T5-base ~1GB + overhead), no es "gratis"
escalar procesos aunque no haya cuello de botella de gpu_lock.
"""
import os
from pathlib import Path

from infrastructure.persistence.csv_user_repository import CsvUserHistoryRepository
from infrastructure.nlp.phonetic_engine import PhoneticEngine
from infrastructure.nlp.context_judge import ContextJudge
from infrastructure.nlp.correction_pipeline import CorrectionPipeline
from application.use_cases.correct_text import CorrectTextUseCase
from application.use_cases.save_feedback import SaveFeedbackUseCase
from application.training_queue import TrainingWorker
from interfaces.api.routes import create_app
import types
import torch

import os
from main import build_app

# Aseguramos que el rol sea API para los hilos web

app = build_app()
if not hasattr(torch.distributed, 'tensor'):
    torch.distributed.tensor = types.ModuleType('tensor')

import os
os.environ["USE_LIBUV"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
from waitress import serve
T5_MODEL_DIR    = "./models/t5_correction"
USER_MODELS_DIR = "./models/users"
LORAS_DIR       = "./models/loras"

ROLE = os.environ.get("ROLE", "all")  # "api" | "worker" | "all"
os.environ["ROLE"] = "api"

def _build_redis_queue():
    from application.training_queue import RedisTrainingQueue
    try:
        return RedisTrainingQueue()
    except Exception as exc:
        raise RuntimeError(
            f"ROLE={ROLE} requiere Redis accesible vía REDIS_URL "
            f"(default redis://localhost:6379/0). Error: {exc}"
        ) from exc


def build_app():
    """
    Composition root para ROLE=api y ROLE=all. Para ROLE=worker usar
    run_worker() más abajo (sin Flask).
    """
    user_repo       = CsvUserHistoryRepository()
    phonetic_engine = PhoneticEngine()
    context_judge   = ContextJudge()

    t5_model     = None
    t5_tokenizer = None
    if Path(T5_MODEL_DIR).exists() and (Path(T5_MODEL_DIR) / "config.json").exists():
        try:
            from infrastructure.ml.t5_model import T5CorrectionModel, T5SpanishTokenizer
            t5_tokenizer = T5SpanishTokenizer(model_name=T5_MODEL_DIR)
            t5_model     = T5CorrectionModel(save_dir=T5_MODEL_DIR)
            print(f"[OK] Modelo T5 base cargado desde {T5_MODEL_DIR}")
        except Exception as exc:
            print(f"[WARN] No se pudo cargar T5 base: {exc}")

    pipeline = CorrectionPipeline(
        phonetic=phonetic_engine,
        judge=context_judge,
        seq2seq=t5_model,
        tokenizer=t5_tokenizer,
    )

    correct_uc = CorrectTextUseCase(
        pipeline=pipeline,
        user_repo=user_repo,
        user_models_dir=USER_MODELS_DIR,
        loras_dir=LORAS_DIR,
    )

    if t5_model and t5_tokenizer:
        correct_uc.set_base_model(t5_model, t5_tokenizer)

    if ROLE == "api":
        # Sin TrainingWorker local: solo empuja tareas a la cola Redis
        # compartida con el/los proceso(s) ROLE=worker.
        queue_backend = _build_redis_queue()
        feedback_uc = SaveFeedbackUseCase(
            user_repo=user_repo,
            training_worker=queue_backend,  # solo necesita .enqueue(), duck-typed
            pipeline=pipeline,
            correct_use_case=correct_uc,
        )
        print("[INFO] ROLE=api — inferencia únicamente, encolando a Redis.")
        return create_app(correct_uc, feedback_uc)

    # ROLE == "all" (comportamiento original: un solo proceso hace todo)
    training_worker = None
    try:
        from application.use_cases.train_model import TrainUserModelUseCase
        from infrastructure.ml.t5_model import T5SpanishTokenizer

        tokenizer_global = t5_tokenizer if t5_tokenizer else T5SpanishTokenizer()

        train_user_uc = TrainUserModelUseCase(
            user_repo=user_repo,
            tokenizer=tokenizer_global,
            base_model_dir=T5_MODEL_DIR,
            loras_dir=LORAS_DIR,
        )

        training_worker = TrainingWorker(
            train_fn=train_user_uc.execute,
            on_train_end=correct_uc.invalidate_user_cache,
        )

    except Exception as exc:
        print(f"[WARN] TrainingWorker desactivado: {exc}")

    feedback_uc = SaveFeedbackUseCase(
        user_repo=user_repo,
        training_worker=training_worker,
        pipeline=pipeline,
        correct_use_case=correct_uc,
    )

    return create_app(correct_uc, feedback_uc)


def run_worker():
    """
    Composition root para ROLE=worker: SIN Flask, SIN modelo de
    inferencia. Solo carga lo necesario para entrenar y consume la
    cola Redis compartida. Bloquea el proceso indefinidamente.
    """
    from application.use_cases.train_model import TrainUserModelUseCase
    from infrastructure.ml.t5_model import T5SpanishTokenizer

    user_repo = CsvUserHistoryRepository()
    tokenizer_global = T5SpanishTokenizer(model_name=T5_MODEL_DIR) \
        if (Path(T5_MODEL_DIR) / "config.json").exists() else T5SpanishTokenizer()

    train_user_uc = TrainUserModelUseCase(
        user_repo=user_repo,
        tokenizer=tokenizer_global,
        base_model_dir=T5_MODEL_DIR,
        loras_dir=LORAS_DIR,
    )

    queue_backend = _build_redis_queue()

    def _on_train_end(user_id: str) -> None:
        # No hay cache local que invalidar aquí: los procesos ROLE=api
        # leen el adaptador LoRA directo de disco en cada corrección
        # (ver generate_with_lora), así que no necesitan aviso.
        print(f"[🔄] '{user_id}' listo — los procesos api lo recogerán "
              f"solos en su próxima corrección (leen de disco).")

    worker = TrainingWorker(
        train_fn=train_user_uc.execute,
        on_train_end=_on_train_end,
        queue_backend=queue_backend,
    )
    print("[INFO] ROLE=worker — esperando tareas en Redis. Ctrl+C para salir.")
    worker._thread.join()


if __name__ == "__main__":
    if ROLE == "worker":
        run_worker()
    else:
        app = build_app()
        port = int(os.environ.get("PORT", 8080))
        print(f"[INFO] Servidor local corriendo en el puerto {port} (ROLE={ROLE})...")

        # Carga condicional: usa Waitress solo si está disponible (entorno local)
        try:
            from waitress import serve

            serve(app, host="0.0.0.0", port=port)
        except ImportError:
            # Si no está Waitress, arranca con el servidor de desarrollo de Flask
            app.run(host="0.0.0.0", port=port)
