"""
main.py - Composition Root
Ensambla todas las capas DDD y arranca el servidor Flask.

Cambio respecto a la versión anterior:
  ANTES → SaveFeedbackUseCase recibía TrainUserModelUseCase directamente
          y lanzaba un thread crudo por feedback → crashes en GPU.

  AHORA → Se crea un TrainingWorker (hilo de fondo persistente) que
          recibe el train_fn y el on_train_end. SaveFeedbackUseCase
          solo llama worker.enqueue(user_id) y responde 204 al instante.
          El worker serializa el acceso a GPU con gpu_lock, que también
          adquiere generate_corrections() → nunca hay concurrencia.
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

T5_MODEL_DIR    = "./models/t5_correction"
USER_MODELS_DIR = "./models/users"


def build_app():
    # ── Infraestructura simbólica (siempre activa) ─────────────────────────────
    user_repo       = CsvUserHistoryRepository()
    phonetic_engine = PhoneticEngine()
    context_judge   = ContextJudge()

    # ── Modelo T5 base (opcional) ──────────────────────────────────────────────
    t5_model     = None
    t5_tokenizer = None
    if Path(T5_MODEL_DIR).exists() and (Path(T5_MODEL_DIR) / "config.json").exists():
        try:
            from infrastructure.ml.t5_model import T5CorrectionModel, T5SpanishTokenizer
            t5_tokenizer = T5SpanishTokenizer(model_name=T5_MODEL_DIR)
            t5_model     = T5CorrectionModel(save_dir=T5_MODEL_DIR)
            print(f"[OK] Modelo T5 base cargado desde {T5_MODEL_DIR}")
        except Exception as exc:
            print(f"[WARN] No se pudo cargar el modelo T5 base: {exc}")

    # ── Pipeline base (compartido entre usuarios sin modelo personalizado) ──────
    pipeline = CorrectionPipeline(
        phonetic=phonetic_engine,
        judge=context_judge,
        seq2seq=t5_model,
        tokenizer=t5_tokenizer,
    )

    # ── Caso de uso de corrección (con soporte de modelo por usuario) ──────────
    correct_uc = CorrectTextUseCase(
        pipeline=pipeline,
        user_repo=user_repo,
        user_models_dir=USER_MODELS_DIR,
    )

    # ── TrainingWorker: único consumidor de GPU para fine-tuning ───────────────
    training_worker = None
    try:
        from application.use_cases.train_model import TrainUserModelUseCase
        from infrastructure.ml.t5_model import T5SpanishTokenizer

        tokenizer_global = t5_tokenizer if t5_tokenizer else T5SpanishTokenizer()

        train_user_uc = TrainUserModelUseCase(
            user_repo=user_repo,
            tokenizer=tokenizer_global,
            base_model_dir=T5_MODEL_DIR,
        )

        # El worker arranca su hilo de fondo aquí.
        # on_train_end invalida el cache para que la siguiente request
        # del usuario cargue el modelo recién entrenado.
        training_worker = TrainingWorker(
            train_fn=train_user_uc.execute,
            on_train_end=correct_uc.invalidate_user_cache,
        )

    except Exception as exc:
        print(f"[WARN] TrainingWorker desactivado: {exc}")

    # ── Feedback: memoria simbólica + cola de fine-tuning ─────────────────────
    feedback_uc = SaveFeedbackUseCase(
        user_repo=user_repo,
        training_worker=training_worker,   # ← cola, no thread directo
        pipeline=pipeline,
        correct_use_case=correct_uc,
    )

    return create_app(correct_uc, feedback_uc)


if __name__ == "__main__":
    app = build_app()
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
