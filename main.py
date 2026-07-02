"""
main.py - Composition Root  v3 → v4 (LoRA)

Cambios:
  - TrainUserModelUseCase recibe loras_dir="./models/loras"
  - correct_uc.set_base_model() inyecta el modelo base para LoRA
  - USER_MODELS_DIR se mantiene solo por compatibilidad legacy
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


def build_app():
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


if __name__ == "__main__":
    app = build_app()
    print("[INFO] Servidor Waitress corriendo en el puerto 8080...")
    serve(app, host="0.0.0.0", port=8080)
