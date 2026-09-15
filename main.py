"""
main.py - Composition Root  v5 (multi-proceso) → v6 (LoRA global)

Un solo proceso, un solo pipeline: reglas + BETO + T5 (base fusionado con el
LoRA gramatical global en models/t5_correction). No hay memoria, inferencia ni
entrenamiento por alumno; el feedback aceptado solo se persiste en CSV para
curación offline (ver train_grammar_lora.py).

Variables de entorno:
  PORT            puerto HTTP local (default 8080).
  ENABLE_T5       "false" para desactivar la capa T5 (default "true").
  MODEL_VERSION   identificador del modelo global que se devuelve en cada
                  respuesta como `modelVersion`. Si no se define, se compone
                  desde los manifiestos (`models/t5_correction/model-manifest.json`
                  + `models/grammar_lora/training-manifest.json`, formato
                  `<base-tag>@<hash8>+<lora-tag>@<hash8>`; ver
                  infrastructure/versioning.py) y, si no existen,
                  `global-lora-unversioned`.

Variables heredadas del lanzador (ROLE, ENABLE_USER_LORA, REDIS_URL) se
ignoran: ya no existen roles, colas ni adaptadores por alumno.
"""
import os
import sys
from pathlib import Path

from infrastructure.persistence.csv_user_repository import CsvUserHistoryRepository
from infrastructure.nlp.phonetic_engine import PhoneticEngine
from infrastructure.nlp.context_judge import ContextJudge
from infrastructure.nlp.correction_pipeline import CorrectionPipeline
from infrastructure.versioning import resolve_model_version
from application.use_cases.correct_text import CorrectTextUseCase
from application.use_cases.save_feedback import SaveFeedbackUseCase
from interfaces.api.routes import create_app
import types
import torch

# Estos parches son específicos de Windows (NCCL no está disponible ahí).
# En Linux (el entorno real de producción, ver docker-compose.yml) no hacen
# falta: NCCL sí está disponible y no queremos pisar su configuración.
if sys.platform.startswith("win"):
    if not hasattr(torch.distributed, 'tensor'):
        torch.distributed.tensor = types.ModuleType('tensor')

    os.environ["USE_LIBUV"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"

T5_MODEL_DIR = "./models/t5_correction"

# models/t5_correction contiene el modelo base + adaptador LoRA de concordancia
# (fusionado), reentrenado correctamente. Actúa como capa 4 de refinamiento
# gramatical (sujeto-verbo) sobre reglas+BETO, con guardas anti-alucinación en
# el pipeline. Se puede desactivar con ENABLE_T5=false si hiciera falta.
ENABLE_T5 = os.environ.get("ENABLE_T5", "true").lower() in ("1", "true", "yes")


def build_app():
    """Composition root: un único pipeline global compartido por todos los alumnos."""
    user_repo       = CsvUserHistoryRepository()
    phonetic_engine = PhoneticEngine()
    context_judge   = ContextJudge()

    t5_model     = None
    t5_tokenizer = None
    if not ENABLE_T5:
        print("[INFO] T5 desactivado (ENABLE_T5=false). Corrección vía reglas + BETO.")
    elif Path(T5_MODEL_DIR).exists() and (Path(T5_MODEL_DIR) / "config.json").exists():
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

    # Compartido con evaluate.py y correct_text.py: MODEL_VERSION (env) >
    # versión compuesta desde los manifiestos de models/ > default.
    model_version = resolve_model_version()
    print(f"[INFO] modelVersion = {model_version}")

    correct_uc  = CorrectTextUseCase(pipeline, model_version=model_version)
    feedback_uc = SaveFeedbackUseCase(user_repo)

    return create_app(correct_uc, feedback_uc)


# ── WSGI entrypoint para gunicorn ────────────────────────────────────────────
# gunicorn hace `import main` y busca la variable `app` a nivel de módulo
# (target: "main:app" en el Dockerfile/CMD). Ese import NO pasa por
# `if __name__ == "__main__"`, así que app debe construirse aquí.
# La guarda evita reconstruir `app` dos veces cuando el archivo se corre
# directo con `python main.py` (ese caso lo maneja el bloque de abajo).
if __name__ != "__main__":
    app = build_app()


if __name__ == "__main__":
    app = build_app()
    port = int(os.environ.get("PORT", 8080))
    print(f"[INFO] Servidor local corriendo en el puerto {port}...")

    # Carga condicional: usa Waitress solo si está disponible (entorno local)
    try:
        from waitress import serve

        serve(app, host="0.0.0.0", port=port)
    except ImportError:
        # Si no está Waitress, arranca con el servidor de desarrollo de Flask
        app.run(host="0.0.0.0", port=port)
