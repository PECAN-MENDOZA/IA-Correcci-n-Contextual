"""
setup_model.py — Descarga el T5 base en español y lo guarda en un directorio local.

Uso:
    python setup_model.py                       # → ./models/t5_base
    OUT=./models/otro python setup_model.py     # → directorio elegido
    MODEL_REVISION=<sha del commit HF> python setup_model.py   # revisión explícita (reproducible)

Necesita red (descarga de Hugging Face): NO fijar `HF_HUB_OFFLINE=1` antes de
ejecutarlo; se fija después, para las evaluaciones. `MODEL_REVISION` fija la
revisión del repositorio de HF (`revision=` de `from_pretrained`), de modo que
el baseline sea el mismo en cualquier máquina; sin ella se descarga `main`.

`models/t5_correction` es el checkpoint GLOBAL (base + LoRA gramatical
fusionado, el que sirve main.py y evalúa evaluate.py): este script se niega a
sobrescribirlo salvo que se pase explícitamente ALLOW_MODEL_OVERWRITE=true.

La descarga solo ocurre al ejecutar el script (no al importarlo): la guarda se
prueba en test_evaluate_metrics.py sin transformers.
"""
import os
import sys
from pathlib import Path

MODEL_NAME = "vgaraujov/t5-base-spanish"
REPO_DIR = Path(__file__).resolve().parent
DEFAULT_SAVE_DIR = "./models/t5_base"
PROTECTED_DIR = REPO_DIR / "models" / "t5_correction"


def resolve_save_dir(env=None) -> Path:
    """
    Directorio de salida (`OUT`, default ./models/t5_base, relativo al repo).
    Termina con error si resuelve al checkpoint global `models/t5_correction`
    y no se ha pasado ALLOW_MODEL_OVERWRITE=true.
    """
    env = os.environ if env is None else env
    out = str(env.get("OUT", "")).strip() or DEFAULT_SAVE_DIR
    save_dir = Path(out)
    if not save_dir.is_absolute():
        save_dir = REPO_DIR / save_dir
    allow = str(env.get("ALLOW_MODEL_OVERWRITE", "")).strip().lower() in ("1", "true", "yes")
    if save_dir.resolve() == PROTECTED_DIR.resolve() and not allow:
        sys.exit(
            f"[ERROR] {save_dir} es el checkpoint global (base + LoRA fusionado) y no se "
            f"sobrescribe. Usa OUT=<otro directorio> o, si de verdad quieres reemplazarlo, "
            f"ALLOW_MODEL_OVERWRITE=true."
        )
    return save_dir


def resolve_revision(env=None):
    """Revisión de HF pedida por `MODEL_REVISION` (o None: `main`)."""
    env = os.environ if env is None else env
    return str(env.get("MODEL_REVISION", "") or "").strip() or None


def main():
    save_dir = resolve_save_dir()
    revision = resolve_revision()

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    print(f"Descargando {MODEL_NAME} (revision={revision or 'main'})...")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, revision=revision)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=revision)

    print(f"Guardando modelo y tokenizador en {save_dir}...")
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    print("¡Listo! El modelo base ha sido guardado correctamente.")


if __name__ == "__main__":
    main()
