"""
infrastructure/versioning.py — `modelVersion` del modelo global, compartido.

Única fuente de verdad para el identificador del modelo que devuelve el
servidor (`main.py`), el caso de uso (`application/use_cases/correct_text.py`)
y el evaluador (`evaluate.py`), de modo que el backend pueda trazar cada
corrección y cada informe al mismo modelo.

Precedencia: valor explícito (p. ej. `--model-version`) > `MODEL_VERSION`
(entorno) > manifiesto del LoRA global (`models/grammar_lora/
training-manifest.json`, y si no existe `manifest.json`, clave `modelVersion`)
> `DEFAULT_MODEL_VERSION`. Los manifiestos los genera el entrenamiento global;
aquí nunca se descarga ni se carga nada: solo librería estándar.
"""
import json
import os
import warnings
from pathlib import Path

DEFAULT_MODEL_VERSION = "global-lora-unversioned"
REPO_DIR = Path(__file__).resolve().parents[1]
MANIFEST_NAMES = ("training-manifest.json", "manifest.json")


def manifest_candidates(repo_dir: Path = REPO_DIR) -> list:
    """Rutas de manifiesto del LoRA global, en orden de preferencia."""
    lora_dir = Path(repo_dir) / "models" / "grammar_lora"
    return [lora_dir / name for name in MANIFEST_NAMES]


def _version_from_manifest(path: Path) -> str:
    """`modelVersion` del manifiesto o "" si no existe o no la trae; avisa si no se puede leer."""
    try:
        if not path.exists():
            return ""
        manifest = json.loads(path.read_text(encoding="utf-8"))
        return str(manifest.get("modelVersion", "")).strip() if isinstance(manifest, dict) else ""
    except Exception as exc:
        warnings.warn(f"No se pudo leer {path}: {exc}", RuntimeWarning, stacklevel=3)
        return ""


def resolve_model_version(explicit=None, env=None, repo_dir: Path = REPO_DIR) -> str:
    """
    `explicit` > `MODEL_VERSION` (env) > manifiesto del LoRA global > default.
    `env` por defecto es `os.environ`; `repo_dir` localiza `models/grammar_lora/`.
    """
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    env = os.environ if env is None else env
    from_env = str(env.get("MODEL_VERSION", "")).strip()
    if from_env:
        return from_env
    for path in manifest_candidates(repo_dir):
        from_manifest = _version_from_manifest(path)
        if from_manifest:
            return from_manifest
    return DEFAULT_MODEL_VERSION
