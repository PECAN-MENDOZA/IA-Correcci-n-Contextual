"""
infrastructure/versioning.py — `modelVersion` del modelo global, compartido.

Única fuente de verdad para el identificador del modelo que devuelve el
servidor (`main.py`), el caso de uso (`application/use_cases/correct_text.py`)
y el evaluador (`evaluate.py`), de modo que el backend pueda trazar cada
corrección y cada informe al mismo modelo.

Formato compuesto: `<base-tag>@<hash8>+<lora-tag>@<hash8>`, donde el primer
hash son los 8 primeros hex del SHA-256 de `models/t5_correction/
model.safetensors` (los pesos T5 que realmente se sirven: base fusionado con
el LoRA) y el segundo los del `adapter_model.safetensors` del LoRA global.
Los hashes vienen de dos manifiestos que escribe `scripts/model_manifest.py`:
`models/t5_correction/model-manifest.json` (clave `model_sha256`, lo escribe
`scripts/merge_grammar_lora.py`) y `models/grammar_lora/training-manifest.json`
(clave `adapter_sha256`, lo escribe `train_grammar_lora.py`).

Precedencia: valor explícito (p. ej. `--model-version`) > `MODEL_VERSION`
(entorno) > versión compuesta (manifiesto del modelo fusionado + manifiesto
del LoRA; el del modelo fusionado ya embebe `adapter_sha256`, así que basta él
solo) > `modelVersion` del manifiesto del LoRA (`training-manifest.json`, y si
no existe `manifest.json`) > `DEFAULT_MODEL_VERSION`. Aquí nunca se descarga
ni se carga nada: solo librería estándar.
"""
import json
import os
import re
import warnings
from pathlib import Path

DEFAULT_MODEL_VERSION = "global-lora-unversioned"
REPO_DIR = Path(__file__).resolve().parents[1]
MANIFEST_NAMES = ("training-manifest.json", "manifest.json")
MODEL_MANIFEST_NAME = "model-manifest.json"

# Etiquetas por defecto de cada mitad de la versión compuesta; un manifiesto
# puede traer las suyas (`base_tag` / `lora_tag`).
BASE_TAG = "beto-t5-base"
LORA_TAG = "global-lora-v1"
HASH8 = 8
_HEX_RE = re.compile(r"^[0-9a-f]{8,64}$")


def manifest_candidates(repo_dir: Path = REPO_DIR) -> list:
    """Rutas de manifiesto del LoRA global, en orden de preferencia."""
    lora_dir = Path(repo_dir) / "models" / "grammar_lora"
    return [lora_dir / name for name in MANIFEST_NAMES]


def model_manifest_path(repo_dir: Path = REPO_DIR) -> Path:
    """Ruta del manifiesto del modelo fusionado (`models/t5_correction/model-manifest.json`)."""
    return Path(repo_dir) / "models" / "t5_correction" / MODEL_MANIFEST_NAME


def _hash8(manifest, key: str, what: str) -> str:
    """Primeros 8 hex del SHA-256 guardado en `manifest[key]`; ValueError si falta o no es hex."""
    if not isinstance(manifest, dict):
        raise ValueError(f"El manifiesto {what} no es un objeto JSON")
    value = str(manifest.get(key, "") or "").strip().lower()
    if not _HEX_RE.match(value):
        raise ValueError(f"El manifiesto {what} no trae un SHA-256 válido en '{key}'")
    return value[:HASH8]


def compose_model_version(base_manifest: dict, lora_manifest: dict,
                          base_tag: str = None, lora_tag: str = None) -> str:
    """
    `<base-tag>@<hash8>+<lora-tag>@<hash8>` a partir de `base_manifest["model_sha256"]`
    (pesos T5 fusionados servidos) y `lora_manifest["adapter_sha256"]` (adaptador
    LoRA). Las etiquetas salen de los argumentos, si no de los manifiestos
    (`base_tag` / `lora_tag`) y si no de `BASE_TAG` / `LORA_TAG`.
    """
    base_hash = _hash8(base_manifest, "model_sha256", "del modelo fusionado")
    lora_hash = _hash8(lora_manifest, "adapter_sha256", "del LoRA")
    base_tag = str(base_tag or base_manifest.get("base_tag") or BASE_TAG).strip()
    lora_tag = str(lora_tag or lora_manifest.get("lora_tag") or LORA_TAG).strip()
    return f"{base_tag}@{base_hash}+{lora_tag}@{lora_hash}"


def _read_manifest(path: Path):
    """Manifiesto JSON (dict) o None si no existe / no es un objeto; avisa si no se puede leer."""
    try:
        if not path.exists():
            return None
        manifest = json.loads(path.read_text(encoding="utf-8"))
        return manifest if isinstance(manifest, dict) else None
    except Exception as exc:
        warnings.warn(f"No se pudo leer {path}: {exc}", RuntimeWarning, stacklevel=3)
        return None


def _version_from_manifest(path: Path) -> str:
    """`modelVersion` del manifiesto o "" si no existe o no la trae; avisa si no se puede leer."""
    manifest = _read_manifest(path)
    return str(manifest.get("modelVersion", "")).strip() if manifest else ""


def _composed_version(repo_dir: Path) -> str:
    """
    Versión compuesta si existe el manifiesto del modelo fusionado. El LoRA se toma
    de `training-manifest.json` si existe; si no, del propio manifiesto fusionado
    (que embebe `adapter_sha256`). Avisa y devuelve "" si faltan los hashes.
    """
    base_path = model_manifest_path(repo_dir)
    base = _read_manifest(base_path)
    if base is None:
        return ""
    lora = _read_manifest(manifest_candidates(repo_dir)[0]) or base
    try:
        return compose_model_version(base, lora)
    except ValueError as exc:
        warnings.warn(f"No se pudo componer modelVersion desde {base_path}: {exc}", RuntimeWarning, stacklevel=3)
        return ""


def resolve_model_version(explicit=None, env=None, repo_dir: Path = REPO_DIR) -> str:
    """
    `explicit` > `MODEL_VERSION` (env) > versión compuesta (manifiesto del modelo
    fusionado + manifiesto del LoRA) > `modelVersion` del manifiesto del LoRA > default.
    `env` por defecto es `os.environ`; `repo_dir` localiza `models/`.
    """
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    env = os.environ if env is None else env
    from_env = str(env.get("MODEL_VERSION", "")).strip()
    if from_env:
        return from_env
    composed = _composed_version(Path(repo_dir))
    if composed:
        return composed
    for path in manifest_candidates(repo_dir):
        from_manifest = _version_from_manifest(path)
        if from_manifest:
            return from_manifest
    return DEFAULT_MODEL_VERSION
