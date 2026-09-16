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
(entorno) > versión compuesta (SOLO desde el manifiesto del modelo fusionado,
que embebe el `adapter_sha256` del adaptador que realmente se fusionó) >
`modelVersion` del manifiesto del LoRA (`training-manifest.json`, y si no
existe `manifest.json`) > `DEFAULT_MODEL_VERSION`.

Nunca se mezclan generaciones: si `training-manifest.json` describe un
adaptador distinto del fusionado (LoRA reentrenado pero todavía no fusionado),
se avisa y la versión sigue describiendo los pesos servidos. Un manifiesto
fusionado AUSENTE deja pasar a los siguientes escalones; uno PRESENTE pero
inválido (JSON ilegible, no es un objeto, sin `model_sha256` o sin
`adapter_sha256` válidos) falla cerrado: `resolve_model_version` lanza
`VersionResolutionError` en lugar de degradar a la versión parcial del LoRA o
al default, que nombrarían pesos que no se sirven. La versión compuesta solo
puede servirse si el T5 fusionado está cargado, y con el T5 cargado solo se
sirve una versión compuesta válida o `MODEL_VERSION` explícita
(`served_model_version`). Aquí nunca se descarga ni se carga nada: solo
librería estándar.
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
# Forma de la versión compuesta: `<base-tag>@<hash8>+<lora-tag>@<hash8>`.
_COMPOSED_RE = re.compile(r"^[^\s@+]+@[0-9a-f]{8}\+[^\s@+]+@[0-9a-f]{8}$")


class VersionResolutionError(ValueError):
    """El manifiesto del modelo fusionado existe pero no permite componer una versión fiable."""


def is_composed_version(value) -> bool:
    """True si `value` tiene la forma compuesta exacta `<tag>@<hash8>+<tag>@<hash8>`."""
    return bool(_COMPOSED_RE.match(str(value or "")))


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


def _read_merged_manifest(path: Path):
    """
    Manifiesto del modelo fusionado: None si NO existe (ausente); dict si es
    válido; `VersionResolutionError` si existe pero no se puede leer o no es
    un objeto JSON (presente pero inválido: nunca se degrada en silencio).
    """
    if not path.exists():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise VersionResolutionError(f"{path} existe pero no se puede leer: {exc}") from exc
    if not isinstance(manifest, dict):
        raise VersionResolutionError(f"{path} existe pero no es un objeto JSON")
    return manifest


def _composed_version(repo_dir: Path) -> str:
    """
    Versión compuesta si existe el manifiesto del modelo fusionado, SOLO a
    partir de sus propios `model_sha256` y `adapter_sha256` (el adaptador que
    de verdad está fusionado en los pesos servidos). Si además existe
    `training-manifest.json` y su `adapter_sha256` es otro (LoRA reentrenado y
    aún no fusionado), avisa "adaptador entrenado ≠ adaptador fusionado" y
    sigue usando el fusionado: nunca se mezclan generaciones. Devuelve "" solo
    si el manifiesto fusionado está AUSENTE; si está presente pero es ilegible
    o le falta algún hash lanza `VersionResolutionError` (fail-closed: el
    `modelVersion` parcial del LoRA describiría un adaptador que quizá no es
    el fusionado).
    """
    base_path = model_manifest_path(repo_dir)
    base = _read_merged_manifest(base_path)
    if base is None:
        return ""
    try:
        version = compose_model_version(base, base)
    except ValueError as exc:
        raise VersionResolutionError(
            f"{base_path} existe pero no permite componer modelVersion ({exc}); no se usa la versión "
            "parcial del LoRA ni el default. Regenera el manifiesto (scripts/model_manifest.py "
            "--write-current) o define MODEL_VERSION.") from exc
    training_path = manifest_candidates(repo_dir)[0]
    training = _read_manifest(training_path)
    if training is not None:
        trained = str(training.get("adapter_sha256", "") or "").strip().lower()
        merged = str(base.get("adapter_sha256", "") or "").strip().lower()
        if trained and trained != merged:
            warnings.warn(
                f"adaptador entrenado ≠ adaptador fusionado: {training_path} describe el adaptador "
                f"{trained[:HASH8]} pero {base_path} fusionó {merged[:HASH8]}; modelVersion describe "
                f"el fusionado ({version}). Fusiona el LoRA nuevo con scripts/merge_grammar_lora.py.",
                RuntimeWarning, stacklevel=3)
    return version


def served_model_version(t5_loaded: bool, explicit, composed: str) -> str:
    """
    Versión que puede anunciar el servidor. `explicit` es `MODEL_VERSION` (entorno)
    y `composed` la resuelta desde los manifiestos (`resolve_model_version`).
    Con una versión explícita se sirve esa (el operador asume la
    trazabilidad). Con el T5 fusionado cargado se sirve `composed` solo si es
    una versión compuesta válida (`<tag>@<hash8>+<tag>@<hash8>`): la parcial
    del LoRA (`global-lora-v1@…`) o el default (`global-lora-unversioned`) no
    describen los pesos fusionados y se rechazan. Sin T5 y sin versión
    explícita se rechaza: el pipeline servido (reglas + BETO) no es el que
    nombra la versión compuesta, ni el LoRA, ni el default `global-lora-*`.
    """
    explicit = str(explicit or "").strip()
    if explicit:
        return explicit
    composed = str(composed or "").strip()
    if t5_loaded:
        if is_composed_version(composed):
            return composed
        raise ValueError(
            f"el T5 fusionado está cargado pero no hay una versión compuesta válida que lo describa "
            f"(resuelta: '{composed}'): hace falta models/t5_correction/model-manifest.json con "
            "model_sha256 y adapter_sha256 (scripts/model_manifest.py --write-current) o MODEL_VERSION explícita.")
    raise ValueError(
        "el T5 fusionado no está cargado (ENABLE_T5=false o fallo de carga) y no hay MODEL_VERSION "
        f"explícita: la versión '{composed}' describiría un modelo que no se está sirviendo. "
        "Define MODEL_VERSION (p. ej. 'rules-beto-only') o corrige la carga del T5.")


def resolve_model_version(explicit=None, env=None, repo_dir: Path = REPO_DIR) -> str:
    """
    `explicit` > `MODEL_VERSION` (env) > versión compuesta (manifiesto del modelo
    fusionado) > `modelVersion` del manifiesto del LoRA > default. Los dos
    últimos escalones solo se alcanzan si el manifiesto fusionado está AUSENTE:
    si existe pero es inválido se lanza `VersionResolutionError` (ValueError).
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
