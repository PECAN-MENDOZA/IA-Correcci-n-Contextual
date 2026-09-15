"""
scripts/model_manifest.py — manifiestos reproducibles del LoRA gramatical global.

Dos artefactos, dos manifiestos (solo librería estándar, sin torch):

- `models/grammar_lora/training-manifest.json` (`training_manifest` +
  `adapter_fields`): lo escribe `train_grammar_lora.py` justo después de guardar
  el adaptador. Registra el modelo base (nombre y revisión de la caché de
  Hugging Face), el SHA-256 y el número de filas del CSV de entrenamiento, los
  índices exactos de la partición train/validación (semilla 42), todos los
  hiperparámetros (épocas, batch, acumulación, learning rate, rank, alpha,
  dropout, módulos objetivo, longitud máxima, prefijo), las versiones de las
  librerías, la hora UTC y el SHA-256 del `adapter_model.safetensors`
  (`adapter_sha256`). Su `modelVersion` es parcial: `<lora-tag>@<hash8>`.
- `models/t5_correction/model-manifest.json` (`merged_model_manifest`): lo
  escribe `scripts/merge_grammar_lora.py` en el directorio temporal antes del
  intercambio. Es una copia del manifiesto de entrenamiento más
  `adapter_sha256` (verificado contra el adaptador fusionado), `model_sha256`
  (SHA-256 del `model.safetensors` fusionado, los pesos que se sirven),
  `merged_at_utc` y el `modelVersion` compuesto
  `<base-tag>@<hash8>+<lora-tag>@<hash8>` (`compose_model_version`, en
  `infrastructure/versioning.py`, que es lo que lee el servidor).

Para los artefactos ya existentes (entrenados antes de que existieran los
manifiestos) `--write-current` solo calcula los hashes de los archivos que ya
están en `models/` y escribe ambos manifiestos con `"provenance":
"reconstructed"`; nunca entrena, fusiona ni descarga nada.

Uso (desde la raíz del repositorio):
    python scripts/model_manifest.py --write-current [--force] [--base-tag X] [--lora-tag Y]
    python scripts/model_manifest.py --show            # versión compuesta actual
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import random
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from infrastructure.versioning import (  # noqa: E402
    BASE_TAG,
    HASH8,
    LORA_TAG,
    MODEL_MANIFEST_NAME,
    compose_model_version,
    model_manifest_path,
    resolve_model_version,
)

__all__ = [
    "BASE_MODEL", "BASE_TAG", "LORA_TAG", "MODEL_MANIFEST_NAME", "REPO_DIR",
    "TRAINING_DEFAULTS", "TRAINING_MANIFEST_NAME", "adapter_fields",
    "compose_model_version", "hf_snapshot_revision", "library_versions",
    "load_pairs", "merged_model_manifest", "read_json",
    "reconstructed_training_manifest", "sha256_file", "split_indices",
    "training_manifest", "utc_now", "validate_merged_dir", "write_current",
    "write_json",
]

SCHEMA = "grammar-lora-manifest/1"
TRAINING_MANIFEST_NAME = "training-manifest.json"
BASE_MODEL = "vgaraujov/t5-base-spanish"
DEFAULT_DATASET = "data/training_pairs_clean.csv"
ADAPTER_WEIGHTS = "adapter_model.safetensors"
MERGED_WEIGHTS = "model.safetensors"
# Archivos que el pipeline exige en `models/t5_correction` (además del manifiesto).
MERGED_REQUIRED = ("config.json", "tokenizer_config.json", "generation_config.json", MERGED_WEIGHTS)
LIBRARIES = ("torch", "transformers", "peft", "safetensors")

# Hiperparámetros por defecto de `train_grammar_lora.py` (única fuente de verdad;
# el script los importa de aquí y admite override por entorno).
TRAINING_DEFAULTS = {
    "seed": 42,
    "epochs": 3,
    "batch": 8,
    "accumulation": 2,        # batch efectivo = batch * accumulation
    "learning_rate": 3e-4,
    "rank": 16,
    "lora_alpha": 32,         # 2 * rank
    "lora_dropout": 0.05,
    "target_modules": ["q", "v"],
    "max_length": 96,
    "validation_size": 300,
    "prefix": "corrige: ",
}


# ── utilidades ───────────────────────────────────────────────────────────────
def sha256_file(path) -> str:
    """SHA-256 (hex) de los bytes exactos del archivo, leído por bloques."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    """Hora UTC en ISO-8601 con sufijo Z y segundos enteros."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload: dict) -> Path:
    """Escribe JSON legible (UTF-8, claves en orden de inserción, salto final)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def load_pairs(csv_path) -> list:
    """Pares (errónea, corregida) con el mismo filtro que `train_grammar_lora.py`."""
    with Path(csv_path).open(encoding="utf-8", newline="") as fh:
        rows = [(r[0], r[1]) for r in csv.reader(fh) if len(r) >= 2][1:]
    return [p for p in rows if p[0] and p[1]]


def split_indices(n: int, seed: int = 42, validation_size: int = 300) -> tuple:
    """
    (índices_train, índices_validación) equivalentes a `random.seed(seed);
    random.shuffle(pares); val, train = pares[:validation_size], pares[validation_size:]`:
    `shuffle` solo depende de la longitud y de la semilla, así que barajar
    `range(n)` produce exactamente la misma permutación.
    """
    order = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(order)
    return order[validation_size:], order[:validation_size]


def library_versions() -> dict:
    """Versiones instaladas (sin importar las librerías); None si no están."""
    versions = {"python": platform.python_version()}
    for name in LIBRARIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def hf_snapshot_revision(model_name: str = BASE_MODEL, hf_home=None):
    """Revisión (`refs/main`) del modelo en la caché de Hugging Face, o None."""
    hf_home = Path(hf_home or os.environ.get("HF_HOME") or (Path.home() / ".cache" / "huggingface"))
    ref = hf_home / "hub" / ("models--" + model_name.replace("/", "--")) / "refs" / "main"
    try:
        return ref.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


# ── manifiesto de entrenamiento ──────────────────────────────────────────────
def training_manifest(dataset_path, parameters: dict, *, base_model: str = BASE_MODEL,
                      base_model_revision=None, provenance: str = "trained",
                      libraries: dict = None, now: str = None) -> dict:
    """
    Manifiesto del entrenamiento del LoRA global: dataset (ruta, SHA-256, filas),
    partición exacta, hiperparámetros (`parameters` sobre `TRAINING_DEFAULTS`),
    modelo base, librerías y hora UTC. El adaptador se añade con `adapter_fields`.
    """
    dataset_path = Path(dataset_path)
    params = dict(TRAINING_DEFAULTS)
    params.update(parameters or {})
    pairs = load_pairs(dataset_path)
    train_idx, val_idx = split_indices(len(pairs), params["seed"], params["validation_size"])
    try:
        shown_path = dataset_path.resolve().relative_to(REPO_DIR).as_posix()
    except ValueError:
        shown_path = dataset_path.as_posix()
    dataset_hash = sha256_file(dataset_path)
    return {
        "schema": SCHEMA,
        "provenance": provenance,
        "created_at_utc": now or utc_now(),
        "base_model": base_model,
        "base_model_revision": base_model_revision,
        "lora_tag": LORA_TAG,
        "dataset_sha256": dataset_hash,
        "dataset": {
            "path": shown_path,
            "sha256": dataset_hash,
            "rows": len(pairs),
            "train_rows": len(train_idx),
            "validation_rows": len(val_idx),
        },
        "parameters": params,
        "split": {
            "method": "random.seed(seed); random.shuffle(pares); validación = primeros validation_size",
            "seed": params["seed"],
            "validation_indices": val_idx,
            "train_indices": train_idx,
        },
        "libraries": libraries if libraries is not None else library_versions(),
    }


def adapter_fields(adapter_dir, lora_tag: str = LORA_TAG) -> dict:
    """
    Hash del adaptador guardado: `adapter_sha256` (de `adapter_model.safetensors`),
    hashes de todos los archivos, resumen de `adapter_config.json` y el
    `modelVersion` parcial `<lora-tag>@<hash8>`.
    """
    adapter_dir = Path(adapter_dir)
    weights = adapter_dir / ADAPTER_WEIGHTS
    if not weights.exists():
        raise FileNotFoundError(f"No existe {weights}")
    files = {p.name: sha256_file(p) for p in sorted(adapter_dir.iterdir()) if p.is_file()}
    config_path = adapter_dir / "adapter_config.json"
    config = read_json(config_path) if config_path.exists() else {}
    keys = ("base_model_name_or_path", "r", "lora_alpha", "lora_dropout", "target_modules",
            "peft_version", "task_type", "bias")
    adapter_hash = files[ADAPTER_WEIGHTS]
    try:
        shown_dir = adapter_dir.resolve().relative_to(REPO_DIR).as_posix()
    except ValueError:
        shown_dir = adapter_dir.as_posix()
    return {
        "lora_tag": lora_tag,
        "adapter_sha256": adapter_hash,
        "adapter": {"dir": shown_dir, "files": files, "config": {k: config[k] for k in keys if k in config}},
        "modelVersion": f"{lora_tag}@{adapter_hash[:HASH8]}",
    }


# ── manifiesto del modelo fusionado ──────────────────────────────────────────
def merged_model_manifest(training: dict, merged_dir, adapter_dir, *, base_tag: str = BASE_TAG,
                          now: str = None) -> dict:
    """
    Copia del manifiesto de entrenamiento + `adapter_sha256` (recalculado sobre el
    adaptador fusionado y comprobado contra el manifiesto), `model_sha256` del
    `model.safetensors` fusionado, hashes de los archivos fusionados,
    `merged_at_utc`, `base_tag` y el `modelVersion` compuesto. No muta `training`.
    """
    merged_dir = Path(merged_dir)
    weights = merged_dir / MERGED_WEIGHTS
    if not weights.exists():
        raise FileNotFoundError(f"No existe {weights}")
    adapter_hash = sha256_file(Path(adapter_dir) / ADAPTER_WEIGHTS)
    recorded = str(training.get("adapter_sha256") or "")
    if recorded and recorded != adapter_hash:
        raise ValueError(
            f"El adaptador {adapter_dir} ({adapter_hash[:HASH8]}) no es el del manifiesto de "
            f"entrenamiento ({recorded[:HASH8]}); reentrena o regenera el manifiesto")
    manifest = dict(training)
    manifest["adapter_sha256"] = adapter_hash
    manifest["lora_tag"] = manifest.get("lora_tag") or LORA_TAG
    manifest["base_tag"] = base_tag
    manifest["merged_at_utc"] = now or utc_now()
    manifest["merged_files"] = {p.name: sha256_file(p) for p in sorted(merged_dir.iterdir())
                                if p.is_file() and p.name != MODEL_MANIFEST_NAME}
    manifest["model_sha256"] = manifest["merged_files"][MERGED_WEIGHTS]
    manifest["modelVersion"] = compose_model_version(manifest, manifest)
    return manifest


def validate_merged_dir(merged_dir) -> list:
    """
    Errores (lista vacía = válido) del directorio fusionado antes del intercambio:
    archivos requeridos, manifiesto JSON legible con `modelVersion` y `model_sha256`
    igual al hash real de `model.safetensors`.
    """
    merged_dir = Path(merged_dir)
    errors = [f"falta {name}" for name in MERGED_REQUIRED if not (merged_dir / name).is_file()]
    manifest_path = merged_dir / MODEL_MANIFEST_NAME
    if not manifest_path.is_file():
        errors.append(f"falta {MODEL_MANIFEST_NAME}")
        return errors
    try:
        manifest = read_json(manifest_path)
        if not isinstance(manifest, dict):
            raise ValueError("no es un objeto JSON")
    except Exception as exc:
        errors.append(f"{MODEL_MANIFEST_NAME} ilegible: {exc}")
        return errors
    if not str(manifest.get("modelVersion", "")).strip():
        errors.append(f"{MODEL_MANIFEST_NAME} sin modelVersion")
    weights = merged_dir / MERGED_WEIGHTS
    if weights.is_file() and manifest.get("model_sha256") != sha256_file(weights):
        errors.append(f"{MODEL_MANIFEST_NAME}: model_sha256 no coincide con {MERGED_WEIGHTS}")
    return errors


# ── reconstrucción para artefactos ya existentes ─────────────────────────────
def reconstructed_training_manifest(repo_dir=REPO_DIR, *, lora_tag: str = LORA_TAG, now: str = None) -> dict:
    """
    Manifiesto de entrenamiento con `provenance: "reconstructed"` para un adaptador
    entrenado antes de que existieran los manifiestos: parámetros conocidos de
    `train_grammar_lora.py` (`TRAINING_DEFAULTS`) corregidos con lo que diga
    `adapter_config.json` (rank, alpha, dropout, módulos, base, versión de PEFT),
    dataset y partición si `data/training_pairs_clean.csv` sigue en el repo, y el
    hash real del adaptador. Solo lee archivos; no entrena nada.
    """
    repo_dir = Path(repo_dir)
    adapter_dir = repo_dir / "models" / "grammar_lora"
    fields = adapter_fields(adapter_dir, lora_tag=lora_tag)
    config = fields["adapter"]["config"]
    params = dict(TRAINING_DEFAULTS)
    for src, dst in (("r", "rank"), ("lora_alpha", "lora_alpha"), ("lora_dropout", "lora_dropout"),
                     ("target_modules", "target_modules")):
        if src in config:
            params[dst] = config[src]
    base_model = config.get("base_model_name_or_path") or BASE_MODEL
    libraries = {"python": None, "torch": None, "transformers": None, "peft": config.get("peft_version"),
                 "safetensors": None}
    merged_config = repo_dir / "models" / "t5_correction" / "config.json"
    if merged_config.exists():
        try:
            libraries["transformers"] = read_json(merged_config).get("transformers_version")
        except Exception:
            pass
    dataset_path = repo_dir / DEFAULT_DATASET
    if dataset_path.exists():
        manifest = training_manifest(
            dataset_path, params, base_model=base_model, provenance="reconstructed", libraries=libraries,
            base_model_revision=hf_snapshot_revision(base_model, repo_dir / "models" / "hf-cache"), now=now)
    else:
        manifest = {
            "schema": SCHEMA, "provenance": "reconstructed", "created_at_utc": now or utc_now(),
            "base_model": base_model,
            "base_model_revision": hf_snapshot_revision(base_model, repo_dir / "models" / "hf-cache"),
            "lora_tag": lora_tag, "dataset_sha256": None,
            "dataset": {"path": DEFAULT_DATASET, "sha256": None, "rows": None,
                        "train_rows": None, "validation_rows": None},
            "parameters": params, "split": None, "libraries": libraries,
        }
    manifest["note"] = ("Reconstruido a partir de los artefactos existentes (adapter_config.json y "
                        "constantes de train_grammar_lora.py); el adaptador no se reentrenó.")
    manifest.update(fields)
    return manifest


def write_current(repo_dir=REPO_DIR, *, base_tag: str = BASE_TAG, lora_tag: str = LORA_TAG,
                  force: bool = False, now: str = None) -> dict:
    """
    Escribe `models/t5_correction/model-manifest.json` (siempre) y, si no existe
    o `force`, `models/grammar_lora/training-manifest.json` reconstruido, solo con
    hashes de los archivos existentes. Un manifiesto de entrenamiento existente
    cuyo `adapter_sha256` no sea el del adaptador actual es un error (salvo
    `force`, que lo reconstruye). Devuelve rutas y el `modelVersion` compuesto.
    """
    repo_dir = Path(repo_dir)
    adapter_dir = repo_dir / "models" / "grammar_lora"
    merged_dir = repo_dir / "models" / "t5_correction"
    for required in (adapter_dir / ADAPTER_WEIGHTS, merged_dir / MERGED_WEIGHTS):
        if not required.exists():
            raise FileNotFoundError(f"No existe {required}: no hay artefactos que versionar")
    lora_path = adapter_dir / TRAINING_MANIFEST_NAME
    training = None
    if lora_path.exists() and not force:
        training = read_json(lora_path)
        current = sha256_file(adapter_dir / ADAPTER_WEIGHTS)
        recorded = str(training.get("adapter_sha256") or "")
        if recorded != current:
            raise ValueError(
                f"{lora_path} describe otro adaptador ({recorded[:HASH8] or '?'} != {current[:HASH8]}); "
                "usa --force para reconstruirlo")
    if training is None:
        training = reconstructed_training_manifest(repo_dir, lora_tag=lora_tag, now=now)
        write_json(lora_path, training)
    merged = merged_model_manifest(training, merged_dir, adapter_dir, base_tag=base_tag, now=now)
    base_path = write_json(model_manifest_path(repo_dir), merged)
    return {"training_manifest": str(lora_path), "model_manifest": str(base_path),
            "modelVersion": merged["modelVersion"]}


# ── CLI ──────────────────────────────────────────────────────────────────────
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write-current", action="store_true",
                        help="escribe los manifiestos de los artefactos existentes en models/ (solo hashes)")
    parser.add_argument("--show", action="store_true", help="muestra el modelVersion que resolvería el servidor")
    parser.add_argument("--force", action="store_true",
                        help="con --write-current: reconstruye training-manifest.json aunque exista")
    parser.add_argument("--base-tag", default=BASE_TAG, help=f"etiqueta del modelo fusionado (default {BASE_TAG})")
    parser.add_argument("--lora-tag", default=LORA_TAG, help=f"etiqueta del LoRA (default {LORA_TAG})")
    parser.add_argument("--repo-dir", default=str(REPO_DIR), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    repo_dir = Path(args.repo_dir)
    if args.write_current:
        try:
            result = write_current(repo_dir, base_tag=args.base_tag, lora_tag=args.lora_tag, force=args.force)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[ERROR] {exc}")
            return 1
        print(f"[OK] {result['training_manifest']}")
        print(f"[OK] {result['model_manifest']}")
        print(f"[OK] modelVersion = {result['modelVersion']}")
        return 0
    if args.show:
        print(resolve_model_version(env={}, repo_dir=repo_dir))
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
