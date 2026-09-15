"""
scripts/merge_grammar_lora.py
Fusiona el adaptador LoRA de concordancia (models/grammar_lora) con el base
LIMPIO vgaraujov/t5-base-spanish y lo guarda como modelo completo en
models/t5_correction (reemplaza el checkpoint divergido, con backup). Así el
pipeline lo carga como capa 4 sin descargas ni adaptadores en runtime.

Propaga el manifiesto: copia models/grammar_lora/training-manifest.json al
directorio temporal como model-manifest.json añadiendo el SHA-256 del
adaptador (comprobado contra el manifiesto), el SHA-256 del model.safetensors
fusionado, la hora de fusión y el `modelVersion` compuesto
`<base-tag>@<hash8>+<lora-tag>@<hash8>` que sirve `main.py`. Valida el
directorio temporal (archivos requeridos + manifiesto) antes del intercambio
con backup; si falla, deja el temporal para inspección y no toca OUT.

Ejecutar desde la raíz del repositorio con el entorno Python preparado:
    python scripts/merge_grammar_lora.py
Variables: BASE, ADAPTER, OUT, BASE_TAG (default beto-t5-base).
"""
import os
import shutil
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import torch.distributed.tensor  # noqa: E402  Expose torch.distributed.tensor for PEFT on Windows.
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig  # noqa: E402
from peft import PeftModel  # noqa: E402

from scripts.model_manifest import (  # noqa: E402
    BASE_TAG as DEFAULT_BASE_TAG,
    MODEL_MANIFEST_NAME,
    TRAINING_MANIFEST_NAME,
    merged_model_manifest,
    read_json,
    validate_merged_dir,
    write_json,
)

BASE     = os.environ.get("BASE", "vgaraujov/t5-base-spanish")
ADAPTER  = os.environ.get("ADAPTER", "models/grammar_lora")
OUT      = os.environ.get("OUT", "models/t5_correction")
BASE_TAG = os.environ.get("BASE_TAG", DEFAULT_BASE_TAG)
TMP      = OUT + "_new"

training_manifest_path = Path(ADAPTER) / TRAINING_MANIFEST_NAME
if not training_manifest_path.exists():
    sys.exit(f"[ERROR] falta {training_manifest_path}: entrena con train_grammar_lora.py o "
             f"reconstrúyelo con `python scripts/model_manifest.py --write-current`")
training = read_json(training_manifest_path)

print(f"[..] base={BASE} adapter={ADAPTER} -> {OUT}", flush=True)
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForSeq2SeqLM.from_pretrained(BASE, use_safetensors=False)
model = PeftModel.from_pretrained(model, ADAPTER).merge_and_unload()

if os.path.exists(TMP):
    shutil.rmtree(TMP)
os.makedirs(TMP, exist_ok=True)
model.save_pretrained(TMP, safe_serialization=True)
tok.save_pretrained(TMP)
GenerationConfig(decoder_start_token_id=0, eos_token_id=1,
                 pad_token_id=0, max_new_tokens=96).save_pretrained(TMP)

# ── manifiesto del modelo fusionado + validación del temporal ───────────────
manifest = merged_model_manifest(training, TMP, ADAPTER, base_tag=BASE_TAG)
write_json(Path(TMP) / MODEL_MANIFEST_NAME, manifest)
errors = validate_merged_dir(TMP)
if errors:
    sys.exit("[ERROR] el directorio fusionado no es válido, no se reemplaza "
             f"{OUT} (revisa {TMP}): " + "; ".join(errors))
print(f"[OK] manifiesto {MODEL_MANIFEST_NAME} escrito; modelVersion = {manifest['modelVersion']}", flush=True)

if os.path.exists(OUT):
    bak = OUT + "_broken_bak"
    if os.path.exists(bak):
        shutil.rmtree(bak)
    shutil.move(OUT, bak)
    print(f"[OK] checkpoint anterior respaldado en {bak}", flush=True)
shutil.move(TMP, OUT)
print(f"[OK] modelo fusionado guardado en {OUT}", flush=True)
