"""
scripts/merge_grammar_lora.py
Fusiona el adaptador LoRA de concordancia (models/grammar_lora_v2) con el base
LIMPIO vgaraujov/t5-base-spanish y lo guarda como modelo completo en
models/t5_correction (reemplaza el checkpoint divergido, con backup). Así el
pipeline lo carga como capa 4 sin descargas ni adaptadores en runtime.

Ejecutar dentro de un contenedor con torch+peft y el volumen ./models montado:
    python scripts/merge_grammar_lora.py
"""
import os
import shutil

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig
from peft import PeftModel

BASE    = os.environ.get("BASE", "vgaraujov/t5-base-spanish")
ADAPTER = os.environ.get("ADAPTER", "models/grammar_lora_v2")
OUT     = os.environ.get("OUT", "models/t5_correction")
TMP     = OUT + "_new"

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

if os.path.exists(OUT):
    bak = OUT + "_broken_bak"
    if os.path.exists(bak):
        shutil.rmtree(bak)
    shutil.move(OUT, bak)
    print(f"[OK] checkpoint anterior respaldado en {bak}", flush=True)
shutil.move(TMP, OUT)
print(f"[OK] modelo fusionado guardado en {OUT}", flush=True)
