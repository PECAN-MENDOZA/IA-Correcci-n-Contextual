"""
scripts/prefetch_beto.py
Descarga y cachea BETO (dccuchile/bert-base-spanish-wwm-cased) dentro
de la imagen Docker durante el build. Antes esto era un
`RUN python -c "..."` multilínea en el Dockerfile, pero esa forma
(comillas abiertas en varias líneas) no la parsean bien todos los
builders (ej. Docker Desktop con BuildKit "bake" en Windows), así que
se separó a un archivo normal.
"""
from transformers import BertTokenizerFast, BertForMaskedLM

MODEL = "dccuchile/bert-base-spanish-wwm-cased"

BertTokenizerFast.from_pretrained(MODEL)
BertForMaskedLM.from_pretrained(MODEL, use_safetensors=False)
print("[OK] BETO descargado y guardado en imagen")
