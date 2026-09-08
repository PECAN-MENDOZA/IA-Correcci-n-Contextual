from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from pathlib import Path

MODEL_NAME = "vgaraujov/t5-base-spanish"
SAVE_DIR = Path("./models/t5_correction")

print(f"Descargando {MODEL_NAME}...")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print(f"Guardando modelo y tokenizador en {SAVE_DIR}...")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print("¡Listo! El modelo base ha sido restaurado correctamente.")