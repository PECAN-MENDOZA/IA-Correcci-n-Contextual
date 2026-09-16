"""
train_grammar_lora.py
Entrena un adaptador LoRA de corrección gramatical/ortográfica sobre el base
LIMPIO `vgaraujov/t5-base-spanish` (NO el checkpoint divergido). Bajo consumo de
VRAM (solo el adaptador + gradient checkpointing), pensado para correr en la T4
compartida sin tumbar el servicio en vivo.

Guarda el adaptador en models/grammar_lora/ junto con `training-manifest.json`
(scripts/model_manifest.py: SHA-256 del dataset, partición exacta, hiperparámetros,
versiones de librerías, hora UTC y SHA-256 del adaptador) y hace un sanity-check
al final. Después, `scripts/merge_grammar_lora.py` fusiona el adaptador y
propaga el manifiesto a models/t5_correction/model-manifest.json.

Reproducibilidad: `configure_seeds(SEED)` siembra `random`, `torch` y CUDA
ANTES de cargar el base y de `get_peft_model` (la inicialización aleatoria del
adaptador queda sembrada), el DataLoader baraja con un `torch.Generator`
sembrado y el manifiesto registra las banderas deterministas efectivas
(`deterministic`).

Uso (dentro de un contenedor con torch+peft):
    python train_grammar_lora.py
Variables: EPOCHS, BATCH, ACCUM, LR, R, SEED por entorno (opcional); los valores
por defecto viven en `scripts.model_manifest.TRAINING_DEFAULTS`.
"""
import os
import random

import torch
import torch.distributed.tensor  # Expose torch.distributed.tensor for PEFT on Windows.
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig
from peft import LoraConfig, get_peft_model, TaskType

from scripts.model_manifest import (
    BASE_MODEL,
    TRAINING_DEFAULTS as D,
    TRAINING_MANIFEST_NAME,
    adapter_fields,
    hf_snapshot_revision,
    load_pairs,
    split_indices,
    training_manifest,
    write_json,
)

BASE      = BASE_MODEL
CSV_PATH  = os.environ.get("CSV_PATH", "data/training_pairs_clean.csv")
OUT_DIR   = os.environ.get("OUT_DIR", "models/grammar_lora")
PREFIX    = D["prefix"]
MAX_LEN   = D["max_length"]
SEED      = int(os.environ.get("SEED", D["seed"]))
EPOCHS    = int(os.environ.get("EPOCHS", D["epochs"]))
BATCH     = int(os.environ.get("BATCH", D["batch"]))
ACCUM     = int(os.environ.get("ACCUM", D["accumulation"]))     # batch efectivo = BATCH*ACCUM
LR        = float(os.environ.get("LR", D["learning_rate"]))
R         = int(os.environ.get("R", D["rank"]))
LORA_DROPOUT   = D["lora_dropout"]
TARGET_MODULES = list(D["target_modules"])
VAL_N     = D["validation_size"]

# Hiperparámetros efectivos: van tal cual al manifiesto de entrenamiento.
PARAMETERS = {
    "seed": SEED, "epochs": EPOCHS, "batch": BATCH, "accumulation": ACCUM,
    "learning_rate": LR, "rank": R, "lora_alpha": 2 * R, "lora_dropout": LORA_DROPOUT,
    "target_modules": TARGET_MODULES, "max_length": MAX_LEN, "validation_size": VAL_N,
    "prefix": PREFIX,
}

# Config de generación segura para T5 (igual que en producción)
GEN_CFG = GenerationConfig(decoder_start_token_id=0, eos_token_id=1,
                           pad_token_id=0, max_new_tokens=MAX_LEN)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] device={device}  seed={SEED} epochs={EPOCHS} batch={BATCH}x{ACCUM} lr={LR} r={R}", flush=True)


def configure_seeds(seed: int) -> dict:
    """
    Siembra toda la aleatoriedad efectiva ANTES de crear el modelo y el
    adaptador (la inicialización de LoRA usa el generador global de torch) y
    fija cuDNN en modo determinista. Devuelve las banderas para el manifiesto.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return {
        "seed": seed,
        "python_random": True,
        "torch_manual_seed": True,
        "cuda_manual_seed_all": True,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "use_deterministic_algorithms": False,       # no se fuerza: algunos kernels CUDA lo rechazan
        "dataloader_generator_seed": seed,
        "seeded_before_model_and_peft_init": True,
    }


DETERMINISTIC = configure_seeds(SEED)

tok   = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForSeq2SeqLM.from_pretrained(BASE, use_safetensors=False)
model.gradient_checkpointing_enable()
model.config.use_cache = False

peft_cfg = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=R, lora_alpha=2 * R,
                      lora_dropout=LORA_DROPOUT, bias="none", target_modules=TARGET_MODULES)
model = get_peft_model(model, peft_cfg)
model.to(device)
model.print_trainable_parameters()

# ── datos ────────────────────────────────────────────────────────────────────
# Partición por índices (misma permutación que `random.seed(SEED); random.shuffle(pairs)`),
# así el manifiesto registra exactamente qué filas fueron de validación.
pairs = load_pairs(CSV_PATH)
train_idx, val_idx = split_indices(len(pairs), SEED, VAL_N)
val_pairs, train_pairs = [pairs[i] for i in val_idx], [pairs[i] for i in train_idx]
print(f"[INFO] train={len(train_pairs)} val={len(val_pairs)}", flush=True)


class PairDS(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        err, cor = self.data[i]
        x = tok(PREFIX + err, max_length=MAX_LEN, truncation=True,
                padding="max_length", return_tensors="pt")
        y = tok(text_target=cor, max_length=MAX_LEN, truncation=True,
                padding="max_length", return_tensors="pt")
        labels = y.input_ids.squeeze(0)
        labels[labels == tok.pad_token_id] = -100
        return {"input_ids": x.input_ids.squeeze(0),
                "attention_mask": x.attention_mask.squeeze(0),
                "labels": labels}


# El barajado de cada época sale de un generador sembrado propio (no del
# global, que ya consumieron la inicialización del adaptador y el dropout).
loader_generator = torch.Generator().manual_seed(SEED)
train_loader = DataLoader(PairDS(train_pairs), batch_size=BATCH, shuffle=True, generator=loader_generator)
val_loader   = DataLoader(PairDS(val_pairs), batch_size=BATCH)
opt = AdamW(model.parameters(), lr=LR, weight_decay=0.01)


def val_loss():
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for b in val_loader:
            b = {k: v.to(device) for k, v in b.items()}
            tot += model(**b).loss.item()
            n += 1
    model.train()
    return tot / max(n, 1)


# ── entrenamiento ────────────────────────────────────────────────────────────
model.train()
for ep in range(EPOCHS):
    running = 0.0
    opt.zero_grad()
    for i, b in enumerate(train_loader):
        b = {k: v.to(device) for k, v in b.items()}
        loss = model(**b).loss
        (loss / ACCUM).backward()
        running += loss.item()
        if (i + 1) % ACCUM == 0:
            opt.step()
            opt.zero_grad()
        if (i + 1) % 300 == 0:
            print(f"  ep{ep+1} paso {i+1}/{len(train_loader)} loss={running/(i+1):.4f}", flush=True)
    print(f"[EPOCH {ep+1}] train_loss={running/len(train_loader):.4f} "
          f"val_loss={val_loss():.4f}", flush=True)

os.makedirs(OUT_DIR, exist_ok=True)
model.save_pretrained(OUT_DIR)
print(f"[OK] adaptador guardado en {OUT_DIR}", flush=True)

# ── manifiesto de entrenamiento (junto al adaptador, tras guardarlo) ─────────
manifest = training_manifest(CSV_PATH, PARAMETERS, base_model=BASE,
                             base_model_revision=hf_snapshot_revision(BASE))
manifest["deterministic"] = DETERMINISTIC        # semillas y banderas efectivas
manifest.update(adapter_fields(OUT_DIR))          # adapter_sha256 + modelVersion parcial
manifest_path = write_json(os.path.join(OUT_DIR, TRAINING_MANIFEST_NAME), manifest)
print(f"[OK] manifiesto escrito en {manifest_path} (dataset {manifest['dataset_sha256'][:8]}, "
      f"adaptador {manifest['adapter_sha256'][:8]})", flush=True)

# ── sanity check ─────────────────────────────────────────────────────────────
model.eval()
tests = [
    "los perros que vi fue muy grandes",
    "las casas que compré era muy caras",
    "los niños del colegio está cansados",
    "mis amigos no vino a la fiesta",
    "los niños juegan en el parque",          # ya correcto: no debe cambiar
    "habian muchos autos en la calle",
]
print("\n=== SANITY CHECK ===", flush=True)
for t in tests:
    x = tok(PREFIX + t, return_tensors="pt").to(device)
    with torch.no_grad():
        g = model.generate(input_ids=x.input_ids, attention_mask=x.attention_mask,
                           generation_config=GEN_CFG, num_beams=4, do_sample=False)
    print(f"  IN : {t}\n  OUT: {tok.decode(g[0], skip_special_tokens=True)}\n", flush=True)
