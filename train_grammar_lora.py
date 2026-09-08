"""
train_grammar_lora.py
Entrena un adaptador LoRA de corrección gramatical/ortográfica sobre el base
LIMPIO `vgaraujov/t5-base-spanish` (NO el checkpoint divergido). Bajo consumo de
VRAM (solo el adaptador + gradient checkpointing), pensado para correr en la T4
compartida sin tumbar el servicio en vivo.

Guarda el adaptador en models/grammar_lora/ y hace un sanity-check al final.

Uso (dentro de un contenedor con torch+peft):
    python train_grammar_lora.py
Variables: EPOCHS, BATCH, ACCUM, LR, R por entorno (opcional).
"""
import csv
import os
import random

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig
from peft import LoraConfig, get_peft_model, TaskType

BASE      = "vgaraujov/t5-base-spanish"
CSV_PATH  = os.environ.get("CSV_PATH", "data/training_pairs_clean.csv")
OUT_DIR   = os.environ.get("OUT_DIR", "models/grammar_lora")
PREFIX    = "corrige: "
MAX_LEN   = 96
EPOCHS    = int(os.environ.get("EPOCHS", 3))
BATCH     = int(os.environ.get("BATCH", 8))
ACCUM     = int(os.environ.get("ACCUM", 2))     # batch efectivo = BATCH*ACCUM
LR        = float(os.environ.get("LR", 3e-4))
R         = int(os.environ.get("R", 16))
VAL_N     = 300

# Config de generación segura para T5 (igual que en producción)
GEN_CFG = GenerationConfig(decoder_start_token_id=0, eos_token_id=1,
                           pad_token_id=0, max_new_tokens=MAX_LEN)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] device={device}  epochs={EPOCHS} batch={BATCH}x{ACCUM} lr={LR} r={R}", flush=True)

tok   = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForSeq2SeqLM.from_pretrained(BASE, use_safetensors=False)
model.gradient_checkpointing_enable()
model.config.use_cache = False

peft_cfg = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=R, lora_alpha=2 * R,
                      lora_dropout=0.05, bias="none", target_modules=["q", "v"])
model = get_peft_model(model, peft_cfg)
model.to(device)
model.print_trainable_parameters()

# ── datos ────────────────────────────────────────────────────────────────────
pairs = [(r[0], r[1]) for r in csv.reader(open(CSV_PATH, encoding="utf-8"))][1:]
pairs = [p for p in pairs if len(p) == 2 and p[0] and p[1]]
random.seed(42)
random.shuffle(pairs)
val_pairs, train_pairs = pairs[:VAL_N], pairs[VAL_N:]
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


train_loader = DataLoader(PairDS(train_pairs), batch_size=BATCH, shuffle=True)
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
