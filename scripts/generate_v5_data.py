"""
scripts/generate_v5_data.py
Datos de entrenamiento de LoRA v5 (escritura infantil).

1. Frases LIMPIAS de data/v5/clean_<fuente>.txt (scripts/build_v5_clean.py): Claude y dominio
   público completas con dos corrupciones distintas cada una (registro escolar: más peso);
   muestras de COWS-L2H (12 000) y subtítulos filtrados (4 000).
2. Corrupción infantil (scripts/corrupt_child.py; 0-6 errores por frase).
3. ENTRADA REAL DEL T5: cada frase corrompida pasa por las capas 0-3 del pipeline (reglas +
   BETO, sin T5) y se guarda el texto que el pipeline le pasaría al T5 (auditoría C: el T5
   no ve el texto crudo, ve el residuo de las capas anteriores).
4. Dos CSV de entrenamiento con distinta proporción de pares idénticos (entrada == objetivo:
   enseñan a no tocar lo correcto, pero en v3 un 7 % hizo que T5 dejara "se callo"):
     data/training_pairs_v5a.csv  ~15 % de identidad
     data/training_pairs_v5b.csv  ~5 % de identidad
   más los bloques dirigidos de v2-v4: concordancia (generador corregido,
   training_pairs_agreement_v5.csv), subjuntivo (training_pairs_subjunctive.csv) y controles
   de indicativo/prospectivo/predicado (training_pairs_v4_extra.csv).
5. Ningún par cuyo objetivo coincida (normalizado) con una frase de un set de evaluación.

Traza completa (entrada cruda, entrada al T5, objetivo, fuente, operaciones):
data/v5/pairs_core.csv. Semilla 2026. Uso (GPU, ~30-60 min):
    HF_HOME=models/hf-cache HF_HUB_OFFLINE=1 python scripts/generate_v5_data.py
"""
import csv
import io
import contextlib
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import corrupt_child as cc                                   # noqa: E402
from build_v5_clean import evaluation_sentences, norm          # noqa: E402

SEED = 2026
V5 = REPO / "data" / "v5"
SOURCES = [  # (fuente, muestra o None = todas, corrupciones por frase)
    ("claude", None, 2), ("dominio_publico", None, 2), ("cowsl2h", 12000, 1), ("subtitulos", 4000, 1),
]
TARGETED = ["training_pairs_agreement_v5.csv", "training_pairs_subjunctive.csv", "training_pairs_v4_extra.csv"]
IDENTITY_SHARE = {"v5a": 0.15, "v5b": 0.05}


class _RecordT5:
    """Seq2seq falso: guarda el texto que el pipeline le pasaría al T5 y no corrige nada."""
    def __init__(self):
        self.last = None

    def generate_corrections(self, text, tokenizer, num_returns=2):
        self.last = text
        return []


def t5_input(pipeline, recorder, text: str) -> str:
    recorder.last = None
    with contextlib.redirect_stdout(io.StringIO()):
        out = pipeline.correct(text, {})
    return recorder.last if recorder.last is not None else out[0]


def read_pairs(path: Path) -> list:
    with open(path, encoding="utf-8") as fh:
        return [(r[0].strip(), r[1].strip()) for r in list(csv.reader(fh))[1:] if len(r) >= 2]


def main() -> None:
    from infrastructure.nlp.phonetic_engine import PhoneticEngine
    from infrastructure.nlp.context_judge import ContextJudge
    from infrastructure.nlp.correction_pipeline import CorrectionPipeline

    rng = random.Random(SEED)
    pe = PhoneticEngine()
    cc.set_lexicon({w for w, f in pe.word_freqs.items() if f >= 1000})
    recorder = _RecordT5()
    pipeline = CorrectionPipeline(pe, ContextJudge(), recorder, object())
    excluded = evaluation_sentences()

    core = []
    for source, sample, copies in SOURCES:
        path = V5 / f"clean_{source}.txt"
        lines = [l.strip() for l in open(path, encoding="utf-8") if l.strip()] if path.exists() else []
        if sample is not None and len(lines) > sample:
            lines = rng.sample(lines, sample)
        for i, clean in enumerate(lines):
            for _ in range(copies):
                raw, target, ops = cc.corrupt(clean, rng)
                try:
                    inp = t5_input(pipeline, recorder, raw)
                except Exception as e:                          # una frase rara no tumba la corrida
                    print(f"[WARN] {source}: {e}")
                    continue
                core.append({"erronea": inp, "corregida": target, "cruda": raw, "fuente": source,
                             "ops": "+".join(ops)})
            if i % 1000 == 0:
                print(f"[{source}] {i}/{len(lines)}", flush=True)

    with open(V5 / "pairs_core.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["erronea", "corregida", "cruda", "fuente", "ops"])
        w.writeheader()
        w.writerows(core)
    identity = [p for p in core if p["erronea"] == p["corregida"]]
    changed = [p for p in core if p["erronea"] != p["corregida"]]
    print(f"[core] {len(core)} pares: {len(changed)} con cambios, {len(identity)} idénticos")

    targeted = []
    for name in TARGETED:
        pairs = [(e, c) for e, c in read_pairs(REPO / "data" / name) if e != c]
        kept = [(e, c) for e, c in pairs if norm(c) not in excluded and norm(e) not in excluded]
        print(f"[{name}] {len(pairs)} -> {len(kept)} (fuera {len(pairs) - len(kept)} que coinciden con evaluación)")
        targeted.extend(kept)

    for variant, share in IDENTITY_SHARE.items():
        vrng = random.Random(f"{SEED}-{variant}")
        base = [(p["erronea"], p["corregida"]) for p in changed] + targeted
        n_identity = min(len(identity), round(share * len(base) / (1 - share)))
        pairs = base + [(p["erronea"], p["corregida"]) for p in vrng.sample(identity, n_identity)]
        seen, final = set(), []
        for pair in pairs:
            if pair not in seen:
                seen.add(pair)
                final.append(pair)
        vrng.shuffle(final)
        out = REPO / "data" / f"training_pairs_{variant}.csv"
        with open(out, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["erronea", "corregida"])
            w.writerows(final)
        ident = sum(1 for e, c in final if e == c)
        print(f"[{variant}] {len(final)} pares ({ident} idénticos = {100 * ident / len(final):.1f} %) -> {out}")


if __name__ == "__main__":
    main()
