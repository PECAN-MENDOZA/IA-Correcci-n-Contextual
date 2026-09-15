"""
scripts/calibrate_alternatives.py

Calibración de la capa 5 (alternativas) con el pipeline REAL (reglas + BETO +
T5 global) sobre `data/ambiguity_calibration.csv` (`text,expected_options,notes`):
para cada frase imprime las sugerencias devueltas por `CorrectionPipeline.correct`
y resume precisión/recall de "ofrece ≥ 2 opciones" frente a la etiqueta
(`expected_options` ≥ 2 = ambigua). Los umbrales son los de
`infrastructure.nlp.alternatives.THRESHOLDS` (constantes de código; este script
no los modifica: sirve para decidir con datos si hay que cambiarlos en código).

Uso (desde la raíz del repo, con el modelo local y sin descargar nada):
    $env:HF_HOME = "models\\hf-cache"
    .venv\\Scripts\\python.exe scripts\\calibrate_alternatives.py [--t5-dir models/t5_correction]
                                                              [--no-beto] [--no-t5]
                                                              [--verbose] [--out reporte.json]

No usa el puerto 5000 ni el servicio: carga los modelos en proceso.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))

# Nunca descargar: los modelos ya están en models/ y models/hf-cache.
os.environ.setdefault("HF_HOME", str(REPO_DIR / "models" / "hf-cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from infrastructure.nlp.alternatives import THRESHOLDS  # noqa: E402


def load_cases(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    cases = []
    for row in rows:
        label = str(row["expected_options"]).strip()
        minimum = int(label.split("-")[0])   # "2-3" → 2
        cases.append({
            "text": row["text"].strip(),
            "expected": label,
            "expected_ambiguous": minimum >= 2,
            "notes": (row.get("notes") or "").strip(),
        })
    return cases


def build_pipeline(args):
    from infrastructure.nlp.phonetic_engine import PhoneticEngine
    from infrastructure.nlp.correction_pipeline import CorrectionPipeline

    phonetic = PhoneticEngine()
    judge = None
    if not args.no_beto:
        from infrastructure.nlp.context_judge import ContextJudge
        judge = ContextJudge()

    seq2seq = tokenizer = None
    if not args.no_t5:
        t5_dir = Path(args.t5_dir)
        if not (t5_dir / "config.json").exists() or not (t5_dir / "tokenizer_config.json").exists():
            raise SystemExit(f"[ERROR] --t5-dir {t5_dir} no contiene config.json y tokenizer_config.json")
        from infrastructure.ml.t5_model import T5CorrectionModel, T5SpanishTokenizer
        tokenizer = T5SpanishTokenizer(model_name=str(t5_dir))
        seq2seq = T5CorrectionModel(save_dir=str(t5_dir))
        print(f"[OK] Modelo T5 cargado desde {t5_dir}")

    return CorrectionPipeline(phonetic=phonetic, judge=judge, seq2seq=seq2seq, tokenizer=tokenizer)


def trace_internals(pipeline, sink: dict):
    """
    Envuelve las capas 2.5 y 4 para registrar la base, los beams de T5 y las
    variantes de BETO con sus scores (solo --verbose). Con `--out`, esos
    internos permiten barrer umbrales offline con el selector puro, sin GPU.
    """
    original_disambiguate = pipeline._disambiguate_context
    original_refine = pipeline._refine_with_model

    def disambiguate(sentence):
        text, variants = original_disambiguate(sentence)
        sink["beto"] = [(v, round(s, 3)) for v, s in variants]
        return text, variants

    def refine(text):
        sink["base"] = text   # texto tras reglas + BETO + gramática (entrada de T5)
        beams = original_refine(text)
        sink["t5"] = [(b, round(s, 3)) for b, s in beams]
        return beams

    pipeline._disambiguate_context = disambiguate
    if pipeline._seq2seq is not None:
        pipeline._refine_with_model = refine


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def main():
    ap = argparse.ArgumentParser(description="Calibración de alternativas (≥ 2 opciones solo si la frase es ambigua).")
    ap.add_argument("--csv", default=str(REPO_DIR / "data" / "ambiguity_calibration.csv"))
    ap.add_argument("--t5-dir", default=str(REPO_DIR / "models" / "t5_correction"))
    ap.add_argument("--no-beto", action="store_true", help="pipeline sin BETO")
    ap.add_argument("--no-t5", action="store_true", help="pipeline sin T5 (solo reglas + BETO)")
    ap.add_argument("--verbose", action="store_true", help="muestra beams de T5 y variantes de BETO por frase")
    ap.add_argument("--out", default=None, help="ruta para guardar los resultados en JSON")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    cases = load_cases(Path(args.csv))
    pipeline = build_pipeline(args)
    internals: dict = {}
    if args.verbose:
        trace_internals(pipeline, internals)

    print(f"\nTHRESHOLDS = {json.dumps(THRESHOLDS)}\n")
    results = []
    for case in cases:
        internals.clear()
        suggestions = pipeline.correct(case["text"], {})
        offered = len(suggestions) >= 2
        results.append({**case, "suggestions": suggestions, "offered_ambiguous": offered,
                        **({"internals": dict(internals)} if args.verbose else {})})
        mark = "OK " if offered == case["expected_ambiguous"] else ("FP " if offered else "FN ")
        print(f"[{mark}] esperado={case['expected']:<3} obtenido={len(suggestions)}  {case['text']}")
        print(f"       → {suggestions}")
        if args.verbose:
            if internals.get("t5") is not None:
                print(f"       T5:   {internals['t5']}")
            if internals.get("beto"):
                print(f"       BETO: {internals['beto']}")
        if mark != "OK ":
            print(f"       ({case['notes']})")

    tp = sum(1 for r in results if r["offered_ambiguous"] and r["expected_ambiguous"])
    fp = sum(1 for r in results if r["offered_ambiguous"] and not r["expected_ambiguous"])
    fn = sum(1 for r in results if not r["offered_ambiguous"] and r["expected_ambiguous"])
    tn = len(results) - tp - fp - fn
    precision, recall, f1 = prf(tp, fp, fn)

    summary = {
        "thresholds": THRESHOLDS,
        "cases": len(results),
        "expectedAmbiguous": tp + fn,
        "offeredAmbiguous": tp + fp,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "maxOptionsOffered": max((len(r["suggestions"]) for r in results), default=0),
    }
    print("\nResumen (ofrece ≥ 2 opciones vs. etiqueta ambigua):")
    print(f"  frases={summary['cases']}  ambiguas esperadas={summary['expectedAmbiguous']}  "
          f"ofrecidas={summary['offeredAmbiguous']}")
    print(f"  TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"  precisión={precision:.3f}  recall={recall:.3f}  F1={f1:.3f}  "
          f"máx. opciones={summary['maxOptionsOffered']}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        print(f"[OK] Resultados guardados en {out_path}")


if __name__ == "__main__":
    main()
