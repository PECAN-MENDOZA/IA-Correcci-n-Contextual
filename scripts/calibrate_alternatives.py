"""
scripts/calibrate_alternatives.py

Calibración de la capa 5 (alternativas) con el pipeline REAL (reglas + BETO +
T5 global) sobre `data/ambiguity_calibration.csv`
(`text,expected_options,forbidden_readings,notes`): para cada frase imprime las
sugerencias devueltas por `CorrectionPipeline.correct` y resume
precisión/recall de "ofrece ≥ 2 opciones" frente a la etiqueta
(`expected_options` ≥ 2 = ambigua). Los umbrales son los de
`infrastructure.nlp.alternatives.THRESHOLDS` (constantes de código; este script
no los modifica: sirve para decidir con datos si hay que cambiarlos en código).

`--check` convierte la calibración en PRUEBA DE ACEPTACIÓN (código de salida 1
si falla): falla si más de `MAX_CLEAR_FP` (2) de las 20 frases claras reciben
≥ 2 opciones, o si alguna alternativa ofrecida (posiciones 1..n; la
recomendada sigue la regla del beam 1 y no se juzga aquí) contiene una
palabra de la columna `forbidden_readings` de su fila (lecturas inválidas
conocidas: `quizas`, `llegan` tras "es posible que", ...). El recall se
informa, no se exige.

Uso (desde la raíz del repo, con el modelo local y sin descargar nada):
    $env:HF_HOME = "models\\hf-cache"
    .venv\\Scripts\\python.exe scripts\\calibrate_alternatives.py [--t5-dir models/t5_correction]
                                                              [--no-beto] [--no-t5] [--check]
                                                              [--verbose] [--out reporte.json]

`--verbose --out` guarda por frase la base, los beams crudos de T5 con el
veredicto de las guardas del beam recomendado (`t5Raw`), los beams que llegan
a la capa 5 (`t5`) y las variantes de BETO, para barrer umbrales y guardas
offline con el selector puro.

No usa el puerto 5000 ni el servicio: carga los modelos en proceso. Las
funciones `load_cases`, `evaluate_case` y `summarize` son puras (se prueban en
test_alternatives.py sin modelo).
"""
import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))

# Nunca descargar: los modelos ya están en models/ y models/hf-cache.
os.environ.setdefault("HF_HOME", str(REPO_DIR / "models" / "hf-cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from infrastructure.ml.guards import is_lexically_plausible_refinement, is_safe_refinement  # noqa: E402
from infrastructure.nlp.alternatives import THRESHOLDS  # noqa: E402

# Prueba de aceptación: como mucho tantas frases claras con ≥ 2 opciones.
MAX_CLEAR_FP = 2

_OUTER_PUNCT_RE = re.compile(r"^[^\w]+|[^\w]+$")


def load_cases(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    cases = []
    for row in rows:
        label = str(row["expected_options"]).strip()
        minimum = int(label.split("-")[0])   # "2-3" → 2
        forbidden = [f.strip() for f in (row.get("forbidden_readings") or "").split(";") if f.strip()]
        cases.append({
            "text": row["text"].strip(),
            "expected": label,
            "expected_ambiguous": minimum >= 2,
            "forbidden": forbidden,
            "notes": (row.get("notes") or "").strip(),
        })
    return cases


def _tokens(text: str) -> set[str]:
    """Palabras de una sugerencia: casefold, sin puntuación exterior, CON tildes."""
    return {_OUTER_PUNCT_RE.sub("", w).casefold() for w in text.split()}


def forbidden_offered(suggestions: list[str], forbidden: list[str]) -> list[tuple[str, str]]:
    """`(alternativa, forma prohibida)` por cada alternativa (no la recomendada) que contenga una lectura inválida."""
    found = []
    for alternative in suggestions[1:]:
        words = _tokens(alternative)
        for form in forbidden:
            if form.casefold() in words:
                found.append((alternative, form))
    return found


def evaluate_case(case: dict, suggestions: list[str]) -> dict:
    """Resultado de una frase: sugerencias, si ofreció ≥ 2 y las lecturas prohibidas ofrecidas."""
    return {**case, "suggestions": list(suggestions), "offered_ambiguous": len(suggestions) >= 2,
            "forbidden_offered": forbidden_offered(suggestions, case.get("forbidden", []))}


def summarize(results: list[dict]) -> dict:
    """Métricas de "ofrece ≥ 2 opciones" vs. etiqueta + veredicto de la prueba de aceptación."""
    tp = sum(1 for r in results if r["offered_ambiguous"] and r["expected_ambiguous"])
    fp = sum(1 for r in results if r["offered_ambiguous"] and not r["expected_ambiguous"])
    fn = sum(1 for r in results if not r["offered_ambiguous"] and r["expected_ambiguous"])
    tn = len(results) - tp - fp - fn
    precision, recall, f1 = prf(tp, fp, fn)
    forbidden = [(r["text"], alternative, form) for r in results for alternative, form in r.get("forbidden_offered", [])]
    reasons = []
    if fp > MAX_CLEAR_FP:
        reasons.append(f"FP en frases claras {fp} > {MAX_CLEAR_FP}")
    for text, alternative, form in forbidden:
        reasons.append(f"lectura prohibida '{form}' ofrecida en '{text}' → '{alternative}'")
    return {
        "thresholds": THRESHOLDS,
        "cases": len(results),
        "expectedAmbiguous": tp + fn,
        "clearSentences": fp + tn,
        "offeredAmbiguous": tp + fp,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "maxOptionsOffered": max((len(r["suggestions"]) for r in results), default=0),
        "forbiddenOffered": [{"text": t, "alternative": a, "form": f} for t, a, f in forbidden],
        "check": {"maxClearFp": MAX_CLEAR_FP, "passed": not reasons, "reasons": reasons},
    }


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
    internos permiten barrer umbrales offline con el selector puro, sin GPU:
      - "t5Raw": TODOS los beams crudos de `generate_corrections`, con su
        score y el resultado de las dos guardas del beam recomendado
        (`safe` = is_safe_refinement, `plausible` =
        is_lexically_plausible_refinement con recommendedMinSimilarity),
        para barrer las guardas sin volver a generar;
      - "t5": los beams que `_refine_with_model` deja pasar a la capa 5
        (vacío si el beam 1 no supera las guardas).
    """
    original_disambiguate = pipeline._disambiguate_context
    original_refine = pipeline._refine_with_model

    def disambiguate(sentence, *args, **kwargs):
        text, variants = original_disambiguate(sentence, *args, **kwargs)
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
        original_generate = pipeline._seq2seq.generate_corrections

        def generate(text, tokenizer, num_returns=2):
            raw = original_generate(text, tokenizer, num_returns=num_returns)
            sink["t5Raw"] = [{
                "text": str(b), "score": round(float(s), 3),
                "safe": is_safe_refinement(text, str(b).strip()),
                "plausible": is_lexically_plausible_refinement(
                    text, str(b).strip(), min_similarity=THRESHOLDS["recommendedMinSimilarity"]),
            } for b, s in raw]
            return raw

        pipeline._seq2seq.generate_corrections = generate


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
    ap.add_argument("--check", action="store_true",
                    help=f"prueba de aceptación: código 1 si FP en frases claras > {MAX_CLEAR_FP} "
                         "o si se ofrece alguna lectura de forbidden_readings")
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
        result = evaluate_case(case, suggestions)
        if args.verbose:
            result["internals"] = dict(internals)
        results.append(result)
        offered = result["offered_ambiguous"]
        mark = "OK " if offered == case["expected_ambiguous"] else ("FP " if offered else "FN ")
        print(f"[{mark}] esperado={case['expected']:<3} obtenido={len(suggestions)}  {case['text']}")
        print(f"       → {suggestions}")
        for alternative, form in result["forbidden_offered"]:
            print(f"       !! lectura prohibida '{form}' en la alternativa '{alternative}'")
        if args.verbose:
            if internals.get("t5") is not None:
                print(f"       T5:   {internals['t5']}")
            blocked = [b for b in internals.get("t5Raw", []) if not (b["safe"] and b["plausible"])]
            if blocked:
                print(f"       T5 descartados: {[(b['text'], b['score'], 'safe' if b['safe'] else 'unsafe', 'plausible' if b['plausible'] else 'lexical') for b in blocked]}")
            if internals.get("beto"):
                print(f"       BETO: {internals['beto']}")
        if mark != "OK ":
            print(f"       ({case['notes']})")

    summary = summarize(results)
    print("\nResumen (ofrece ≥ 2 opciones vs. etiqueta ambigua):")
    print(f"  frases={summary['cases']}  ambiguas esperadas={summary['expectedAmbiguous']}  "
          f"claras={summary['clearSentences']}  ofrecidas={summary['offeredAmbiguous']}")
    print(f"  TP={summary['tp']} FP={summary['fp']} FN={summary['fn']} TN={summary['tn']}")
    print(f"  precisión={summary['precision']:.3f}  recall={summary['recall']:.3f}  F1={summary['f1']:.3f}  "
          f"máx. opciones={summary['maxOptionsOffered']}")
    print(f"  lecturas prohibidas ofrecidas={len(summary['forbiddenOffered'])}")
    check = summary["check"]
    verdict = "PASA" if check["passed"] else "FALLA"
    print(f"  aceptación (FP claras ≤ {MAX_CLEAR_FP}/{summary['clearSentences']} y sin lecturas prohibidas): {verdict}")
    for reason in check["reasons"]:
        print(f"    - {reason}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        print(f"[OK] Resultados guardados en {out_path}")

    if args.check and not check["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
