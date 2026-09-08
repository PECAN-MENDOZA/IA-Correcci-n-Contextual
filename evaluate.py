"""
evaluate.py — Evaluación offline del pipeline de corrección (WER / CER / exactitud).

No necesita el servidor: por defecto construye el pipeline en el mismo proceso
(reglas + BETO) y lo evalúa contra el set gold. También puede medir un servidor
en vivo por HTTP. Imprime métricas por categoría y globales, y guarda un JSON.

Uso:
  python evaluate.py                                   # pipeline en-proceso (reglas + BETO)
  python evaluate.py --no-beto                         # baseline: solo reglas (para comparar)
  python evaluate.py --source http --url http://35.224.215.77
  python evaluate.py --dataset data/eval_gold.csv --out reports/eval.json

Métricas (mismas definiciones que application.use_cases.evaluate_model, pero
implementadas aquí en Python puro para que el harness corra sin instalar
jiwer/Levenshtein — útil en el contenedor de producción, en local o en CI):
  - WER  : Word Error Rate  entre predicción y gold (menor es mejor)
  - CER  : Character Error Rate                       (menor es mejor)
  - improvement : cuánto acorta la distancia al gold respecto al input crudo (mayor mejor)
  - exact: fracción de frases idénticas al gold
"""
import argparse
import json
import os
import sys
import time
from collections import OrderedDict


def _edit_distance(a, b) -> int:
    """Distancia de Levenshtein entre dos secuencias (listas de palabras o strings)."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def _corpus_rate(refs, hyps, unit) -> float:
    """WER/CER a nivel de corpus: total de ediciones / total de unidades de referencia."""
    edits = total = 0
    for r, h in zip(refs, hyps):
        rs = r.split() if unit == "word" else r
        hs = h.split() if unit == "word" else h
        edits += _edit_distance(rs, hs)
        total += len(rs)
    return edits / max(total, 1)


def load_dataset(path: str) -> list:
    rows = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split("|")
            if len(parts) != 3:
                print(f"[WARN] línea {ln} ignorada (esperaba 3 campos): {line!r}")
                continue
            cat, inp, exp = (p.strip() for p in parts)
            if cat.lower() == "categoria":       # cabecera
                continue
            rows.append((cat, inp, exp))
    return rows


def build_predictor(args):
    """Devuelve una función predict(text) -> corrección, según el origen elegido."""
    if args.source == "http":
        import urllib.request

        def predict(text: str) -> str:
            payload = json.dumps({"originalText": text, "studentId": "eval"}).encode("utf-8")
            req = urllib.request.Request(
                args.url.rstrip("/") + "/interno/corregir",
                data=payload, headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=40) as r:
                return json.loads(r.read().decode("utf-8")).get("correctedText", "")
        return predict

    # --- pipeline en-proceso ---
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from infrastructure.nlp.phonetic_engine import PhoneticEngine
    from infrastructure.nlp.correction_pipeline import CorrectionPipeline

    phonetic = PhoneticEngine()
    judge = None
    if not args.no_beto:
        from infrastructure.nlp.context_judge import ContextJudge
        judge = ContextJudge()
    pipeline = CorrectionPipeline(phonetic=phonetic, judge=judge, seq2seq=None, tokenizer=None)

    def predict(text: str) -> str:
        return pipeline.correct(text, {})[0]
    return predict


def summarize(subset: list) -> dict:
    golds  = [r["gold"]  for r in subset]
    inputs = [r["input"] for r in subset]
    preds  = [r["pred"]  for r in subset]
    # improvement: cuánto acerca la predicción al gold vs el input crudo (a nivel char)
    dist_before = sum(_edit_distance(g, i) for g, i in zip(golds, inputs))
    dist_after  = sum(_edit_distance(g, p) for g, p in zip(golds, preds))
    improvement = (dist_before - dist_after) / dist_before if dist_before > 0 else 0.0
    return {
        "wer": _corpus_rate(golds, preds, "word"),
        "cer": _corpus_rate(golds, preds, "char"),
        "improvement_ratio": improvement,
        "exact": sum(r["exact"] for r in subset) / len(subset),
        "n": len(subset),
    }


def main():
    ap = argparse.ArgumentParser(description="Evaluación offline del pipeline de corrección.")
    ap.add_argument("--dataset", default="data/eval_gold.csv")
    ap.add_argument("--source", choices=["pipeline", "http"], default="pipeline")
    ap.add_argument("--url", default="http://35.224.215.77")
    ap.add_argument("--no-beto", action="store_true", help="pipeline sin BETO (baseline solo reglas)")
    ap.add_argument("--out", default=None, help="ruta para guardar el reporte JSON")
    ap.add_argument("--show-fails", action="store_true", help="imprime cada caso fallido")
    args = ap.parse_args()

    # Consola robusta (evita UnicodeEncodeError con los caracteres de caja en
    # terminales Windows con codificación cp1252).
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    rows = load_dataset(args.dataset)
    label = "HTTP " + args.url if args.source == "http" else \
            ("reglas (sin BETO)" if args.no_beto else "reglas + BETO")
    print(f"\n{'='*74}\n  EVALUACIÓN — {len(rows)} casos — fuente: {label}\n{'='*74}")

    predict = build_predictor(args)

    results = []
    for cat, inp, exp in rows:
        t0 = time.time()
        try:
            pred = predict(inp)
        except Exception as e:
            print(f"[ERROR] '{inp[:40]}...' -> {e}")
            pred = inp
        results.append({
            "cat": cat, "input": inp, "gold": exp, "pred": pred,
            "exact": pred.strip() == exp.strip(), "ms": int((time.time() - t0) * 1000),
        })

    by_cat = OrderedDict()
    for r in results:
        by_cat.setdefault(r["cat"], []).append(r)

    print(f"\n{'categoría':<14}{'n':>3}  {'WER':>6} {'CER':>6} {'mejora':>7} {'exactos':>9}")
    print("-" * 74)
    report = {"source": label, "por_categoria": {}, "casos": results}
    for cat, subset in by_cat.items():
        m = summarize(subset)
        report["por_categoria"][cat] = m
        print(f"{cat:<14}{m['n']:>3}  {m['wer']:>6.3f} {m['cer']:>6.3f} "
              f"{m['improvement_ratio']:>6.0%} {int(m['exact']*m['n']):>4}/{m['n']:<4}")
    overall = summarize(results)
    report["global"] = overall
    print("-" * 74)
    print(f"{'GLOBAL':<14}{overall['n']:>3}  {overall['wer']:>6.3f} {overall['cer']:>6.3f} "
          f"{overall['improvement_ratio']:>6.0%} {int(overall['exact']*overall['n']):>4}/{overall['n']:<4}")
    avg_ms = sum(r["ms"] for r in results) // max(len(results), 1)
    print(f"\n  latencia media: {avg_ms} ms/frase")

    if args.show_fails:
        print(f"\n{'─'*74}\n  FALLOS\n{'─'*74}")
        for r in results:
            if not r["exact"]:
                print(f"[{r['cat']}]\n  IN : {r['input']}\n  OUT: {r['pred']}\n  ESP: {r['gold']}\n")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"  reporte guardado en {args.out}")


if __name__ == "__main__":
    main()
