"""
evaluate.py — Evaluación offline del pipeline de corrección.

Métrica principal: Precisión, Recall y F0.5 **a nivel de edición** con el
scorer `exact_token_edits_v1` (ver `extract_edits` / `edit_scores`). WER, CER,
mejora, exactitud y latencia se conservan como diagnósticos.

No necesita el servidor: por defecto construye el pipeline en el mismo proceso
(reglas + BETO, y T5 si se pasa `--t5-dir`) y lo evalúa contra el dataset
gold. También puede medir un servidor en vivo por HTTP. Imprime métricas por
categoría y globales y guarda un informe JSON versionado que el panel Vue
valida y el backend acepta (`POST /api/v1/research/technical-evaluations`).

Uso:
  python evaluate.py                                   # pipeline en-proceso (reglas + BETO)
  python evaluate.py --no-beto                         # baseline: solo reglas (para comparar)
  python evaluate.py --t5-dir models/t5_correction     # reglas + BETO + T5 (modelo global fusionado)
  python evaluate.py --source http --url http://35.224.215.77
  python evaluate.py --dataset data/holdout.csv --model-version "t5@abcd1234+grammar@ef567890" \
                     --out reports/holdout.json

Dataset: `categoria|entrada|esperado_1|esperado_2...` (una o más referencias;
menos de tres campos es un error). El informe registra el SHA-256 de los bytes
exactos del archivo: `data/eval_gold.csv` y `pruebas.txt` son de desarrollo y
el informe se marca `"development": true` (también con `--development`).

Scorer `exact_token_edits_v1`:
  - tokens = `re.findall(r"\\w+|[^\\w\\s]", texto)` (palabras y signos sueltos);
  - ediciones = opcodes no-equal de `difflib.SequenceMatcher` entre la entrada
    y el texto, como `(inicio_en_entrada, fin_en_entrada, tokens_reemplazo)`;
    un bloque de reemplazo con igual número de tokens a ambos lados se divide
    palabra a palabra (ver `extract_edits`);
  - por frase se elige la referencia con mejor F0.5 (en empate, más TP y menos
    FP+FN); TP = ediciones predichas ∩ gold, FP = predichas − gold,
    FN = gold − predichas, sumadas sobre el corpus;
  - P = TP/(TP+FP), R = TP/(TP+FN), F0.5 = 1.25·P·R/(0.25·P+R); cuando el
    denominador es 0 el valor es 0.0 (convención del backend, tolerancia 1e-6).

Este módulo no carga torch ni modelos al importarse (los tests puros de
`test_evaluate_metrics.py` dependen de ello): el pipeline se construye en
`build_predictor`, desde `main()`.
"""
import argparse
import difflib
import hashlib
import json
import os
import re
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

SCORER_VERSION = "exact_token_edits_v1"
DEFAULT_MODEL_VERSION = "global-lora-unversioned"   # igual que application.use_cases.correct_text
REPO_DIR = Path(__file__).resolve().parent
GRAMMAR_LORA_MANIFEST = REPO_DIR / "models" / "grammar_lora" / "manifest.json"
# Datasets de desarrollo: sus informes nunca se registran como evaluación final.
DEVELOPMENT_DATASETS = {"eval_gold.csv", "pruebas.txt"}
MAX_CATEGORIES = 50       # límite del backend (TechnicalEvaluationRequest.categories)
MAX_CATEGORY_NAME = 80    # CategoryResult.category

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


# ───────────────────────────── scorer exact_token_edits_v1 ─────────────────────────────

def tokenize(text: str) -> list:
    """Palabras y signos de puntuación como tokens independientes."""
    return _TOKEN_RE.findall(text)


def extract_edits(source: str, target: str) -> set:
    """
    Ediciones exactas que transforman `source` en `target`, como conjunto de
    `(inicio, fin, tokens_reemplazo)` sobre los tokens de `source`. Una
    inserción tiene inicio == fin; un borrado, reemplazo vacío.

    difflib funde reemplazos contiguos en un solo bloque (`arbol esta` →
    `árbol está`); cuando el bloque tiene el mismo número de tokens a ambos
    lados se divide palabra a palabra, para que una corrección parcial cuente
    como TP + FN y no como FP + FN. Los bloques de distinta longitud
    (`ala` → `a la`, `uillos` → `niños`) se mantienen como una sola edición.
    """
    src = tokenize(source)
    tgt = tokenize(target)
    matcher = difflib.SequenceMatcher(None, src, tgt, autojunk=False)
    edits = set()
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace" and i2 - i1 == j2 - j1:
            for k in range(i2 - i1):
                edits.add((i1 + k, i1 + k + 1, (tgt[j1 + k],)))
        else:
            edits.add((i1, i2, tuple(tgt[j1:j2])))
    return edits


def prf(tp: int, fp: int, fn: int) -> tuple:
    """(precisión, recall, F0.5) con la convención del backend: 0.0 si el denominador es 0."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    denominator = 0.25 * precision + recall
    f0_5 = 1.25 * precision * recall / denominator if denominator else 0.0
    return precision, recall, f0_5


def score_sentence(source: str, references: list, prediction: str) -> dict:
    """
    TP/FP/FN de una frase frente a la mejor de sus referencias (mayor F0.5; en
    empate, más TP y menos FP+FN; luego la primera). Devuelve también la
    referencia elegida para los diagnósticos WER/CER.
    """
    if not references:
        raise ValueError("cada frase necesita al menos una referencia")
    predicted = extract_edits(source, prediction)
    best = None
    for reference in references:
        gold = extract_edits(source, reference)
        tp = len(predicted & gold)
        fp = len(predicted - gold)
        fn = len(gold - predicted)
        f0_5 = prf(tp, fp, fn)[2]
        key = (f0_5, tp, -(fp + fn))
        if best is None or key > best[0]:
            best = (key, {"tp": tp, "fp": fp, "fn": fn, "gold": reference})
    return best[1]


def edit_scores(inputs: list, golds: list, predictions: list) -> dict:
    """
    Precisión / Recall / F0.5 a nivel de edición sobre el corpus.
    `golds` es una lista de listas de referencias (una o más por frase).
    """
    if not (len(inputs) == len(golds) == len(predictions)):
        raise ValueError("inputs, golds y predictions deben tener la misma longitud")
    tp = fp = fn = 0
    for source, references, prediction in zip(inputs, golds, predictions):
        scored = score_sentence(source, references, prediction)
        tp += scored["tp"]
        fp += scored["fp"]
        fn += scored["fn"]
    precision, recall, f0_5 = prf(tp, fp, fn)
    return {
        "scorer": SCORER_VERSION,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f0_5": f0_5,
        "n": len(inputs),
    }


# ───────────────────────────── diagnósticos WER / CER ─────────────────────────────

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


# ───────────────────────────── dataset ─────────────────────────────

def load_dataset(path: str) -> list:
    """
    Lee `categoria|entrada|esperado_1|esperado_2...` y devuelve
    `[(categoria, entrada, [referencias]), ...]`. Ignora líneas vacías, `#` y
    la cabecera; una línea con menos de tres campos o sin referencia es un
    error (el informe se versiona por el hash del archivo, así que no se
    ignoran filas en silencio).
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 3:
                raise ValueError(f"{path}:{ln}: se esperaban al menos 3 campos "
                                 f"(categoria|entrada|esperado...): {line!r}")
            cat, inp, references = parts[0], parts[1], [p for p in parts[2:] if p]
            if cat.lower() == "categoria":       # cabecera
                continue
            if not references:
                raise ValueError(f"{path}:{ln}: la fila no tiene ninguna referencia: {line!r}")
            rows.append((cat, inp, references))
    return rows


def dataset_sha256(path: str) -> str:
    """SHA-256 (hex minúsculas) de los bytes exactos del archivo de dataset."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_model_version(explicit, env=None, manifest_path: Path = GRAMMAR_LORA_MANIFEST) -> str:
    """
    `--model-version` > MODEL_VERSION (env) > manifest.json del LoRA global >
    default. Misma regla que `main.resolve_model_version()`, replicada aquí
    porque importar `main` construye la app (carga BETO/T5) a nivel de módulo.
    """
    if explicit and explicit.strip():
        return explicit.strip()
    env = os.environ if env is None else env
    from_env = str(env.get("MODEL_VERSION", "")).strip()
    if from_env:
        return from_env
    try:
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            from_manifest = str(manifest.get("modelVersion", "")).strip()
            if from_manifest:
                return from_manifest
    except Exception as exc:
        print(f"[WARN] No se pudo leer {manifest_path}: {exc}")
    return DEFAULT_MODEL_VERSION


# ───────────────────────────── predictor ─────────────────────────────

def build_predictor(args):
    """Devuelve una función predict(text) -> corrección recomendada, según el origen elegido."""
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

    # --- pipeline en-proceso (torch/modelos se cargan solo aquí) ---
    sys.path.insert(0, str(REPO_DIR))
    from infrastructure.nlp.phonetic_engine import PhoneticEngine
    from infrastructure.nlp.correction_pipeline import CorrectionPipeline

    phonetic = PhoneticEngine()
    judge = None
    if not args.no_beto:
        from infrastructure.nlp.context_judge import ContextJudge
        judge = ContextJudge()

    seq2seq = tokenizer = None
    if args.t5_dir:
        t5_dir = Path(args.t5_dir)
        # Solo directorios locales completos: T5CorrectionModel descargaría el
        # modelo base si faltara config.json, y aquí nunca se descarga nada.
        if not (t5_dir / "config.json").exists() or not (t5_dir / "tokenizer_config.json").exists():
            raise SystemExit(f"[ERROR] --t5-dir {t5_dir} no contiene config.json y tokenizer_config.json")
        from infrastructure.ml.t5_model import T5CorrectionModel, T5SpanishTokenizer
        tokenizer = T5SpanishTokenizer(model_name=str(t5_dir))
        seq2seq = T5CorrectionModel(save_dir=str(t5_dir))
        print(f"[OK] Modelo T5 cargado desde {t5_dir}")

    pipeline = CorrectionPipeline(phonetic=phonetic, judge=judge, seq2seq=seq2seq, tokenizer=tokenizer)

    def predict(text: str) -> str:
        # La primera sugerencia es la recomendada (las alternativas no se evalúan).
        return pipeline.correct(text, {})[0]
    return predict


# ───────────────────────────── informe ─────────────────────────────

def summarize(subset: list) -> dict:
    """Métricas de un subconjunto de casos ya puntuados (edición + diagnósticos)."""
    golds  = [r["gold"]  for r in subset]
    inputs = [r["input"] for r in subset]
    preds  = [r["pred"]  for r in subset]
    tp = sum(r["tp"] for r in subset)
    fp = sum(r["fp"] for r in subset)
    fn = sum(r["fn"] for r in subset)
    precision, recall, f0_5 = prf(tp, fp, fn)
    # improvement: cuánto acerca la predicción al gold vs el input crudo (a nivel char)
    dist_before = sum(_edit_distance(g, i) for g, i in zip(golds, inputs))
    dist_after  = sum(_edit_distance(g, p) for g, p in zip(golds, preds))
    improvement = (dist_before - dist_after) / dist_before if dist_before > 0 else 0.0
    return {
        "n": len(subset),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f0_5": f0_5,
        "wer": _corpus_rate(golds, preds, "word"),
        "cer": _corpus_rate(golds, preds, "char"),
        "improvement_ratio": improvement,
        "exact": sum(r["exact"] for r in subset) / len(subset) if subset else 0.0,
    }


def _score_cases(cases: list) -> list:
    """Añade tp/fp/fn, la referencia elegida y `exact` a cada caso {cat, input, golds, pred}."""
    scored = []
    for case in cases:
        s = score_sentence(case["input"], case["golds"], case["pred"])
        pred = case["pred"].strip()
        matching = [g for g in case["golds"] if g.strip() == pred]
        exact = bool(matching)
        gold = matching[0] if exact else s["gold"]
        row = OrderedDict(case)
        row.update({"gold": gold, "exact": exact, "tp": s["tp"], "fp": s["fp"], "fn": s["fn"]})
        scored.append(row)
    return scored


def _category_results(by_cat: "OrderedDict") -> list:
    """Bloque `categories` del contrato: nombres recortados, únicos y como mucho 50."""
    merged = OrderedDict()
    for cat, subset in by_cat.items():
        name = cat.strip()[:MAX_CATEGORY_NAME]
        merged.setdefault(name, []).extend(subset)
    if len(merged) > MAX_CATEGORIES:
        print(f"[WARN] {len(merged)} categorías; el backend acepta {MAX_CATEGORIES}, "
              f"se registran las primeras")
    categories = []
    for name, subset in list(merged.items())[:MAX_CATEGORIES]:
        tp = sum(r["tp"] for r in subset)
        fp = sum(r["fp"] for r in subset)
        fn = sum(r["fn"] for r in subset)
        precision, recall, f0_5 = prf(tp, fp, fn)
        categories.append({"category": name, "tp": tp, "fp": fp, "fn": fn,
                           "precision": precision, "recall": recall, "f05": f0_5})
    return categories


def build_report(cases: list, *, model_version: str, dataset_path: str, dataset_hash: str,
                 model_dir, development: bool, source: str, latencies_ms=None) -> dict:
    """
    Informe JSON versionado. `cases` = [{cat, input, golds, pred}, ...] en el
    orden del dataset. El bloque `technicalEvaluation` es exactamente lo que
    valida el panel Vue y acepta el backend; el resto son diagnósticos.
    """
    scored = _score_cases(cases)
    latencies_ms = list(latencies_ms or [])
    for row, ms in zip(scored, latencies_ms):
        row["ms"] = ms

    by_cat = OrderedDict()
    for r in scored:
        by_cat.setdefault(r["cat"], []).append(r)
    overall = summarize(scored)
    technical = OrderedDict([
        ("modelVersion", model_version),
        ("datasetSha256", dataset_hash),
        ("scorerVersion", SCORER_VERSION),
        ("precision", overall["precision"]),
        ("recall", overall["recall"]),
        ("fZeroFive", overall["f0_5"]),
        ("truePositives", overall["tp"]),
        ("falsePositives", overall["fp"]),
        ("falseNegatives", overall["fn"]),
        ("categories", _category_results(by_cat)),
    ])
    return OrderedDict([
        ("development", bool(development)),
        ("timestamp", datetime.now(timezone.utc).isoformat(timespec="seconds")),
        ("source", source),
        ("modelVersion", model_version),
        ("modelDir", str(model_dir) if model_dir else None),
        ("scorerVersion", SCORER_VERSION),
        ("dataset", str(dataset_path)),
        ("datasetSha256", dataset_hash),
        ("technicalEvaluation", technical),
        ("global", overall),
        ("por_categoria", OrderedDict((cat, summarize(subset)) for cat, subset in by_cat.items())),
        ("latency_ms_avg", sum(latencies_ms) // len(latencies_ms) if latencies_ms else 0),
        ("casos", scored),
    ])


def _print_table(report: dict):
    print(f"\n{'categoría':<12}{'n':>3} {'TP':>4}{'FP':>4}{'FN':>4}  {'P':>6} {'R':>6} {'F0.5':>6}  "
          f"{'WER':>6} {'CER':>6} {'mejora':>7} {'exactos':>8}")
    print("-" * 96)
    rows = list(report["por_categoria"].items()) + [("GLOBAL", report["global"])]
    for i, (cat, m) in enumerate(rows):
        if i == len(rows) - 1:
            print("-" * 96)
        print(f"{cat:<12}{m['n']:>3} {m['tp']:>4}{m['fp']:>4}{m['fn']:>4}  "
              f"{m['precision']:>6.3f} {m['recall']:>6.3f} {m['f0_5']:>6.3f}  "
              f"{m['wer']:>6.3f} {m['cer']:>6.3f} {m['improvement_ratio']:>6.0%} "
              f"{int(round(m['exact']*m['n'])):>4}/{m['n']:<4}")


def main():
    ap = argparse.ArgumentParser(description="Evaluación offline del pipeline de corrección (F0.5 por edición).")
    ap.add_argument("--dataset", default="data/eval_gold.csv")
    ap.add_argument("--source", choices=["pipeline", "http"], default="pipeline")
    ap.add_argument("--url", default="http://35.224.215.77")
    ap.add_argument("--no-beto", action="store_true", help="pipeline sin BETO (baseline solo reglas)")
    ap.add_argument("--t5-dir", default=None,
                    help="directorio local del T5 global (p. ej. models/t5_correction); sin él, reglas(+BETO)")
    ap.add_argument("--model-version", default=None,
                    help="identificador del modelo evaluado (default: MODEL_VERSION > manifiesto > "
                         f"{DEFAULT_MODEL_VERSION})")
    ap.add_argument("--development", action="store_true",
                    help="marca el informe como de desarrollo (automático para eval_gold.csv / pruebas.txt)")
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
    dataset_hash = dataset_sha256(args.dataset)
    model_version = resolve_model_version(args.model_version)
    development = args.development or Path(args.dataset).name in DEVELOPMENT_DATASETS
    if args.source == "http":
        label = "HTTP " + args.url
    else:
        label = "reglas" + ("" if args.no_beto else " + BETO") + (" + T5" if args.t5_dir else "")
    print(f"\n{'='*96}\n  EVALUACIÓN — {len(rows)} casos — fuente: {label}\n"
          f"  modelVersion: {model_version}   scorer: {SCORER_VERSION}\n"
          f"  dataset: {args.dataset}  sha256: {dataset_hash}"
          f"{'   [DESARROLLO]' if development else ''}\n{'='*96}")

    predict = build_predictor(args)

    cases, latencies = [], []
    for cat, inp, references in rows:
        t0 = time.time()
        try:
            pred = predict(inp)
        except Exception as e:
            print(f"[ERROR] '{inp[:40]}...' -> {e}")
            pred = inp
        latencies.append(int((time.time() - t0) * 1000))
        cases.append({"cat": cat, "input": inp, "golds": references, "pred": pred})

    report = build_report(
        cases, model_version=model_version, dataset_path=args.dataset, dataset_hash=dataset_hash,
        model_dir=args.t5_dir, development=development, source=label, latencies_ms=latencies,
    )
    _print_table(report)
    te = report["technicalEvaluation"]
    print(f"\n  F0.5 = {te['fZeroFive']:.4f}  (P = {te['precision']:.4f}, R = {te['recall']:.4f}; "
          f"TP {te['truePositives']}, FP {te['falsePositives']}, FN {te['falseNegatives']})")
    print(f"  latencia media: {report['latency_ms_avg']} ms/frase")

    if args.show_fails:
        print(f"\n{'─'*96}\n  FALLOS\n{'─'*96}")
        for r in report["casos"]:
            if not r["exact"]:
                print(f"[{r['cat']}] tp={r['tp']} fp={r['fp']} fn={r['fn']}\n"
                      f"  IN : {r['input']}\n  OUT: {r['pred']}\n  ESP: {r['gold']}\n")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"  reporte guardado en {args.out}")


if __name__ == "__main__":
    main()
