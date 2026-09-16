"""
evaluate.py — Evaluación offline del pipeline de corrección.

Métrica principal: Precisión, Recall y F0.5 **a nivel de edición** con el
scorer `exact_token_edits_v1` (ver `align_tokens` / `extract_edits` /
`edit_scores`). WER, CER, mejora, exactitud y latencia se conservan como
diagnósticos.

No necesita el servidor: por defecto construye el pipeline en el mismo proceso
(reglas + BETO, y T5 si se pasa `--t5-dir`) y lo evalúa contra el dataset
gold. También puede medir un servidor en vivo por HTTP. Imprime métricas por
categoría y globales y guarda un informe JSON versionado que se sube **tal
cual** al panel Vue (que lo valida) y que el backend acepta
(`POST /api/v1/research/technical-evaluations`).

Uso:
  python evaluate.py --model-version rules-beto        # pipeline en-proceso (reglas + BETO)
  python evaluate.py --no-beto --model-version rules   # baseline: solo reglas (para comparar)
  python evaluate.py --t5-dir models/t5_correction     # reglas + BETO + T5 (modelo global fusionado)
  python evaluate.py --source http --url http://35.224.215.77
  python evaluate.py --dataset data/holdout.csv --t5-dir models/t5_base --model-version "t5-base@abcd1234" \
                     --out reports/holdout.json

`modelVersion` del informe (`evaluation_model_version`): `--model-version` >
`MODEL_VERSION` (entorno) > con `--t5-dir` y BETO, la versión compuesta desde
`<t5-dir>/model-manifest.json` (el manifiesto del modelo EVALUADO, nunca el de
`models/t5_correction` por defecto) > en `--source http`, el `modelVersion` que
devuelve el servidor. Sin `--t5-dir` o con `--no-beto` la versión debe ser
explícita (código de salida 2 si falta): el pipeline evaluado no es el que
nombra ninguna versión compuesta.

Dataset: `categoria|entrada|esperado_1|esperado_2...` (una o más referencias;
menos de tres campos es un error). El informe registra el SHA-256 de los bytes
exactos del archivo: `data/eval_gold.csv` y `pruebas.txt` son de desarrollo y
el informe se marca `"development": true` (también con `--development`).

Scorer `exact_token_edits_v1`:
  - tokens = `re.findall(r"\\w+|[^\\w\\s]", texto)` (palabras y signos sueltos);
  - alineación entrada→texto por programación dinámica sobre tokens
    (`align_tokens`): igual = 0; sustitución 1:1 = 0.1 si solo cambian tildes
    o mayúsculas, la distancia de Levenshtein de caracteres normalizada si es
    ≤ 0.5 y 1 en otro caso; división 1:m / unión k:1 (m, k ≤ 4) solo cuando
    las letras coinciden sin tildes ni mayúsculas (`ala` → `a la`), coste 0.1;
    inserción = borrado = 1;
  - ediciones = cada sustitución, división o unión es una edición
    `(inicio_en_entrada, fin_en_entrada, tokens_reemplazo)`; una racha de
    inserciones/borrados consecutivos es una sola edición (inserción:
    inicio == fin; borrado: reemplazo vacío);
  - por frase se elige la referencia con mejor F0.5 (en empate, más TP y menos
    FP+FN); TP = ediciones predichas ∩ gold, FP = predichas − gold,
    FN = gold − predichas, sumadas sobre el corpus;
  - P = TP/(TP+FP), R = TP/(TP+FN), F0.5 = 1.25·P·R/(0.25·P+R); cuando el
    denominador es 0 el valor es 0.0 (convención del backend, tolerancia 1e-6).

Si el modelo lanza una excepción en un caso, la predicción es la entrada (se
puntúa como "sin cambios"), el caso lleva `error`, el informe lleva `errors`
(conteo) y `errorList`, y un informe que no es de desarrollo termina con
código 2.

Este módulo no carga torch ni modelos al importarse (los tests puros de
`test_evaluate_metrics.py` dependen de ello): el pipeline se construye en
`build_predictor`, desde `main()`. `HF_HUB_OFFLINE=1` se fija en `main()` si
no estaba definido, para que ni BETO ni T5 intenten descargar nada.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

from infrastructure.versioning import (  # noqa: F401
    DEFAULT_MODEL_VERSION,
    MODEL_MANIFEST_NAME,
    REPO_DIR,
    compose_model_version,
    resolve_model_version,
)

SCORER_VERSION = "exact_token_edits_v1"
# Datasets de desarrollo: sus informes nunca se registran como evaluación final.
DEVELOPMENT_DATASETS = {"eval_gold.csv", "pruebas.txt"}
MAX_CATEGORIES = 50       # límite del backend (TechnicalEvaluationRequest.categories)
MAX_CATEGORY_NAME = 80    # CategoryResult.category
MAX_MODEL_VERSION = 160   # TechnicalEvaluationRequest.modelVersion
EXIT_MODEL_ERRORS = 2     # código de salida si un informe final tuvo errores del modelo
EXIT_BAD_VERSION = 2      # código de salida si no se puede determinar el modelVersion del informe

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

# Costes de la alineación (ver docstring del módulo).
COST_CHEAP = 0.1          # misma forma sin tildes/mayúsculas (también divisiones y uniones)
CLOSE_DISTANCE = 0.5      # sustitución 1:1 "cercana": distancia normalizada ≤ 0.5
COST_FAR_SUB = 1.0        # sustitución 1:1 entre tokens distintos
COST_INDEL = 1.0          # inserción o borrado de un token
MAX_SPLIT = 4             # tokens como mucho en una división 1:m o unión k:1


# ───────────────────────────── scorer exact_token_edits_v1 ─────────────────────────────

def tokenize(text: str) -> list:
    """Palabras y signos de puntuación como tokens independientes."""
    return _TOKEN_RE.findall(text)


def _strip_form(text: str) -> str:
    """Forma sin tildes/diacríticos ni mayúsculas (`Árbol` → `arbol`, `mañana` → `manana`)."""
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).casefold()


def _normalized_distance(a: str, b: str) -> float:
    """Levenshtein de caracteres dividido por la longitud mayor (0 = iguales, 1 = nada en común)."""
    longest = max(len(a), len(b))
    return _edit_distance(a, b) / longest if longest else 0.0


def _substitution_cost(source_token: str, target_token: str):
    """Coste de sustituir un token por otro (None nunca: 1:1 siempre se permite)."""
    if _strip_form(source_token) == _strip_form(target_token):
        return COST_CHEAP
    distance = _normalized_distance(source_token, target_token)
    return distance if distance <= CLOSE_DISTANCE else COST_FAR_SUB


def _segmentation_cost(source_tokens: list, target_tokens: list):
    """Coste de una división 1:m o unión k:1, o None si las letras no coinciden."""
    if _strip_form("".join(source_tokens)) == _strip_form("".join(target_tokens)):
        return COST_CHEAP
    return None


def align_tokens(source: list, target: list) -> list:
    """
    Alineación de coste mínimo entre dos listas de tokens, como lista de
    operaciones `(tipo, i1, i2, j1, j2)` sobre `source[i1:i2]` → `target[j1:j2]`:
    `match` (tokens idénticos), `sub` (1:1), `split` (1:m), `join` (k:1),
    `ins` (0:1) y `del` (1:0). Determinista: en empate gana la primera
    operación considerada (borrado, inserción, diagonal, división, unión).
    """
    n, m = len(source), len(target)
    inf = float("inf")
    cost = [[inf] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            candidates = []
            if i > 0:
                candidates.append((cost[i - 1][j] + COST_INDEL, ("del", i - 1, i, j, j)))
            if j > 0:
                candidates.append((cost[i][j - 1] + COST_INDEL, ("ins", i, i, j - 1, j)))
            if i > 0 and j > 0:
                if source[i - 1] == target[j - 1]:
                    candidates.append((cost[i - 1][j - 1], ("match", i - 1, i, j - 1, j)))
                else:
                    step = _substitution_cost(source[i - 1], target[j - 1])
                    candidates.append((cost[i - 1][j - 1] + step, ("sub", i - 1, i, j - 1, j)))
            if i > 0:
                for k in range(2, min(MAX_SPLIT, j) + 1):        # 1 token de entrada → k de salida
                    step = _segmentation_cost(source[i - 1:i], target[j - k:j])
                    if step is not None:
                        candidates.append((cost[i - 1][j - k] + step, ("split", i - 1, i, j - k, j)))
            if j > 0:
                for k in range(2, min(MAX_SPLIT, i) + 1):        # k tokens de entrada → 1 de salida
                    step = _segmentation_cost(source[i - k:i], target[j - 1:j])
                    if step is not None:
                        candidates.append((cost[i - k][j - 1] + step, ("join", i - k, i, j - 1, j)))
            best_cost, best_op = candidates[0]
            for candidate_cost, op in candidates[1:]:
                if candidate_cost < best_cost:
                    best_cost, best_op = candidate_cost, op
            cost[i][j] = best_cost
            back[i][j] = best_op

    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        op = back[i][j]
        ops.append(op)
        i, j = op[1], op[3]
    ops.reverse()
    return ops


def extract_edits(source: str, target: str) -> set:
    """
    Ediciones exactas que transforman `source` en `target`, como conjunto de
    `(inicio, fin, tokens_reemplazo)` sobre los tokens de `source`. Una
    inserción tiene inicio == fin; un borrado, reemplazo vacío.

    Cada sustitución, división (`ala` → `a la`) o unión (`a la` → `ala`) de
    `align_tokens` es una edición; las inserciones/borrados consecutivos se
    funden en una sola. Así una corrección vecina de otra (`árbol` junto a
    `está`, o una inserción al lado) nunca cambia cómo se cuentan las demás.
    """
    src = tokenize(source)
    tgt = tokenize(target)
    edits = set()
    run = None            # racha abierta de inserciones/borrados: [i1, i2, tokens]

    def close_run():
        nonlocal run
        if run is not None:
            edits.add((run[0], run[1], tuple(run[2])))
            run = None

    for kind, i1, i2, j1, j2 in align_tokens(src, tgt):
        if kind == "match":
            close_run()
        elif kind in ("sub", "split", "join"):
            close_run()
            edits.add((i1, i2, tuple(tgt[j1:j2])))
        else:                                   # ins / del
            if run is None:
                run = [i1, i2, list(tgt[j1:j2])]
            else:
                run[1] = i2
                run[2].extend(tgt[j1:j2])
    close_run()
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


def is_development(dataset_path: str, flag: bool) -> bool:
    """`--development` o un dataset de desarrollo (`eval_gold.csv`, `pruebas.txt`)."""
    return bool(flag) or Path(dataset_path).name in DEVELOPMENT_DATASETS


# ───────────────────────────── guardas ─────────────────────────────

def validate_model_version(value) -> str:
    """`modelVersion` como lo exige el backend: texto no vacío de hasta 160 caracteres (recortado)."""
    text = value.strip() if isinstance(value, str) else ""
    if not text:
        raise ValueError("modelVersion no puede estar vacío")
    if len(text) > MAX_MODEL_VERSION:
        raise ValueError(f"modelVersion supera los {MAX_MODEL_VERSION} caracteres ({len(text)})")
    return text


def compose_from_t5_dir(t5_dir) -> str:
    """
    Versión compuesta `<base-tag>@<hash8>+<lora-tag>@<hash8>` desde
    `<t5_dir>/model-manifest.json` (sus propios `model_sha256` y `adapter_sha256`).
    ValueError si el manifiesto no existe, no se puede leer o no trae los hashes.
    """
    manifest_path = Path(t5_dir) / MODEL_MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError(f"no existe {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"{manifest_path} ilegible: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"{manifest_path} no es un objeto JSON")
    return compose_model_version(manifest, manifest)


def evaluation_model_version(explicit, *, source: str, t5_dir, no_beto: bool,
                             server_version=None, env=None) -> str:
    """
    `modelVersion` del informe. Precedencia: `explicit` (`--model-version`) >
    `MODEL_VERSION` (entorno) > según lo evaluado:
      - `--source http`: el `modelVersion` que devolvió el servidor;
      - pipeline con `--t5-dir` y BETO: compuesta desde `<t5-dir>/model-manifest.json`;
      - pipeline sin `--t5-dir` o con `--no-beto`: hace falta `--model-version`.
    Nunca se hereda la versión de `models/t5_correction` para un pipeline que no
    la sirve. ValueError con el motivo cuando no se puede determinar.
    """
    explicit = str(explicit or "").strip()
    if explicit:
        return explicit
    env = os.environ if env is None else env
    from_env = str(env.get("MODEL_VERSION", "") or "").strip()
    if from_env:
        return from_env
    if source == "http":
        server_version = str(server_version or "").strip()
        if not server_version:
            raise ValueError("el servidor no devolvió `modelVersion`; pasa --model-version")
        return server_version
    if t5_dir is None:
        raise ValueError("sin --t5-dir el pipeline evaluado es reglas(+BETO): pasa --model-version "
                         "(p. ej. rules-dev o rules-beto-dev); no se hereda la versión de models/")
    if no_beto:
        raise ValueError("con --no-beto el pipeline evaluado no es el modelo global (reglas + BETO + T5): "
                         "pasa --model-version explícita")
    try:
        return compose_from_t5_dir(t5_dir)
    except ValueError as exc:
        raise ValueError(f"no se pudo componer modelVersion desde {Path(t5_dir) / MODEL_MANIFEST_NAME} "
                         f"({exc}); pasa --model-version") from exc


def git_commit(repo_dir=REPO_DIR):
    """
    Commit HEAD del checkout (hex), con sufijo `-dirty` si hay cambios sin
    commitear en archivos versionados (el informe no debe atribuirse a un
    commit limpio que no corrió) y `-unknown` si `git status` falla (un
    estado desconocido nunca se reporta como limpio); None si git no está
    disponible.
    """
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo_dir), capture_output=True,
                              text=True, timeout=10)
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=str(repo_dir),
                                capture_output=True, text=True, timeout=10)
    except Exception:
        return None
    value = head.stdout.strip().lower() if head.returncode == 0 else ""
    if not re.fullmatch(r"[0-9a-f]{7,40}", value):
        return None
    if status.returncode != 0:
        return value + "-unknown"
    return value + ("-dirty" if status.stdout.strip() else "")


def pipeline_info(commit=None) -> dict:
    """
    Bloque diagnóstico `pipeline` del informe: los umbrales de la capa 5
    (`alternatives.THRESHOLDS`, constantes de código de ESTE checkout) y el
    commit. `modelVersion` identifica pesos, no código: dos commits con reglas o
    umbrales distintos y el mismo `modelVersion` dan informes distintos.
    Import diferido: `alternatives` no carga torch.
    """
    from infrastructure.nlp.alternatives import THRESHOLDS
    return {"thresholds": dict(THRESHOLDS), "commit": commit if commit is not None else git_commit()}


CLIENT_PIPELINE_NOTE = (
    "metadata del checkout cliente que ejecutó evaluate.py (--source http): el servidor solo "
    "devuelve modelVersion; sus reglas, umbrales y commit no se conocen y no se atribuyen aquí")


def report_pipeline_blocks(source: str, info: dict) -> tuple:
    """
    `(pipeline, clientPipeline)` para `build_report` según el origen. En
    proceso (`pipeline`), `info` describe el código que corrigió: va en
    `pipeline`. Por HTTP el código que corrigió es el del servidor, del que
    solo llega `modelVersion`: `pipeline` queda en None y `info` (umbrales y
    commit del checkout que evalúa) se etiqueta como `clientPipeline` con nota.
    """
    if source == "http":
        return None, {**dict(info), "note": CLIENT_PIPELINE_NOTE}
    return dict(info), None


def require_local_t5_dir(path) -> Path:
    """
    Directorio local completo del T5 (`config.json` y `tokenizer_config.json`).
    Se comprueba ANTES de importar torch: `T5CorrectionModel` descargaría el
    modelo base si faltara `config.json`, y aquí nunca se descarga nada.
    """
    t5_dir = Path(path)
    missing = [name for name in ("config.json", "tokenizer_config.json") if not (t5_dir / name).is_file()]
    if missing:
        raise SystemExit(f"[ERROR] --t5-dir {t5_dir} no contiene {' ni '.join(missing)}")
    return t5_dir


def ensure_offline(env=None) -> None:
    """Fija `HF_HUB_OFFLINE=1` si no estaba definido: BETO/T5 solo se leen de la caché local."""
    env = os.environ if env is None else env
    env.setdefault("HF_HUB_OFFLINE", "1")


def exit_status(errors: int, development: bool) -> int:
    """Un informe final (no de desarrollo) con errores del modelo termina con código 2."""
    return EXIT_MODEL_ERRORS if errors > 0 and not development else 0


# ───────────────────────────── predictor ─────────────────────────────

def build_predictor(args):
    """
    Devuelve una función predict(text) -> corrección recomendada, según el origen
    elegido. En modo http, `predict.server_info["modelVersions"]` acumula los
    `modelVersion` distintos que devolvió el servidor (la fuente de la versión
    del informe).
    """
    if args.source == "http":
        import urllib.request

        server_info = {"modelVersions": []}

        def predict(text: str) -> str:
            payload = json.dumps({"originalText": text, "studentId": "eval"}).encode("utf-8")
            req = urllib.request.Request(
                args.url.rstrip("/") + "/interno/corregir",
                data=payload, headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=40) as r:
                body = json.loads(r.read().decode("utf-8"))
            version = str(body.get("modelVersion", "") or "").strip()
            if version and version not in server_info["modelVersions"]:
                server_info["modelVersions"].append(version)
            return body.get("correctedText", "")

        predict.server_info = server_info
        return predict

    # --- pipeline en-proceso (torch/modelos se cargan solo aquí) ---
    t5_dir = require_local_t5_dir(args.t5_dir) if args.t5_dir else None
    sys.path.insert(0, str(REPO_DIR))
    from infrastructure.nlp.phonetic_engine import PhoneticEngine
    from infrastructure.nlp.correction_pipeline import CorrectionPipeline

    phonetic = PhoneticEngine()
    judge = None
    if not args.no_beto:
        from infrastructure.nlp.context_judge import ContextJudge
        judge = ContextJudge()

    seq2seq = tokenizer = None
    if t5_dir is not None:
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
    """Añade tp/fp/fn, la referencia elegida y `exact` a cada caso {cat, input, golds, pred[, error]}."""
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


def _category_name(cat: str) -> str:
    """Nombre como lo verán panel y backend: recortado a 80 y sin espacios en los extremos."""
    return cat.strip()[:MAX_CATEGORY_NAME].strip()


def _category_results(by_cat: "OrderedDict") -> list:
    """Bloque `categories` del contrato: nombres únicos recortados; más de 50 es un error del dataset."""
    if len(by_cat) > MAX_CATEGORIES:
        raise ValueError(f"{len(by_cat)} categorías; el backend acepta como mucho {MAX_CATEGORIES}")
    categories = []
    for name, subset in by_cat.items():
        tp = sum(r["tp"] for r in subset)
        fp = sum(r["fp"] for r in subset)
        fn = sum(r["fn"] for r in subset)
        precision, recall, f0_5 = prf(tp, fp, fn)
        categories.append({"category": name, "tp": tp, "fp": fp, "fn": fn,
                           "precision": precision, "recall": recall, "f05": f0_5})
    return categories


def build_report(cases: list, *, model_version: str, dataset_path: str, dataset_hash: str,
                 model_dir, development: bool, source: str, latencies_ms=None, pipeline=None,
                 client_pipeline=None) -> dict:
    """
    Informe JSON versionado. `cases` = [{cat, input, golds, pred[, error]}, ...]
    en el orden del dataset. Las claves del contrato (`modelVersion`,
    `datasetSha256`, `scorerVersion`, `precision`, `recall`, `fZeroFive`,
    `truePositives`, `falsePositives`, `falseNegatives`, `categories`) van en
    el **nivel superior**, que es lo que valida el panel Vue y acepta el
    backend; `technicalEvaluation` las repite agrupadas, para leerlas de un
    vistazo. El resto son diagnósticos, incluido `pipeline` (`pipeline_info()`:
    umbrales de la capa 5 y commit del código que corrió) y, solo en modo
    HTTP, `clientPipeline` (la misma información pero del checkout que evalúa,
    que no es el que corrigió; ver `report_pipeline_blocks`).
    """
    model_version = validate_model_version(model_version)
    scored = _score_cases(cases)
    latencies_ms = list(latencies_ms or [])
    for row, ms in zip(scored, latencies_ms):
        row["ms"] = ms

    by_cat = OrderedDict()
    for r in scored:
        by_cat.setdefault(_category_name(r["cat"]), []).append(r)
    overall = summarize(scored)
    error_list = [{"cat": r["cat"], "input": r["input"], "error": r["error"]} for r in scored if "error" in r]
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
    report = OrderedDict([
        ("development", bool(development)),
        ("timestamp", datetime.now(timezone.utc).isoformat(timespec="seconds")),
        ("source", source),
        ("modelDir", str(model_dir) if model_dir else None),
        ("dataset", str(dataset_path)),
        ("pipeline", dict(pipeline) if pipeline else None),
    ])
    if client_pipeline is not None:
        report["clientPipeline"] = dict(client_pipeline)
    report.update(technical)
    report.update([
        ("technicalEvaluation", technical),
        ("errors", len(error_list)),
        ("errorList", error_list),
        ("global", overall),
        ("por_categoria", OrderedDict((cat, summarize(subset)) for cat, subset in by_cat.items())),
        ("latency_ms_avg", sum(latencies_ms) // len(latencies_ms) if latencies_ms else 0),
        ("casos", scored),
    ])
    return report


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
                    help="identificador del modelo evaluado (default: MODEL_VERSION > con --t5-dir, el "
                         f"{MODEL_MANIFEST_NAME} de ese directorio > en http, el del servidor; obligatorio "
                         "sin --t5-dir o con --no-beto)")
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

    ensure_offline()
    rows = load_dataset(args.dataset)
    dataset_hash = dataset_sha256(args.dataset)
    development = is_development(args.dataset, args.development)
    if args.source == "http":
        label = "HTTP " + args.url
    else:
        label = "reglas" + ("" if args.no_beto else " + BETO") + (" + T5" if args.t5_dir else "")

    # La versión del informe describe lo evaluado (ver evaluation_model_version).
    # En modo http la da el servidor y solo se conoce tras la primera respuesta.
    def resolve_report_version(server_version=None) -> str:
        try:
            return validate_model_version(evaluation_model_version(
                args.model_version, source=args.source, t5_dir=args.t5_dir, no_beto=args.no_beto,
                server_version=server_version))
        except ValueError as exc:
            print(f"[ERROR] modelVersion: {exc}", file=sys.stderr)
            sys.exit(EXIT_BAD_VERSION)

    model_version = resolve_report_version() if args.source != "http" else None
    print(f"\n{'='*96}\n  EVALUACIÓN — {len(rows)} casos — fuente: {label}\n"
          f"  modelVersion: {model_version or '(la devuelve el servidor)'}   scorer: {SCORER_VERSION}\n"
          f"  dataset: {args.dataset}  sha256: {dataset_hash}"
          f"{'   [DESARROLLO]' if development else ''}\n{'='*96}")

    predict = build_predictor(args)

    cases, latencies = [], []
    for cat, inp, references in rows:
        t0 = time.time()
        case = {"cat": cat, "input": inp, "golds": references}
        try:
            case["pred"] = predict(inp)
        except Exception as e:
            print(f"[ERROR] '{inp[:40]}...' -> {type(e).__name__}: {e}")
            case["pred"] = inp                      # se puntúa como "sin cambios"
            case["error"] = f"{type(e).__name__}: {e}"
        latencies.append(int((time.time() - t0) * 1000))
        cases.append(case)

    if model_version is None:                      # http: la versión la dio el servidor
        versions = getattr(predict, "server_info", {}).get("modelVersions", [])
        if len(versions) > 1:
            print(f"[ERROR] modelVersion: el servidor devolvió varias versiones durante la evaluación: {versions}",
                  file=sys.stderr)
            sys.exit(EXIT_BAD_VERSION)
        model_version = resolve_report_version(versions[0] if versions else None)
        print(f"  modelVersion (servidor): {model_version}")

    # En proceso, `pipeline` describe el código que corrigió; por HTTP solo se
    # conoce el modelVersion del servidor y la metadata local va como cliente.
    pipeline, client_pipeline = report_pipeline_blocks(args.source, pipeline_info())
    report = build_report(
        cases, model_version=model_version, dataset_path=args.dataset, dataset_hash=dataset_hash,
        model_dir=args.t5_dir, development=development, source=label, latencies_ms=latencies,
        pipeline=pipeline, client_pipeline=client_pipeline,
    )
    _print_table(report)
    print(f"\n  F0.5 = {report['fZeroFive']:.4f}  (P = {report['precision']:.4f}, R = {report['recall']:.4f}; "
          f"TP {report['truePositives']}, FP {report['falsePositives']}, FN {report['falseNegatives']})")
    print(f"  latencia media: {report['latency_ms_avg']} ms/frase")
    if report["errors"]:
        print(f"  [ERROR] {report['errors']} caso(s) con excepción del modelo (ver `errorList`)")

    if args.show_fails:
        print(f"\n{'─'*96}\n  FALLOS\n{'─'*96}")
        for r in report["casos"]:
            if not r["exact"]:
                print(f"[{r['cat']}] tp={r['tp']} fp={r['fp']} fn={r['fn']}"
                      f"{'  ERROR: ' + r['error'] if 'error' in r else ''}\n"
                      f"  IN : {r['input']}\n  OUT: {r['pred']}\n  ESP: {r['gold']}\n")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"  reporte guardado en {args.out}")

    status = exit_status(report["errors"], development)
    if status:
        print(f"  informe final con errores del modelo: código de salida {status}")
    sys.exit(status)


if __name__ == "__main__":
    main()
