"""
scripts/analyze_user_experiment.py

Reproduce offline las métricas del estudio con usuarios (PEO, PPM, TAS y TAS
aceptada) a partir del CSV de análisis del backend
(`GET /api/v1/research/studies/{id}/analysis.csv`) con las mismas fórmulas y
reglas de muestra que `StudyMetricsService` (backend), de modo que el análisis
independiente valide los números de `GET …/results`.

Entrada: CSV de 19 columnas, una fila por ejecución completada × sugerencia
evaluada (`pseudonym, condition, task, protocol_version, included, excluded,
run_id, duration_ms, incident_reasons, word_count, orthography_errors,
orthography_batch_id, final_text, suggestion_index, original_text,
suggestion, semantic_score, accepted, semantic_batch_id`).

Palabras: se usa la columna `word_count` del backend (autoritativa) siempre
que venga; la regex `WORD` solo sirve de comprobación (aviso si no coincide)
y de respaldo si la columna falta. Es una APROXIMACIÓN de la clase `\w` de
Java con `UNICODE_CHARACTER_CLASS`: difiere en marcas combinantes (por eso se
normaliza a NFC antes de contar), en numerales Nl/No y en conectores Pc.

Fórmulas (idénticas al backend):
  * PEO = errores ortográficos adjudicados / palabras del texto final × 100,
    por ejecución; media por participante-condición; Δ = asistida − sin
    asistencia; IC 95 % t de Student, prueba t emparejada bilateral y d_z.
  * PPM = palabras / (duración_ms / 60000); sin palabras contables vale 0.
  * TAS = sugerencias adjudicadas con puntaje 0 / evaluadas × 100 agrupado,
    con intervalo de Wilson (z = 1.959964); media ± sd por participante
    (descriptiva, IC t acotado a [0, 100]). TAS aceptada: solo `accepted=true`.
  * Muestra: par completo = al menos una ejecución COMPLETED no excluida por
    condición. PEO solo participantes con palabras contables en ambas
    condiciones; PPM toda la cohorte incluida; TAS ejecuciones ASSISTED con al
    menos una sugerencia evaluada.

Solo librería estándar: la distribución t (cuantil y función de distribución)
se implementa con la función beta incompleta regularizada (`math.lgamma` +
fracción continua de Lentz), porque `scipy` no está instalado en el entorno.

Uso:
    .venv\\Scripts\\python.exe scripts/analyze_user_experiment.py analysis.csv \\
        [--ppm-margin 2.0] [--tas-limit 10] [--json out.json] [--markdown out.md] \\
        [--compare-results results.json] [--tolerance 1e-6]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import unicodedata
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

FORMULA_VERSION = "user_study_v1"
"""Versión de fórmulas/parser; cambia si cambia cualquier regla de cálculo o de muestra."""

Z_95 = 1.959964
"""Cuantil 0.975 de la normal estándar usado por el backend en el intervalo de Wilson."""

CONFIDENCE = 0.95

WORD = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*", re.UNICODE)
"""Misma regla que `WordTokenizer.WORD` del backend (letras/dígitos Unicode con apóstrofes o guiones
internos); aproximación de `\w` de Java (ver docstring del módulo): la columna `word_count` manda."""

CONDITIONS = ("ASSISTED", "UNASSISTED")

COLUMNS = (
    "pseudonym", "condition", "task", "protocol_version", "included", "excluded", "run_id", "duration_ms",
    "incident_reasons", "word_count", "orthography_errors", "orthography_batch_id", "final_text",
    "suggestion_index", "original_text", "suggestion", "semantic_score", "accepted", "semantic_batch_id",
)
OPTIONAL_COLUMNS = {"word_count", "incident_reasons", "task", "protocol_version"}
RUN_LEVEL_COLUMNS = (
    "pseudonym", "condition", "task", "protocol_version", "included", "excluded", "duration_ms",
    "incident_reasons", "word_count", "orthography_errors", "orthography_batch_id", "final_text",
)

PAIRED_KEYS = (
    "n", "assistedMean", "assistedSd", "unassistedMean", "unassistedSd", "meanDelta", "sdDelta",
    "ci95Lower", "ci95Upper", "tStatistic", "pValue", "cohenDz",
)
TAS_KEYS = (
    "participantsEvaluated", "participantsWithoutDenominator", "runsAnalyzed", "suggestionsEvaluated",
    "harmfulSuggestions", "pooledRate", "pooledCi95Lower", "pooledCi95Upper", "participantMean",
    "participantSd", "participantCi95Lower", "participantCi95Upper", "descriptive", "limit", "upperCiBelowLimit",
)
PARTICIPANT_KEYS = (
    "pseudonym", "peoAssisted", "peoUnassisted", "peoDelta", "peoRelativeReduction", "ppmAssisted",
    "ppmUnassisted", "ppmDelta", "tas", "tasAccepted",
)


class AnalysisValidationError(ValueError):
    """El CSV no cumple el contrato del backend; el mensaje identifica la fila y la columna."""


# --------------------------------------------------------------------------- distribución t (sin scipy)


def _betacf(a: float, b: float, x: float) -> float:
    """Fracción continua de la beta incompleta (Lentz modificado, Numerical Recipes)."""
    tiny = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 500):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return h


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """I_x(a, b) con precisión ~1e-15; base de la función de distribución t."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                 + a * math.log(x) + b * math.log(1.0 - x))
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: int) -> float:
    """Función de distribución de la t de Student con `df` grados de libertad."""
    if df <= 0:
        raise ValueError("df must be positive")
    if t == 0.0:
        return 0.5
    x = df / (df + t * t)
    tail = 0.5 * regularized_incomplete_beta(df / 2.0, 0.5, x)
    return 1.0 - tail if t > 0 else tail


def t_quantile(df: int, p: float) -> float:
    """Cuantil `p` de la t de Student (equivale a `TDistribution.inverseCumulativeProbability`)."""
    if df <= 0:
        raise ValueError("df must be positive")
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    if p == 0.5:
        return 0.0
    if p < 0.5:
        return -t_quantile(df, 1.0 - p)
    lower, upper = 0.0, 1.0
    while t_cdf(upper, df) < p:
        upper *= 2.0
        if upper > 1e12:
            break
    for _ in range(300):
        mid = 0.5 * (lower + upper)
        if t_cdf(mid, df) < p:
            lower = mid
        else:
            upper = mid
        if upper - lower <= 1e-14 * max(1.0, upper):
            break
    return 0.5 * (lower + upper)


# --------------------------------------------------------------------------- estadística descriptiva


def _mean(values: Sequence[float]) -> float:
    """Media con suma secuencial, como `StudyMetricsService.mean`."""
    total = 0.0
    for value in values:
        total += value
    return total / len(values)


def _sample_sd(values: Sequence[float]) -> Optional[float]:
    """Desviación muestral (n − 1); `None` con menos de dos valores, como el backend."""
    if len(values) < 2:
        return None
    mean = _mean(values)
    squares = 0.0
    for value in values:
        squares += (value - mean) * (value - mean)
    return math.sqrt(squares / (len(values) - 1))


def _clamp_percent(value: Optional[float]) -> Optional[float]:
    return None if value is None else max(0.0, min(100.0, value))


def interval(values: Sequence[float]) -> Dict[str, Any]:
    """Media, sd muestral e IC 95 % (t) de un valor por participante; inferencia solo con n ≥ 2 y sd > 0."""
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "lower": None, "upper": None}
    mean = _mean(values)
    if n < 2:
        return {"n": n, "mean": mean, "sd": None, "lower": None, "upper": None}
    sd = _sample_sd(values)
    if sd == 0.0:
        return {"n": n, "mean": mean, "sd": sd, "lower": None, "upper": None}
    half = t_quantile(n - 1, 1 - (1 - CONFIDENCE) / 2) * sd / math.sqrt(n)
    return {"n": n, "mean": mean, "sd": sd, "lower": mean - half, "upper": mean + half}


def paired_summary(pairs: Sequence[Tuple[float, float]]) -> Dict[str, Any]:
    """Comparación emparejada `assisted − unassisted` (`PairedSummary` del backend)."""
    n = len(pairs)
    if n == 0:
        return {key: (0 if key == "n" else None) for key in PAIRED_KEYS}
    assisted = [a for a, _ in pairs]
    unassisted = [u for _, u in pairs]
    deltas = [a - u for a, u in pairs]
    delta = interval(deltas)
    t_stat = p_value = dz = None
    if delta["lower"] is not None:
        # TTest.pairedT / pairedTTest de commons-math: t = media(Δ) / (sd(Δ) / √n), p bilateral.
        t_stat = delta["mean"] / (delta["sd"] / math.sqrt(n))
        p_value = 2.0 * t_cdf(-abs(t_stat), n - 1)
        dz = delta["mean"] / delta["sd"]
    return {
        "n": n,
        "assistedMean": _mean(assisted),
        "assistedSd": _sample_sd(assisted),
        "unassistedMean": _mean(unassisted),
        "unassistedSd": _sample_sd(unassisted),
        "meanDelta": delta["mean"],
        "sdDelta": delta["sd"],
        "ci95Lower": delta["lower"],
        "ci95Upper": delta["upper"],
        "tStatistic": t_stat,
        "pValue": p_value,
        "cohenDz": dz,
    }


def wilson(successes: int, n: int) -> Tuple[Optional[float], Optional[float]]:
    """Intervalo de Wilson (95 %) de una proporción agregada, en porcentaje y acotado a [0, 100]."""
    if n <= 0:
        return None, None
    p = successes / n
    z2n = Z_95 * Z_95 / n
    center = (p + z2n / 2.0) / (1.0 + z2n)
    half = Z_95 / (1.0 + z2n) * math.sqrt(p * (1.0 - p) / n + Z_95 * Z_95 / (4.0 * n * n))
    return _clamp_percent(100.0 * (center - half)), _clamp_percent(100.0 * (center + half))


def relative_reduction(unassisted: float, assisted: float) -> Optional[float]:
    """`(sin asistencia − asistida) / sin asistencia × 100`; no definido cuando el PEO sin asistencia es 0."""
    return None if unassisted == 0.0 else 100.0 * (unassisted - assisted) / unassisted


# --------------------------------------------------------------------------- fórmulas por fila


def word_count(text: Optional[str]) -> int:
    """Palabras contables del texto según `WORD` (aproximación de la regla del backend; NFC primero)."""
    if not text or not text.strip():
        return 0
    return sum(1 for _ in WORD.finditer(unicodedata.normalize("NFC", text)))


def _parse_int(value: Optional[str], column: str, minimum: int = 0, blank_ok: bool = False) -> Optional[int]:
    text = (value or "").strip()
    if text == "":
        if blank_ok:
            return None
        raise AnalysisValidationError(f"{column} is blank")
    try:
        number = int(text)
    except ValueError as exc:
        raise AnalysisValidationError(f"{column} must be an integer (was {text!r})") from exc
    if number < minimum:
        raise AnalysisValidationError(f"{column} must be >= {minimum} (was {number})")
    return number


def _parse_bool(value: Optional[str], column: str, blank_ok: bool = False) -> Optional[bool]:
    text = (value or "").strip().lower()
    if text == "":
        if blank_ok:
            return None
        raise AnalysisValidationError(f"{column} is blank")
    if text not in ("true", "false"):
        raise AnalysisValidationError(f"{column} must be true|false (was {value!r})")
    return text == "true"


def row_metrics(row: Dict[str, str]) -> Dict[str, Any]:
    """Palabras, PEO y PPM de una ejecución sobre las columnas reales del CSV.

    Usa `word_count` cuando viene informado; si no, cuenta con `WORD`. PEO es `None` sin palabras contables o
    sin puntaje adjudicado (`orthography_errors` en blanco). Duración no positiva → error, como el backend.
    """
    duration = _parse_int(row.get("duration_ms"), "duration_ms", minimum=1)
    counted = _parse_int(row.get("word_count"), "word_count", blank_ok=True)
    words = counted if counted is not None else word_count(row.get("final_text"))
    errors = _parse_int(row.get("orthography_errors"), "orthography_errors", blank_ok=True)
    peo = None if (errors is None or words == 0) else 100.0 * errors / words
    ppm = words / (duration / 60000.0)
    return {"words": words, "peo": peo, "ppm": ppm}


def _is_harmful(row: Dict[str, str]) -> bool:
    return (row.get("semantic_score") or "").strip() == "0"


def _is_accepted(row: Dict[str, str]) -> bool:
    return (row.get("accepted") or "").strip().lower() == "true"


def semantic_rates(rows: Iterable[Dict[str, str]]) -> Dict[str, Optional[float]]:
    """TAS y TAS aceptada agrupadas sobre filas de sugerencia con puntaje adjudicado (`None` sin denominador)."""
    evaluated = [row for row in rows if (row.get("semantic_score") or "").strip() != ""]
    accepted = [row for row in evaluated if _is_accepted(row)]
    harmful = sum(1 for row in evaluated if _is_harmful(row))
    harmful_accepted = sum(1 for row in accepted if _is_harmful(row))
    tas = None if not evaluated else 100.0 * harmful / len(evaluated)
    tas_accepted = None if not accepted else 100.0 * harmful_accepted / len(accepted)
    return {"tas": tas, "tas_accepted": tas_accepted}


# --------------------------------------------------------------------------- carga y validación


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(path: Path) -> Tuple[List[Dict[str, str]], str]:
    """Lee y valida el CSV; devuelve las filas (con `_line`) y el SHA-256 del archivo."""
    path = Path(path)
    digest = sha256_of(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        missing = [column for column in COLUMNS if column not in header and column not in OPTIONAL_COLUMNS]
        if missing:
            raise AnalysisValidationError(f"missing column(s): {', '.join(missing)}")
        unknown = [column for column in header if column not in COLUMNS]
        if unknown:
            raise AnalysisValidationError(f"unknown column(s): {', '.join(unknown)}")
        rows: List[Dict[str, str]] = []
        for index, raw in enumerate(reader, start=2):
            if raw.get(None):
                raise AnalysisValidationError(
                    f"line {index}: too many fields ({len(header) + len(raw[None])} > {len(header)})")
            row = {column: (raw.get(column) or "") for column in COLUMNS}
            row["_line"] = index
            rows.append(row)
    _validate(rows)
    return rows, digest


def _validate(rows: List[Dict[str, str]]) -> None:
    """Reglas deterministas: tipos, rangos, duplicados, coherencia por ejecución y bandera `included`."""
    seen_suggestions = set()
    run_rows: Dict[str, List[Dict[str, str]]] = OrderedDict()
    for row in rows:
        line = row["_line"]
        try:
            if not row["pseudonym"].strip():
                raise AnalysisValidationError("pseudonym is blank")
            if row["condition"] not in CONDITIONS:
                raise AnalysisValidationError(f"condition must be ASSISTED|UNASSISTED (was {row['condition']!r})")
            if not row["run_id"].strip():
                raise AnalysisValidationError("run_id is blank")
            _parse_bool(row["included"], "included")
            _parse_bool(row["excluded"], "excluded")
            _parse_int(row["protocol_version"], "protocol_version", blank_ok=True)
            metrics = row_metrics(row)
            errors = _parse_int(row["orthography_errors"], "orthography_errors", blank_ok=True)
            if errors is not None and errors > metrics["words"]:
                raise AnalysisValidationError(
                    f"orthography_errors ({errors}) exceeds the word count ({metrics['words']})")
            index = _parse_int(row["suggestion_index"], "suggestion_index", blank_ok=True)
            score = _parse_int(row["semantic_score"], "semantic_score", blank_ok=True)
            if score is not None and score > 2:
                raise AnalysisValidationError(f"semantic_score must be 0|1|2 (was {score})")
            _parse_bool(row["accepted"], "accepted", blank_ok=True)
            if index is None and (score is not None or row["accepted"].strip()):
                raise AnalysisValidationError("semantic_score/accepted without suggestion_index")
            key = (row["run_id"], index)
            if key in seen_suggestions:
                raise AnalysisValidationError(
                    f"duplicate row for run_id {row['run_id']} suggestion_index {index if index is not None else ''}")
            seen_suggestions.add(key)
        except AnalysisValidationError as exc:
            raise AnalysisValidationError(f"line {line}: {exc}") from None
        run_rows.setdefault(row["run_id"], []).append(row)

    for run_id, group in run_rows.items():
        first = group[0]
        for other in group[1:]:
            for column in RUN_LEVEL_COLUMNS:
                if first[column] != other[column]:
                    raise AnalysisValidationError(
                        f"line {other['_line']}: run_id {run_id} has inconsistent {column} across its rows "
                        f"({first[column]!r} at line {first['_line']} vs {other[column]!r})")
        blank = [row for row in group if not row["suggestion_index"].strip()]
        if blank and len(group) > 1:
            raise AnalysisValidationError(
                f"line {blank[0]['_line']}: run_id {run_id} mixes a row without suggestion with suggestion rows")

    cohort = build_cohort(rows)
    for participant in cohort["participants"]:
        for run in participant["runs"]:
            if run["included"] != participant["complete_pair"] and not run["excluded"]:
                raise AnalysisValidationError(
                    f"line {run['line']}: included={str(run['included']).lower()} for run_id {run['run_id']} but the "
                    f"complete-pair rule gives {str(participant['complete_pair']).lower()} for {participant['pseudonym']}")
            if run["included"] and run["excluded"]:
                raise AnalysisValidationError(
                    f"line {run['line']}: run_id {run['run_id']} is both included and excluded")


# --------------------------------------------------------------------------- cohorte


def build_cohort(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    """Agrupa filas en ejecuciones y participantes y aplica las reglas de muestra del backend (`Cohort.of`)."""
    runs: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    for row in rows:
        run = runs.get(row["run_id"])
        if run is None:
            metrics = row_metrics(row)
            errors = _parse_int(row["orthography_errors"], "orthography_errors", blank_ok=True)
            run = {
                "run_id": row["run_id"],
                "line": row["_line"],
                "pseudonym": row["pseudonym"],
                "condition": row["condition"],
                "task": row["task"],
                "protocol_version": _parse_int(row["protocol_version"], "protocol_version", blank_ok=True),
                "included": _parse_bool(row["included"], "included"),
                "excluded": _parse_bool(row["excluded"], "excluded"),
                "duration_ms": int(row["duration_ms"]),
                "incident_reasons": row["incident_reasons"],
                "words": metrics["words"],
                "counted_words": word_count(row["final_text"]),
                "errors": errors,
                "orthography_batch_id": row["orthography_batch_id"],
                "peo": metrics["peo"],
                "ppm": metrics["ppm"],
                "suggestions": [],
            }
            runs[row["run_id"]] = run
        if row["suggestion_index"].strip():
            score = _parse_int(row["semantic_score"], "semantic_score", blank_ok=True)
            run["suggestions"].append({
                "index": int(row["suggestion_index"]),
                "score": score,
                "harmful": score == 0,
                "accepted": _is_accepted(row),
                "semantic_batch_id": row["semantic_batch_id"],
            })

    by_participant: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    for run in runs.values():
        by_participant.setdefault(run["pseudonym"], []).append(run)

    participants = []
    for pseudonym in sorted(by_participant):
        completed = by_participant[pseudonym]
        eligible = {condition: [run for run in completed if not run["excluded"] and run["condition"] == condition]
                    for condition in CONDITIONS}
        complete_pair = all(eligible[condition] for condition in CONDITIONS)
        peo_runs = {condition: [run for run in eligible[condition] if run["words"] > 0] for condition in CONDITIONS}
        participants.append({
            "pseudonym": pseudonym,
            "runs": completed,
            "eligible": eligible,
            "complete_pair": complete_pair,
            "eligible_runs": eligible["ASSISTED"] + eligible["UNASSISTED"],
            "peo_runs": peo_runs,
            "peo_pair": all(peo_runs[condition] for condition in CONDITIONS),
        })
    included = [p for p in participants if p["complete_pair"]]
    incomplete = [p for p in participants if not p["complete_pair"] and p["eligible_runs"]]
    without_eligible = [p for p in participants if not p["eligible_runs"]]
    included_runs = [run for p in included for run in p["eligible_runs"]]
    sample = {
        "participantsTotal": len(participants),
        "participantsIncluded": len(included),
        "participantsWithIncompletePair": len(incomplete),
        "participantsWithoutEligibleRun": len(without_eligible),
        "runsCompleted": sum(len(p["runs"]) for p in participants),
        "runsIncluded": len(included_runs),
        "runsExcluded": sum(1 for p in participants for run in p["runs"] if run["excluded"]),
        "runsInIncompletePairs": sum(len(p["eligible_runs"]) for p in incomplete),
        "runsWithoutCountableWords": sum(1 for run in included_runs if run["words"] == 0),
    }
    return {
        "runs": runs,
        "participants": participants,
        "included": included,
        "incomplete": incomplete,
        "without_eligible": without_eligible,
        "included_runs": included_runs,
        "sample": sample,
    }


# --------------------------------------------------------------------------- análisis


def _stream_mean(values: Sequence[float]) -> float:
    """Media al estilo `DoubleStream.average()` (suma compensada); equivalente a `fsum / n` en la práctica."""
    return math.fsum(values) / len(values)


def _run_rate(run: Dict[str, Any], accepted_only: bool) -> Optional[float]:
    suggestions = [s for s in run["suggestions"] if not accepted_only or s["accepted"]]
    if not suggestions:
        return None
    return 100.0 * sum(1 for s in suggestions if s["harmful"]) / len(suggestions)


def _participant_rate(participant: Dict[str, Any], accepted_only: bool) -> Tuple[Optional[float], int]:
    """Media de la tasa por ejecución ASSISTED elegible y número de ejecuciones con denominador."""
    rates = [rate for rate in (_run_rate(run, accepted_only) for run in participant["eligible"]["ASSISTED"])
             if rate is not None]
    return (_stream_mean(rates) if rates else None), len(rates)


def _orthography_status(cohort: Dict[str, Any]) -> Dict[str, Any]:
    if not cohort["included"]:
        return {"status": "NO_SAMPLE", "batchIds": [], "message": "No participant has a complete pair of included runs"}
    peo_runs = [run for p in cohort["included"] if p["peo_pair"] for c in CONDITIONS for run in p["peo_runs"][c]]
    if not peo_runs:
        return {"status": "NOT_APPLICABLE", "batchIds": [],
                "message": "No included participant has countable words in both conditions"}
    batches = sorted({run["orthography_batch_id"] for run in peo_runs if run["orthography_batch_id"]})
    covered = sum(1 for run in peo_runs if run["errors"] is not None)
    if covered == 0:
        return {"status": "NO_BATCH", "batchIds": batches,
                "message": "No adjudicated orthography score in the export"}
    if covered < len(peo_runs):
        return {"status": "INCOMPLETE_COVERAGE", "batchIds": batches,
                "message": f"{covered} of {len(peo_runs)} included runs have an adjudicated orthography score"}
    return {"status": "ADJUDICATED", "batchIds": batches,
            "message": f"Adjudicated orthography scores cover all {len(peo_runs)} included runs"}


def _semantic_status(cohort: Dict[str, Any]) -> Dict[str, Any]:
    if not cohort["included"]:
        return {"status": "NO_SAMPLE", "batchIds": [], "message": "No participant has a complete pair of included runs"}
    suggestions = [s for run in cohort["included_runs"] for s in run["suggestions"]]
    if not suggestions:
        return {"status": "NOT_APPLICABLE", "batchIds": [],
                "message": "The included ASSISTED runs received no suggestions to evaluate"}
    batches = sorted({s["semantic_batch_id"] for s in suggestions if s["semantic_batch_id"]})
    covered = sum(1 for s in suggestions if s["score"] is not None)
    if covered == 0:
        return {"status": "NO_BATCH", "batchIds": batches, "message": "No adjudicated semantic score in the export"}
    if covered < len(suggestions):
        return {"status": "INCOMPLETE_COVERAGE", "batchIds": batches,
                "message": f"{covered} of {len(suggestions)} included suggestions have an adjudicated semantic score"}
    return {"status": "ADJUDICATED", "batchIds": batches,
            "message": f"Adjudicated semantic scores cover all {len(suggestions)} included suggestions"}


def _tas_result(values: List[float], without: int, runs: int, evaluated: int, harmful: int,
                limit: Optional[float]) -> Dict[str, Any]:
    """`TasResult` del backend: Wilson agregado (criterio) e IC t por participante acotado (descriptivo)."""
    participants = interval(values)
    pooled = None if evaluated == 0 else 100.0 * harmful / evaluated
    lower, upper = wilson(harmful, evaluated)
    below = None if (limit is None or upper is None) else upper < limit
    return {
        "participantsEvaluated": participants["n"],
        "participantsWithoutDenominator": without,
        "runsAnalyzed": runs,
        "suggestionsEvaluated": evaluated,
        "harmfulSuggestions": harmful,
        "pooledRate": pooled,
        "pooledCi95Lower": lower,
        "pooledCi95Upper": upper,
        "participantMean": participants["mean"],
        "participantSd": participants["sd"],
        "participantCi95Lower": _clamp_percent(participants["lower"]),
        "participantCi95Upper": _clamp_percent(participants["upper"]),
        "descriptive": limit is None,
        "limit": limit,
        "upperCiBelowLimit": below,
    }


def analyze(rows: List[Dict[str, str]], input_sha256: str = "", input_name: str = "",
            ppm_margin: Optional[float] = None, tas_limit: Optional[float] = None) -> Dict[str, Any]:
    """Informe completo con la forma de `StudyResultsResponse` más versión de fórmulas, hash y exclusiones."""
    if ppm_margin is not None and not (math.isfinite(ppm_margin) and ppm_margin > 0):
        raise ValueError("ppm_margin must be a finite number greater than 0")
    if tas_limit is not None and not (math.isfinite(tas_limit) and 0.0 <= tas_limit <= 100.0):
        raise ValueError("tas_limit must be a finite percentage between 0 and 100")
    cohort = build_cohort(rows)
    orthography = _orthography_status(cohort)
    semantic = _semantic_status(cohort)
    peo_ready = orthography["status"] == "ADJUDICATED"
    tas_ready = semantic["status"] == "ADJUDICATED"

    participants_out = []
    peo_pairs: List[Tuple[float, float]] = []
    peo_without_words = 0
    reductions: List[float] = []
    reductions_skipped = 0
    ppm_pairs: List[Tuple[float, float]] = []
    tas_values: List[float] = []
    tas_without = tas_runs = 0
    tas_acc_values: List[float] = []
    tas_acc_without = tas_acc_runs = 0

    for participant in cohort["included"]:
        ppm_assisted = _stream_mean([run["ppm"] for run in participant["eligible"]["ASSISTED"]])
        ppm_unassisted = _stream_mean([run["ppm"] for run in participant["eligible"]["UNASSISTED"]])
        ppm_pairs.append((ppm_assisted, ppm_unassisted))
        peo_assisted = peo_unassisted = peo_delta = reduction = None
        if peo_ready and participant["peo_pair"]:
            peo_assisted = _stream_mean([run["peo"] for run in participant["peo_runs"]["ASSISTED"]])
            peo_unassisted = _stream_mean([run["peo"] for run in participant["peo_runs"]["UNASSISTED"]])
            peo_delta = peo_assisted - peo_unassisted
            peo_pairs.append((peo_assisted, peo_unassisted))
            reduction = relative_reduction(peo_unassisted, peo_assisted)
            if reduction is None:
                reductions_skipped += 1
            else:
                reductions.append(reduction)
        elif peo_ready:
            peo_without_words += 1
        tas = tas_accepted = None
        if tas_ready:
            tas, runs_with = _participant_rate(participant, False)
            tas_accepted, runs_with_accepted = _participant_rate(participant, True)
            tas_runs += runs_with
            tas_acc_runs += runs_with_accepted
            if tas is None:
                tas_without += 1
            else:
                tas_values.append(tas)
            if tas_accepted is None:
                tas_acc_without += 1
            else:
                tas_acc_values.append(tas_accepted)
        participants_out.append({
            "pseudonym": participant["pseudonym"],
            "peoAssisted": peo_assisted,
            "peoUnassisted": peo_unassisted,
            "peoDelta": peo_delta,
            "peoRelativeReduction": reduction,
            "ppmAssisted": ppm_assisted,
            "ppmUnassisted": ppm_unassisted,
            "ppmDelta": ppm_assisted - ppm_unassisted,
            "tas": tas,
            "tasAccepted": tas_accepted,
        })

    peo = None
    if peo_ready and peo_pairs:
        paired = paired_summary(peo_pairs)
        peo_runs = sum(len(p["peo_runs"][c]) for p in cohort["included"] if p["peo_pair"] for c in CONDITIONS)
        peo = {
            "paired": paired,
            "participantsAnalyzed": len(peo_pairs),
            "participantsWithoutCountableWords": peo_without_words,
            "runsAnalyzed": peo_runs,
            "relativeReductionMean": _mean(reductions) if reductions else None,
            "relativeReductionN": len(reductions),
            "relativeReductionSkipped": reductions_skipped,
            "upperCiBelowZero": None if paired["ci95Upper"] is None else paired["ci95Upper"] < 0,
        }
    ppm = None
    if ppm_pairs:
        paired = paired_summary(ppm_pairs)
        non_inferior = None if (ppm_margin is None or paired["ci95Lower"] is None) else paired["ci95Lower"] > -ppm_margin
        ppm = {
            "paired": paired,
            "participantsAnalyzed": len(ppm_pairs),
            "runsAnalyzed": len(cohort["included_runs"]),
            "descriptive": ppm_margin is None,
            "nonInferiorityMargin": ppm_margin,
            "nonInferior": non_inferior,
        }
    tas = tas_accepted = None
    if tas_ready:
        all_suggestions = [s for run in cohort["included_runs"] for s in run["suggestions"]]
        accepted = [s for s in all_suggestions if s["accepted"]]
        tas = _tas_result(tas_values, tas_without, tas_runs, len(all_suggestions),
                          sum(1 for s in all_suggestions if s["harmful"]), tas_limit)
        tas_accepted = _tas_result(tas_acc_values, tas_acc_without, tas_acc_runs, len(accepted),
                                   sum(1 for s in accepted if s["harmful"]), tas_limit)

    warnings = []
    mismatches = [run for run in cohort["runs"].values() if run["words"] != run["counted_words"]]
    for run in mismatches:
        warnings.append(f"run {run['run_id']} ({run['pseudonym']}): word_count={run['words']} in the CSV but the "
                        f"WORD regex counts {run['counted_words']} (the CSV value was used)")

    return {
        "formulaVersion": FORMULA_VERSION,
        "inputFile": input_name,
        "inputSha256": input_sha256,
        "rowsRead": len(rows),
        "sample": cohort["sample"],
        "excluded": {
            "participantsWithIncompletePair": [p["pseudonym"] for p in cohort["incomplete"]],
            "participantsWithoutEligibleRun": [p["pseudonym"] for p in cohort["without_eligible"]],
            "runsExcluded": [
                {"runId": run["run_id"], "pseudonym": run["pseudonym"], "condition": run["condition"],
                 "incidentReasons": run["incident_reasons"]}
                for p in cohort["participants"] for run in p["runs"] if run["excluded"]
            ],
        },
        "peo": peo,
        "ppm": ppm,
        "tas": tas,
        "tasAccepted": tas_accepted,
        "orthographyAnnotation": orthography,
        "semanticAnnotation": semantic,
        "participants": participants_out,
        "provenance": {
            "protocolVersions": sorted({run["protocol_version"] for run in cohort["included_runs"]
                                        if run["protocol_version"] is not None}),
            "orthographyBatchIds": orthography["batchIds"],
            "semanticBatchIds": semantic["batchIds"],
            "ppmNonInferiorityMargin": ppm_margin,
            "tasLimit": tas_limit,
            "tDistribution": "pure-python regularized incomplete beta (no scipy)",
        },
        "warnings": warnings,
    }


# --------------------------------------------------------------------------- reconciliación con el backend


def _close(a: Any, b: Any, tolerance: float) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= tolerance
    return a == b


def _compare_block(offline: Optional[Dict[str, Any]], backend: Optional[Dict[str, Any]], prefix: str,
                   keys: Sequence[str], tolerance: float, out: List[str]) -> None:
    if offline is None or backend is None:
        if not (offline is None and backend is None):
            out.append(f"{prefix}: offline={'null' if offline is None else 'present'} "
                       f"backend={'null' if backend is None else 'present'}")
        return
    for key in keys:
        if not _close(offline.get(key), backend.get(key), tolerance):
            out.append(f"{prefix}.{key}: offline={offline.get(key)!r} backend={backend.get(key)!r}")


NOT_ADJUDICATED_OFFLINE = ("NO_BATCH", "INCOMPLETE_COVERAGE")
"""Estados offline compatibles con `NOT_ADJUDICATED` del backend (el CSV no distingue el lote sin adjudicar)."""


def _compare_status(block: str, offline: Optional[str], backend: Optional[str], out: List[str],
                    notes: List[str]) -> None:
    """Compara la CLASE de adjudicación, no el literal.

    `ADJUDICATED` debe coincidir en ambos lados; `NO_SAMPLE` y `NOT_APPLICABLE` se derivan del CSV y se exigen
    literales; `NOT_ADJUDICATED` (el lote más reciente no tiene adjudicación vigente) no es derivable del CSV,
    que solo puede reportar `NO_BATCH` o `INCOMPLETE_COVERAGE`: se acepta con una nota.
    """
    if offline == backend:
        return
    if backend == "NOT_ADJUDICATED" and offline in NOT_ADJUDICATED_OFFLINE:
        notes.append(f"{block}.status: backend reports NOT_ADJUDICATED, which the CSV cannot distinguish from "
                     f"{offline} (no adjudicated score in the export); same adjudication class")
        return
    out.append(f"{block}.status: offline={offline!r} backend={backend!r}")


def compare_results(report: Dict[str, Any], backend: Dict[str, Any],
                    tolerance: float = 1e-6) -> Tuple[List[str], List[str]]:
    """Compara el informe offline con `GET …/results`; devuelve (discrepancias, notas informativas).

    El CSV solo exporta participantes con al menos una ejecución completada, así que `participantsTotal` y
    `participantsWithoutEligibleRun` pueden ser menores que en el backend exactamente en el número de
    participantes sin ejecuciones; esa diferencia se anota, no se considera discrepancia. Los estados de
    anotación se comparan por clase (`_compare_status`: `NOT_ADJUDICATED` del backend equivale a
    `NO_BATCH`/`INCOMPLETE_COVERAGE` offline, con nota). Todo lo demás (partición de la muestra, PEO, PPM,
    TAS, TAS aceptada y filas por participante) debe coincidir con tolerancia `tolerance`.
    """
    out: List[str] = []
    notes: List[str] = []
    sample_offline = report["sample"]
    sample_backend = backend.get("sample") or {}
    for key in ("participantsIncluded", "participantsWithIncompletePair", "runsCompleted", "runsIncluded",
                "runsExcluded", "runsInIncompletePairs", "runsWithoutCountableWords"):
        if not _close(sample_offline.get(key), sample_backend.get(key), tolerance):
            out.append(f"sample.{key}: offline={sample_offline.get(key)!r} backend={sample_backend.get(key)!r}")
    total_gap = (sample_backend.get("participantsTotal") or 0) - sample_offline["participantsTotal"]
    without_gap = (sample_backend.get("participantsWithoutEligibleRun") or 0) - sample_offline["participantsWithoutEligibleRun"]
    if total_gap != without_gap or total_gap < 0:
        out.append(f"sample.participantsTotal/participantsWithoutEligibleRun: offline="
                   f"{sample_offline['participantsTotal']}/{sample_offline['participantsWithoutEligibleRun']} backend="
                   f"{sample_backend.get('participantsTotal')!r}/{sample_backend.get('participantsWithoutEligibleRun')!r}")
    elif total_gap > 0:
        notes.append(f"sample.participantsTotal: backend counts {total_gap} participant(s) without any completed run "
                     f"that the CSV does not export (offline {sample_offline['participantsTotal']}, backend "
                     f"{sample_backend.get('participantsTotal')}); the partition identity still holds")

    for block in ("peo", "ppm"):
        offline_block = report.get(block)
        backend_block = backend.get(block)
        if offline_block is None or backend_block is None:
            _compare_block(offline_block, backend_block, block, (), tolerance, out)
            continue
        _compare_block(offline_block.get("paired"), backend_block.get("paired"), f"{block}.paired", PAIRED_KEYS,
                       tolerance, out)
        extra = (("participantsAnalyzed", "participantsWithoutCountableWords", "runsAnalyzed", "relativeReductionMean",
                  "relativeReductionN", "relativeReductionSkipped", "upperCiBelowZero") if block == "peo"
                 else ("participantsAnalyzed", "runsAnalyzed", "descriptive", "nonInferiorityMargin", "nonInferior"))
        _compare_block(offline_block, backend_block, block, extra, tolerance, out)
    for block in ("tas", "tasAccepted"):
        _compare_block(report.get(block), backend.get(block), block, TAS_KEYS, tolerance, out)
    for block in ("orthographyAnnotation", "semanticAnnotation"):
        _compare_status(block, (report.get(block) or {}).get("status"), (backend.get(block) or {}).get("status"),
                        out, notes)

    offline_rows = {row["pseudonym"]: row for row in report["participants"]}
    backend_rows = {row["pseudonym"]: row for row in backend.get("participants") or []}
    for pseudonym in sorted(set(offline_rows) | set(backend_rows)):
        _compare_block(offline_rows.get(pseudonym), backend_rows.get(pseudonym), f"participants[{pseudonym}]",
                       PARTICIPANT_KEYS[1:], tolerance, out)
    return out, notes


# --------------------------------------------------------------------------- informe Markdown


def _fmt(value: Any, digits: int = 4, null: str = "—") -> str:
    if value is None:
        return null
    if isinstance(value, bool):
        return "sí" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _paired_lines(paired: Dict[str, Any], unit: str) -> List[str]:
    return [
        f"- n = {paired['n']} participantes (pares completos).",
        f"- Asistida: media {_fmt(paired['assistedMean'])} {unit}, sd {_fmt(paired['assistedSd'])}.",
        f"- Sin asistencia: media {_fmt(paired['unassistedMean'])} {unit}, sd {_fmt(paired['unassistedSd'])}.",
        f"- Δ (asistida − sin asistencia): {_fmt(paired['meanDelta'])} {unit}, sd {_fmt(paired['sdDelta'])}, "
        f"IC 95 % [{_fmt(paired['ci95Lower'])}, {_fmt(paired['ci95Upper'])}].",
        f"- t emparejada = {_fmt(paired['tStatistic'])}, p = {_fmt(paired['pValue'], 6)}, d_z = {_fmt(paired['cohenDz'])}.",
    ]


def _tas_lines(tas: Dict[str, Any]) -> List[str]:
    return [
        f"- Sugerencias evaluadas: {tas['suggestionsEvaluated']}, perjudiciales (puntaje 0): {tas['harmfulSuggestions']}; "
        f"ejecuciones ASSISTED con denominador: {tas['runsAnalyzed']}.",
        f"- Tasa agrupada: {_fmt(tas['pooledRate'])} %, IC 95 % de Wilson [{_fmt(tas['pooledCi95Lower'])}, "
        f"{_fmt(tas['pooledCi95Upper'])}] (ignora el agrupamiento por participante).",
        f"- Por participante (n = {tas['participantsEvaluated']}, sin denominador: {tas['participantsWithoutDenominator']}): "
        f"media {_fmt(tas['participantMean'])} %, sd {_fmt(tas['participantSd'])}, IC 95 % t acotado "
        f"[{_fmt(tas['participantCi95Lower'])}, {_fmt(tas['participantCi95Upper'])}] (descriptivo).",
        f"- Límite configurado: {_fmt(tas['limit'])}; límite superior de Wilson por debajo del límite: "
        f"{_fmt(tas['upperCiBelowLimit'])}.",
    ]


def render_markdown(report: Dict[str, Any], reconciliation: Optional[Tuple[List[str], List[str]]] = None,
                    results_name: str = "", tolerance: float = 1e-6) -> str:
    """Resumen legible del informe (mismos números que el JSON); `tolerance` es la usada en la reconciliación."""
    sample = report["sample"]
    lines = [
        "# Análisis offline del estudio con usuarios",
        "",
        f"- Archivo: `{report['inputFile']}`",
        f"- SHA-256 de la entrada: `{report['inputSha256']}`",
        f"- Versión de fórmulas: `{report['formulaVersion']}`",
        f"- Filas leídas: {report['rowsRead']}",
        "",
        "## Muestra",
        "",
        f"- Participantes: {sample['participantsTotal']} en el CSV; incluidos (par completo): "
        f"{sample['participantsIncluded']}; con par incompleto: {sample['participantsWithIncompletePair']}; "
        f"sin ejecución elegible: {sample['participantsWithoutEligibleRun']}.",
        f"- Ejecuciones completadas: {sample['runsCompleted']} = incluidas {sample['runsIncluded']} + excluidas "
        f"{sample['runsExcluded']} + en pares incompletos {sample['runsInIncompletePairs']}; incluidas sin palabras "
        f"contables: {sample['runsWithoutCountableWords']}.",
    ]
    excluded = report["excluded"]
    if excluded["participantsWithIncompletePair"]:
        lines.append(f"- Par incompleto: {', '.join(excluded['participantsWithIncompletePair'])}.")
    if excluded["participantsWithoutEligibleRun"]:
        lines.append(f"- Sin ejecución elegible: {', '.join(excluded['participantsWithoutEligibleRun'])}.")
    for run in excluded["runsExcluded"]:
        lines.append(f"- Ejecución excluida: {run['runId']} ({run['pseudonym']}, {run['condition']}"
                     f"{', incidencias: ' + run['incidentReasons'] if run['incidentReasons'] else ''}).")

    lines += ["", "## PEO (errores ortográficos por 100 palabras)", ""]
    lines.append(f"- Anotación ortográfica: `{report['orthographyAnnotation']['status']}` — "
                 f"{report['orthographyAnnotation']['message']}.")
    peo = report["peo"]
    if peo is None:
        lines.append("- PEO no calculado (ver estado de anotación).")
    else:
        lines += _paired_lines(peo["paired"], "errores/100 palabras")
        lines.append(f"- Participantes analizados: {peo['participantsAnalyzed']} (sin palabras contables en ambas "
                     f"condiciones: {peo['participantsWithoutCountableWords']}); ejecuciones: {peo['runsAnalyzed']}.")
        lines.append(f"- Reducción relativa media: {_fmt(peo['relativeReductionMean'])} % (n = {peo['relativeReductionN']}, "
                     f"omitidas por PEO sin asistencia = 0: {peo['relativeReductionSkipped']}).")
        lines.append(f"- Límite superior del IC 95 % < 0: {_fmt(peo['upperCiBelowZero'])}.")

    lines += ["", "## PPM (palabras por minuto)", ""]
    ppm = report["ppm"]
    if ppm is None:
        lines.append("- PPM no calculado: ningún participante con par completo.")
    else:
        lines += _paired_lines(ppm["paired"], "ppm")
        lines.append(f"- Participantes analizados: {ppm['participantsAnalyzed']}; ejecuciones: {ppm['runsAnalyzed']}.")
        lines.append(f"- Margen de no inferioridad: {_fmt(ppm['nonInferiorityMargin'])}; descriptivo: "
                     f"{_fmt(ppm['descriptive'])}; no inferior (IC inferior > −margen): {_fmt(ppm['nonInferior'])}.")

    lines += ["", "## TAS (tasa de sugerencias perjudiciales)", ""]
    lines.append(f"- Anotación semántica: `{report['semanticAnnotation']['status']}` — "
                 f"{report['semanticAnnotation']['message']}.")
    if report["tas"] is None:
        lines.append("- TAS no calculada (ver estado de anotación).")
    else:
        lines += _tas_lines(report["tas"])
        lines += ["", "### TAS aceptada (solo sugerencias aceptadas)", ""]
        lines += _tas_lines(report["tasAccepted"])

    lines += ["", "## Participantes incluidos", "",
              "| Seudónimo | PEO asist. | PEO sin asist. | ΔPEO | Red. rel. % | PPM asist. | PPM sin asist. | ΔPPM | TAS | TAS acept. |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for row in report["participants"]:
        lines.append("| " + " | ".join(_fmt(row[key]) for key in PARTICIPANT_KEYS) + " |")

    provenance = report["provenance"]
    lines += ["", "## Procedencia", "",
              f"- Versiones de protocolo: {provenance['protocolVersions']}",
              f"- Lotes ortográficos: {provenance['orthographyBatchIds'] or '—'}; lotes semánticos: "
              f"{provenance['semanticBatchIds'] or '—'}",
              f"- Distribución t: {provenance['tDistribution']}"]
    if report["warnings"]:
        lines += ["", "## Advertencias", ""] + [f"- {warning}" for warning in report["warnings"]]
    if reconciliation is not None:
        discrepancies, notes = reconciliation
        lines += ["", "## Reconciliación con el backend", "", f"- Resultados comparados: `{results_name}`"]
        if discrepancies:
            lines.append(f"- **{len(discrepancies)} discrepancia(s)** (tolerancia {tolerance:g}):")
            lines += [f"  - {item}" for item in discrepancies]
        else:
            lines.append("- PEO, PPM, TAS, TAS aceptada, partición de la muestra y filas por participante "
                         "coinciden con el backend: sin discrepancias.")
        lines += [f"- Nota: {note}" for note in notes]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- CLI


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Reproduce PEO, PPM y TAS offline desde analysis.csv del backend.")
    parser.add_argument("csv", type=Path, help="analysis.csv exportado por el backend")
    parser.add_argument("--ppm-margin", type=float, default=None, help="margen δ de no inferioridad de PPM (> 0)")
    parser.add_argument("--tas-limit", type=float, default=None, help="límite de TAS en porcentaje (0–100)")
    parser.add_argument("--json", type=Path, default=None, help="ruta del informe JSON")
    parser.add_argument("--markdown", type=Path, default=None, help="ruta del informe Markdown")
    parser.add_argument("--compare-results", type=Path, default=None,
                        help="JSON de GET …/results del mismo estudio; falla si PEO/PPM/TAS difieren")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="tolerancia absoluta de la comparación (>= 0)")
    args = parser.parse_args(argv)
    if not (math.isfinite(args.tolerance) and args.tolerance >= 0):
        print(f"error: --tolerance must be a finite number >= 0 (was {args.tolerance})", file=sys.stderr)
        return 2
    for stream in (sys.stdout, sys.stderr):
        # Consolas con página de códigos limitada (cp1252) no deben abortar el análisis por un carácter.
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    try:
        rows, digest = load_rows(args.csv)
        report = analyze(rows, input_sha256=digest, input_name=args.csv.name,
                         ppm_margin=args.ppm_margin, tas_limit=args.tas_limit)
    except (AnalysisValidationError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    reconciliation = None
    if args.compare_results is not None:
        try:
            backend = json.loads(args.compare_results.read_text(encoding="utf-8"))
            if not isinstance(backend, dict):
                raise ValueError("the results file is not a JSON object")
        except (OSError, ValueError) as exc:
            print(f"error: cannot read {args.compare_results}: {exc}", file=sys.stderr)
            return 2
        reconciliation = compare_results(report, backend, tolerance=args.tolerance)
        report["reconciliation"] = {
            "resultsFile": args.compare_results.name,
            "tolerance": args.tolerance,
            "discrepancies": reconciliation[0],
            "notes": reconciliation[1],
        }

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(report, reconciliation, args.compare_results.name
                                                 if args.compare_results else "", tolerance=args.tolerance),
                                 encoding="utf-8")

    def cfmt(value: Any, digits: int = 4) -> str:
        """Formato ASCII para la consola (el Markdown conserva los guiones largos)."""
        return _fmt(value, digits, null="null")

    sample = report["sample"]
    print(f"{args.csv.name}: sha256={digest} formulaVersion={FORMULA_VERSION}")
    print(f"participantes incluidos={sample['participantsIncluded']} (total en CSV {sample['participantsTotal']}, "
          f"par incompleto {sample['participantsWithIncompletePair']}, sin elegible {sample['participantsWithoutEligibleRun']})")
    for block, label in (("peo", "PEO"), ("ppm", "PPM")):
        data = report[block]
        if data is None:
            print(f"{label}: no calculado ({report['orthographyAnnotation']['status'] if block == 'peo' else 'sin muestra'})")
        else:
            paired = data["paired"]
            print(f"{label}: n={paired['n']} delta={cfmt(paired['meanDelta'])} IC95=[{cfmt(paired['ci95Lower'])}, "
                  f"{cfmt(paired['ci95Upper'])}] p={cfmt(paired['pValue'], 6)} d_z={cfmt(paired['cohenDz'])}")
    for block, label in (("tas", "TAS"), ("tasAccepted", "TAS aceptada")):
        data = report[block]
        if data is None:
            print(f"{label}: no calculada ({report['semanticAnnotation']['status']})")
        else:
            print(f"{label}: {data['harmfulSuggestions']}/{data['suggestionsEvaluated']} = {cfmt(data['pooledRate'])} % "
                  f"Wilson=[{cfmt(data['pooledCi95Lower'])}, {cfmt(data['pooledCi95Upper'])}]")
    for warning in report["warnings"]:
        print(f"advertencia: {warning}")
    if reconciliation is not None:
        discrepancies, notes = reconciliation
        for note in notes:
            print(f"nota: {note}")
        if discrepancies:
            print(f"RECONCILIACION FALLIDA: {len(discrepancies)} discrepancia(s) con {args.compare_results.name}")
            for item in discrepancies:
                print(f"  - {item}")
            return 1
        print(f"reconciliacion OK: el analisis offline coincide con {args.compare_results.name} (tolerancia {args.tolerance})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
