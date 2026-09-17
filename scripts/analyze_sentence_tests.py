"""
scripts/analyze_sentence_tests.py

Reproduce offline los resultados de una prueba de oraciones
(`GET /api/v1/research/tests/{id}/results`) a partir del CSV de exportación
del backend (`GET /api/v1/research/tests/{id}/export.csv`, 21 columnas, una
fila por respuesta de cada intento COMPLETED, los excluidos con
`excluded=true`) con las mismas fórmulas, el mismo orden de acumulación y el
mismo bootstrap determinista que `TestResultsService` (backend), de modo que
el análisis independiente valide número a número el JSON del backend.

Contrato numérico (ver backend/docs/research-api.md §6.2):
  * Por alumno y condición: errorsPer100Words = 100 · Σerrores / Σpalabras
    (solo respuestas con `word_count > 0` y `error_count` conocido) y
    wordsPerMinute = Σpalabras / (Σduration_first_key_ms / 60000) (solo con
    `duration_first_key_ms > 0`). Las sumas son enteras y secuenciales.
  * Los valores por alumno se ordenan por `student_username` ascendente y se
    promedian con suma secuencial. IC 95 % bootstrap percentil: SplitMix64
    con semilla 42, 2000 remuestreos, n índices por remuestreo con
    `nextIndex(n) = nextLong() mod n` (resto sin signo), cuantil R-7 sobre las
    medias ordenadas en 0.025 y 0.975. n < 2 → sin límites.
  * Diferencia pareada con − sin (alumnos con ambas condiciones): el mismo
    bootstrap sobre las diferencias y, además, IC t, t, p bilateral y d_z.
  * Tasa de aceptación (solo con ayuda): Σaceptadas / Σofrecidas con intervalo
    de Wilson (z = 1.959964), acotado a [0, 100].
  * Una respuesta `skipped=true` no aporta a ninguna métrica (ni a los
    contadores de sugerencias ni a `participants`); sí cuenta en
    `sentences[].n` y `skippedCount`. Los intentos `excluded=true` solo cuentan
    en `sample.completed` / `sample.excluded`.

Solo librería estándar (la t de Student se implementa con la beta incompleta
regularizada, como en el análisis del estudio anterior).

Uso:
    python scripts/analyze_sentence_tests.py responses.csv
        [--compare-results results.json] [--out report.json] [--tolerance 1e-6]
        [--min-sample 8]
Código de salida: 0 sin diferencias, 1 si `--compare-results` detecta alguna,
2 si el CSV o el JSON no se pueden leer o validar.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

FORMULA_VERSION = "sentence_tests_v1"

# Cuantil 0.975 de la normal estándar (Stats.Z_95 del backend).
Z_95 = 1.959964
CONFIDENCE = 0.95
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 42
DEFAULT_MIN_SAMPLE = 8

CONDITIONS = ("ASSISTED", "UNASSISTED")
KINDS = ("DICTATED", "FREE")
ERROR_SOURCES = ("AUTO", "ANNOTATED", "PENDING")

# Las 21 columnas del CSV de exportación, en el orden exacto de TestExportCsv.COLUMNS.
COLUMNS = (
    "test_code", "student_username", "attempt_id", "position", "kind", "assistance", "reference_text",
    "final_text", "skipped", "word_count", "error_count", "error_source", "duration_first_key_ms",
    "duration_start_ms", "suggestions_offered", "suggestions_accepted", "suggestions_rejected",
    "suggestions_undone", "model_version", "app_version", "excluded",
)

# Claves de cada bloque del informe, en el orden de los records Java.
METRIC_INTERVAL_KEYS = ("n", "mean", "lower", "upper")
ACCEPTANCE_KEYS = ("accepted", "offered", "ratePct", "wilsonLower", "wilsonUpper")
PAIRED_KEYS = ("n", "meanAssisted", "meanUnassisted", "meanDelta", "bootstrapLower", "bootstrapUpper",
               "tLower", "tUpper", "t", "p", "dz")
SENTENCE_KEYS = ("position", "kind", "assistance", "n", "meanErrors", "meanDurationFirstKeyMs", "skippedCount")
SAMPLE_COMPARED_KEYS = ("completed", "excluded", "unannotatedFree")


class AnalysisValidationError(ValueError):
    """CSV inválido (cabecera, tipos o valores fuera de dominio)."""


# --------------------------------------------------------------------------- PRNG y bootstrap


class SplitMix64:
    """SplitMix64 (Steele, Lea y Flood 2014) con la misma aritmética de 64 bits que la clase Java."""

    MASK = (1 << 64) - 1

    def __init__(self, seed: int) -> None:
        self.state = seed & self.MASK

    def next_long(self) -> int:
        """Siguiente valor como entero con signo de 64 bits (igual que `long` en Java)."""
        self.state = (self.state + 0x9E3779B97F4A7C15) & self.MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & self.MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & self.MASK
        z ^= z >> 31
        return z - (1 << 64) if z >= (1 << 63) else z

    def next_index(self, n: int) -> int:
        """Índice uniforme en [0, n): resto sin signo (`Long.remainderUnsigned`)."""
        return (self.next_long() & self.MASK) % n


def _mean(values: Sequence[float]) -> float:
    """Media con suma secuencial, como `Bootstrap.mean`."""
    total = 0.0
    for value in values:
        total += value
    return total / len(values)


def quantile(sorted_values: Sequence[float], p: float) -> float:
    """Cuantil R-7 (el de numpy por defecto) sobre una muestra ya ordenada."""
    k = len(sorted_values)
    idx = p * (k - 1)
    lo = int(math.floor(idx))
    frac = idx - lo
    hi = min(lo + 1, k - 1)
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])


def bootstrap_mean_interval(values: Sequence[float], resamples: int = BOOTSTRAP_RESAMPLES,
                            seed: int = BOOTSTRAP_SEED) -> Dict[str, Any]:
    """`Bootstrap.meanInterval`: media e IC 95 % bootstrap percentil determinista; n < 2 → sin límites."""
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "lower": None, "upper": None}
    mean = _mean(values)
    if n < 2:
        return {"n": n, "mean": mean, "lower": None, "upper": None}
    rng = SplitMix64(seed)
    means: List[float] = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(n):
            total += values[rng.next_index(n)]
        means.append(total / n)
    means.sort()
    return {"n": n, "mean": mean, "lower": quantile(means, 0.025), "upper": quantile(means, 0.975)}


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


# --------------------------------------------------------------------------- estadística clásica (Stats.java)


def _sample_sd(values: Sequence[float]) -> float:
    mean = _mean(values)
    squares = 0.0
    for value in values:
        squares += (value - mean) * (value - mean)
    return math.sqrt(squares / (len(values) - 1))


def _clamp_percent(value: float) -> float:
    return max(0.0, min(100.0, value))


def interval(values: Sequence[float]) -> Dict[str, Any]:
    """`Stats.interval`: media, sd muestral e IC 95 % (t); inferencia solo con n ≥ 2 y sd > 0."""
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
    """`Stats.pairedSummary` (con − sin) con las claves de `PairedDelta` salvo el bootstrap."""
    n = len(pairs)
    if n == 0:
        return {"n": 0, "meanAssisted": None, "meanUnassisted": None, "meanDelta": None, "sdDelta": None,
                "tLower": None, "tUpper": None, "t": None, "p": None, "dz": None}
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
    return {"n": n, "meanAssisted": _mean(assisted), "meanUnassisted": _mean(unassisted),
            "meanDelta": delta["mean"], "sdDelta": delta["sd"], "tLower": delta["lower"], "tUpper": delta["upper"],
            "t": t_stat, "p": p_value, "dz": dz}


def wilson(successes: int, n: int) -> Tuple[Optional[float], Optional[float]]:
    """`Stats.wilson`: intervalo de Wilson (95 %) de una proporción, en porcentaje y acotado a [0, 100]."""
    if n <= 0:
        return None, None
    p = successes / n
    z2n = Z_95 * Z_95 / n
    center = (p + z2n / 2.0) / (1.0 + z2n)
    half = Z_95 / (1.0 + z2n) * math.sqrt(p * (1.0 - p) / n + Z_95 * Z_95 / (4.0 * n * n))
    return _clamp_percent(100.0 * (center - half)), _clamp_percent(100.0 * (center + half))


# --------------------------------------------------------------------------- carga y validación del CSV


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _parse_int(value: str, column: str, line: int, blank_ok: bool = False) -> Optional[int]:
    text = value.strip()
    if text == "":
        if blank_ok:
            return None
        raise AnalysisValidationError(f"line {line}: {column} is blank")
    try:
        number = int(text)
    except ValueError as exc:
        raise AnalysisValidationError(f"line {line}: {column} is not an integer: {text!r}") from exc
    if number < 0:
        raise AnalysisValidationError(f"line {line}: {column} must be >= 0")
    return number


def _parse_bool(value: str, column: str, line: int) -> bool:
    text = value.strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise AnalysisValidationError(f"line {line}: {column} must be true or false: {value!r}")


def _parse_enum(value: str, column: str, line: int, allowed: Sequence[str]) -> str:
    text = value.strip()
    if text not in allowed:
        raise AnalysisValidationError(f"line {line}: {column} must be one of {', '.join(allowed)}: {text!r}")
    return text


def _parse_text(value: str, column: str, line: int, required: bool) -> Optional[str]:
    if value == "":
        if required:
            raise AnalysisValidationError(f"line {line}: {column} is blank")
        return None
    return value


def _parse_row(raw: List[str], line: int) -> Dict[str, Any]:
    cells = dict(zip(COLUMNS, raw))
    row: Dict[str, Any] = {"_line": line}
    row["test_code"] = _parse_text(cells["test_code"], "test_code", line, True)
    row["student_username"] = _parse_text(cells["student_username"], "student_username", line, True)
    row["attempt_id"] = _parse_text(cells["attempt_id"], "attempt_id", line, True)
    row["position"] = _parse_int(cells["position"], "position", line)
    if row["position"] < 1:
        raise AnalysisValidationError(f"line {line}: position must be >= 1")
    row["kind"] = _parse_enum(cells["kind"], "kind", line, KINDS)
    row["assistance"] = _parse_enum(cells["assistance"], "assistance", line, CONDITIONS)
    row["reference_text"] = cells["reference_text"]
    row["final_text"] = cells["final_text"]
    row["skipped"] = _parse_bool(cells["skipped"], "skipped", line)
    row["word_count"] = _parse_int(cells["word_count"], "word_count", line)
    row["error_count"] = _parse_int(cells["error_count"], "error_count", line, blank_ok=True)
    row["error_source"] = _parse_enum(cells["error_source"], "error_source", line, ERROR_SOURCES)
    row["duration_first_key_ms"] = _parse_int(cells["duration_first_key_ms"], "duration_first_key_ms", line,
                                              blank_ok=True)
    row["duration_start_ms"] = _parse_int(cells["duration_start_ms"], "duration_start_ms", line, blank_ok=True)
    for column in ("suggestions_offered", "suggestions_accepted", "suggestions_rejected", "suggestions_undone"):
        row[column] = _parse_int(cells[column], column, line)
    row["model_version"] = _parse_text(cells["model_version"], "model_version", line, False)
    row["app_version"] = _parse_text(cells["app_version"], "app_version", line, False)
    row["excluded"] = _parse_bool(cells["excluded"], "excluded", line)
    return row


def load_rows(path: Path) -> Tuple[List[Dict[str, Any]], str]:
    """Lee y valida el CSV de exportación; devuelve las filas tipadas y el SHA-256 de los bytes del archivo."""
    path = Path(path)
    digest = sha256_of(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None or tuple(header) != COLUMNS:
            raise AnalysisValidationError(
                f"header must be exactly the {len(COLUMNS)} export columns: {', '.join(COLUMNS)}")
        rows: List[Dict[str, Any]] = []
        for line, raw in enumerate(reader, start=2):
            if not raw:
                continue
            if len(raw) != len(COLUMNS):
                raise AnalysisValidationError(f"line {line}: expected {len(COLUMNS)} fields, found {len(raw)}")
            rows.append(_parse_row(raw, line))
    _validate(rows)
    return rows, digest


def _validate(rows: List[Dict[str, Any]]) -> None:
    codes = {row["test_code"] for row in rows}
    if len(codes) > 1:
        raise AnalysisValidationError(f"test_code must be the same in every row: {', '.join(sorted(codes))}")
    seen = set()
    attempt_owner: Dict[str, Tuple[str, bool]] = {}
    for row in rows:
        key = (row["attempt_id"], row["position"])
        if key in seen:
            raise AnalysisValidationError(f"line {row['_line']}: duplicated attempt_id/position {key}")
        seen.add(key)
        owner = (row["student_username"], row["excluded"])
        previous = attempt_owner.setdefault(row["attempt_id"], owner)
        if previous != owner:
            raise AnalysisValidationError(
                f"line {row['_line']}: attempt {row['attempt_id']} has inconsistent student_username/excluded")
        if row["kind"] == "DICTATED" and row["error_source"] != "AUTO":
            raise AnalysisValidationError(f"line {row['_line']}: DICTATED rows must have error_source AUTO")
        if row["kind"] == "FREE" and row["error_source"] == "AUTO":
            raise AnalysisValidationError(f"line {row['_line']}: FREE rows cannot have error_source AUTO")
        if row["error_source"] == "PENDING" and row["error_count"] is not None:
            raise AnalysisValidationError(f"line {row['_line']}: PENDING rows cannot have error_count")
        if row["error_source"] != "PENDING" and row["error_count"] is None:
            raise AnalysisValidationError(f"line {row['_line']}: error_count is blank but error_source is not PENDING")


# --------------------------------------------------------------------------- acumuladores por alumno


def _analyzed(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filas de intentos completados no excluidos (las únicas que entran en las métricas)."""
    return [row for row in rows if not row["excluded"]]


def per_student_condition(rows: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, int]]:
    """
    Acumuladores de `TestResultsService.StudentCondition` por (username, condición) sobre filas no excluidas
    y no omitidas: `words`/`errors` (solo `word_count > 0` y `error_count` conocido), `ppm_words`/`duration_ms`
    (solo `duration_first_key_ms > 0`), `offered`/`accepted` y `pending_free` (libres sin anotar).
    """
    acc: Dict[Tuple[str, str], Dict[str, int]] = {}
    for row in _analyzed(rows):
        if row["skipped"]:
            continue
        key = (row["student_username"], row["assistance"])
        bucket = acc.setdefault(key, {"words": 0, "errors": 0, "ppm_words": 0, "duration_ms": 0,
                                      "offered": 0, "accepted": 0, "pending_free": 0})
        bucket["offered"] += row["suggestions_offered"]
        bucket["accepted"] += row["suggestions_accepted"]
        words = row["word_count"]
        errors = row["error_count"]
        if errors is None and row["kind"] == "FREE":
            bucket["pending_free"] += 1
        if words > 0 and errors is not None:
            bucket["errors"] += errors
            bucket["words"] += words
        duration = row["duration_first_key_ms"]
        if duration is not None and duration > 0:
            bucket["ppm_words"] += words
            bucket["duration_ms"] += duration
    return acc


def _errors_per_100_words(bucket: Dict[str, int]) -> Optional[float]:
    return None if bucket["words"] == 0 else 100.0 * bucket["errors"] / bucket["words"]


def _words_per_minute(bucket: Dict[str, int]) -> Optional[float]:
    return None if bucket["duration_ms"] == 0 else bucket["ppm_words"] / (bucket["duration_ms"] / 60000.0)


METRICS = (("errorsPer100Words", _errors_per_100_words), ("wordsPerMinute", _words_per_minute))


# --------------------------------------------------------------------------- análisis completo


def _condition_values(acc, usernames, condition, metric) -> List[float]:
    values = []
    for username in usernames:
        bucket = acc.get((username, condition))
        value = None if bucket is None else metric(bucket)
        if value is not None:
            values.append(value)
    return values


def _paired(acc, usernames, metric) -> Dict[str, Any]:
    pairs: List[Tuple[float, float]] = []
    for username in usernames:
        with_help = acc.get((username, "ASSISTED"))
        without = acc.get((username, "UNASSISTED"))
        a = None if with_help is None else metric(with_help)
        u = None if without is None else metric(without)
        if a is not None and u is not None:
            pairs.append((a, u))
    if not pairs:
        return {key: (0 if key == "n" else None) for key in PAIRED_KEYS}
    bootstrap = bootstrap_mean_interval([a - u for a, u in pairs])
    summary = paired_summary(pairs)
    return {"n": summary["n"], "meanAssisted": summary["meanAssisted"], "meanUnassisted": summary["meanUnassisted"],
            "meanDelta": summary["meanDelta"], "bootstrapLower": bootstrap["lower"],
            "bootstrapUpper": bootstrap["upper"], "tLower": summary["tLower"], "tUpper": summary["tUpper"],
            "t": summary["t"], "p": summary["p"], "dz": summary["dz"]}


def _sentence_stats(analyzed: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """`SentenceStat` por posición (orden ascendente) sobre las filas analizadas, omitidas incluidas en n."""
    by_position: Dict[int, Dict[str, Any]] = {}
    for row in analyzed:
        stat = by_position.setdefault(row["position"], {
            "position": row["position"], "kind": row["kind"], "assistance": row["assistance"],
            "n": 0, "errors": 0, "errorCount": 0, "duration": 0, "durationCount": 0, "skippedCount": 0})
        if (stat["kind"], stat["assistance"]) != (row["kind"], row["assistance"]):
            raise AnalysisValidationError(
                f"line {row['_line']}: position {row['position']} has inconsistent kind/assistance")
        stat["n"] += 1
        if row["skipped"]:
            stat["skippedCount"] += 1
            continue
        if row["error_count"] is not None:
            stat["errors"] += row["error_count"]
            stat["errorCount"] += 1
        duration = row["duration_first_key_ms"]
        if duration is not None and duration > 0:
            stat["duration"] += duration
            stat["durationCount"] += 1
    sentences = []
    for position in sorted(by_position):
        stat = by_position[position]
        sentences.append({
            "position": position, "kind": stat["kind"], "assistance": stat["assistance"], "n": stat["n"],
            "meanErrors": None if stat["errorCount"] == 0 else stat["errors"] / stat["errorCount"],
            "meanDurationFirstKeyMs": None if stat["durationCount"] == 0 else stat["duration"] / stat["durationCount"],
            "skippedCount": stat["skippedCount"]})
    return sentences


def _distinct_sorted(rows: Sequence[Dict[str, Any]], column: str) -> List[str]:
    return sorted({row[column] for row in rows if row[column] is not None})


def analyze(rows: Sequence[Dict[str, Any]], input_sha256: str, input_name: str,
            min_sample: int = DEFAULT_MIN_SAMPLE) -> Dict[str, Any]:
    """
    Informe con la estructura de `TestResultsResponse`. Los campos que el CSV no puede conocer
    (`testId`, `title`, `status`, `sample.assigned/cancelled/inProgress`, `provenance.backendVersions`) van a
    `null`; `minSample` es el parámetro (por defecto el del backend, 8).
    """
    analyzed = _analyzed(rows)
    acc = per_student_condition(rows)
    usernames = sorted({username for username, _ in acc})
    attempts_all = {row["attempt_id"] for row in rows}
    attempts_excluded = {row["attempt_id"] for row in rows if row["excluded"]}
    unannotated_free = sum(bucket["pending_free"] for bucket in acc.values())

    conditions: Dict[str, Any] = {}
    for condition in CONDITIONS:
        participants = {username for username, cond in acc if cond == condition}
        acceptance = None
        if condition == "ASSISTED":
            offered = sum(bucket["offered"] for (_, cond), bucket in acc.items() if cond == condition)
            accepted = sum(bucket["accepted"] for (_, cond), bucket in acc.items() if cond == condition)
            lower, upper = wilson(accepted, offered)
            acceptance = {"accepted": accepted, "offered": offered,
                          "ratePct": None if offered == 0 else 100.0 * accepted / offered,
                          "wilsonLower": lower, "wilsonUpper": upper}
        block: Dict[str, Any] = {"participants": len(participants)}
        for name, metric in METRICS:
            block[name] = bootstrap_mean_interval(_condition_values(acc, usernames, condition, metric))
        block["acceptanceRate"] = acceptance
        conditions[condition] = block

    analyzed_attempts = len(attempts_all) - len(attempts_excluded)
    return {
        "formulaVersion": FORMULA_VERSION,
        "inputName": input_name,
        "testId": None,
        "code": rows[0]["test_code"] if rows else None,
        "title": None,
        "status": None,
        "computedAt": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "sample": {"assigned": None, "completed": len(attempts_all), "excluded": len(attempts_excluded),
                   "cancelled": None, "inProgress": None, "unannotatedFree": unannotated_free},
        "incomplete": unannotated_free > 0,
        "sampleInsufficient": analyzed_attempts < min_sample,
        "minSample": min_sample,
        "designManual": True,
        "conditions": conditions,
        "paired": {name: _paired(acc, usernames, metric) for name, metric in METRICS},
        "sentences": _sentence_stats(analyzed),
        "provenance": {"modelVersions": _distinct_sorted(analyzed, "model_version"),
                       "appVersions": _distinct_sorted(analyzed, "app_version"), "backendVersions": None},
        "datasetSha256": input_sha256,
    }


# --------------------------------------------------------------------------- reconciliación con el backend


def _close(offline: Any, backend: Any, tol: float) -> bool:
    if offline is None or backend is None:
        return offline is None and backend is None
    if isinstance(offline, bool) or isinstance(backend, bool):
        return offline == backend
    if isinstance(offline, (int, float)) and isinstance(backend, (int, float)):
        if isinstance(offline, int) and isinstance(backend, int):
            return offline == backend
        return math.isclose(offline, backend, rel_tol=0.0, abs_tol=tol)
    return offline == backend


def _compare_keys(offline: Optional[Dict[str, Any]], backend: Any, prefix: str, keys: Sequence[str],
                  tol: float, out: List[str]) -> None:
    if not isinstance(backend, dict):
        out.append(f"{prefix}: backend value is not an object ({backend!r})")
        return
    for key in keys:
        mine = None if offline is None else offline.get(key)
        theirs = backend.get(key)
        if not _close(mine, theirs, tol):
            out.append(f"{prefix}.{key}: offline {mine!r} vs backend {theirs!r}")


def compare(report: Dict[str, Any], results_json_path: Path, tol: float = 1e-6) -> List[str]:
    """
    Diferencias campo a campo entre el informe offline y `TestResultsResponse` (JSON del backend):
    `sample.{completed,excluded,unannotatedFree}`, `incomplete`, por condición `participants`, los dos
    `MetricInterval`, `acceptanceRate`, `paired.*`, `sentences[]` y `datasetSha256`. Enteros, booleanos y
    cadenas exactos; flotantes con tolerancia absoluta `tol`; `null` ⇔ `None`. [] si todo coincide.
    """
    backend = json.loads(Path(results_json_path).read_text(encoding="utf-8"))
    out: List[str] = []
    _compare_keys(report["sample"], backend.get("sample"), "sample", SAMPLE_COMPARED_KEYS, tol, out)
    if not _close(report["incomplete"], backend.get("incomplete"), tol):
        out.append(f"incomplete: offline {report['incomplete']!r} vs backend {backend.get('incomplete')!r}")

    conditions = backend.get("conditions") or {}
    for condition in CONDITIONS:
        mine = report["conditions"][condition]
        theirs = conditions.get(condition)
        prefix = f"conditions.{condition}"
        if not isinstance(theirs, dict):
            out.append(f"{prefix}: missing in backend results")
            continue
        _compare_keys(mine, theirs, prefix, ("participants",), tol, out)
        for name, _ in METRICS:
            _compare_keys(mine[name], theirs.get(name), f"{prefix}.{name}", METRIC_INTERVAL_KEYS, tol, out)
        if mine["acceptanceRate"] is None or theirs.get("acceptanceRate") is None:
            if not (mine["acceptanceRate"] is None and theirs.get("acceptanceRate") is None):
                out.append(f"{prefix}.acceptanceRate: offline {mine['acceptanceRate']!r} "
                           f"vs backend {theirs.get('acceptanceRate')!r}")
        else:
            _compare_keys(mine["acceptanceRate"], theirs["acceptanceRate"], f"{prefix}.acceptanceRate",
                          ACCEPTANCE_KEYS, tol, out)

    paired = backend.get("paired") or {}
    for name, _ in METRICS:
        _compare_keys(report["paired"][name], paired.get(name), f"paired.{name}", PAIRED_KEYS, tol, out)

    sentences = backend.get("sentences")
    if not isinstance(sentences, list):
        out.append(f"sentences: backend value is not a list ({sentences!r})")
    elif len(sentences) != len(report["sentences"]):
        out.append(f"sentences: offline has {len(report['sentences'])} entries, backend {len(sentences)}")
    else:
        for index, (mine, theirs) in enumerate(zip(report["sentences"], sentences)):
            _compare_keys(mine, theirs, f"sentences[{index}]", SENTENCE_KEYS, tol, out)

    if report["datasetSha256"] != backend.get("datasetSha256"):
        out.append(f"datasetSha256: input CSV {report['datasetSha256']!r} vs backend {backend.get('datasetSha256')!r}")
    return out


# --------------------------------------------------------------------------- CLI


def _fmt(value: Any, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def _summary_lines(report: Dict[str, Any]) -> List[str]:
    sample = report["sample"]
    lines = [
        f"Prueba {report['code'] or '?'} ({report['inputName']}, sha256 {report['datasetSha256'][:12]}...)",
        f"  completados {sample['completed']}, excluidos {sample['excluded']}, "
        f"libres sin anotar {sample['unannotatedFree']}"
        + (" (incompleto)" if report["incomplete"] else "")
        + (f", muestra insuficiente (< {report['minSample']})" if report["sampleInsufficient"] else ""),
    ]
    for condition in CONDITIONS:
        block = report["conditions"][condition]
        errors = block["errorsPer100Words"]
        ppm = block["wordsPerMinute"]
        lines.append(f"  {condition:<10} n={block['participants']}  err/100 {_fmt(errors['mean'])} "
                     f"[{_fmt(errors['lower'])}, {_fmt(errors['upper'])}]  ppm {_fmt(ppm['mean'])} "
                     f"[{_fmt(ppm['lower'])}, {_fmt(ppm['upper'])}]")
        acceptance = block["acceptanceRate"]
        if acceptance is not None:
            lines.append(f"             aceptación {acceptance['accepted']}/{acceptance['offered']} = "
                         f"{_fmt(acceptance['ratePct'])} % [{_fmt(acceptance['wilsonLower'])}, "
                         f"{_fmt(acceptance['wilsonUpper'])}]")
    for name, _ in METRICS:
        paired = report["paired"][name]
        lines.append(f"  delta {name}: n={paired['n']} media {_fmt(paired['meanDelta'])} "
                     f"bootstrap [{_fmt(paired['bootstrapLower'])}, {_fmt(paired['bootstrapUpper'])}] "
                     f"t [{_fmt(paired['tLower'])}, {_fmt(paired['tUpper'])}] p={_fmt(paired['p'], 4)} "
                     f"dz={_fmt(paired['dz'])}")
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce offline los resultados de una prueba de oraciones a partir del CSV de exportación.")
    parser.add_argument("csv", type=Path, help="CSV de exportación (GET /api/v1/research/tests/{id}/export.csv)")
    parser.add_argument("--compare-results", type=Path, metavar="JSON",
                        help="JSON de GET /api/v1/research/tests/{id}/results; salida 1 si hay diferencias")
    parser.add_argument("--out", type=Path, metavar="JSON", help="escribe el informe offline en JSON")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="tolerancia absoluta para flotantes")
    parser.add_argument("--min-sample", type=int, default=DEFAULT_MIN_SAMPLE,
                        help="app.tests.min-sample del backend (solo afecta a sampleInsufficient)")
    args = parser.parse_args(argv)
    if args.tolerance < 0:
        parser.error("--tolerance must be >= 0")

    try:
        rows, digest = load_rows(args.csv)
    except (OSError, UnicodeDecodeError, AnalysisValidationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    report = analyze(rows, digest, args.csv.name, args.min_sample)
    for line in _summary_lines(report):
        print(line)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                            encoding="utf-8")
        print(f"informe escrito en {args.out}")

    if args.compare_results:
        try:
            differences = compare(report, args.compare_results, args.tolerance)
        except (OSError, UnicodeDecodeError, ValueError, KeyError, TypeError) as exc:
            print(f"error: cannot read backend results: {exc}", file=sys.stderr)
            return 2
        if differences:
            print(f"{len(differences)} diferencia(s) con el backend:")
            for difference in differences:
                print(f"  - {difference}")
            return 1
        print("reconciliación con el backend: sin diferencias")
    return 0


if __name__ == "__main__":
    sys.exit(main())
