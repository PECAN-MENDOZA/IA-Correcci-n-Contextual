"""
infrastructure/nlp/candidates.py
Capa 2: elección del candidato de SymSpell con un coste de edición informado
por los errores disléxicos (Rello et al., DysList) en vez de "distancia entera
y luego frecuencia", y protección de formas verbales regulares que no están
en el léxico es_50k.

Motivación (fallos reales, 2026-09-19): `jugan` -> `jugar` (SymSpell prefiere
el infinitivo por frecuencia; `juegan` es la inserción de una vocal),
`aruz` -> `cruz` (en vez de `arroz`), `camera` -> `cadera` (en vez de
`cámara`), `nadaremos` -> `daremos` (forma válida fuera de las 50k palabras).
"""
import re
import unicodedata

# ── Costes de edición ────────────────────────────────────────────────────────
# Sustituciones "baratas": las confusiones típicas (fonéticas y visuales).
_VOWELS = set("aeiou")
_CONFUSABLE = [
    {"b", "v"}, {"c", "s", "z"}, {"c", "k", "q"}, {"g", "j"}, {"m", "n"},
    {"r", "l"}, {"d", "t"}, {"p", "b"}, {"y", "i"}, {"g", "c"}, {"n", "ñ"},
]
_COST_ACCENT = 0.1          # esta -> está
_COST_VOWEL_SUB = 0.5       # camera -> camara
_COST_CONFUSABLE = 0.5      # baca -> vaca, sapato -> zapato
_COST_SUB = 1.0
_COST_VOWEL_INDEL = 0.5     # jugan -> juegan (empate con vocal<->vocal: decide la frecuencia)
_COST_DOUBLE = 0.5          # aroz -> arroz, cassa -> casa
_COST_H = 0.3               # ombre -> hombre
_COST_INDEL = 1.0
_COST_TRANSPOSE = 0.5       # gaot -> gato
_COST_DIPHTHONG = 0.5       # poden -> pueden, pensan -> piensan, juegar -> jugar (o/u<->ue, e<->ie)
_COST_YEISMO = 0.5          # rodia -> rodilla, tiyo -> tío: "ll"/"y" junto a vocal se omite o se añade
_DIPHTHONGS = {"ue": {"o", "u"}, "ie": {"e"}}
_PENALTY_FIRST_LETTER = 0.5 # aruz -> cruz (no si la 1.ª letra es confundible: kasa -> casa)
_PENALTY_SHORT = 0.25       # candidatos de <= 3 letras MÁS CORTOS que la palabra (got, ava): ruido del corpus
# Frecuencia mínima del candidato: es_50k trae typos residuales de baja
# frecuencia (dugan, arun, reba...) que no deben ganar por coste.
MIN_CANDIDATE_COUNT = 150


def _strip(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _sub_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    sa, sb = _strip(a), _strip(b)
    if sa == sb:
        return _COST_ACCENT
    if sa in _VOWELS and sb in _VOWELS:
        return _COST_VOWEL_SUB
    for group in _CONFUSABLE:
        if sa in group and sb in group:
            return _COST_CONFUSABLE
    return _COST_SUB


def _indel_cost(ch: str, neighbor: str) -> float:
    s = _strip(ch)
    if s == "h":
        return _COST_H
    if s == "y" and _strip(neighbor) in _VOWELS:
        return _COST_YEISMO
    if neighbor and _strip(neighbor) == s:
        return _COST_DOUBLE
    if s in _VOWELS:
        return _COST_VOWEL_INDEL
    return _COST_INDEL


def dyslexic_cost(word: str, candidate: str) -> float:
    """Distancia de Damerau-Levenshtein ponderada entre la palabra escrita y
    un candidato (ambas en minúsculas). Menor = más plausible como error."""
    a, b = word.lower(), candidate.lower()
    n, m = len(a), len(b)
    d = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = d[i - 1][0] + _indel_cost(a[i - 1], a[i - 2] if i > 1 else "")
    for j in range(1, m + 1):
        d[0][j] = d[0][j - 1] + _indel_cost(b[j - 1], b[j - 2] if j > 1 else "")
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best = min(
                d[i - 1][j] + _indel_cost(a[i - 1], a[i - 2] if i > 1 else ""),        # borrado
                d[i][j - 1] + _indel_cost(b[j - 1], b[j - 2] if j > 1 else ""),        # inserción
                d[i - 1][j - 1] + _sub_cost(a[i - 1], b[j - 1]),                        # sustitución
            )
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1] and a[i - 1] != a[i - 2]:
                best = min(best, d[i - 2][j - 2] + _COST_TRANSPOSE)                     # transposición
            # Cambio de raíz o/u->ue, e->ie (y a la inversa) como una sola operación.
            if j > 1 and b[j - 2:j] in _DIPHTHONGS and a[i - 1] in _DIPHTHONGS[b[j - 2:j]]:
                best = min(best, d[i - 1][j - 2] + _COST_DIPHTHONG)
            if i > 1 and a[i - 2:i] in _DIPHTHONGS and b[j - 1] in _DIPHTHONGS[a[i - 2:i]]:
                best = min(best, d[i - 2][j - 1] + _COST_DIPHTHONG)
            # Yeísmo: "ll" entera omitida o añadida (rodia -> rodilla).
            if j > 1 and b[j - 2:j] == "ll":
                best = min(best, d[i][j - 2] + _COST_YEISMO)
            if i > 1 and a[i - 2:i] == "ll":
                best = min(best, d[i - 2][j] + _COST_YEISMO)
            if j > 1 and a[i - 1] == "y" and b[j - 2:j] == "ll":      # cabayo -> caballo
                best = min(best, d[i - 1][j - 2] + _COST_YEISMO)
            if i > 1 and b[j - 1] == "y" and a[i - 2:i] == "ll":      # llo -> yo
                best = min(best, d[i - 2][j - 1] + _COST_YEISMO)
            d[i][j] = best
    cost = d[n][m]
    yeismo_start = {a[:1], b[:1]} == {"y", "l"} and (a.startswith("ll") or b.startswith("ll"))
    if a and b and _strip(a[0]) != _strip(b[0]) and _sub_cost(a[0], b[0]) >= _COST_SUB and not yeismo_start:
        cost += _PENALTY_FIRST_LETTER
    if len(b) <= 3 and len(b) < len(a):
        cost += _PENALTY_SHORT
    return cost


def rank_candidates(word: str, candidates, min_count: int = MIN_CANDIDATE_COUNT) -> list:
    """`candidates` = [(término, frecuencia)] (los de SymSpell a distancia <= 2)
    ordenados por coste disléxico y, a igual coste, por frecuencia:
    [(coste, término)]."""
    word = word.lower()
    ranked = [(round(dyslexic_cost(word, term), 3), -count, term)
              for term, count in candidates if term != word and count >= min_count]
    ranked.sort()
    return [(cost, term) for cost, _, term in ranked]


def pick_candidate(word: str, candidates, min_count: int = MIN_CANDIDATE_COUNT) -> str | None:
    """El de menor coste disléxico de `rank_candidates`; empate -> más
    frecuente. None si no hay candidato aceptable."""
    ranked = rank_candidates(word, candidates, min_count)
    return ranked[0][1] if ranked else None


# ── Formas verbales regulares fuera del léxico ───────────────────────────────
# Terminaciones que NO sufren cambio de raíz (o->ue, e->ie, e->i) y no llevan
# tilde: si el infinitivo existe, la forma es válida aunque es_50k no la tenga
# (nadaremos, dibujaremos, pintábamos no: lleva tilde y la resuelve SymSpell).
# Se excluyen a propósito el presente sg/3.ª pl. (juga/jugan), el subjuntivo,
# el pretérito de 3.ª (durmió) y los participios (escribido, rompido).
_REGULAR_ENDINGS = (
    # (terminación, vocal temática del infinitivo)
    ("aremos", "ar"), ("eremos", "er"), ("iremos", "ir"),
    ("aban", "ar"), ("abas", "ar"), ("aba", "ar"),
    ("aste", "ar"), ("iste", "er"), ("iste", "ir"),
    ("ando", "ar"), ("iendo", "er"),
    ("amos", "ar"), ("emos", "er"), ("imos", "ir"),
)
_MIN_STEM = 2
_MIN_INFINITIVE_COUNT = 100


def is_regular_verb_form(word: str, word_freqs: dict) -> bool:
    """True si `word` (minúsculas, sin tilde) es stem + terminación regular con
    el infinitivo en `word_freqs` (>= _MIN_INFINITIVE_COUNT)."""
    w = word.lower()
    if len(w) < 6 or not re.fullmatch(r"[a-zñ]+", w):
        return False
    for ending, theme in _REGULAR_ENDINGS:
        if w.endswith(ending) and len(w) - len(ending) >= _MIN_STEM:
            infinitive = w[: -len(ending)] + theme
            if word_freqs.get(infinitive, 0) >= _MIN_INFINITIVE_COUNT:
                return True
    return False


__all__ = ["dyslexic_cost", "pick_candidate", "rank_candidates", "is_regular_verb_form",
           "MIN_CANDIDATE_COUNT"]
