"""
infrastructure/nlp/confusions.py
Conjuntos de confusión: pares de palabras REALES que el alumno confunde y que
las capas léxicas no pueden separar (las dos existen en el diccionario), así
que solo el contexto decide. Dos puntos de entrada:

  - `fix_lexical_confusions` (capa 0.2, tras las amalgamas y antes de las
    capas léxicas y de BETO): r/rr con marco sintáctico inequívoco.
        1. determinante + `pero`  -> `perro`   ("el pero ladra")
        2. determinante + `caro`  -> `carro`   ("el caro de mi papá")
        3. determinante singular + `aros` -> `arroz` ("el aros con pollo")
    Va antes de BETO porque BETO, con "el pero ...", convierte el artículo en
    pronombre ("él pero").

  - `resolve_final_confusions` (capa 6, tras T5, sobre la recomendada y las
    alternativas): decisiones que T5 deshace si se toman antes.
        4. caer/callar (yeísmo): `se callo` -> `se cayó` / `se calló`.
           T5 aprendió `se callo -> se calló` como respuesta fija (500 pares
           `se + Xo -> se Xó` del corpus v3) y la escribe incluso en "se callo
           de la bicicleta". Se decide por marco: dativo ("se me cayó"),
           preposición de lugar u origen detrás ("se cayó de/en/al ..."),
           palabras-pista de caída o de silencio en la frase y, si nada
           decide, BETO entre las dos formas (acierta 6/6 en silencio).
        5. `yo` + pretérito de 3.ª persona -> 1.ª persona ("ayer yo comió" ->
           "ayer yo comí"); T5 corrige la tilde pero no la persona.
    Las palabras que el alumno escribió con tilde se respetan (como en las
    capas 1 y 0.5).
"""
import re
import unicodedata

from infrastructure.nlp.grammar_rules import _lexicon, _reword

_PUNCT_RE = re.compile(r"^(\W*)(.*?)(\W*)$", re.UNICODE)


def _core(token: str) -> str:
    m = _PUNCT_RE.match(token)
    return m.group(2) if m else token


def _norm(word: str) -> str:
    w = _core(word).lower()
    return "".join(c for c in unicodedata.normalize("NFD", w)
                   if unicodedata.category(c) != "Mn")


# ── Capa 0.2: r / rr ─────────────────────────────────────────────────────────

_DET_SG = {"el", "un", "mi", "tu", "su", "este", "ese", "aquel", "del", "al",
           "nuestro", "otro", "cada"}
_DET_PL = {"los", "unos", "mis", "tus", "sus", "estos", "esos", "aquellos",
           "nuestros", "otros"}
# (palabra escrita, corrección) según el número del determinante.
_RR_AFTER_SG = {"pero": "perro", "caro": "carro", "aros": "arroz"}
_RR_AFTER_PL = {"peros": "perros", "caros": "carros"}


def fix_lexical_confusions(text: str) -> str:
    """r/rr tras determinante (ver reglas 1-3 del módulo)."""
    tokens = text.split()
    for i in range(1, len(tokens)):
        prev_tok = tokens[i - 1]
        if prev_tok != _core(prev_tok):          # "el," / "mi." cierran sintagma
            continue
        prev, low = _norm(prev_tok), _norm(tokens[i])
        if prev in _DET_SG and low in _RR_AFTER_SG:
            tokens[i] = _reword(tokens[i], _RR_AFTER_SG[low])
        elif prev in _DET_PL and low in _RR_AFTER_PL:
            tokens[i] = _reword(tokens[i], _RR_AFTER_PL[low])
    return " ".join(tokens)


# ── Capa 6: caer / callar ────────────────────────────────────────────────────

_CAER_CALLAR = {"callo", "cayo"}          # sin tildes; `callo` de "yo me callo" no va tras "se"
_DATIVES = {"me", "te", "le", "nos", "les", "os"}
# Tras el verbo: el lugar o el origen de la caída ("se cayó de la cama").
_FALL_NEXT = {"de", "del", "desde", "al", "a", "en", "sobre", "encima", "hacia",
              "contra", "dentro", "abajo", "arriba", "fuera", "adentro", "afuera"}
# Pistas en el resto de la frase (sin tildes, por prefijo).
_FALL_CUES = ("lastim", "golpe", "suelo", "piso", "rodilla", "herid", "sangr",
              "rompi", "tropez", "resbal", "empuj", "dolor", "duele")
_SILENCE_CUES = ("silencio", "callad", "habl", "dij", "decir", "grit", "llor",
                 "ruido", "voz", "boca", "pregunt", "respond", "contest", "rega")
_CAYO, _CALLO = "cayó", "calló"


def _decide_caer_callar(tokens: list, i: int, judge) -> str | None:
    """`cayó`/`calló` para `tokens[i]` o None si no es el caso."""
    if _norm(tokens[i]) not in _CAER_CALLAR:
        return None
    prev1 = _norm(tokens[i - 1]) if i >= 1 else ""
    prev2 = _norm(tokens[i - 2]) if i >= 2 else ""
    if prev1 in _DATIVES and prev2 == "se":
        return _CAYO                              # "se me cayó el helado"
    if prev1 != "se":
        return None
    nxt = _norm(tokens[i + 1]) if i + 1 < len(tokens) and tokens[i] == _core(tokens[i]) else ""
    if nxt in _FALL_NEXT:
        return _CAYO
    rest = [_norm(t) for j, t in enumerate(tokens) if j != i]
    fall = any(w.startswith(_FALL_CUES) for w in rest)
    silence = any(w.startswith(_SILENCE_CUES) for w in rest)
    if fall != silence:
        return _CAYO if fall else _CALLO
    if judge is None:
        return None
    pref_suff = _PUNCT_RE.match(tokens[i])
    pref, suff = (pref_suff.group(1), pref_suff.group(3)) if pref_suff else ("", "")
    scored = judge.score_candidates(tokens, i, [pref + _CAYO + suff, pref + _CALLO + suff])
    return _core(scored[0][0]).lower()


# ── Capa 6: yo + pretérito de 3.ª persona ────────────────────────────────────

# Irregulares y de cambio de raíz (claves sin tilde). Se omite "rio" (río).
# "cayo" solo llega aquí si caer/callar no decidió (no hay "se" delante).
_YO_PRETERITE = {
    "cayo": "caí",
    "fue": "fui", "tuvo": "tuve", "hizo": "hice", "estuvo": "estuve",
    "pudo": "pude", "dijo": "dije", "vino": "vine", "quiso": "quise",
    "puso": "puse", "supo": "supe", "trajo": "traje", "dio": "di", "vio": "vi",
    "anduvo": "anduve", "condujo": "conduje", "leyo": "leí", "oyo": "oí",
    "creyo": "creí", "durmio": "dormí", "murio": "morí", "pidio": "pedí",
    "sintio": "sentí", "siguio": "seguí", "sirvio": "serví", "vistio": "vestí",
    "repitio": "repetí", "midio": "medí", "prefirio": "preferí",
    "consiguio": "conseguí", "eligio": "elegí", "divirtio": "divertí",
}
# Pueden ir entre "yo" y el verbo: "yo no fue", "yo me cayó", "yo ya lo hizo".
_YO_GAP = {"no", "me", "te", "lo", "la", "le", "los", "las", "les", "nos", "ya",
           "tambien", "nunca", "siempre", "solo"}


def _first_person_preterite(token: str) -> str | None:
    """1.ª persona del pretérito de `token` (3.ª persona), validada en es_50k."""
    low, core = _norm(token), _core(token).lower()
    if low in _YO_PRETERITE:
        return _YO_PRETERITE[low]
    lex = _lexicon()
    # Regulares: SOLO con la tilde escrita ("comió"); "comio"/"estudio" sin
    # tilde puede ser presente ("yo estudio") y no se toca.
    if core.endswith("ió") and len(core) > 3:
        for cand in (core[:-2] + "í",             # comió -> comí
                     core[:-1] + "é"):            # estudió -> estudié (-iar)
            if cand in lex:
                return cand
        return None
    if core.endswith("ó") and len(core) > 3:
        stem = core[:-1]
        if stem.endswith("g"):
            stem += "u"                           # jugó -> jugué
        elif stem.endswith("c"):
            stem = stem[:-1] + "qu"               # tocó -> toqué
        elif stem.endswith("z"):
            stem = stem[:-1] + "c"                # empezó -> empecé
        cand = stem + "é"
        return cand if cand in lex else None
    return None


def _yo_before(tokens: list, i: int) -> bool:
    j = i - 1
    while j >= 0 and _norm(tokens[j]) in _YO_GAP and i - j <= 2:
        j -= 1
    return j >= 0 and _norm(tokens[j]) == "yo" and tokens[j] == _core(tokens[j])


def resolve_final_confusions(text: str, written: set | None = None, judge=None) -> str:
    """Reglas 4-5 sobre el texto final (tras T5). `written` son las palabras
    (minúsculas) que el alumno escribió con tilde: no se cambian."""
    written = written or set()
    tokens = text.split()
    for i, tok in enumerate(tokens):
        core_low = _core(tok).lower()
        if core_low in written:
            continue
        choice = _decide_caer_callar(tokens, i, judge)
        if choice is None and _yo_before(tokens, i):
            choice = _first_person_preterite(tok)
        if choice is not None and choice != core_low:
            tokens[i] = _reword(tok, choice)
    return " ".join(tokens)


__all__ = ["fix_lexical_confusions", "resolve_final_confusions"]
