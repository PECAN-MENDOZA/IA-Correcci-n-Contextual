"""
infrastructure/nlp/segmentation.py
Segmentación de palabras: el error más frecuente de la escritura infantil real
(30 de 99 casos en data/test_ninos_reales.csv, 2/30 resueltos por v4). Dos casos:

  * HIPERSEGMENTACIÓN — una palabra partida en dos: `en contró`, `al morzar`,
    `con tenta`, `to do`, `a bajo`, `aun que`. Va antes de las capas léxicas
    (capa 0.1), que si no "corrigen" cada trozo por separado (contró->contra,
    morzar->marcar). Se une si la unión es palabra y
      - uno de los trozos no es palabra española (`contró`, `tenta`; `to`/`do`
        están en es_50k por los subtítulos, pero no son español): sin más; o
      - los dos son palabras (`a bajo`, `aun que`): decide BETO (ver abajo).
  * HIPOSEGMENTACIÓN — palabras pegadas: `queno`, `selas`, `poreso`, `apartir`,
    `estabapreparando`, `asufamilia`. Va en la capa 2, para palabras fuera del
    diccionario que la búsqueda fonética no resolvió: se proponen las
    particiones en palabras españolas más probables (unigramas de es_50k) y, si
    SymSpell también propone una sola palabra (`queno` -> `quemo`), BETO elige.

Cómo decide BETO: NO por la probabilidad de la propia ranura (su media por
subtoken favorece siempre la versión partida: `a bajo` > `abajo` incluso en
"lo metió a bajo de la cama", porque `a` es muy predecible), sino por cuánto
hace cada variante probable el CONTEXTO (las `CONTEXT_WINDOW` palabras a cada
lado, que son las mismas en todas las variantes): "de la cama" encaja tras
`abajo`, "precio" tras `a bajo`.

`porque`/`por qué`, `sino`/`si no` y `a ver`/`haber` no se tocan aquí: los
resuelve amalgams.py con reglas propias.
"""
import math
import unicodedata

from infrastructure.nlp.amalgams import _JOINED as _AMALGAMS
from infrastructure.nlp.phonetic_engine import to_phonetic

# Palabras españolas de una o dos letras (es_50k trae también `to`, `do`, `in`...).
SHORT_WORDS = {
    "a", "o", "y", "e", "u",
    "al", "as", "da", "ir", "de", "di", "el", "en", "es", "fe", "ha", "he", "la", "le", "lo",
    "me", "mi", "ni", "no", "os", "se", "si", "su", "te", "ti", "tu", "un", "va",
    "ve", "vi", "ya", "yo", "dé", "sé", "sí", "tú", "él", "mí",
}
# Inglés frecuente en los subtítulos de es_50k con 3+ letras.
_ENGLISH_NOISE = {"the", "you", "and", "for", "test", "yes", "hey", "okay", "what", "this",
                  "that", "with", "your", "not", "are", "was", "all", "get", "one"}
# Trozos de una letra admitidos al separar (`teley` -> `tele y`, `asufamilia`).
_SPLIT_SINGLE = {"a", "y", "o"}
# Frecuencia mínima en es_50k para que una palabra de 3+ letras cuente como
# española: más exigente cuanto más corta, que es donde está el ruido de los
# subtítulos (`der` 1 222, `can` 4 668, `boy` 4 462; `oso` 10 243 sí cuenta);
# `caperucita` (378) sí cuenta.
MIN_WORD_FREQ_3 = 10000         # 3 letras
MIN_WORD_FREQ_4 = 1000          # 4 letras
MIN_WORD_FREQ = 300             # 5+ letras
# La palabra unida debe tener al menos esta frecuencia (alcancía: 186).
MIN_JOINED_FREQ = 150
MAX_SPLIT_PIECES = 3
# Unir dos palabras reales solo si la unión es una palabra larga y frecuente:
# `a ir` -> `air`, `y yo` -> `yyo`, `ir a` -> `ira` no.
MIN_JOINED_REAL_LEN = 5
MIN_JOINED_REAL_FREQ = 1000
SPLIT_CANDIDATES = 3
# Palabras funcionales: dos de ellas juntas no se unen nunca (`a la` -> `ala`,
# `de más`, `se a`): demasiado frecuentes y legítimas por separado.
_FUNCTION = {
    "a", "o", "y", "e", "u", "al", "de", "del", "el", "en", "la", "las", "le", "les",
    "lo", "los", "me", "mi", "mis", "nos", "que", "se", "si", "su", "sus", "te",
    "tu", "tus", "un", "una", "con", "por", "para", "sin", "más", "mas",
}
# Trozos cortos (<= 3 letras) admitidos DETRÁS del primero al separar: palabras
# funcionales que el niño pega (`queno`, `selas`, `asufamilia`, `poreso`). Evita
# particiones como `imaginativos` -> `imagina ti vos`.
_SPLIT_SHORT_TAIL = _FUNCTION | {"no", "eso", "esa", "ese", "ya", "yo", "hay", "muy", "les"}
# Pronombres enclíticos: `confiarse`, `diciéndole` no se separan tras infinitivo o gerundio.
_CLITICS = {"me", "te", "se", "lo", "la", "le", "nos", "os", "los", "las", "les"}
_CLITIC_HOSTS = ("ar", "er", "ir", "ár", "ér", "ír", "ando", "iendo", "yendo")
# Prefijos de compuestos (`sobreentrenamiento`, `latinoamericanos`, `autocontrol`).
_COMPOUND_PREFIXES = {"sobre", "latino", "latinos", "auto", "contra", "super", "entre", "anti",
                      "multi", "inter", "ultra", "micro", "macro", "extra", "semi", "mini",
                      "hispano", "afro", "euro", "ibero", "centro", "norte", "sud", "sur"}
# Nunca se forman aquí (tienen regla propia en amalgams.py o son ambiguas).
_NEVER_JOIN = {"sino", "porque", "porqué", "conque", "haber", "aver", "asimismo", "adonde",
               *_AMALGAMS}                       # porfavor, enserio, talvez...
# Tampoco uniendo dos palabras reales: "lo hizo tan bien" es legítimo y BETO
# prefiere "también" por frecuencia (`tam bien` sí se une: `tam` no es palabra).
_NEVER_JOIN_REAL = {"también", "sobretodo"}   # "sobre todo me gusta dibujar"
# Márgenes en suma de log-prob media del contexto (calibrados el 2026-09-30:
# abajo +1,0, afuera +1,2, adentro +2,7, aunque +3,0 frente a `de bajo
# calidad` -1,8 y `a bajo precio` -13,9).
JOIN_MARGIN = 0.5
SPLIT_MARGIN = 0.5
CONTEXT_WINDOW = 3
_SENTENCE_PUNCT = ".,;:!?¡¿\"'()-—«»"


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def lookup(word: str, freqs: dict, accent_dict: dict) -> str | None:
    """Forma del diccionario de `word` (minúsculas), restaurando la tilde si hace falta."""
    w = word.lower()
    if w in freqs:
        return w
    plain = strip_accents(w)
    if plain in accent_dict:
        return accent_dict[plain]
    return plain if plain in freqs else None


def is_spanish_word(word: str, freqs: dict, accent_dict: dict) -> bool:
    w = word.lower()
    if len(w) <= 2:
        return w in SHORT_WORDS or strip_accents(w) in SHORT_WORDS
    if w in _ENGLISH_NOISE:
        return False
    form = lookup(w, freqs, accent_dict)
    floor = {3: MIN_WORD_FREQ_3, 4: MIN_WORD_FREQ_4}.get(len(w), MIN_WORD_FREQ)
    return form is not None and freqs.get(form, 0) >= floor


def _match_case(original: str, replacement: str) -> str:
    return replacement[:1].upper() + replacement[1:] if original[:1].isupper() else replacement


def _split_punct(token: str) -> tuple:
    i, j = 0, len(token)
    while i < j and token[i] in _SENTENCE_PUNCT:
        i += 1
    while j > i and token[j - 1] in _SENTENCE_PUNCT:
        j -= 1
    return token[:i], token[i:j], token[j:]


def context_score(judge, tokens: list, slot: int, window: int = CONTEXT_WINDOW) -> float:
    """Suma de la log-prob media (BETO) de las palabras vecinas de `slot`."""
    total = 0.0
    for j in range(max(0, slot - window), min(len(tokens), slot + window + 1)):
        if j != slot:
            total += judge.score_candidates(tokens, j, [tokens[j]])[0][1]
    return total


def _best_variant(judge, tokens: list, slot: int, variants: list) -> tuple:
    """[(variante, score de contexto)] de mejor a peor para la ranura `slot`."""
    scored = []
    for v in variants:
        t = list(tokens)
        t[slot] = v
        scored.append((v, context_score(judge, t, slot)))
    scored.sort(key=lambda vs: -vs[1])
    return scored


def _joined_form(a: str, b: str, freqs: dict, accent_dict: dict) -> str | None:
    if not (a.isalpha() and b.isalpha()):
        return None
    form = lookup(a + b, freqs, accent_dict)
    if form is None or form in _NEVER_JOIN or freqs.get(form, 0) < MIN_JOINED_FREQ:
        return None
    return form


def _is_misspelled_word(piece: str, phonetic_dict: dict) -> bool:
    """`boy` (voy), `ise` (hice): suena como una palabra real de 3+ letras, así que
    es una falta de ortografía y no el trozo de una palabra partida. De dos
    letras, solo si es la palabra sin la h muda: `oy e echo` es `hoy he hecho`,
    no `oye echo`."""
    key = to_phonetic(piece.lower())
    if len(piece) >= 3:
        return key in phonetic_dict
    word = strip_accents(phonetic_dict.get(key, ""))
    return len(piece) == 2 and word == "h" + strip_accents(piece.lower())


def join_split_words(text: str, freqs: dict, accent_dict: dict, judge=None,
                     phonetic_dict: dict | None = None) -> str:
    """Capa 0.1: une las palabras partidas de `text` (ver docstring del módulo)."""
    phonetic_dict = phonetic_dict or {}
    tokens = text.split()
    i = 0
    while i + 1 < len(tokens):
        pref_a, a, suff_a = _split_punct(tokens[i])
        pref_b, b, suff_b = _split_punct(tokens[i + 1])
        # Nombres propios y siglas no se unen ("a Ra", "Ma Ri", "de PTO").
        names = b[:1].isupper() or (len(a) > 1 and a.isupper()) or (len(b) > 1 and b.isupper())
        form = None if (suff_a or pref_b or names) else _joined_form(a, b, freqs, accent_dict)
        if form is None:
            i += 1
            continue
        a_word = is_spanish_word(a, freqs, accent_dict)
        b_word = is_spanish_word(b, freqs, accent_dict)
        join = False
        if not b_word and a_word and i + 2 < len(tokens):
            # `a to do`: `to` va con el vecino que forme la palabra más frecuente.
            _, c, _ = _split_punct(tokens[i + 2])
            other = None if suff_b else _joined_form(b, c, freqs, accent_dict)
            if other and freqs.get(other, 0) > freqs.get(form, 0):
                i += 1
                continue
        if not (a_word and b_word):
            # el trozo que no es palabra tiene al menos dos letras (`i se` no es
            # `ise`) y no es una falta de ortografía de otra palabra (`boy a` no es `boya`)
            pieces = [x for x, ok in ((a, a_word), (b, b_word)) if not ok]
            join = all(len(x) >= 2 and not _is_misspelled_word(x, phonetic_dict) for x in pieces)
        elif (judge is not None and form not in _NEVER_JOIN_REAL
              and len(form) >= MIN_JOINED_REAL_LEN and freqs.get(form, 0) >= MIN_JOINED_REAL_FREQ
              and form not in _ENGLISH_NOISE
              and not (a.lower() in _FUNCTION and b.lower() in _FUNCTION)):
            split_cand = pref_a + f"{a} {b}" + suff_b
            join_cand = pref_a + _match_case(a, form) + suff_b
            try:
                ranked = dict(_best_variant(judge, tokens[:i] + [split_cand] + tokens[i + 2:], i,
                                            [split_cand, join_cand]))
                join = ranked[join_cand] - ranked[split_cand] >= JOIN_MARGIN
            except Exception as e:                       # BETO caído: no se une
                print(f"[WARN] Segmentación (BETO) falló: {e}")
                join = False
        if join:
            tokens[i:i + 2] = [pref_a + _match_case(a, form) + suff_b]
        else:
            i += 1
    return " ".join(tokens)


def split_candidates(word: str, freqs: dict, accent_dict: dict,
                     k: int = SPLIT_CANDIDATES, phonetic_dict: dict | None = None) -> list[list[str]]:
    """Hasta `k` particiones de `word` en 2..MAX_SPLIT_PIECES palabras españolas,
    de mayor a menor probabilidad unigrama (Norvig). Si no hay ninguna y se da
    `phonetic_dict`, un trozo de 3+ letras puede ser una palabra funcional
    escrita como suena (`porezo` -> `por eso`; SymSpell daba `porrazo`)."""
    w = word.lower()
    if len(w) < 4:
        return []
    found = _walk_splits(w, freqs, accent_dict, None)
    if not found and phonetic_dict:
        found = _walk_splits(w, freqs, accent_dict, phonetic_dict)
    if not found:
        return []
    # Solo las particiones con menos trozos (`poreso` -> `por eso`, no `por es o`).
    fewest = min(len(p) for _, p in found)
    found = [(lp, p) for lp, p in found if len(p) == fewest]
    found.sort(key=lambda lp: -lp[0])
    return [p for _, p in found[:k]]


def _walk_splits(w: str, freqs: dict, accent_dict: dict, phonetic_dict: dict | None) -> list:
    """[(log-prob, trozos)] de las particiones plausibles de `w`."""
    compound = _looks_compound(w)
    total = sum(freqs.values()) or 1

    def piece(p: str) -> str | None:
        if len(p) == 1:
            return p if p in _SPLIT_SINGLE else None
        if is_spanish_word(p, freqs, accent_dict):
            return lookup(p, freqs, accent_dict) or p
        if phonetic_dict and len(p) >= 3:
            sound = phonetic_dict.get(to_phonetic(p))
            if sound in _SPLIT_SHORT_TAIL:
                return sound
        return None

    found = []

    def walk(start: int, pieces: list, logp: float) -> None:
        if start == len(w):
            if len(pieces) >= 2:
                found.append((logp, pieces))
            return
        if len(pieces) >= MAX_SPLIT_PIECES:
            return
        for end in range(start + 1, len(w) + 1):
            form = piece(w[start:end])
            if form is not None:
                walk(end, pieces + [form], logp + math.log(max(freqs.get(form, 1), 1) / total))

    walk(0, [], 0.0)
    found = [(lp, p) for lp, p in found if _plausible_split(p)]
    if compound:
        # `sobreentrenamiento` no se parte; `sobrelamesa` sí: tras el prefijo
        # viene una palabra funcional (sobre la mesa, contra la pared).
        found = [(lp, p) for lp, p in found
                 if len(p) >= 3 and p[0] in _COMPOUND_PREFIXES and p[1] in _FUNCTION]
    return found


def _plausible_split(pieces: list) -> bool:
    """Descarta particiones de palabras que sí son válidas aunque falten en es_50k."""
    if any(len(p) <= 3 and p not in _SPLIT_SHORT_TAIL for p in pieces[1:]):
        return False                               # imagina ti vos
    if pieces[-1] in _CLITICS and len(pieces) >= 2 and pieces[-2].endswith(_CLITIC_HOSTS):
        return False                               # confiar se
    return True


FUNCTION_WORDS = frozenset(_FUNCTION)


def _looks_compound(word: str) -> bool:
    """`sobreentrenamiento`, `latinoamericanos`: prefijo de compuesto + resto largo."""
    return any(word.startswith(p) and len(word) - len(p) >= 5 for p in _COMPOUND_PREFIXES)


def choose_split(tokens: list, i: int, splits: list, pick: str | None, judge=None,
                 pick_cost: float | None = None) -> str | None:
    """Capa 2: devuelve la partición (texto con espacios, sin puntuación) que
    sustituye a tokens[i], o None si gana la corrección de una palabra `pick`.
    Sin BETO: gana la partición más probable salvo que `pick` sea barata."""
    if not splits:
        return None
    pref, core, suff = _split_punct(tokens[i])
    options = [" ".join(s) for s in splits]
    # La corrección de una palabra es uno de los trozos (`tecuento` -> `cuento`
    # frente a `te cuento`): borra letras que forman palabra; gana la partición.
    for s, o in zip(splits, options):
        if pick is not None and pick.lower() in s:
            return o
    if judge is None:
        if pick is None or (pick_cost is not None and pick_cost > SPLIT_MARGIN):
            return options[0]
        return None
    variants = [pref + _match_case(core, o) + suff for o in options]
    pick_variant = pref + _match_case(core, pick) + suff if pick else None
    try:
        ranked = _best_variant(judge, tokens, i, variants + ([pick_variant] if pick_variant else []))
    except Exception as e:                               # BETO caído: gana la palabra
        print(f"[WARN] Segmentación (BETO) falló: {e}")
        return None
    best, best_score = next((v, s) for v, s in ranked if v != pick_variant)
    if pick_variant is not None:
        pick_score = dict(ranked)[pick_variant]
        if best_score - pick_score < SPLIT_MARGIN:
            return None
    return options[variants.index(best)]


__all__ = ["join_split_words", "split_candidates", "choose_split", "is_spanish_word",
           "lookup", "context_score", "SHORT_WORDS"]
