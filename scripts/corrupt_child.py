"""
scripts/corrupt_child.py
Corruptor de ESCRITURA INFANTIL para generar los datos de LoRA v5 a partir de frases limpias.

Tipología (pesos relativos) guiada por la escritura infantil publicada (Peláez 2016;
Querejeta 2007/2009; Minedu 2004/2013; DysList) y por la distribución del set de desarrollo
test_ninos_reales_dev (la segmentación es el error más frecuente; luego b/v, h, c/s/z, tildes):
  segmentación (partir y pegar), b/v, h, c/s/z, g/j, ll/y, r/rr, m/n ante p/b, qu -> k/q,
  letra omitida, haber (ha/he/hay/a ver), porque/por qué, concordancia de número del verbo,
  género del artículo, tilde omitida en una palabra, minúscula en nombre propio.
Además, por frase y sin contar como error: quitar TODAS las tildes (muchos niños no las
usan) o quitar la puntuación.

Densidad (errores contados por frase): 30 % sin errores (controles de copia), 25 % 1-2,
30 % 3-4, 15 % 5-6 (en el set real la media es 2,97 y 28/61 frases tienen 3 o más).
Cada operación devuelve None si no encuentra dónde aplicarse; se reintenta con otra.
"""
import random
import re
import unicodedata

_ACCENTS = str.maketrans("áéíóúÁÉÍÓÚ", "aeiouAEIOU")
_WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+")
_FUNCTION_JOIN = {"a", "al", "de", "del", "en", "la", "el", "lo", "le", "se", "me", "te", "que",
                  "por", "con", "su", "mi", "tu", "un", "y", "los", "las", "no", "es"}
_JOIN_PREFIXES = ("a", "al", "de", "en", "con", "es", "por", "para", "sin")   # trozos que son palabra
_PLURAL_VERB = [("ieron", "ió"), ("aron", "ó"), ("aban", "aba"), ("ían", "ía"), ("eron", "ó")]

DENSITY = [(0.30, (0, 0)), (0.25, (1, 2)), (0.30, (3, 4)), (0.15, (5, 6))]


def _words(text):
    return list(_WORD_RE.finditer(text))


def _replace_span(text, start, end, new):
    return text[:start] + new + text[end:]


def _pick(rng, items):
    return rng.choice(items) if items else None


def _sub_in_word(rng, text, pattern, repl, min_len=3):
    """Aplica `pattern` -> `repl` (regex) dentro de una palabra al azar que lo contenga."""
    cands = [m for m in _words(text) if len(m.group()) >= min_len and re.search(pattern, m.group())]
    m = _pick(rng, cands)
    if not m:
        return None
    word = m.group()
    hits = list(re.finditer(pattern, word))
    h = rng.choice(hits)
    new = word[:h.start()] + h.expand(repl) + word[h.end():]
    return None if new == word else _replace_span(text, m.start(), m.end(), new)


LEXICON: set = set()          # palabras válidas (minúsculas); lo carga set_lexicon()


def set_lexicon(words) -> None:
    LEXICON.clear()
    LEXICON.update(words)


def op_split(rng, text):
    """en contró, a bajo, con tenta, al morzar: el niño corta donde el primer trozo es un
    prefijo o una palabra corta, o donde uno de los dos trozos es palabra (nunca al azar)."""
    cands = [m for m in _words(text) if len(m.group()) >= 5]
    rng.shuffle(cands)
    for m in cands:
        w = m.group()
        low = w.lower()
        cuts = [len(p) for p in _JOIN_PREFIXES if low.startswith(p) and len(low) - len(p) >= 3]
        cuts += [i for i in range(3, len(low) - 2)
                 if low[:i] in LEXICON and low[i:] in LEXICON]      # ambos trozos, 3+ letras
        if cuts:
            cut = rng.choice(cuts)
            return _replace_span(text, m.start(), m.end(), w[:cut] + " " + w[cut:])
    return None


def op_join(rng, text):
    """sefue, ala, queno, poreso, asufamilia."""
    ws = _words(text)
    cands = [(a, b) for a, b in zip(ws, ws[1:])
             if a.group().lower() in _FUNCTION_JOIN and text[a.end():b.start()] == " "]
    if not cands:
        cands = [(a, b) for a, b in zip(ws, ws[1:]) if text[a.end():b.start()] == " " and len(a.group()) <= 4]
    # Nunca una unión que sea otra palabra real ("se paró" -> "separó", "en seguida").
    cands = [(a, b) for a, b in cands if (a.group() + b.group()).lower() not in LEXICON]
    pair = _pick(rng, cands)
    if not pair:
        return None
    a, b = pair
    return text[:a.end()] + text[b.start():]


def op_bv(rng, text):
    return _sub_in_word(rng, text, r"b", "v") if rng.random() < 0.5 else _sub_in_word(rng, text, r"v", "b")


def op_h(rng, text):
    if rng.random() < 0.8:
        return _sub_in_word(rng, text, r"^[Hh]", "", min_len=3) or _sub_in_word(rng, text, r"(?<=[aeiou])h", "")
    return _sub_in_word(rng, text, r"^([aeiou])(?=[a-z]{2})", r"h\1", min_len=3)


def op_csz(rng, text):
    choice = rng.random()
    if choice < 0.35:
        return _sub_in_word(rng, text, r"c(?=[eiéí])", "s")
    if choice < 0.6:
        return _sub_in_word(rng, text, r"z", "s")
    if choice < 0.85:
        return _sub_in_word(rng, text, r"s(?=[aeiouáéíóú])", "c" if rng.random() < 0.5 else "z")
    return _sub_in_word(rng, text, r"c(?=[eiéí])", "z")


def op_gj(rng, text):
    return (_sub_in_word(rng, text, r"g(?=[eiéí])", "j") if rng.random() < 0.6
            else _sub_in_word(rng, text, r"j(?=[eiéí])", "g"))


def op_lly(rng, text):
    return (_sub_in_word(rng, text, r"ll", "y") if rng.random() < 0.6
            else _sub_in_word(rng, text, r"(?<=[aeiou])y(?=[aeiou])", "ll"))


def op_rr(rng, text):
    return (_sub_in_word(rng, text, r"rr", "r") if rng.random() < 0.8
            else _sub_in_word(rng, text, r"(?<=[aeiou])r(?=[aeiou])", "rr"))


def op_mn(rng, text):
    return _sub_in_word(rng, text, r"m(?=[pb])", "n")


def op_qu(rng, text):
    return _sub_in_word(rng, text, r"qu(?=[eiéí])", "k" if rng.random() < 0.5 else "q", min_len=2)


def op_omit(rng, text):
    """cidado, pimero, estudiate: una letra interior omitida."""
    cands = [m for m in _words(text) if len(m.group()) >= 5]
    m = _pick(rng, cands)
    if not m:
        return None
    w = m.group()
    i = rng.randint(1, len(w) - 2)
    return _replace_span(text, m.start(), m.end(), w[:i] + w[i + 1:])


def op_haber(rng, text):
    options = [
        (r"\b[Hh]a\b(?= \w+[ai]do\b)", "a"), (r"\b[Hh]e\b(?= \w+[ai]do\b)", "e"), (r"\b[Hh]ay\b", "ay"),
        (r"\b[Hh]aber\b", "aber"), (r"\ba ver\b", "aver"), (r"\b[Hh]aya\b", "aya"),
        (r"\b[Hh]abía\b", "abia"), (r"\b[Hh]as\b(?= \w+[ai]do\b)", "as"),
    ]
    rng.shuffle(options)
    for pattern, repl in options:
        if re.search(pattern, text):
            return re.sub(pattern, repl, text, count=1)
    return None


def op_porque(rng, text):
    for pattern, repl in [(r"\bpor qué\b", "por que"), (r"\bporque\b", "por que"), (r"\bpor qué\b", "porque"),
                          (r"\bPor qué\b", "Porque")]:
        if re.search(pattern, text):
            return re.sub(pattern, repl, text, count=1)
    return None


def op_number(rng, text):
    """Verbo plural -> singular (estaban -> estaba, comieron -> comió): error de concordancia."""
    cands = [m for m in _words(text) if any(m.group().endswith(e) for e, _ in _PLURAL_VERB) and len(m.group()) >= 5]
    m = _pick(rng, cands)
    if not m:
        return None
    w = m.group()
    for ending, sg in _PLURAL_VERB:
        if w.endswith(ending):
            return _replace_span(text, m.start(), m.end(), w[: -len(ending)] + sg)
    return None


def op_gender(rng, text):
    swaps = [(r"\bla\b(?= \w+o\b)", "el"), (r"\bel\b(?= \w+a\b)", "la"), (r"\buna\b", "un"), (r"\bun\b(?= \w+a\b)", "una")]
    rng.shuffle(swaps)
    for pattern, repl in swaps:
        if re.search(pattern, text):
            return re.sub(pattern, repl, text, count=1)
    return None


def op_accent(rng, text):
    cands = [m for m in _words(text) if re.search(r"[áéíóú]", m.group())]
    m = _pick(rng, cands)
    return None if not m else _replace_span(text, m.start(), m.end(), m.group().translate(_ACCENTS))


def op_lower(rng, text):
    cands = [m for m in _words(text) if m.group()[:1].isupper() and m.start() > 0]
    m = _pick(rng, cands)
    return None if not m else _replace_span(text, m.start(), m.end(), m.group().lower())


OPERATIONS = [
    (op_split, 0.13), (op_join, 0.12), (op_bv, 0.10), (op_h, 0.09), (op_csz, 0.09), (op_accent, 0.08),
    (op_omit, 0.05), (op_gj, 0.04), (op_lly, 0.04), (op_haber, 0.04), (op_number, 0.04), (op_rr, 0.03),
    (op_gender, 0.03), (op_mn, 0.02), (op_qu, 0.02), (op_porque, 0.02),
]


def n_errors(rng):
    r, acc = rng.random(), 0.0
    for p, (lo, hi) in DENSITY:
        acc += p
        if r < acc:
            return rng.randint(lo, hi)
    return 0


def _strip_punctuation(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[¿¡,;:]", "", text).rstrip(".")).strip()


def corrupt(text: str, rng: random.Random, k: int | None = None) -> tuple:
    """Devuelve (texto corrompido, objetivo, [operaciones aplicadas]).

    La puntuación y la mayúscula inicial son NEUTRAS: si se quitan de la entrada, se
    quitan también del objetivo. El corrector respeta la puntuación del niño
    (intervención mínima, como las referencias de evaluación); v5a, que las dejaba en
    el objetivo, aprendió a añadir puntos finales, comas y mayúsculas (2026-09-30)."""
    k = n_errors(rng) if k is None else k
    applied, out, target = [], text, text
    if k == 0:
        return out, target, applied
    if rng.random() < 0.35:
        out = out.translate(_ACCENTS)
        applied.append("sin_tildes")
    if rng.random() < 0.30:
        out, target = _strip_punctuation(out), _strip_punctuation(target)
        applied.append("sin_puntuacion")
    ops, weights = zip(*OPERATIONS)
    tries = 0
    while len([a for a in applied if a not in ("sin_tildes", "sin_puntuacion")]) < k and tries < 4 * k + 8:
        tries += 1
        op = rng.choices(ops, weights)[0]
        if op in (op_split, op_join) and applied.count(op.__name__[3:]) >= 2:
            continue                                     # como mucho 2 cortes y 2 uniones por frase
        new = op(rng, out)
        if new and new != out:
            out = new
            applied.append(op.__name__[3:])
    if rng.random() < 0.25 and out[:1].isupper():
        out = out[0].lower() + out[1:]
        target = target[0].lower() + target[1:]
        applied.append("minuscula_inicial")
    return out, target, applied


__all__ = ["corrupt", "set_lexicon", "OPERATIONS", "DENSITY"]
