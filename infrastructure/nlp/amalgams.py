"""
infrastructure/nlp/amalgams.py
Amalgamas: palabras que el alumno escribe pegadas (o separadas donde no toca) y
que las capas léxicas NO pueden arreglar, porque la corrección cambia el número
de palabras y SymSpell solo sustituye una por una. Sin esta capa, `aver si
vienes` acaba en `ver si vienes` (SymSpell escoge la palabra más cercana).

Se ejecuta antes que todo lo demás (capa 0) porque cambia la tokenización.

Reglas (cada una con la condición mínima de contexto que la hace segura y el
ejemplo de alumno que la motiva):

  1. `aver`   -> `a ver`     — "aver" no existe en español: incondicional.
  2. `haber si` -> `a ver si` — "haber" + "si" solo es legítimo tras un modal
                                ("puede haber si quieres"); si no, es la
                                amalgama de "a ver".
  3. `asique` -> `así que`   — incondicional (no existe).
  4. `sino`   -> `si no`     — solo ante verbo conjugado o clítico
                                ("sino vienes"); la conjunción adversativa
                                ("no esto sino aquello", "sino que") se conserva.
  5. `porque` -> `por qué`   — solo cuando abre una interrogativa
                                ("porque no vienes?"); el causal se conserva.

La concordancia con colectivos (`la gente son` -> `la gente es`) vive en
grammar_rules.py, que es donde está el resto de la concordancia.
"""
import re
import unicodedata

# Formas verbales frecuentes tras "si no" (2.ª/3.ª persona de presente y
# pretérito). Sin tildes: se comparan contra `_norm`.
_SINO_VERBS = {
    "vienes", "viene", "vienen", "vas", "va", "van", "quieres", "quiere",
    "quieren", "puedes", "puede", "pueden", "tienes", "tiene", "tienen",
    "sabes", "sabe", "saben", "haces", "hace", "hacen", "estas", "esta",
    "estan", "eres", "es", "son", "llegas", "llega", "llegan", "sales",
    "sale", "salen", "estudias", "estudia", "estudian", "comes", "come",
    "comen", "hablas", "habla", "hablan", "traes", "trae", "traen", "dices",
    "dice", "dicen", "vuelves", "vuelve", "vuelven", "apuras", "apura",
    "terminas", "termina", "terminan", "empiezas", "empieza", "empiezan",
    "ayudas", "ayuda", "ayudan", "escuchas", "escucha", "escuchan",
}
# Pronombres átonos: "sino te apuras" -> "si no te apuras".
_SINO_CLITICS = {"me", "te", "se", "le", "nos", "les", "lo", "la", "los", "las"}

# Verbos que legitiman el infinitivo "haber" delante de "si".
_HABER_MODALS = {
    "puede", "pueden", "podia", "podria", "podrian", "debe", "deben", "debia",
    "deberia", "suele", "suelen", "tiene", "va", "iba", "parece", "de", "al",
}
_PARTICIPLE_RE = re.compile(r"(ado|ados|ada|adas|ido|idos|ida|idas)$", re.IGNORECASE)
_IRREGULAR_PARTICIPLES = {
    "hecho", "dicho", "visto", "puesto", "vuelto", "escrito", "roto", "abierto",
    "muerto", "cubierto", "descubierto", "resuelto", "impreso", "frito", "habido",
}
_PUNCT_RE = re.compile(r"^(\W*)(.*?)(\W*)$", re.UNICODE)
# Cierra una oración: tras esto, la palabra siguiente vuelve a "abrir" frase.
_SENTENCE_END = (".", "!", "?", "¿", "¡", ";", ":")


def _split(token: str) -> tuple:
    m = _PUNCT_RE.match(token)
    return (m.group(1), m.group(2), m.group(3)) if m else ("", token, "")


def _norm(word: str) -> str:
    w = word.strip(".,;:!?¿¡\"'()").lower()
    return "".join(c for c in unicodedata.normalize("NFD", w)
                   if unicodedata.category(c) != "Mn")


def _match_case(original: str, replacement: str) -> str:
    """Traslada la mayúscula inicial de `original` a la primera palabra."""
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _is_participle(word: str) -> bool:
    w = _norm(word)
    return bool(w in _IRREGULAR_PARTICIPLES or _PARTICIPLE_RE.search(w))


def expand_amalgams(text: str) -> str:
    """Separa (o une) las amalgamas de `text` y devuelve la frase resultante.

    Conserva la puntuación adyacente y la mayúscula inicial de cada token. Si
    no hay ninguna amalgama, devuelve `text` tal cual.
    """
    tokens = text.split()
    if not tokens:
        return text

    out: list[str] = []
    for i, token in enumerate(tokens):
        # ¿abre oración? primer token, tras cierre de frase, o tras '¿'/'¡'.
        opens_sentence = (
            i == 0
            or tokens[i - 1].endswith(_SENTENCE_END)
            or token.startswith(("¿", "¡"))
        )
        pref, core, suff = _split(token)
        low = _norm(core)
        prev = _norm(tokens[i - 1]) if i > 0 else ""
        nxt = _norm(tokens[i + 1]) if i + 1 < len(tokens) else ""
        replacement = None

        # 1. "aver si vienes" -> "a ver si vienes"
        if low == "aver":
            replacement = "a ver"

        # 2. "haber si vienes" -> "a ver si vienes" (no tras modal, no + participio)
        elif low == "haber" and nxt == "si" and prev not in _HABER_MODALS:
            replacement = "a ver"

        # 3. "asique me fui" -> "así que me fui"
        elif low in ("asique", "asiq"):
            replacement = "así que"

        # 4. "sino vienes" -> "si no vienes" (ante verbo o clítico; la
        #    adversativa "no esto sino aquello" y "sino que" se conservan)
        elif low == "sino" and (nxt in _SINO_VERBS or nxt in _SINO_CLITICS):
            replacement = "si no"

        # 5. "porque no vienes?" -> "por qué no vienes?" (solo si abre la
        #    interrogativa; el causal "no vine porque estaba enfermo" se conserva)
        elif low == "porque" and opens_sentence and ("?" in text or "¿" in text):
            replacement = "por qué"

        if replacement is not None:
            out.append(pref + _match_case(core, replacement) + suff)
        else:
            out.append(token)

    return " ".join(out)


__all__ = ["expand_amalgams"]
