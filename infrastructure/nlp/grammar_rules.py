"""
infrastructure/nlp/grammar_rules.py
Corrección gramatical por reglas de alta precisión (capa 3, mientras el T5 se
reentrena). Cubre errores frecuentes del español expresables sin parsing:

  1. Haber impersonal:  "habían muchas personas" -> "había muchas personas"
  2. Concordancia de gustar/encantar con objeto plural:
                        "me gusta los dulces"     -> "me gustan los dulces"
  3. Número tras cuantificador: "dos gato"         -> "dos gatos"

Cada regla es conservadora: solo dispara cuando el patrón es inequívoco, para
no introducir regresiones (verificado con evaluate.py sobre el set gold).
La concordancia sujeto-verbo general (fue->fueron) NO se intenta aquí: requiere
identificar el sujeto y es territorio del modelo generativo.
"""
import re
import unicodedata

# ── Utilidades ────────────────────────────────────────────────────────────────

_IRREGULAR_PARTICIPLES = {
    "hecho", "dicho", "visto", "puesto", "vuelto", "escrito", "roto", "abierto",
    "muerto", "cubierto", "descubierto", "resuelto", "impreso", "frito",
}
_PARTICIPLE_RE = re.compile(r"(ado|ados|ada|adas|ido|idos|ida|idas)$", re.IGNORECASE)
_GERUND_RE     = re.compile(r"(ando|endo)$", re.IGNORECASE)


def _clean(word: str) -> str:
    return word.strip(".,;:!?¿¡\"'()").lower()


def _norm(word: str) -> str:
    """Minúsculas, sin puntuación y SIN tildes: para emparejar aunque el texto
    llegue sin acentuar (p. ej. 'habian' antes de restaurar la tilde)."""
    w = _clean(word)
    return "".join(c for c in unicodedata.normalize("NFD", w)
                   if unicodedata.category(c) != "Mn")


def _is_verb_chain(word: str) -> bool:
    """True si la palabra parece participio o gerundio (uso auxiliar de haber)."""
    w = _norm(word)
    return bool(w in _IRREGULAR_PARTICIPLES or _PARTICIPLE_RE.search(w) or _GERUND_RE.search(w))


def _match_case(original: str, corrected: str) -> str:
    if original[:1].isupper():
        return corrected[:1].upper() + corrected[1:]
    return corrected


def _pluralize(noun: str) -> str:
    """Pluralización morfológica básica del español."""
    w = noun
    if re.search(r"[sx]$", w, re.IGNORECASE):
        return w                                  # ya plural o invariable
    if re.search(r"z$", w, re.IGNORECASE):
        return w[:-1] + "ces"
    if re.search(r"[aeiouáéíóú]$", w, re.IGNORECASE):
        return w + "s"
    return w + "es"


# ── Reglas ────────────────────────────────────────────────────────────────────

# 1. Haber impersonal: la forma plural es siempre incorrecta cuando NO va seguida
#    de participio/gerundio (que sería su uso auxiliar legítimo: "habían comido").
#    Claves SIN tilde (se emparejan con _norm); los valores llevan la tilde correcta.
_HABER_IMPERSONAL = {"habian": "había", "habrian": "habría", "hubieron": "hubo"}

# Cuantificadores que fuerzan plural en el sustantivo siguiente.
_QUANTIFIERS = {
    "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez",
    "varios", "varias", "muchos", "muchas", "pocos", "pocas", "algunos", "algunas",
    "unos", "unas",
}
# Palabras que NO se deben pluralizar aunque sigan a un número.
# Sin tildes: se comparan contra _norm(palabra).
_QUANT_STOP = {
    "mil", "millon", "millones", "por", "mas", "menos", "veces", "y", "o", "de",
}

# 2. Gustar/encantar: verbo singular + objeto plural -> verbo plural.
#    Solo presente/imperfecto (inequívocos). Se omite "gustó" a propósito: su
#    forma sin tilde choca con el sustantivo "gusto".
_GUSTAR_SG = {
    "gusta": "gustan", "gustaba": "gustaban",
    "encanta": "encantan", "encantaba": "encantaban",
}
_PLURAL_DET = _QUANTIFIERS | {
    "los", "las", "estos", "estas", "esos", "esas", "aquellos", "aquellas", "sus", "mis", "tus",
}
_CLITICS = {"me", "te", "le", "nos", "os", "les"}

# 4-6. Tilde diacrítica y auxiliares monosílabos.
# BETO NO sirve aquí: su probabilidad favorece la forma SIN tilde (más común en
# corpus), así que se resuelve con marcos sintácticos de alta precisión.
_PREPS = {"a", "para", "de", "por", "con", "sin", "hacia", "hasta", "sobre",
          "entre", "ante", "segun"}
# "se" -> "sé" (verbo saber) solo ante interrogativo/negación; NUNCA ante clítico
# (evita el falso "no se lo dije" -> "sé lo", que sí es pronombre).
_SE_PREV = {"no", "yo", "ya"}
_SE_NEXT = {"que", "si", "donde", "como", "cuando", "cuanto", "nada", "nadie",
            "quien", "por"}
# "mi" -> "mí" (pronombre) tras preposición y ante clítico o fin de frase
# ("a mí me gustan", "es para mí"); NO ante sustantivo ("a mi casa").
_MI_NEXT = _CLITICS | {"mismo", "misma"}


def correct_grammar(text: str) -> str:
    """Aplica las reglas gramaticales sobre `text` y devuelve la frase corregida."""
    tokens = text.split()
    if not tokens:
        return text

    def next_content(idx: int) -> str:
        return tokens[idx + 1] if idx + 1 < len(tokens) else ""

    for i, tok in enumerate(tokens):
        low = _norm(tok)

        # 1. Haber impersonal
        if low in _HABER_IMPERSONAL and not _is_verb_chain(next_content(i)):
            tokens[i] = _reword(tok, _HABER_IMPERSONAL[low])
            continue

        # 2. Concordancia de gustar/encantar con objeto plural
        if low in _GUSTAR_SG and _norm(next_content(i)) in _PLURAL_DET:
            prev = _norm(tokens[i - 1]) if i > 0 else ""
            if prev in _CLITICS or prev == "no":
                tokens[i] = _reword(tok, _GUSTAR_SG[low])
                continue

        # 3. Número + sustantivo singular -> pluralizar el sustantivo
        if low in _QUANTIFIERS:
            nxt = next_content(i)
            nxt_norm = _norm(nxt)
            if nxt_norm and nxt_norm not in _QUANT_STOP and not re.search(r"[sx]$", nxt_norm):
                tokens[i + 1] = _reword(nxt, _pluralize(_clean(nxt)))

        prev = _norm(tokens[i - 1]) if i > 0 else ""
        nxt  = _norm(next_content(i))

        # 4. mí (pronombre) tras preposición, ante clítico o fin de frase
        if low == "mi" and prev in _PREPS and (nxt in _MI_NEXT or nxt == ""):
            tokens[i] = _reword(tok, "mí")
            continue

        # 5. sé (verbo saber) ante interrogativo/negación (no ante clítico)
        if low == "se" and prev in _SE_PREV and nxt in _SE_NEXT:
            tokens[i] = _reword(tok, "sé")
            continue

        # 6. he (auxiliar): "e" + participio -> "he" ("e comido" -> "he comido")
        if low == "e" and _is_verb_chain(next_content(i)):
            tokens[i] = _reword(tok, "he")
            continue

    return " ".join(tokens)


def _reword(original_token: str, replacement: str) -> str:
    """Sustituye el núcleo de `original_token` por `replacement` preservando
    puntuación adyacente y capitalización."""
    m = re.match(r"^(\W*)(.*?)(\W*)$", original_token, re.UNICODE)
    pref, core, suff = (m.group(1), m.group(2), m.group(3)) if m else ("", original_token, "")
    return pref + _match_case(core, replacement) + suff
