"""
infrastructure/nlp/amalgams.py
Amalgamas: palabras que el alumno escribe pegadas (o separadas donde no toca) y
que las capas léxicas NO pueden arreglar, porque la corrección cambia el número
de palabras y SymSpell solo sustituye una por una. Sin esta capa, `aver si
vienes` acaba en `ver si vienes` (SymSpell escoge la palabra más cercana).

Se ejecuta antes que todo lo demás (capa 0) porque cambia la tokenización.

Reglas (cada una con la condición mínima de contexto que la hace segura y el
ejemplo de alumno que la motiva):

  1. `aver`   -> `a ver`     — "aver" no existe en español: incondicional,
                                salvo ante participio: `aver pasado` ->
                                `haber pasado` (el infinitivo de haber).
  2. `haber si` -> `a ver si` — "haber" + "si" solo es legítimo tras un modal
                                ("puede haber si quieres"); si no, es la
                                amalgama de "a ver".
  3. `asique` -> `así que`   — incondicional (no existe).
  4. `sino`   -> `si no`     — ante verbo conjugado o clítico ("sino
                                vienes") o abriendo la oración ("Sino llueve,
                                ..."); la conjunción adversativa ("no esto
                                sino aquello", "sino que") se conserva.
  5. `porque` -> `por qué`   — solo cuando abre una interrogativa
                                ("porque no vienes?"); el causal se conserva.
     `por que` -> `por qué`   — abriendo la oración (tras `y`/`pero` como mucho):
                                "y por que tienes la nariz tan grande".
     `por que` -> `porque`    — en mitad de la frase ante sujeto, negación,
                                clítico o determinante, salvo tras un verbo
                                de pregunta o saber ("no sé por qué"):
                                "no entendía nada por que yo no quería".
  6. Locuciones pegadas que no existen como palabra (`porfavor`, `enserio`,
     `aveces`, `derrepente`, `talvez`, `osea`, `nose`...): incondicional.
  7. `con migo/tigo/sigo` -> `conmigo/contigo/consigo`: incondicional.
  8. `haber` tras verbo de movimiento: `ir haber a mi abuela` -> `ir a ver a
     mi abuela` (ante `a`, `si`, interrogativo o determinante definido);
     `va haber una fiesta` -> `va a haber una fiesta` (ante indefinido o
     cuantificador: el "haber" existencial).

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
    "llueve", "nieva", "funciona", "pasa", "sirve", "gusta", "importa",
    "entiendes", "entiende", "cambias", "cambia", "corres", "corre", "juegas",
    "juega", "duermes", "duerme", "aprendes", "aprende", "lees", "lee",
    "escribes", "escribe", "respondes", "responde", "contestas", "contesta",
    "practicas", "practica", "trabajas", "trabaja", "limpias", "limpia",
    "ordenas", "ordena", "guardas", "guarda", "pagas", "paga", "llamas",
    "llama", "esperas", "espera", "consigues", "consigue", "logras", "logra",
    "subes", "sube", "bajas", "baja", "entras", "entra", "abres", "abre",
    "cierras", "cierra", "ganas", "gana", "pierdes", "pierde", "cuidas",
    "cuida", "avisas", "avisa", "obedeces", "obedece",
    "viniste", "vino", "vinieron", "fuiste", "fue", "fueron", "hiciste",
    "hizo", "hicieron", "pudiste", "pudo", "pudieron", "quisiste", "quiso",
    "tuviste", "tuvo", "llegaste", "llego", "terminaste", "termino",
    "ayudaste", "ayudo", "estudiaste", "estudio",
}
# Pronombres átonos: "sino te apuras" -> "si no te apuras".
_SINO_CLITICS = {"me", "te", "se", "le", "nos", "les", "lo", "la", "los", "las"}

# Verbos que legitiman el infinitivo "haber" delante de "si".
_HABER_MODALS = {
    "puede", "pueden", "podia", "podria", "podrian", "debe", "deben", "debia",
    "deberia", "suele", "suelen", "tiene", "va", "iba", "parece", "de", "al",
}
# 6. Locuciones escritas en una palabra (ninguna existe en español).
_JOINED = {
    "porfavor": "por favor", "enserio": "en serio",
    "aveces": "a veces", "derrepente": "de repente", "depronto": "de pronto",
    "talvez": "tal vez", "osea": "o sea", "apesar": "a pesar",
    "atraves": "a través", "atravez": "a través", "deveras": "de veras",
    "nose": "no sé", "sinembargo": "sin embargo", "encambio": "en cambio",
    "almenos": "al menos", "porsupuesto": "por supuesto", "enfin": "en fin",
}
# 7. Pronombres con "con" escritos separados.
_CON_JOIN = {"migo": "conmigo", "tigo": "contigo", "sigo": "consigo"}
# 8. Verbos de movimiento que rigen "a + infinitivo" ("voy a ver", "va a haber").
_MOTION = {
    "ir", "voy", "vas", "va", "vamos", "van", "fui", "fue", "fuimos", "fueron",
    "iba", "ibas", "iban", "ibamos", "vine", "vino", "vinimos", "vinieron",
    "vengo", "viene", "vienen", "venimos",
}
# Sin "si": "haber si" lo resuelve la regla 2 (respeta los modales).
_VER_NEXT = {
    "a", "al", "que", "como", "quien", "quienes", "cuando", "donde",
    "el", "la", "los", "las", "mi", "mis", "tu", "tus", "su", "sus",
    "esto", "eso", "este", "esta", "ese", "esa", "estos", "esas",
}
_HABER_EXIST_NEXT = {
    "un", "una", "unos", "unas", "mucho", "mucha", "muchos", "muchas", "mas",
    "poco", "poca", "pocos", "pocas", "algo", "alguien", "nada", "nadie",
    "clases", "clase", "examen", "fiesta", "lluvia", "tiempo", "problemas",
}
_PARTICIPLE_RE = re.compile(r"(ado|ados|ada|adas|ido|idos|ida|idas)$", re.IGNORECASE)
# 5b. "por que" causal: lo que sigue abre una oración (sujeto, negación,
#     clítico, determinante); lo que precede no pide interrogativa indirecta.
_CAUSAL_NEXT = {
    "yo", "tu", "el", "ella", "nosotros", "nosotras", "ellos", "ellas", "usted",
    "ustedes", "no", "me", "te", "se", "le", "les", "lo", "la", "los", "las", "nos",
    "mi", "mis", "su", "sus", "un", "una", "ya", "estaba", "era", "tenia", "habia", "hay",
}
_ASK_PREV = {
    "pregunto", "pregunta", "pregunte", "pregunto", "preguntaba", "preguntar", "preguntan",
    "se", "sabe", "sabes", "saber", "sabia", "sabemos", "saben", "entiendo", "entiende",
    "entender", "entendia", "explica", "explico", "explicar", "dime", "digas", "averiguar",
    "el", "un", "su", "del", "al",           # "el por qué" (sustantivo)
}
_CLAUSE_OPENERS = {"y", "e", "pero", "entonces"}
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

    # 7. "con migo" -> "conmigo": une antes del resto porque cambia el número
    #    de tokens (la puntuación de "migo," pasa a la palabra unida).
    joined: list[str] = []
    for token in tokens:
        pref, core, suff = _split(token)
        if (joined and _norm(core) in _CON_JOIN and not pref
                and _norm(joined[-1]) == "con" and _split(joined[-1])[2] == ""):
            prev_pref, prev_core, _ = _split(joined[-1])
            joined[-1] = prev_pref + _match_case(prev_core, _CON_JOIN[_norm(core)]) + suff
        else:
            joined.append(token)
    tokens = joined

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

        # 1. "aver pasado" -> "haber pasado"; "aver si vienes" -> "a ver si vienes"
        if low == "aver" and _is_participle(nxt):
            replacement = "haber"
        elif low == "aver":
            replacement = "a ver"

        # 2. "haber si vienes" -> "a ver si vienes" (no tras modal, no + participio)
        elif low == "haber" and nxt == "si" and prev not in _HABER_MODALS:
            replacement = "a ver"

        # 8. "ir haber a mi abuela" -> "ir a ver a mi abuela";
        #    "va haber una fiesta" -> "va a haber una fiesta"
        elif low == "haber" and prev in _MOTION and nxt in _VER_NEXT:
            replacement = "a ver"
        elif low == "haber" and prev in _MOTION and nxt in _HABER_EXIST_NEXT:
            replacement = "a haber"

        # 6. "porfavor" -> "por favor", "enserio" -> "en serio"...
        elif low in _JOINED:
            replacement = _JOINED[low]

        # 3. "asique me fui" -> "así que me fui"
        elif low in ("asique", "asiq"):
            replacement = "así que"

        # 4. "sino vienes" -> "si no vienes" (ante verbo o clítico, o abriendo
        #    la oración: la adversativa "no esto sino aquello" nunca abre frase;
        #    "sino que" se conserva)
        elif low == "sino" and (
            nxt in _SINO_VERBS or nxt in _SINO_CLITICS
            or (opens_sentence and nxt and nxt != "que")
        ):
            replacement = "si no"

        # 5. "porque no vienes?" -> "por qué no vienes?" (solo si abre la
        #    interrogativa; el causal "no vine porque estaba enfermo" se conserva)
        elif low == "porque" and opens_sentence and ("?" in text or "¿" in text):
            replacement = "por qué"

        # 5b. "por que": interrogativo abriendo la oración, causal en mitad.
        elif low == "por" and nxt == "que" and not suff and i + 1 < len(tokens):
            answers_question = i > 0 and tokens[i - 1].endswith("?")   # "¿Por qué...? Por que me caí"
            opens_clause = not answers_question and (opens_sentence or (
                i == 1 and prev in _CLAUSE_OPENERS) or (
                i >= 1 and prev in _CLAUSE_OPENERS and (i == 1 or tokens[i - 2].endswith(_SENTENCE_END))))
            after = _norm(tokens[i + 2]) if i + 2 < len(tokens) else ""
            que_pref, que_core, que_suff = _split(tokens[i + 1])
            if opens_clause:
                out.append(pref + core + suff)
                tokens[i + 1] = que_pref + "qué" + que_suff
                continue
            if after in _CAUSAL_NEXT and prev not in _ASK_PREV and not que_suff:
                out.append(pref + _match_case(core, "porque") + suff)
                tokens[i + 1] = ""
                continue

        if replacement is not None:
            out.append(pref + _match_case(core, replacement) + suff)
        elif token:
            out.append(token)

    return " ".join(out)


__all__ = ["expand_amalgams"]
