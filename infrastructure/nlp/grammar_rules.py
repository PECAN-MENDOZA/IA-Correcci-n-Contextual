"""
infrastructure/nlp/grammar_rules.py
Corrección gramatical por reglas de alta precisión (capa 3, mientras el T5 se
reentrena). Cubre errores frecuentes del español expresables sin parsing:

  1. Haber impersonal:  "habían muchas personas" -> "había muchas personas"
  2. Concordancia de gustar/encantar con objeto plural:
                        "me gusta los dulces"     -> "me gustan los dulces"
  3. Número tras cuantificador: "dos gato"         -> "dos gatos"
  9. Colectivo singular:  "la gente son amables"   -> "la gente es amable"
                          (irregulares por tabla; regulares por sufijo, con
                          el léxico es_50k como guarda para el presente)
 10. Ortografía vigente:  "sólo" -> "solo" (RAE 2010; `normalize_modern_spelling`,
                          se aplica al final del pipeline)
 11. Adverbio de lugar:   "mi mochila está hay"    -> "mi mochila está ahí"
 12. Artículo femenino:   "el canción"             -> "la canción"
                          (solo sufijos siempre femeninos: -ción, -dad...)
 13. Posesivo plural:     "mi amigos"              -> "mis amigos"
 14. Auxiliar haber:      "a habido muchos asaltos" -> "ha habido" ("a" +
                          participio masculino, salvo sustantivos en -ado/-ido
                          y tras verbo de movimiento); "le aya pasado" -> "haya"

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
_HABER_IMPERSONAL = {
    "habian": "había", "habrian": "habría", "hubieron": "hubo",
    "habran": "habrá", "hayan": "haya", "hubieran": "hubiera", "hubiesen": "hubiese",
}

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

# "el"/"tu" -> "él"/"tú" (pronombre sujeto) cuando les sigue un VERBO. El
# artículo "el" y el posesivo "tu" van siempre ante sustantivo. Conjuntos
# curados (sin tilde, se comparan con _norm) de formas verbales frecuentes.
# Para "el" se usan solo formas de 3ª persona que NO coinciden con sustantivos
# comunes (se evitan "canto/juego/trabajo/jugó", que serían nombres).
_EL_VERBS = {
    "es", "era", "fue", "sera", "esta", "estaba", "estuvo", "tiene", "tenia",
    "tuvo", "hace", "hacia", "hizo", "va", "iba", "viene", "vino", "venia",
    "dice", "dijo", "decia", "sabe", "sabia", "supo", "quiere", "queria",
    "quiso", "puede", "podia", "pudo", "debe", "debia", "cree", "creia", "ve",
    "vio", "da", "dio", "llega", "come", "corre", "vive", "vivia", "habla",
    "mira", "deja", "sale", "entra", "gana", "lee", "canta", "baila", "duerme",
    "siente", "parece", "necesita", "busca", "encuentra", "lleva", "trae",
    "pone", "piensa", "conoce", "entiende", "prefiere",
}
# Para "tú": 2ª persona. Presente termina en -s (tienes, sabes, eres...) y el
# posesivo "tu" nunca precede palabra en -s (sería "tus"), así que "tu"+(-s) es
# pronombre; se excluyen sustantivos singulares en -s (crisis, país, tos...).
_TU_VERBS = {
    "eres", "fuiste", "estuviste", "tuviste", "hiciste", "viste", "diste",
    "dijiste", "quisiste", "pudiste", "supiste", "comiste", "corriste",
    "viviste", "hablaste", "llegaste", "jugaste", "trabajaste", "pensaste",
    "saliste", "ganaste", "leiste", "escribiste", "miraste", "viniste",
    "volviste", "conociste", "seguiste", "entendiste", "trajiste", "dormiste",
    "pediste", "empezaste", "terminaste", "ayudaste", "estudiaste", "pusiste",
}
_S_NOUN_STOP = {
    "crisis", "analisis", "tos", "virus", "atlas", "lunes", "martes",
    "miercoles", "jueves", "viernes", "mes", "pais", "interes", "gas",
    "sintesis", "dosis", "iris", "bilis",
}

# 9. Colectivos singulares + verbo plural inmediato: "la gente son amables" ->
#    "la gente es amable". La norma exige el singular con estos sustantivos;
#    se excluyen a propósito los cuantificadores partitivos ("la mayoría",
#    "parte de", "un montón de"), donde la concordancia ad sensum ("la mayoría
#    llegaron") está aceptada y corregirla sería un falso positivo.
_COLLECTIVE_NOUNS = {
    "gente", "familia", "grupo", "equipo", "publico", "gentio", "multitud",
    "muchedumbre", "policia", "ejercito", "gobierno", "jurado", "alumnado",
    "profesorado",
}
# Formas irregulares o que no se singularizan por sufijo. Las regulares las
# resuelve `_collective_singular` por morfología (ver abajo).
_PLURAL_TO_SG_VERB = {
    "son": "es", "estan": "está", "eran": "era", "estaban": "estaba",
    "fueron": "fue", "seran": "será", "estaran": "estará",
    "tienen": "tiene", "tenian": "tenía", "tuvieron": "tuvo",
    "van": "va", "iban": "iba", "fueran": "fuera",
    "vienen": "viene", "venian": "venía", "vinieron": "vino",
    "hacen": "hace", "hacian": "hacía", "hicieron": "hizo",
    "dicen": "dice", "decian": "decía", "dijeron": "dijo",
    "quieren": "quiere", "querian": "quería", "quisieron": "quiso",
    "pueden": "puede", "podian": "podía", "pudieron": "pudo",
    "deben": "debe", "debian": "debía", "saben": "sabe", "sabian": "sabía",
    "llegan": "llega", "llegaron": "llegó", "salen": "sale", "salieron": "salió",
    "ganan": "gana", "ganaron": "ganó", "juegan": "juega", "jugaron": "jugó",
    "comen": "come", "comieron": "comió", "viven": "vive", "vivian": "vivía",
    "hablan": "habla", "hablaban": "hablaba", "piensan": "piensa",
    "parecen": "parece", "parecian": "parecía", "necesitan": "necesita",
    "trabajan": "trabaja", "esperan": "espera", "entran": "entra",
    "salieran": "saliera", "gritan": "grita", "aplauden": "aplaude",
    # Futuros irregulares (el sufijo no es -arán/-erán/-irán).
    "vendran": "vendrá", "tendran": "tendrá", "pondran": "pondrá",
    "saldran": "saldrá", "podran": "podrá", "querran": "querrá",
    "sabran": "sabrá", "habran": "habrá", "haran": "hará", "diran": "dirá",
    "valdran": "valdrá", "cabran": "cabrá",
}
# Singularización morfológica del verbo plural (claves sin tilde). Los
# sufijos de pretérito, imperfecto y futuro son inequívocamente verbales; el
# presente (-an/-en) choca con sustantivos ("examen", "pan"), así que solo se
# acepta si la forma singular existe en el léxico es_50k y no está en la
# lista de colisiones conocidas.
_COLLECTIVE_VERB_SUFFIXES = (
    # (sufijo plural, sufijo singular, exige léxico)
    ("ieron", "ió", False), ("yeron", "yó", False), ("aron", "ó", False),
    ("aban", "aba", False),
    ("aran", "ará", False), ("eran", "erá", False), ("iran", "irá", False),
    ("an", "a", True), ("en", "e", True),
)
_COLLECTIVE_SG_STOP = {"tre", "crime", "resume", "orde", "image", "volume", "ta", "pa", "sa"}
_LEXICON_PATH = "./es_50k.txt"
_lexicon_cache: set | None = None


def _lexicon() -> set:
    """Palabras del corpus es_50k (primera columna), cargadas una sola vez.
    Sin el archivo devuelve un conjunto vacío: las reglas que exigen léxico
    simplemente no disparan."""
    global _lexicon_cache
    if _lexicon_cache is None:
        words: set = set()
        try:
            with open(_LEXICON_PATH, encoding="utf-8") as fh:
                for line in fh:
                    parts = line.split()
                    if parts:
                        words.add(parts[0].lower())
        except OSError:
            pass
        _lexicon_cache = words
    return _lexicon_cache


def _collective_singular(verb: str) -> str | None:
    """Forma singular (con tilde) de un verbo plural tras un colectivo, o None
    si no se reconoce con seguridad. `verb` es el token original (puede traer
    tilde); la clave se normaliza."""
    low = _norm(verb)
    if low in _PLURAL_TO_SG_VERB:
        return _PLURAL_TO_SG_VERB[low]
    if len(low) < 4:
        return None
    # -ían (imperfecto: tenían -> tenía) vs -ian (presente: cambian -> cambia).
    # Sin tilde son indistinguibles: gana la lectura cuyo singular está en el léxico.
    if low.endswith("ian"):
        stem = low[:-3]
        if "ían" in _clean(verb) or stem + "ía" in _lexicon():
            return stem + "ía"
        if stem + "ia" in _lexicon() and stem + "ia" not in _COLLECTIVE_SG_STOP:
            return stem + "ia"
        return None
    for plural, singular, needs_lexicon in _COLLECTIVE_VERB_SUFFIXES:
        if low.endswith(plural) and len(low) > len(plural) + 1:
            candidate = low[: -len(plural)] + singular
            if needs_lexicon and (candidate not in _lexicon() or candidate in _COLLECTIVE_SG_STOP):
                return None
            return candidate
    return None


# 11. "hay" -> "ahí": verbos de ubicación que piden un adverbio de lugar
#     detrás, y preposiciones que solo admiten "ahí" a final de frase.
_AHI_VERBS = {
    "esta", "estan", "estaba", "estaban", "estoy", "estas", "estamos", "estuvo",
    "estuvieron", "queda", "quedo", "quedan", "quedaron", "deje", "dejo",
    "dejaste", "puse", "puso", "pusiste", "pon", "ponlo", "ponla", "dejalo", "dejala",
}
_AHI_PREPS = {"por", "de", "hasta", "desde"}
# "por hay" + verbo u otra cosa es "por ahí" ("por hay voy a entrar"); ante
# determinante o cuantificador puede ser el existencial ("por hay muchos
# perros" = "por ahí hay..."): se deja. "hasta hay gente que..." es
# gramatical, por eso las demás preposiciones siguen pidiendo final de frase.
_AHI_ALWAYS_PREPS = {"por"}
_EXISTENTIAL_NEXT = {
    "un", "una", "unos", "unas", "mucho", "mucha", "muchos", "muchas", "poco", "poca",
    "pocos", "pocas", "varios", "varias", "algunos", "algunas", "algo", "alguien", "nada",
    "nadie", "mas", "tanto", "tanta", "tantos", "tantas", "dos", "tres", "cuatro", "cinco",
    "que", "de",
}

# 14. "a" + participio masculino singular -> "ha". Sustantivos en -ado/-ido
#     que sí van tras la preposición "a" ("a pedido de", "a lado", "a cuidado").
_A_PARTICIPLE_NOUNS = {
    "lado", "pedido", "partido", "mercado", "sentido", "cuidado", "helado", "pescado",
    "soldado", "abogado", "ruido", "vestido", "marido", "apellido", "sonido", "contenido",
    "significado", "resultado", "grado", "cunado", "prado", "punado", "tejado", "bocado",
    "teclado", "nido", "olvido", "oido", "latido", "gemido", "silbido", "ladrido",
    "chillido", "menudo", "medio", "estadio", "senado", "juzgado", "arado", "venado",
    "ganado", "pasado", "futuro", "cercado", "comunicado", "tratado", "mandado", "recado",
    "nado",
}
_A_PARTICIPLE_MASC = re.compile(r"(ado|ido)$", re.IGNORECASE)
# Tras un verbo de movimiento "a" es preposición ("fue a cuidado de...").
_MOTION_PREV = {
    "voy", "vas", "va", "vamos", "van", "fui", "fue", "fuimos", "fueron", "iba", "iban",
    "ir", "llego", "llegue", "llegamos", "llegaron", "vino", "vine", "volvio", "volvi",
    "sali", "salio", "entro", "entre", "paso", "subio", "bajo",
}
_HAYA_MISSPELLINGS = {"aya"}
_AHI_PREP_NEXT = {"que", "mismo", "nomas", "no"}

# 12. Sufijos siempre femeninos (-ción, -sión, -dad, -tad, -tud, -umbre) y la
#     forma femenina del determinante masculino que los precede.
_FEM_SUFFIXES = ("cion", "sion", "dad", "tad", "tud", "umbre")
_FEM_ARTICLE = {"el": "la", "un": "una", "del": "de la", "al": "a la",
                "este": "esta", "ese": "esa", "aquel": "aquella"}

# 9 (género). Colectivos femeninos: su predicado va en femenino.
_FEM_COLLECTIVES = {"gente", "familia", "policia", "multitud", "muchedumbre"}

# 13. Posesivo singular ante sustantivo plural. Singulares terminados en
#     -os/-as que no deben pluralizar el posesivo (su forma sin -s existe).
# Sin "tu": "tu" + palabra en -s es el pronombre "tú" (regla 8: "tu juegas").
_POSSESSIVE_SG = {"mi", "su"}
_POSSESSIVE_PLURAL_STOP = {"dios", "caos", "cosmos", "atlas", "tras", "pues", "jamas", "adios"}

# Intensificadores que pueden separar el verbo del predicado.
_INTENSIFIERS = {"muy", "tan", "bastante", "demasiado", "poco", "algo", "super"}
# Lista blanca de adjetivos que se singularizan detrás del verbo corregido. Es
# una lista cerrada a propósito: singularizar cualquier palabra en -s rompería
# sintagmas nominales ("la gente son mis vecinos").
_COLLECTIVE_ADJECTIVES = {
    "amables", "buenos", "buenas", "malos", "malas", "felices", "tranquilos",
    "tranquilas", "ruidosos", "ruidosas", "simpaticos", "simpaticas",
    "groseros", "groseras", "educados", "educadas", "generosos", "generosas",
    "grandes", "pequenos", "pequenas", "jovenes", "viejos", "viejas",
    "pobres", "ricos", "ricas", "cansados", "cansadas", "contentos",
    "contentas", "listos", "listas", "altos", "altas", "bajos", "bajas",
    "alegres", "serios", "serias", "curiosos", "curiosas", "puntuales",
    "responsables", "unidos", "unidas", "fuertes", "nerviosos", "nerviosas",
}


def _singularize(word: str) -> str:
    """Singularización morfológica básica (inversa de `_pluralize`)."""
    w = word
    if re.search(r"ces$", w, re.IGNORECASE):
        return w[:-3] + "z"                       # felices -> feliz
    if re.search(r"[aeiouáéíóú]s$", w, re.IGNORECASE):
        return w[:-1]                             # amables -> amable
    if re.search(r"es$", w, re.IGNORECASE):
        return w[:-2]                             # papeles -> papel
    return w


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

        # 9. Colectivo singular + verbo plural inmediato -> verbo (y predicado
        #    de la lista blanca) en singular: "la gente son muy amables".
        singular = _collective_singular(next_content(i)) if low in _COLLECTIVE_NOUNS else None
        if singular is not None:
            tokens[i + 1] = _reword(tokens[i + 1], singular)
            j = i + 2
            if j < len(tokens) and _norm(tokens[j]) in _INTENSIFIERS:
                j += 1
            if j < len(tokens) and _norm(tokens[j]) in _COLLECTIVE_ADJECTIVES:
                adjective = _singularize(_clean(tokens[j]))
                # Colectivo femenino: "la gente está contenta" (no "contento").
                if low in _FEM_COLLECTIVES and adjective.endswith("o"):
                    adjective = adjective[:-1] + "a"
                tokens[j] = _reword(tokens[j], adjective)

        prev = _norm(tokens[i - 1]) if i > 0 else ""
        nxt  = _norm(next_content(i))

        # 13. Posesivo singular + sustantivo plural en -os/-as: "mi amigos" ->
        #     "mis amigos". El singular (sin la -s) debe existir en es_50k; se
        #     excluyen los singulares en -s ("mi dios") y los nombres propios.
        nxt_tok = _clean(next_content(i))
        if (low in _POSSESSIVE_SG and len(nxt) > 3 and nxt.endswith(("os", "as"))
                and not nxt.endswith("mos")
                and nxt not in _S_NOUN_STOP and nxt not in _POSSESSIVE_PLURAL_STOP
                and next_content(i)[:1].islower() and nxt_tok[:-1] in _lexicon()):
            tokens[i] = _reword(tok, low + "s")
            continue

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

        # 14. ha / haya (auxiliar): "a habido" -> "ha habido", "le aya pasado"
        #     -> "le haya pasado". Solo participio masculino singular (el de
        #     los tiempos compuestos) y nunca tras puntuación en "a".
        if (low == "a" and tok == tok.rstrip(".,;:!?") and prev not in _MOTION_PREV
                and next_content(i)[:1].islower() and nxt not in _A_PARTICIPLE_NOUNS
                and (_A_PARTICIPLE_MASC.search(nxt) or nxt in _IRREGULAR_PARTICIPLES)):
            tokens[i] = _reword(tok, "ha")
            continue
        if low in _HAYA_MISSPELLINGS and (_PARTICIPLE_RE.search(nxt) or nxt in _IRREGULAR_PARTICIPLES):
            tokens[i] = _reword(tok, "haya")
            continue

        # 7. él (pronombre) vs el (artículo): "el" + verbo -> "él"
        if low == "el" and nxt in _EL_VERBS:
            tokens[i] = _reword(tok, "él")
            continue

        # 8. tú (pronombre) vs tu (posesivo): "tu" + verbo (o palabra en -s
        #    que no sea sustantivo singular) -> "tú"
        if low == "tu" and (
            nxt in _TU_VERBS
            or (len(nxt) > 2 and nxt.endswith("s") and nxt not in _S_NOUN_STOP)
        ):
            tokens[i] = _reword(tok, "tú")
            continue

        # 11. ahí (lugar) vs hay (haber): "está hay" -> "está ahí"; "por hay"
        #     a final de frase -> "por ahí". Un signo tras el verbo corta el
        #     marco ("está, hay ...").
        if low == "hay" and i > 0 and _clean(tokens[i - 1]) == tokens[i - 1].lower():
            ends = nxt == "" or tok != tok.rstrip(".,;:!?")
            if (prev in _AHI_VERBS or (prev in _AHI_ALWAYS_PREPS and nxt not in _EXISTENTIAL_NEXT)
                    or (prev in _AHI_PREPS and (ends or nxt in _AHI_PREP_NEXT))):
                tokens[i] = _reword(tok, "ahí")
                continue

        # 12. Artículo masculino ante sustantivo femenino por sufijo:
        #     "el canción" -> "la canción", "al estación" -> "a la estación".
        if low in _FEM_ARTICLE and len(nxt) > 3 and nxt.endswith(_FEM_SUFFIXES):
            tokens[i] = _reword(tok, _FEM_ARTICLE[low])
            continue

    return " ".join(tokens)


# 10. Ortografía vigente (RAE 2010): "sólo", "guión" y los demostrativos con
#     tilde ya no se acentúan. El corpus de subtítulos con el que se entrenó el
#     T5 y el léxico es_50k son anteriores a la reforma y las reponen; en
#     COWS-L2H el profesor las marca como error. Se normaliza al final del
#     pipeline salvo que el propio alumno haya escrito la tilde.
_MODERN_SPELLING = {
    "sólo": "solo", "guión": "guion", "truhán": "truhan",
    "éste": "este", "ésta": "esta", "éstos": "estos", "éstas": "estas",
    "ése": "ese", "ésa": "esa", "ésos": "esos", "ésas": "esas",
    "aquél": "aquel", "aquélla": "aquella", "aquéllos": "aquellos", "aquéllas": "aquellas",
}


# Monosílabos que nunca llevan tilde (ni antes de 2010 en la norma): "se fué"
# -> "se fue". Se quitan aunque las haya escrito el alumno.
_MONOSYLLABLE_ACCENT = {"fué": "fue", "fuí": "fui", "dió": "dio", "vió": "vio", "ví": "vi",
                        "tí": "ti", "dí": "di"}


def normalize_modern_spelling(text: str, written: set | None = None) -> str:
    """Quita las tildes abolidas en 2010 (ver _MODERN_SPELLING). `written` son
    las palabras (en minúsculas) que el alumno escribió ya con tilde: esas se
    respetan, igual que en la capa 1, salvo las de _MONOSYLLABLE_ACCENT."""
    written = written or set()
    out = []
    for tok in text.split():
        core = _clean(tok)
        if core in _MONOSYLLABLE_ACCENT:
            out.append(_reword(tok, _MONOSYLLABLE_ACCENT[core]))
        elif core in _MODERN_SPELLING and core not in written:
            out.append(_reword(tok, _MODERN_SPELLING[core]))
        else:
            out.append(tok)
    return " ".join(out)


def _reword(original_token: str, replacement: str) -> str:
    """Sustituye el núcleo de `original_token` por `replacement` preservando
    puntuación adyacente y capitalización."""
    m = re.match(r"^(\W*)(.*?)(\W*)$", original_token, re.UNICODE)
    pref, core, suff = (m.group(1), m.group(2), m.group(3)) if m else ("", original_token, "")
    return pref + _match_case(core, replacement) + suff
