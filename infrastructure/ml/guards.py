"""
infrastructure/ml/guards.py

Guardas anti-alucinación para cualquier generación de texto basada en modelo
(T5 base o adaptador LoRA). Deliberadamente SIN dependencias de
torch/transformers/peft: son funciones puras sobre strings, para que se
puedan testear sin necesitar el modelo real ni GPU.

Usado por:
  - infrastructure/nlp/correction_pipeline.py  (refinamiento T5 global)
  - infrastructure/ml/t5_model.py              (generate_with_lora)
"""
import difflib
import re
import unicodedata

_OUTER_PUNCT_RE = re.compile(r"^[^\w]+|[^\w]+$")


def _lexical_key(word: str) -> str:
    """Palabra sin tildes, en minúsculas y sin puntuación exterior ("Está," → "esta")."""
    stripped = "".join(
        c for c in unicodedata.normalize("NFD", word)
        if unicodedata.category(c) != "Mn"
    )
    return _OUTER_PUNCT_RE.sub("", stripped).casefold()


def is_safe_refinement(base_text: str, candidate: str) -> bool:
    """
    Un candidato generado por el modelo (T5 base o LoRA) solo se acepta si es
    una variación CONSERVADORA de `base_text`:
      - no vacío
      - longitud entre 0.6x y 1.5x del original (en palabras)
      - no cambia más de max(3, len//3) palabras respecto al original

    Si no cumple esto, se asume que el modelo alucinó/inventó y el llamador
    debe descartar `candidate` y quedarse con `base_text`.
    """
    if not candidate:
        return False

    base_len, cand_len = len(base_text.split()), len(candidate.split())
    if base_len == 0 or not (0.6 <= cand_len / base_len <= 1.5):
        return False

    changed = (
        sum(1 for a, b in zip(base_text.split(), candidate.split()) if a != b)
        + abs(cand_len - base_len)
    )
    if changed > max(3, base_len // 3):
        return False

    return True


# Palabras que se pueden insertar o borrar sin cambiar el sentido: artículos,
# preposiciones, conjunciones, clíticos y las contracciones al/del (claves
# léxicas: sin tildes ni mayúsculas).
ALLOWED_FUNCTION_WORDS = frozenset(
    "el la los las un una unos unas a de en con por para y e o u que se me te le lo al del".split()
)

# Negadores y cuantificadores: insertarlos, borrarlos o sustituirlos invierte
# o cambia el alcance del enunciado ("no quiero ir" → "quiero ir", "lo quiero
# ir" → "no quiero ir"); prohibidos siempre, incluso dentro de un bloque que
# por lo demás sería permitido y aunque la sustitución 1:1 sea "cercana"
# (lo/no, yo/no, ni/mi tienen similitud 0.5).
FORBIDDEN_INDEL_WORDS = frozenset(
    "no nunca jamas nadie nada ni ningun ninguna ninguno tampoco todos todas siempre".split()
)

# Únicas equivalencias admitidas dentro del conjunto protegido: flexiones de
# número/género de la MISMA palabra (todos/todas, ningún/ninguna/ninguno).
# nunca/jamás, nada/nadie o no/ni son palabras distintas y no se intercambian.
_PROTECTED_FAMILY = {
    "todos": "todos", "todas": "todos",
    "ningun": "ningun", "ninguna": "ningun", "ninguno": "ningun",
}


def _protected_family(word: str):
    """Familia de una palabra protegida (clave léxica) o None si no está protegida."""
    if word not in FORBIDDEN_INDEL_WORDS:
        return None
    return _PROTECTED_FAMILY.get(word, word)


# Marca de 2.ª persona del singular en pretérito (-aste/-iste): "viniste",
# "comiste", "jugaste". El T5 a veces la cambia de persona ("tú viniste
# conmigo" -> "tú vino conmigo"): la similitud la deja pasar (viniste/vino
# 0.55) pero cambia de quién habla la frase. Solo se bloquea PERDER la marca;
# ganarla ("tú comio mucho" -> "comiste") es una corrección de concordancia
# legítima.
# Excepciones: palabras que acaban igual sin ser pretérito de 2.ª persona
# (3.ª de presente en -siste, sustantivos y adjetivos, subjuntivos de -ar).
_SECOND_PERSON_EXCEPTIONS = frozenset(
    "existe insiste asiste consiste resiste persiste desiste subsiste embiste "
    "reviste triste chiste baste gaste coste poste oeste viste contraste "
    "desgaste".split()
)


def _drops_second_person(word_a: str, word_b: str) -> bool:
    """True si el reemplazo borra la marca de 2.ª persona del pretérito."""
    return (
        len(word_a) >= 5
        and word_a.endswith(("aste", "iste"))
        and word_a not in _SECOND_PERSON_EXCEPTIONS
        and not word_b.endswith("ste")
    )


def _protected_swap_is_allowed(word_a: str, word_b: str) -> bool:
    """
    Un reemplazo 1:1 que toca una palabra protegida solo se acepta si ambos
    lados son la misma palabra protegida (o su flexión de número/género):
    todos→todas sí; lo→no, no→yo, ni→mi, nunca→jamás no.
    """
    family_a, family_b = _protected_family(word_a), _protected_family(word_b)
    if family_a is None and family_b is None:
        return True
    return family_a is not None and family_a == family_b


def _is_adjacent_duplicate_deletion(base_words: list, i1: int, i2: int) -> bool:
    """Borrado de tokens iguales a un vecino inmediato ("muy muy" → "muy", "a a" → "a")."""
    neighbours = set()
    if i1 > 0:
        neighbours.add(base_words[i1 - 1])
    if i2 < len(base_words):
        neighbours.add(base_words[i2])
    return all(w in neighbours for w in base_words[i1:i2])


def _is_allowed_block(base_words: list, i1: int, i2: int, cand_words: list, j1: int, j2: int) -> bool:
    """
    Un bloque que no es un reemplazo n:n (inserción, borrado, 1:n, n:1, n:m)
    solo se acepta en la lista blanca; cualquier otra cosa se bloquea
    (fail-closed). Las claves vacías son puntuación suelta y no cuentan.
    """
    removed = [w for w in base_words[i1:i2] if w]
    added = [w for w in cand_words[j1:j2] if w]
    # Un negador/cuantificador nunca se inserta ni se borra, ni dentro de un
    # bloque permitido (se comparan por familia: todos/todas cuentan igual).
    if [_protected_family(w) for w in removed if w in FORBIDDEN_INDEL_WORDS] != \
            [_protected_family(w) for w in added if w in FORBIDDEN_INDEL_WORDS]:
        return False
    if not removed and not added:                      # solo puntuación
        return True
    if removed and added and "".join(removed) == "".join(added):   # ala → a la, por que → porque
        return True
    if not added and _is_adjacent_duplicate_deletion(base_words, i1, i2):
        return True
    return all(w in ALLOWED_FUNCTION_WORDS for w in removed + added)


def is_lexically_plausible_refinement(base_text: str, candidate: str, min_similarity: float = 0.3) -> bool:
    """
    Guarda léxica del beam RECOMENDADO: `is_safe_refinement` mide cuánto
    cambia el candidato (cantidad), esta mide qué cambia (calidad). Se compara
    palabra a palabra con `base_text` (opcodes de `SequenceMatcher` sobre las
    palabras sin tildes, mayúsculas ni puntuación exterior):

    - Reemplazo n:n: cada par de palabras debe ser una variante léxica cercana
      (`difflib.SequenceMatcher(None, a, b).ratio()` ≥ `min_similarity`). Con
      0.3 se bloquean las sustituciones léxicas del T5 (pasto→maíz 0.22,
      pasto→carne 0.20, verde→rojo 0.22, voy→iré 0.0) y se conservan las
      flexiones, incluso irregulares (es→son 0.40, hizo→hicieron 0.50,
      viene→vengan 0.55, fue→fueron 0.67; esta→está 1.0). Si alguno de los
      dos lados es un negador o cuantificador (`FORBIDDEN_INDEL_WORDS`), la
      similitud no basta: ambos deben ser la misma palabra protegida o su
      flexión de número/género (todos→todas, ningún→ninguna); "lo quiero ir"
      → "no quiero ir", "no quiero ir" → "yo quiero ir", "ni" → "mi" o
      "nunca" → "jamás" se bloquean aunque sean "cercanos". Tampoco se puede
      perder la marca de 2.ª persona del pretérito (viniste→vino, jugaste→jugó:
      cambian de quién habla la frase); ganarla sí (comio→comiste).
    - Inserción, borrado y bloques 1:n / n:1 / n:m: cerrados salvo lista
      blanca explícita: puntuación suelta; artículos, preposiciones,
      conjunciones, clíticos y al/del (`ALLOWED_FUNCTION_WORDS`: "llego tarde
      a clase" → "llegué tarde a la clase"); borrado de un duplicado adyacente
      ("muy muy bien" → "muy bien"); unión o división con las mismas letras
      ("ala" → "a la", "por que" → "porque"). Insertar o borrar un negador o
      cuantificador (`FORBIDDEN_INDEL_WORDS`: "no quiero ir" → "quiero ir",
      "quiero ir" → "no quiero ir") se bloquea siempre; insertar o borrar una
      palabra de contenido ("fui parque" → "fui al gran parque", "voy a ir"
      → "iré") también.
    """
    base_words = [_lexical_key(w) for w in base_text.split()]
    cand_words = [_lexical_key(w) for w in candidate.split()]
    matcher = difflib.SequenceMatcher(a=base_words, b=cand_words, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace" and i2 - i1 == j2 - j1:
            for word_a, word_b in zip(base_words[i1:i2], cand_words[j1:j2]):
                if not _protected_swap_is_allowed(word_a, word_b):
                    return False
                if _drops_second_person(word_a, word_b):
                    return False
                if difflib.SequenceMatcher(None, word_a, word_b, autojunk=False).ratio() < min_similarity:
                    return False
            continue
        if not _is_allowed_block(base_words, i1, i2, cand_words, j1, j2):
            return False
    return True


# ───────────── reversión de sustituciones léxicas (2026-09-30) ─────────────
# `is_lexically_plausible_refinement` acepta un reemplazo con similitud ≥ 0,3,
# y el T5 cambia así palabras CORRECTAS por otras (sinónimos, paráfrasis):
# alcancía→caja, aula→clase, madre→padre, tele→televisión, sanguche→viaje.
# En vez de descartar el beam entero (y perder sus correcciones buenas de la
# misma frase), se revierte solo la palabra mal cambiada. Criterio medido sobre
# los 183 reemplazos del beam 1 en los sets de desarrollo (.local/t5_pairs.py):
# los buenos son flexiones de la misma palabra (fue→fueron, viene→venga,
# feliz→felices), homófonos (ves→vez, a→ha) o irregulares de ser/ir/haber.

# Formas irregulares que el T5 intercambia legítimamente (claves léxicas),
# agrupadas por lema Y TIEMPO: cambiar persona, número o modo dentro del mismo
# tiempo es concordancia (es→son, eres→seas, tenía→tuviera); cambiar de tiempo
# (es→fue) cambia lo que el niño dijo y se revierte (auditoría B, 30-sep).
_IRREGULAR_FAMILIES = [
    set("es son soy eres somos sea seas sean seamos".split()),                       # ser presente
    set("era eras eran eramos fuera fueras fueran".split()),                         # ser imperfecto
    set("fue fueron fui fuiste fuimos".split()),                                     # ser/ir pretérito
    set("ser sido siendo".split()),
    set("va van vas voy vamos vaya vayas vayan vayamos".split()),                    # ir presente
    set("iba ibas iban ibamos fuera fueras fueran".split()),                         # ir imperfecto
    set("ha han has he hemos hay haya hayas hayan".split()),                         # haber presente
    set("habia habian habias hubiera hubieran hubieras".split()),                    # haber imperfecto
    set("sabe saben sabes se sepa sepas sepan".split()),                             # saber presente
    set("sabia sabian supiera supieran".split()),                                    # saber imperfecto
    set("tiene tienen tienes tengo tenga tengas tengan".split()),                    # tener presente
    set("tenia tenian tenias tuviera tuvieran tuvieras".split()),                    # tener imperfecto
    # Determinantes, posesivos y demostrativos: solo género y número.
    set("el la los las lo".split()), set("un una unos unas".split()),
    set("mi mis".split()), set("tu tus".split()), set("su sus".split()),
    set("este esta estos estas".split()), set("ese esa esos esas".split()),
]
# Los pronombres átonos (se->le) NO van como familia: medido el 2026-09-30, no
# ganan ningún acierto y añaden un falso positivo en COWS-L2H.
# Parecido mínimo (SequenceMatcher) de un reemplazo de palabra válida con el
# mismo comienzo (SAME_WORD_PREFIX letras: la flexión cambia el final, no el
# principio; casa→cosa, caza→cosa no), y de uno de palabra fuera del
# diccionario (una falta que el T5 arregla: dicieron→dijeron 0,80;
# sanguche→viaje 0,31 no).
SAME_WORD_MIN_SIMILARITY = 0.5
SAME_WORD_PREFIX = 2
MAX_INFLECTION_TAIL = 4
# Pronombres enclíticos (y sus combinaciones): quitarlos cambia lo que el niño dijo.
_ENCLITICS = {c + d for c in ("", "me", "te", "se", "nos") for d in
              ("", "lo", "la", "los", "las", "le", "les")} - {""} | {"me", "te", "se", "nos", "os"}
NEAR_IDENTICAL_SIMILARITY = 0.85     # special→especial 0,93; madre→padre 0,80 no
UNKNOWN_WORD_MIN_SIMILARITY = 0.6
# tele→televisión: misma raíz + >= 4 letras nuevas es otra palabra, no una
# flexión (fue→fueron +3, llegó→llegaron +3).
MAX_INFLECTION_GROWTH = 3


def drops_enclitic(word_a: str, word_b: str) -> bool:
    """True si `word_b` es `word_a` sin su pronombre enclítico (leyendolo -> leyendo)."""
    return word_a.startswith(word_b) and word_a[len(word_b):] in _ENCLITICS


def is_same_word_variant(word_a: str, word_b: str, sound=None) -> bool:
    """True si `word_b` (claves léxicas) es una variante de la MISMA palabra
    `word_a`: flexión, homófono o irregular de ser/ir/haber."""
    if word_a == word_b:
        return True
    if word_b.startswith(word_a) and len(word_b) - len(word_a) > MAX_INFLECTION_GROWTH:
        return False
    if drops_enclitic(word_a, word_b):
        return False                                     # leyendolo -> leyendo: pierde "lo"
    if sound is not None and sound(word_a) == sound(word_b):
        return True
    if any(word_a in fam and word_b in fam for fam in _IRREGULAR_FAMILIES):
        return True
    ratio = difflib.SequenceMatcher(None, word_a, word_b, autojunk=False).ratio()
    if ratio >= NEAR_IDENTICAL_SIMILARITY:
        return True
    # Raíz con diptongo reducido (vienes/vengas, puede/podamos); en palabras
    # de <= 3 letras basta la inicial (das/des, mi/mis).
    stem_a = word_a.replace("ie", "e").replace("ue", "o")
    stem_b = word_b.replace("ie", "e").replace("ue", "o")
    # La flexión cambia el final: el prefijo común cubre todo salvo, como mucho,
    # MAX_INFLECTION_TAIL letras de la más corta (haces/hagas, llegamos/lleguemos sí;
    # sanguche/sandwich, "san" de 8, no).
    shortest = min(len(stem_a), len(stem_b))
    prefix = 1 if min(len(word_a), len(word_b)) <= 3 else max(SAME_WORD_PREFIX, shortest - MAX_INFLECTION_TAIL)
    return stem_a[:prefix] == stem_b[:prefix] and ratio >= SAME_WORD_MIN_SIMILARITY


def revert_lexical_substitutions(base_text: str, candidate: str, is_known_word, sound=None) -> str:
    """Devuelve `candidate` con cada reemplazo 1:1 de palabra que NO es una
    variante de la misma palabra devuelto a la palabra de `base_text` (con su
    puntuación y mayúsculas). `is_known_word(clave)` dice si la palabra base
    está en el diccionario: si lo está, se exige `is_same_word_variant`; si no
    (falta de ortografía), basta eso o un parecido ≥ UNKNOWN_WORD_MIN_SIMILARITY.
    Los demás bloques (inserciones, 1:n) no se tocan: los juzga
    `is_lexically_plausible_refinement`."""
    base_tokens, cand_tokens = base_text.split(), candidate.split()
    base_words = [_lexical_key(w) for w in base_tokens]
    cand_words = [_lexical_key(w) for w in cand_tokens]
    out = list(cand_tokens)
    matcher = difflib.SequenceMatcher(a=base_words, b=cand_words, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "replace" or i2 - i1 != j2 - j1:
            continue
        for k in range(i2 - i1):
            word_a, word_b = base_words[i1 + k], cand_words[j1 + k]
            if not word_a or not word_b:
                continue
            keep = is_same_word_variant(word_a, word_b, sound)
            if not keep and not is_known_word(word_a) and not drops_enclitic(word_a, word_b):
                keep = difflib.SequenceMatcher(None, word_a, word_b, autojunk=False).ratio() \
                    >= UNKNOWN_WORD_MIN_SIMILARITY
            if not keep:
                out[j1 + k] = base_tokens[i1 + k]
    return " ".join(out)
