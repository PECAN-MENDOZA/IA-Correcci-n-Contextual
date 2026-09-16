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
      "nunca" → "jamás" se bloquean aunque sean "cercanos".
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
                if difflib.SequenceMatcher(None, word_a, word_b, autojunk=False).ratio() < min_similarity:
                    return False
            continue
        if not _is_allowed_block(base_words, i1, i2, cand_words, j1, j2):
            return False
    return True
