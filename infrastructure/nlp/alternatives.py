"""
infrastructure/nlp/alternatives.py

Selección pura de las alternativas que se ofrecen al alumno: como mucho
`max_options` (3) y SOLO cuando la oración admite lecturas distintas. En el
caso común el resultado es una sola recomendación. Sin torch: solo
difflib/re/unicodedata y la guarda `is_safe_refinement`.

La RECOMENDADA (posición 0) es el beam 1 de T5 si es seguro (como antes de la
Task 5: `is_safe_refinement` respecto a la base y, desde el cierre, la guarda
léxica `is_lexically_plausible_refinement`, ambas en `_refine_with_model`) y
en cualquier otro caso (sin T5, T5 falla, beam 1 inseguro o implausible) el
texto base (reglas + BETO). Nunca se salta al beam 2 para recomendar. Esta
capa no la cambia: solo se le aplica (a). Una segunda lectura de BETO jamás
es la recomendada.

Las OPCIONES EXTRA (la base cuando difiere de la recomendada, los demás beams
y las segundas lecturas de BETO) se ofrecen únicamente si:
  (a) pasan `is_safe_refinement` respecto al texto base (no alucinan);
  (b) tras normalizar (espacios y puntuación final) son distintas del texto
      original y de las ya elegidas — el original nunca es una "corrección";
      la mayúscula inicial sí cuenta aquí ("El niño juega." corrige
      "el niño juega");
  (c) su resultado de edición respecto al original (qué tramos cambian y por
      qué palabras, con tildes, sin la mayúscula inicial) es distinto del de
      las ya elegidas: "juegan" y "jueguen" son dos lecturas; "Está bien." y
      "Está bien!" son la misma;
  (d) su score (log-prob de beam, o la de BETO para una segunda lectura) está
      a menos de `score_margin` del mejor candidato; la base recibe el mejor
      score y nunca cae por (d);
  (e) cada tramo que editan respecto a la BASE (lo que T5 tocó realmente) es
      una variante léxica cercana (similitud sin tildes ≥ MIN_SPAN_SIMILARITY):
      "juega/jueguen", "esta/está", "tubo/tuvo" son otras lecturas de la misma
      palabra; "luego/después" o "voy/iré" son paráfrasis, no correcciones;
  (f) SEÑAL POSITIVA DE AMBIGÜEDAD (ola final): estar cerca del mejor beam no
      prueba que la frase admita dos lecturas. Un beam de T5 (o la base) solo
      es alternativa si
        (i)  difiere de la RECOMENDADA en UNA SOLA PALABRA (un reemplazo 1:1;
             cualquier otra edición, p. ej. "mucho/muchos" además del verbo,
             lo descalifica), esa palabra es una flexión del mismo lexema
             (prefijo común sin tildes ≥ INFLECTION_MIN_PREFIX del más corto
             y similitud ≥ MIN_SPAN_SIMILARITY) y el contraste es de MODO
             (indicativo ↔ subjuntivo: juegan/jueguen, llegan/lleguen,
             gana/gane, comen/coman; `_mood_pair`) o SOLO DE TILDE
             (callo/calló, papa/papá, como/cómo, esta/está), y la alternativa
             no vuelve a la forma del original (el original nunca es una
             corrección, tampoco por tramo). Un contraste de modo exige que
             la raíz sea un verbo de `lexicon` con una sola conjugación
             decidible (jug-ar, lleg-ar, com-er) y, si `lexicon` trae
             frecuencias, que ninguna de las dos formas sea más frecuente que
             su infinitivo (casa ≫ casar, esta ≫ estar: lectura nominal o
             demostrativa, no verbo); sin léxico no hay contraste de modo.
             Corregir número o persona (juega/juegan, fue/fueron,
             lleguen/llegue) lo fija el sujeto y nunca es "otra lectura"; una
             inserción o borrado tampoco;
        (ii) o es una segunda lectura de BETO (empate < betoTieMargin en un
             homófono, `variants=`), que trae su propia señal.
  (g) FILTRO DE MODO: si la recomendada contiene un disparador de subjuntivo
      obligatorio (`SUBJUNCTIVE_TRIGGERS`: ojalá, es posible que, es
      necesario que, espero que, quiero que, me alegra que, me gusta que,
      para que, antes de que, dudo que, no creo que), la variante en
      INDICATIVO del verbo que gobierna (tras el disparador) no se ofrece: solo
      cabe el subjuntivo, como recomendada o como alternativa. La conjugación
      (-ar: subjuntivo en -e; -er/-ir: en -a) se decide buscando el infinitivo
      en `lexicon` (las frecuencias de `PhoneticEngine`); si no se puede
      decidir, el par no se ofrece (fail-closed). No cambia la recomendada.

La base va justo detrás de la recomendada (el tope no la elimina); el resto
por score descendente. En la práctica la base no aparece como alternativa:
solo volvería a la forma del original en el tramo que T5 corrigió.
"""
import difflib
import re
import unicodedata
from collections.abc import Mapping

from infrastructure.ml.guards import is_safe_refinement

# Umbrales de la capa 5 y de la señal de ambigüedad de BETO (capa 2.5). Son
# constantes de código, sin override por entorno: el mismo `modelVersion` debe
# comportarse igual en el servicio, en evaluate.py y en una ejecución manual.
# evaluate.py los registra en el bloque `pipeline` del informe junto con el
# commit (el manifiesto del modelo no los incluye: identifica pesos, no
# código). Calibración: data/ambiguity_calibration.csv +
# scripts/calibrate_alternatives.py (`--check` es la prueba de aceptación).
THRESHOLDS = {
    # (d) Margen de score (log-prob de beam normalizada por longitud) respecto
    # al mejor candidato para ofrecer una alternativa. Beams reales del T5
    # fusionado: otra lectura válida juega→juegan 0.09 · juega→jueguen 0.14 ·
    # fue→fueron 0.25; paráfrasis/ruido voy→iré 0.37 · luego→después 0.40 ·
    # ayer→anoche 0.48 · mi→mis hermanos 0.75 · ellos→él 0.91.
    "scoreMargin": 0.3,
    # (e) Similitud mínima (difflib.SequenceMatcher.ratio, sin tildes) entre
    # un tramo de la base y el tramo que lo sustituye para considerarlo "otra
    # lectura": se/sé, esta/está 1.0 · tubo/tuvo 0.75 · fue/fueron, boy/voy,
    # a/ha 0.67 · hoy/holly 0.5 · ellos/él 0.29 · luego/después 0.15 ·
    # voy/iré 0.0. Solo filtra alternativas, nunca la recomendada.
    "minSpanSimilarity": 0.6,
    # Capa 2.5: una segunda lectura más de N veces más rara (corpus es_50k)
    # que la palabra escrita no cuenta como ambigüedad (la PLL media de BETO
    # infla las palabras de varios subtokens: hoy/holly −8.16/−8.09). Ratios
    # reales: se/sé 3.4 · tuvo/tubo 9.5 (se conservan) · hoy/holly 33 ·
    # mis/miss 35 · voy/boy 120 (se descartan).
    "maxFreqRatio": 20,
    # Capa 2.5: diferencia de PLL media de BETO (log-prob por subtoken) por
    # debajo de la cual dos lecturas de un homófono se consideran empatadas y
    # la segunda se ofrece como alternativa. Es independiente de los márgenes
    # de sobrescritura (0.5 / 3.0 / 5.0): tubo/tuvo con dif 1.0 o se/sé con
    # dif 1.4 no se sobrescriben pero tampoco son empate; esta/está con dif
    # 0.13 sí.
    "betoTieMargin": 0.3,
    # Capa 4: guarda léxica del beam RECOMENDADO
    # (`guards.is_lexically_plausible_refinement`): cada reemplazo 1:1 de
    # palabra respecto a la base debe tener similitud sin tildes ≥ este valor.
    # Más bajo que (e) para no perder concordancias irregulares: bloquea
    # pasto→maíz 0.22 · pasto→carne 0.20 · verde→rojo 0.22 · voy→iré 0.0;
    # conserva es→son 0.40 · hizo→hicieron 0.50 · viene→vengan 0.55 ·
    # fue→fueron 0.67. No es una guarda general contra sustituciones léxicas:
    # luego→después 0.33, tuvo→provocó 0.36 y ayer→anoche 0.40 pasan.
    "recommendedMinSimilarity": 0.3,
    # (f)(i): dos palabras son flexiones del mismo lexema si su prefijo común
    # (sin tildes) cubre al menos esta fracción de la más corta (y la
    # similitud es ≥ minSpanSimilarity): llegan/lleguen 0.67 · gana/gane 0.75
    # · están/estén 0.6 · juegan/jueguen 0.67; voy/iré 0.0 · luego/después 0.0.
    "inflectionMinPrefix": 0.6,
}

SCORE_MARGIN               = THRESHOLDS["scoreMargin"]
MIN_SPAN_SIMILARITY        = THRESHOLDS["minSpanSimilarity"]
RECOMMENDED_MIN_SIMILARITY = THRESHOLDS["recommendedMinSimilarity"]
INFLECTION_MIN_PREFIX      = THRESHOLDS["inflectionMinPrefix"]

# (g) Disparadores de subjuntivo obligatorio (palabras sin tildes ni
# mayúsculas, en orden): tras ellos la subordinada solo admite subjuntivo.
SUBJUNCTIVE_TRIGGERS = (
    ("ojala",),
    ("es", "posible", "que"),
    ("es", "necesario", "que"),
    ("espero", "que"),
    ("quiero", "que"),
    ("me", "alegra", "que"),
    ("me", "gusta", "que"),
    ("para", "que"),
    ("antes", "de", "que"),
    ("dudo", "que"),
    ("no", "creo", "que"),
)

# Terminaciones del presente cuyo contraste de vocal temática distingue
# indicativo y subjuntivo (misma persona): -ar canta/cante, cantan/canten;
# -er/-ir come/coma, comen/coman.
_MOOD_ENDINGS = (("a", "e"), ("an", "en"), ("as", "es"), ("amos", "emos"), ("ais", "eis"))
_MIN_STEM = 2

_TRAILING_PUNCT_RE = re.compile(r"[\s.!?…]+$")
_SPACES_RE         = re.compile(r"\s+")
_OUTER_PUNCT_RE    = re.compile(r"^[^\w]+|[^\w]+$")


def _normalize(text: str) -> str:
    """Colapsa espacios y quita la puntuación final (`.`, `!`, `?`, `…`)."""
    return _TRAILING_PUNCT_RE.sub("", _SPACES_RE.sub(" ", text.strip()))


def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def _words_for_edits(text: str) -> list[str]:
    """
    Palabras sobre las que se calculan las ediciones. Se compara CON tildes
    (`esta/está` es precisamente una edición) pero sin la mayúscula inicial
    de la oración: poner mayúscula no es otra interpretación.
    """
    words = _normalize(text).split()
    if words:
        words[0] = words[0][:1].lower() + words[0][1:]
    return words


def _norm_word(word: str) -> str:
    """Palabra resultante de una edición: NFC, casefold, sin puntuación exterior; CON tildes."""
    return _OUTER_PUNCT_RE.sub("", unicodedata.normalize("NFC", word)).casefold()


def _similarity(a: str, b: str) -> float:
    a, b = _strip_accents(a).lower(), _strip_accents(b).lower()
    return difflib.SequenceMatcher(a=a, b=b, autojunk=False).ratio()


def _edit_signature(original_words: list[str], cand_words: list[str]) -> frozenset:
    """
    Resultado de edición del candidato respecto al original:
    {(i1, i2, resultado)} sobre los opcodes no-`equal` de SequenceMatcher,
    donde [i1, i2) es el tramo del original y `resultado` las palabras
    normalizadas (con tildes) que lo sustituyen; inserción ante la posición i
    = (i, i, "+palabras"); borrado = (i1, i2, "-"). Solo decide si dos
    candidatos son la misma lectura; (a)/(d)/(e) siguen frenando paráfrasis.
    """
    matcher = difflib.SequenceMatcher(a=original_words, b=cand_words, autojunk=False)
    signature = set()
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        result = " ".join(_norm_word(w) for w in cand_words[j1:j2])
        if i2 == i1:            # inserción
            signature.add((i1, i1, "+" + result))
        elif j2 == j1:          # borrado
            signature.add((i1, i2, "-"))
        else:                   # reemplazo
            signature.add((i1, i2, result))
    return frozenset(signature)


def _is_duplicate_deletion(words: list[str], i1: int, i2: int) -> bool:
    """Borrado de tokens iguales (sin tildes ni mayúsculas) a un vecino inmediato: "muy muy" → "muy"."""
    neighbours = set()
    if i1 > 0:
        neighbours.add(_strip_accents(words[i1 - 1]).lower())
    if i2 < len(words):
        neighbours.add(_strip_accents(words[i2]).lower())
    return all(_strip_accents(w).lower() in neighbours for w in words[i1:i2])


def _span_similarity(base_words: list[str], cand_words: list[str]) -> float:
    """
    Similitud mínima entre cada tramo de la base que el candidato edita y lo
    que pone en su lugar (1.0 si no edita nada). Un tramo con el mismo número
    de palabras se compara palabra a palabra (dos edits contiguos no se
    diluyen entre sí); si cambia el número de palabras se comparan los tramos
    completos ("a el" → "al"). Borrar un token duplicado vale 1.0; cualquier
    otro borrado o inserción se compara contra la cadena vacía (0.0).
    """
    matcher    = difflib.SequenceMatcher(a=base_words, b=cand_words, autojunk=False)
    similarity = 1.0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if j2 == j1 and _is_duplicate_deletion(base_words, i1, i2):
            continue
        if i2 - i1 == j2 - j1:
            pairs = zip(base_words[i1:i2], cand_words[j1:j2])
        else:
            pairs = [(" ".join(base_words[i1:i2]), " ".join(cand_words[j1:j2]))]
        for word_a, word_b in pairs:
            similarity = min(similarity, _similarity(word_a, word_b))
    return similarity


# ---------------------------------------------------------------------------
# (f) señal positiva de ambigüedad y (g) filtro de modo
# ---------------------------------------------------------------------------

def _norm_words(text: str) -> list[str]:
    """Palabras normalizadas (`_norm_word`: con tildes, casefold, sin puntuación exterior)."""
    return [_norm_word(w) for w in _words_for_edits(text)]


def _plain(word: str) -> str:
    return _strip_accents(word).lower()


def _is_inflection(a: str, b: str) -> bool:
    """Flexión del mismo lexema según el brief: prefijo común sin tildes ≥ INFLECTION_MIN_PREFIX y similitud ≥ MIN_SPAN_SIMILARITY."""
    pa, pb = _plain(a), _plain(b)
    shorter = min(len(pa), len(pb))
    if shorter == 0:
        return False
    prefix = 0
    for ca, cb in zip(pa, pb):
        if ca != cb:
            break
        prefix += 1
    return prefix >= INFLECTION_MIN_PREFIX * shorter and _similarity(a, b) >= MIN_SPAN_SIMILARITY


def _canonical_stem(stem: str, vowel: str) -> str:
    """
    Raíz comparable entre la forma ante `a` y la forma ante `e`: ante e la
    ortografía escribe gu/qu/g/c donde ante a escribe g/c/j/z (llega/llegue,
    busca/busque, coge/coja, cruza/cruce).
    """
    if vowel != "e":
        return stem
    if stem.endswith("gu"):
        return stem[:-2] + "g"
    if stem.endswith("qu"):
        return stem[:-2] + "c"
    if stem.endswith("g"):
        return stem[:-1] + "j"
    if stem.endswith("c"):
        return stem[:-1] + "z"
    return stem


def _orthographic_mood_split(a: str, b: str):
    """
    `(vocal_a, vocal_b, raíz_ante_a, raíz_ante_e)` si `a` y `b` (sin tildes)
    son la misma persona del presente con vocal temática distinta (a ↔ e)
    sobre la misma raíz; None si no (número/persona distintos, raíz distinta o
    demasiado corta, solo tildes). Puramente ortográfico: casa/case también
    lo cumple; `_mood_split` añade la comprobación con el léxico.
    """
    pa, pb = _plain(a), _plain(b)
    if pa == pb:
        return None
    for ending_a, ending_e in _MOOD_ENDINGS:
        for x, y, ex, ey in ((pa, pb, ending_a, ending_e), (pa, pb, ending_e, ending_a)):
            if x.endswith(ex) and y.endswith(ey):
                stem_x, stem_y = x[:-len(ex)], y[:-len(ey)]
                if len(stem_x) >= _MIN_STEM and len(stem_y) >= _MIN_STEM \
                        and _canonical_stem(stem_x, ex[0]) == _canonical_stem(stem_y, ey[0]):
                    stems = {ex[0]: stem_x, ey[0]: stem_y}
                    return ex[0], ey[0], stems["a"], stems["e"]
    return None


def _is_noun_reading(form: str, infinitive: str, lexicon) -> bool:
    """
    Con un léxico de frecuencias (Mapping), una forma más frecuente que su
    propio infinitivo se toma como lectura nominal/demostrativa (casa 496977
    ≫ casar 10364; esta 897814 ≫ estar 338633; cosa ≫ coser) y no da
    contraste de modo. llega 29024 < llegar 88969, juegan 3507 < jugar 52990,
    estan 18914 < estar sí son verbos. Sin frecuencias (un set) no se aplica.
    """
    if not isinstance(lexicon, Mapping):
        return False
    return lexicon.get(form, 0) > lexicon.get(infinitive, 0)


def _mood_split(a: str, b: str, lexicon):
    """
    `(vocal_a, vocal_b, raíz_ante_a, raíz_ante_e)` si `a` y `b` son dos formas
    de modo (indicativo ↔ subjuntivo) del MISMO verbo: cumplen el contraste
    ortográfico (`_orthographic_mood_split`), la raíz tiene un solo infinitivo
    decidible en `lexicon` (`_infinitive`) y ninguna de las dos formas es una
    lectura nominal más frecuente que ese infinitivo (`_is_noun_reading`).
    None en cualquier otro caso, incluido `lexicon` vacío (fail-closed).
    """
    split = _orthographic_mood_split(a, b)
    if split is None:
        return None
    _, _, stem_a, stem_e = split
    infinitive = _infinitive(stem_a, stem_e, lexicon)
    if infinitive is None:
        return None
    if _is_noun_reading(_plain(a), infinitive, lexicon) or _is_noun_reading(_plain(b), infinitive, lexicon):
        return None
    return split


def _mood_pair(a: str, b: str, lexicon):
    """
    `(vocal_a, vocal_b)` si `a` y `b` son un contraste de modo del mismo
    verbo según `lexicon`: llegan/lleguen → ("a", "e"); jueguen/juegan →
    ("e", "a"); None si no (casa/case, número, raíz desconocida, sin léxico).
    """
    split = _mood_split(a, b, lexicon)
    return None if split is None else (split[0], split[1])


def _stem_variants(stem: str) -> list[str]:
    """Raíz tal cual y con el diptongo/cierre vocálico deshecho (jueg→jug, vien→ven, sigu→segu)."""
    variants = [stem]
    for old, new in (("ie", "e"), ("ue", "o"), ("ue", "u"), ("i", "e")):
        index = stem.rfind(old)
        if index >= 0:
            variants.append(stem[:index] + new + stem[index + len(old):])
    return variants


def _infinitives(stem_a: str, stem_e: str, lexicon) -> tuple[list[str], list[str]]:
    """Infinitivos en -ar (raíz ante a) y en -er/-ir (raíz ante e) presentes en `lexicon`."""
    if not lexicon:
        return [], []
    ar = [v + "ar" for v in _stem_variants(stem_a) if v + "ar" in lexicon]
    er = [v + suffix for v in _stem_variants(stem_e) for suffix in ("er", "ir") if v + suffix in lexicon]
    return ar, er


def _infinitive(stem_a: str, stem_e: str, lexicon):
    """El único infinitivo decidible (-ar o -er/-ir) de la raíz, o None si ninguno o ambos."""
    ar, er = _infinitives(stem_a, stem_e, lexicon)
    if ar and not er:
        return ar[0]
    if er and not ar:
        return er[0]
    return None


def _verb_class(stem_a: str, stem_e: str, lexicon):
    """
    "ar" si la raíz (tal como se escribe ante a) + "ar" está en `lexicon`,
    "er" si la raíz (ante e) + "er"/"ir" lo está; None si ninguna o ambas
    (no se puede saber cuál de las dos formas es el indicativo).
    """
    infinitive = _infinitive(stem_a, stem_e, lexicon)
    if infinitive is None:
        return None
    return "ar" if infinitive.endswith("ar") else "er"


def _candidate_is_indicative(rec_word: str, cand_word: str, lexicon):
    """
    Para un par de modo (recomendada, candidata): True si la candidata es el
    indicativo, False si es el subjuntivo, None si `lexicon` no decide la
    conjugación (-ar: indicativo en a; -er/-ir: indicativo en e).
    """
    split = _mood_split(rec_word, cand_word, lexicon)
    if split is None:
        return None
    _, cand_vowel, stem_a, stem_e = split
    verb_class = _verb_class(stem_a, stem_e, lexicon)
    if verb_class is None:
        return None
    return cand_vowel == ("a" if verb_class == "ar" else "e")


def _aligned_original(original_words: list[str], rec_words: list[str]) -> dict:
    """{índice en la recomendada: palabra del original alineada} para tramos iguales o reemplazos 1:1."""
    aligned = {}
    matcher = difflib.SequenceMatcher(a=original_words, b=rec_words, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal" or (tag == "replace" and i2 - i1 == j2 - j1):
            for offset in range(j2 - j1):
                aligned[j1 + offset] = original_words[i1 + offset]
    return aligned


def _reading_contrasts(rec_words: list[str], cand_words: list[str]):
    """
    Reemplazos 1:1 `(índice, palabra_recomendada, palabra_candidata)` entre la
    recomendada y el candidato, o None si difieren por algo que no sea un
    reemplazo 1:1 (inserción, borrado, 1:n).
    """
    matcher = difflib.SequenceMatcher(a=rec_words, b=cand_words, autojunk=False)
    contrasts = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag != "replace" or i2 - i1 != j2 - j1:
            return None
        contrasts.extend((i1 + k, rec_words[i1 + k], cand_words[j1 + k]) for k in range(i2 - i1))
    return contrasts


def _has_reading_signal(contrasts, aligned_original: dict, lexicon=None) -> bool:
    """
    (f)(i): el candidato difiere de la recomendada en UNA sola palabra (un
    único reemplazo 1:1: cualquier edición extra lo descalifica), esa palabra
    es una flexión del mismo lexema, el contraste es de modo (según `lexicon`)
    o solo de tilde, y la forma candidata es distinta de la del original en
    esa posición (un tramo alineado desconocido no da señal).
    """
    if not contrasts or len(contrasts) != 1:
        return False
    index, rec_word, cand_word = contrasts[0]
    if not _is_inflection(rec_word, cand_word):
        return False
    original_word = aligned_original.get(index)
    if original_word is None or cand_word == original_word:
        return False
    return _plain(rec_word) == _plain(cand_word) or _mood_pair(rec_word, cand_word, lexicon) is not None


def _trigger_end(rec_words: list[str]):
    """Índice de la primera palabra tras un disparador de subjuntivo obligatorio, o None."""
    plain = [_plain(w) for w in rec_words]
    for trigger in SUBJUNCTIVE_TRIGGERS:
        n = len(trigger)
        for start in range(len(plain) - n + 1):
            if tuple(plain[start:start + n]) == trigger:
                return start + n
    return None


def _indicative_after_trigger(contrasts, trigger_end, lexicon) -> bool:
    """
    (g): True si algún contraste de modo tras el disparador ofrece el
    indicativo (o no se puede decidir cuál es el indicativo: fail-closed).
    """
    if trigger_end is None or not contrasts:
        return False
    for index, rec_word, cand_word in contrasts:
        if index < trigger_end or _mood_pair(rec_word, cand_word, lexicon) is None:
            continue
        indicative = _candidate_is_indicative(rec_word, cand_word, lexicon)
        if indicative is None or indicative:
            return True
    return False


def select_alternatives(
    original_text: str,
    base_text: str,
    candidates: list[tuple[str, float]],
    max_options: int = 3,
    score_margin: float = 1.0,
    recommended: str | None = None,
    variants: list[tuple[str, float]] | None = None,
    lexicon=None,
) -> list[str]:
    """
    Devuelve la lista final de sugerencias (recomendada primero, nunca vacía)
    a partir de `candidates` = [(texto, score)] (los beams de T5 en orden de
    generación) y `variants` = [(texto, score)] (las segundas lecturas de
    BETO, ya ancladas a la escala de los beams).

    `recommended` es el texto que ocupa la posición 0. Si es None se deduce
    con la misma regla que el pipeline: `candidates[0]` (el beam 1) si es
    seguro respecto a la base (`is_safe_refinement`) y, si no lo es o no hay
    beams, la base; nunca el beam 2. Las `variants` no participan en esa
    deducción: una segunda lectura de BETO jamás ocupa la posición 0 (solo
    puede ir detrás, como cualquier extra). `CorrectionPipeline.correct` pasa
    `recommended` explícito (beam 1 seguro y léxicamente plausible, o base).
    Si la recomendada explícita no es segura, se recomienda la base.

    `lexicon` (cualquier contenedor con `in`, p. ej. `PhoneticEngine.word_freqs`;
    si es un Mapping palabra → frecuencia también descarta lecturas nominales)
    decide qué pares son contrastes de modo, en la señal (f) y en el filtro
    (g); sin él no hay contraste de modo (solo señal de tilde o de BETO).
    """
    base   = base_text.strip()
    scored = [(str(t).strip(), float(s)) for t, s in candidates if t and str(t).strip()]
    variant_texts: set[str] = set()          # segundas lecturas de BETO: señal (ii) propia
    for text, score in (variants or []):
        text = str(text or "").strip()
        if text:
            variant_texts.add(text)
            scored.append((text, float(score)))
    # Una variante idéntica a la base (el empate lo absorbió una capa
    # posterior, p. ej. "mi/mí" y luego la regla de gustar) no es una segunda
    # lectura: se trata como la base y necesita la señal (i).
    variant_texts.discard(base)

    if recommended is None:
        first = str(candidates[0][0] or "").strip() if candidates else ""
        recommended = first if first and is_safe_refinement(base, first) else base
    else:
        recommended = str(recommended).strip()
        if not recommended or not is_safe_refinement(base, recommended):
            recommended = base

    if max_options <= 0:
        return []

    best_score = max((s for _, s in scored), default=0.0)

    # Opciones extra: la base primero (con el mejor score: nunca cae por (d)
    # y el tope no la elimina), después el resto por score descendente
    # (orden estable: a igual score, el orden de generación).
    extras = [(base, best_score)] + sorted(scored, key=lambda ts: -ts[1])

    original_norm  = _normalize(original_text)
    original_words = _words_for_edits(original_text)
    base_words     = _words_for_edits(base)
    rec_norm_words = _norm_words(recommended)
    aligned_orig   = _aligned_original(_norm_words(original_text), rec_norm_words)
    trigger_end    = _trigger_end(rec_norm_words)

    chosen: list[str]                 = [recommended]
    chosen_norms: set[str]            = {_normalize(recommended)}
    chosen_signatures: list[frozenset] = [_edit_signature(original_words, _words_for_edits(recommended))]

    for text, score in extras:
        if len(chosen) >= max_options:
            break
        norm = _normalize(text)
        if not norm or norm == original_norm or norm in chosen_norms:     # (b)
            continue
        if text != base:
            if score < best_score - score_margin:                          # (d)
                continue
            if not is_safe_refinement(base, text):                         # (a)
                continue
        cand_words = _words_for_edits(text)
        signature  = _edit_signature(original_words, cand_words)
        if signature in chosen_signatures:                                 # (c)
            continue
        if _span_similarity(base_words, cand_words) < MIN_SPAN_SIMILARITY: # (e)
            continue
        contrasts = _reading_contrasts(rec_norm_words, _norm_words(text))
        if text not in variant_texts and not _has_reading_signal(contrasts, aligned_orig, lexicon):   # (f)(i)
            continue
        if _indicative_after_trigger(contrasts, trigger_end, lexicon):     # (g)
            continue

        chosen.append(text)
        chosen_norms.add(norm)
        chosen_signatures.append(signature)

    return chosen
