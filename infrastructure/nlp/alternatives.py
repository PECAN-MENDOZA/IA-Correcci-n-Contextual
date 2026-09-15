"""
infrastructure/nlp/alternatives.py

Selección pura de las alternativas que se ofrecen al alumno: como mucho
`max_options` (3) y SOLO cuando la oración admite lecturas distintas. En el
caso común el resultado es una sola recomendación (o refinado + base, como
siempre). Sin torch: solo difflib/re/unicodedata y la guarda
`is_safe_refinement`.

La RECOMENDADA (posición 0) se decide exactamente como antes de la Task 5:
el primer beam seguro de T5 (`is_safe_refinement` respecto a la base) o, si
no hay ninguno, el texto base (reglas + BETO). Nunca la cambia esta capa: solo
se le aplica (a). Una segunda lectura de BETO jamás es la recomendada.

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
      palabra; "luego/después" o "voy/iré" son paráfrasis, no correcciones.
      Borrar un token igual a su vecino ("muy muy" → "muy") cuenta como
      edición válida.

La base va justo detrás de la recomendada (el tope no la elimina); el resto
por score descendente.
"""
import difflib
import re
import unicodedata

from infrastructure.ml.guards import is_safe_refinement

# Umbrales de la capa 5 y de la señal de ambigüedad de BETO (capa 2.5). Son
# constantes de código, sin override por entorno: el mismo `modelVersion` debe
# comportarse igual en el servicio, en evaluate.py y en una ejecución manual.
# evaluate.py y el manifiesto del modelo los registran tal cual. Calibración:
# data/ambiguity_calibration.csv + scripts/calibrate_alternatives.py.
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
}

SCORE_MARGIN        = THRESHOLDS["scoreMargin"]
MIN_SPAN_SIMILARITY = THRESHOLDS["minSpanSimilarity"]

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


def select_alternatives(
    original_text: str,
    base_text: str,
    candidates: list[tuple[str, float]],
    max_options: int = 3,
    score_margin: float = 1.0,
    recommended: str | None = None,
) -> list[str]:
    """
    Devuelve la lista final de sugerencias (recomendada primero, nunca vacía)
    a partir de `candidates` = [(texto, score)] (beams de T5 en orden de
    generación y segundas lecturas de BETO).

    `recommended` es el texto que ocupa la posición 0. Si es None se deduce
    como antes de la Task 5: el primer candidato seguro respecto a la base
    (`is_safe_refinement`), en el orden dado, o la base si no hay ninguno.
    El llamador (`CorrectionPipeline.correct`) lo pasa explícito con el primer
    beam seguro de T5, de modo que una variante de BETO nunca se recomienda.
    Si la recomendada no es segura, se recomienda la base.
    """
    base   = base_text.strip()
    scored = [(str(t).strip(), float(s)) for t, s in candidates if t and str(t).strip()]

    if recommended is None:
        recommended = next((t for t, _ in scored if is_safe_refinement(base, t)), base)
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

        chosen.append(text)
        chosen_norms.add(norm)
        chosen_signatures.append(signature)

    return chosen
