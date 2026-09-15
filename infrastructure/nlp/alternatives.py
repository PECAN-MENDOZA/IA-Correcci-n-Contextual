"""
infrastructure/nlp/alternatives.py

Selección pura de las alternativas que se ofrecen al alumno: como mucho
`max_options` (3) y SOLO cuando la oración admite lecturas distintas. En el
caso común el resultado es una sola recomendación (o refinado + base, como
siempre). Sin torch: solo difflib/re/unicodedata y la guarda
`is_safe_refinement`.

Un candidato `(texto, score)` se ofrece únicamente si:
  (a) pasa `is_safe_refinement` respecto al texto base (no alucina);
  (b) tras normalizar (espacios y puntuación final) es distinto del texto
      original y de los ya elegidos — el original nunca es una "corrección";
      la mayúscula inicial sí cuenta aquí ("El niño juega." corrige
      "el niño juega");
  (c) su conjunto de ediciones (índices de palabra cambiados respecto al
      original, ignorando la mayúscula inicial) es distinto del de los ya
      elegidos: dos textos que corrigen las mismas palabras no son dos
      interpretaciones, solo una y su variante;
  (d) su score (log-prob de beam, o la de BETO para una segunda lectura) está
      a menos de `score_margin` del mejor candidato;
  (e) cada tramo que edita es una variante léxica cercana del original
      (similitud sin tildes ≥ MIN_SPAN_SIMILARITY): "juega/jueguen",
      "esta/está", "tubo/tuvo" son otras lecturas de la misma palabra;
      "luego/después" o "voy/iré" son paráfrasis, no correcciones.

El texto base (reglas + BETO) entra siempre justo detrás del mejor candidato,
con su mismo score, y solo se filtra por (b): si el refinado difiere de la
base, la base se conserva como alternativa (red de seguridad frente a un T5
equivocado).
"""
import difflib
import re
import unicodedata

from infrastructure.ml.guards import is_safe_refinement

# Similitud mínima (difflib.SequenceMatcher.ratio, sin tildes) entre un tramo
# del original y el tramo que lo sustituye para considerarlo "otra lectura":
#   se/sé, esta/está 1.0 · tubo/tuvo 0.75 · fue/fueron, boy/voy, a/ha 0.67
#   hoy/holly 0.5 · ellos/él 0.29 · luego/después 0.15 · voy/iré 0.0
MIN_SPAN_SIMILARITY = 0.6

_TRAILING_PUNCT_RE = re.compile(r"[\s.!?…]+$")
_SPACES_RE         = re.compile(r"\s+")


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
    Palabras sobre las que se calcula el conjunto de ediciones. Se compara CON
    tildes (`esta/está` es precisamente una edición) pero sin la mayúscula
    inicial de la oración: poner mayúscula no es otra interpretación.
    """
    words = _normalize(text).split()
    if words:
        words[0] = words[0][:1].lower() + words[0][1:]
    return words


def _similarity(a: str, b: str) -> float:
    a, b = _strip_accents(a).lower(), _strip_accents(b).lower()
    return difflib.SequenceMatcher(a=a, b=b, autojunk=False).ratio()


def _edits(original_words: list[str], cand_words: list[str]) -> tuple[frozenset, float]:
    """
    Devuelve (índices de palabra del original que el candidato modifica —
    reemplazo, borrado o posición ante la que inserta—, similitud mínima entre
    lo editado y su reemplazo). Similitud 1.0 si no hay ediciones. Un tramo
    con el mismo número de palabras se compara palabra a palabra (dos edits
    contiguos no se diluyen entre sí); si cambia el número de palabras, se
    comparan los tramos completos ("a el" → "al").
    """
    matcher    = difflib.SequenceMatcher(a=original_words, b=cand_words, autojunk=False)
    edited     = set()
    similarity = 1.0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if i2 > i1:
            edited.update(range(i1, i2))
        else:  # inserción: se anota la posición del original donde entra
            edited.add(min(i1, max(len(original_words) - 1, 0)))
        if i2 - i1 == j2 - j1:
            pairs = zip(original_words[i1:i2], cand_words[j1:j2])
        else:
            pairs = [(" ".join(original_words[i1:i2]), " ".join(cand_words[j1:j2]))]
        for word_a, word_b in pairs:
            similarity = min(similarity, _similarity(word_a, word_b))
    return frozenset(edited), similarity


def select_alternatives(
    original_text: str,
    base_text: str,
    candidates: list[tuple[str, float]],
    max_options: int = 3,
    score_margin: float = 1.0,
) -> list[str]:
    """
    Devuelve la lista final de sugerencias (recomendada primero) a partir de
    `candidates` = [(texto, score)] (beams de T5 y variantes homófonas de BETO).
    Si ningún candidato cumple las reglas devuelve [] y el llamador debe caer
    al comportamiento actual (`[base_text]`).
    """
    if max_options <= 0:
        return []

    scored = [(str(t).strip(), float(s)) for t, s in candidates if t and str(t).strip()]
    best_score = max((s for _, s in scored), default=0.0)

    # Orden por score descendente (estable); la base va justo detrás del mejor
    # candidato con su mismo score: nunca lo desplaza como recomendada y el
    # tope de `max_options` no la elimina.
    base    = base_text.strip()
    ordered = sorted(scored, key=lambda ts: -ts[1])
    ordered.insert(1, (base, best_score))

    original_norm  = _normalize(original_text)
    original_words = _words_for_edits(original_text)

    chosen: list[str]             = []
    chosen_norms: set[str]        = set()
    chosen_edits: list[frozenset] = []

    for text, score in ordered:
        is_base = text == base
        if not is_base:
            if score < best_score - score_margin:          # (d)
                continue
            if not is_safe_refinement(base, text):         # (a)
                continue
        norm = _normalize(text)
        if not norm or norm == original_norm or norm in chosen_norms:   # (b)
            continue
        edits, similarity = _edits(original_words, _words_for_edits(text))
        if not is_base:
            if edits in chosen_edits:                      # (c)
                continue
            if similarity < MIN_SPAN_SIMILARITY:           # (e)
                continue

        chosen.append(text)
        chosen_norms.add(norm)
        chosen_edits.append(edits)
        if len(chosen) >= max_options:
            break

    return chosen
