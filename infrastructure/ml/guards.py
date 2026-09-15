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


def is_lexically_plausible_refinement(base_text: str, candidate: str, min_similarity: float = 0.3) -> bool:
    """
    Guarda léxica del beam RECOMENDADO: `is_safe_refinement` mide cuánto
    cambia el candidato (cantidad), esta mide qué cambia (calidad). Cada
    reemplazo 1:1 de palabra respecto a `base_text` (opcodes de palabra de
    `SequenceMatcher`, comparando sin tildes ni mayúsculas ni puntuación
    exterior) debe ser una variante léxica cercana: la similitud de caracteres
    (`difflib.SequenceMatcher(None, a, b).ratio()` sobre las palabras sin
    tildes) tiene que ser ≥ `min_similarity`.

    Con el umbral 0.3 se bloquean las sustituciones léxicas del T5
    (pasto→maíz 0.22, pasto→carne 0.20, verde→rojo 0.22, voy→iré 0.0) y se
    conservan las flexiones, incluso irregulares (es→son 0.40, hizo→hicieron
    0.50, viene→vengan 0.55, fue→fueron 0.67; esta→está 1.0). Un bloque n:n se
    juzga palabra a palabra; las inserciones, los borrados y los bloques
    1:n / n:1 ("a el" → "al", "voy a ir" → "iré") NO se juzgan aquí (de eso
    se ocupan `is_safe_refinement` y la regla (e) de las alternativas).
    """
    base_words = [_lexical_key(w) for w in base_text.split()]
    cand_words = [_lexical_key(w) for w in candidate.split()]
    matcher = difflib.SequenceMatcher(a=base_words, b=cand_words, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "replace" or i2 - i1 != j2 - j1:
            continue
        for word_a, word_b in zip(base_words[i1:i2], cand_words[j1:j2]):
            if difflib.SequenceMatcher(None, word_a, word_b, autojunk=False).ratio() < min_similarity:
                return False
    return True
