"""
infrastructure/ml/guards.py

Guardas anti-alucinación para cualquier generación de texto basada en modelo
(T5 base o adaptador LoRA por-usuario). Deliberadamente SIN dependencias de
torch/transformers/peft: son funciones puras sobre strings, para que se
puedan testear sin necesitar el modelo real ni GPU.

Usado por:
  - infrastructure/nlp/correction_pipeline.py  (refinamiento T5 base)
  - infrastructure/ml/t5_model.py              (generate_with_lora)
  - application/use_cases/correct_text.py      (_LoraProxyPipeline)
  - application/use_cases/train_model.py       (canario post-entrenamiento)
"""


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
