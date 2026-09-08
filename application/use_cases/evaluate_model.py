"""
application/use_cases/evaluate_model.py
Caso de uso: evaluar el rendimiento del modelo sobre un batch de predicciones.
"""
from typing import List, Dict
import Levenshtein
from jiwer import wer, cer


class EvaluateModelUseCase:
    """
    Calcula métricas estándar de evaluación NLP para el sistema de corrección.

    Métricas:
      - WER  (Word Error Rate)
      - CER  (Character Error Rate)
      - improvement_ratio: qué tan cerca llegamos del ground truth vs el input original
    """

    def execute(
        self,
        originals: List[str],
        inputs: List[str],
        predictions: List[str],
    ) -> Dict[str, float]:
        dist_before = sum(Levenshtein.distance(o, i) for o, i in zip(originals, inputs))
        dist_after  = sum(Levenshtein.distance(o, p) for o, p in zip(originals, predictions))

        improvement = (dist_before - dist_after) / dist_before if dist_before > 0 else 0.0

        return {
            "wer": wer(originals, predictions),
            "cer": cer(originals, predictions),
            "improvement_ratio": improvement,
        }
