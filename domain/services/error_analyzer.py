"""
domain/services/error_analyzer.py
Servicio de dominio: clasifica el tipo de error ortográfico detectado.
No depende de ninguna librería de infraestructura (solo Levenshtein como utilidad matemática).
"""
from typing import List
import Levenshtein

from domain.entities.correction import WordError


class ErrorAnalyzerService:
    """
    Analiza pares (original, corregida) a nivel de palabra y clasifica
    el tipo de error según patrones clínicos de dislexia / disgrafía.
    """

    def __init__(self):
        self._visual_errors = {
            ('b', 'd'), ('d', 'b'), ('p', 'q'), ('q', 'p'),
            ('u', 'n'), ('n', 'u')
        }
        self._phonological_errors = {
            ('b', 'v'), ('v', 'b'), ('c', 's'), ('s', 'c'),
            ('z', 's'), ('s', 'z'), ('ll', 'y'), ('y', 'll')
        }

    def analyze(self, original: str, corrected: str) -> List[WordError]:
        """
        Compara palabra a palabra y devuelve la lista de errores encontrados.
        """
        orig_words = original.lower().split()
        corr_words = corrected.lower().split()
        errors: List[WordError] = []

        for o_word, c_word in zip(orig_words, corr_words):
            if o_word == c_word:
                continue

            error_type = self._classify(o_word, c_word)
            errors.append(WordError(
                original_word=o_word,
                corrected_word=c_word,
                error_type=error_type
            ))

        return errors

    def _classify(self, o_word: str, c_word: str) -> str:
        if len(o_word) == len(c_word) and Levenshtein.distance(o_word, c_word) == 1:
            diff = [(o, c) for o, c in zip(o_word, c_word) if o != c]
            if diff:
                pair = diff[0]
                if pair in self._visual_errors:
                    return "Dislexia Visual (Espejamiento)"
                if pair in self._phonological_errors:
                    return "Disgrafía Fonológica"
                if pair[0] in 'aeiou' and pair[1] in 'aeiou':
                    return "Sustitución de Vocales"
        return "Otro (Gramática/Léxico)"
