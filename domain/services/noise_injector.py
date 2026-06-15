"""
domain/services/noise_injector.py
Servicio de dominio: inyecta ruido clínico en texto limpio para data augmentation.
Encapsula el conocimiento de patrones de dislexia y disgrafía.
"""
import random
from typing import List


class NoiseInjectorService:
    """
    Genera variaciones con errores típicos de dislexia y disgrafía
    a partir de texto correcto. Usado exclusivamente para entrenamiento.
    """

    def __init__(self):
        self._visual_map = {
            'b': 'd', 'd': 'b', 'p': 'q', 'q': 'p',
            'n': 'u', 'u': 'n', 'm': 'w', 'w': 'm'
        }
        self._phonetic_map = {
            'v': 'b', 'b': 'v', 's': 'c', 'c': 's',
            'z': 's', 'j': 'g', 'g': 'j', 'y': 'll', 'll': 'y'
        }
        self._vowels = list('aeiouáéíóú')

    def inject(self, text: str) -> str:
        words = text.split()
        noisy_words: List[str] = []
        errors_injected = 0
        max_errors = random.randint(1, 3)

        for word in words:
            if errors_injected >= max_errors or len(word) < 2:
                noisy_words.append(word)
                continue
            if random.random() < 0.30:
                word = self._apply_clinical_rule(word)
                errors_injected += 1
            noisy_words.append(word)

        if random.random() < 0.15 and len(noisy_words) > 2:
            return self._apply_boundary_error(noisy_words)

        return " ".join(noisy_words)

    def _apply_clinical_rule(self, word: str) -> str:
        chars = list(word)
        rule = random.choices(
            ['phonetic', 'visual', 'h_rule', 'omission'],
            weights=[0.40, 0.30, 0.15, 0.15]
        )[0]

        if rule == 'visual':
            for i, ch in enumerate(chars):
                if ch.lower() in self._visual_map and random.random() < 0.6:
                    replacement = self._visual_map[ch.lower()]
                    chars[i] = replacement.upper() if ch.isupper() else replacement
                    break

        elif rule == 'phonetic':
            for i, ch in enumerate(chars):
                if ch.lower() in self._phonetic_map and random.random() < 0.6:
                    replacement = self._phonetic_map[ch.lower()]
                    chars[i] = replacement.upper() if ch.isupper() else replacement
                    break

        elif rule == 'h_rule':
            if chars[0].lower() == 'h':
                chars = chars[1:]
            elif chars[0].lower() in self._vowels:
                chars.insert(0, 'H' if chars[0].isupper() else 'h')

        elif rule == 'omission' and len(chars) > 3:
            idx = random.randint(1, len(chars) - 2)
            chars.pop(idx)

        return "".join(chars)

    @staticmethod
    def _apply_boundary_error(words: List[str]) -> str:
        idx = random.randint(0, len(words) - 2)
        words[idx] = words[idx] + words[idx + 1]
        words.pop(idx + 1)
        return " ".join(words)
