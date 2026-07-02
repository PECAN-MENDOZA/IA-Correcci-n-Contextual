"""
domain/entities/correction.py
Entidades del dominio: representan los objetos de negocio centrales.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class WordError:
    """Representa un error detectado en una palabra individual."""
    original_word: str
    corrected_word: str
    error_type: str  # "Dislexia Visual", "Disgrafía Fonológica", "Otro", etc.


@dataclass
class Correction:
    """Resultado completo de una corrección de texto."""
    input_text: str
    corrected_text: str
    user_id: str
    errors: List[WordError] = field(default_factory=list)

    @property
    def was_modified(self) -> bool:
        return self.input_text != self.corrected_text
