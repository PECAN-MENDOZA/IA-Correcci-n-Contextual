"""
application/dtos.py
Data Transfer Objects: estructuras simples para comunicación entre capas.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class CorrectionRequestDTO:
    text: str
    user_id: str = "default_user"


@dataclass
class WordErrorDTO:
    word: str
    correction: str
    type: str


@dataclass
class CorrectionResponseDTO:
    input: str
    original: str
    corrected: str
    errors: List[WordErrorDTO] = field(default_factory=list)


@dataclass
class FeedbackRequestDTO:
    user_id: str
    original: str
    corrected: str
    accepted: bool


@dataclass
class TrainingPairsDTO:
    pairs: List[tuple]
    source: str = "dataset"
