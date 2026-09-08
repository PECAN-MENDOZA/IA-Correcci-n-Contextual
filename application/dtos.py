"""
application/dtos.py
Data Transfer Objects ajustados al contrato de arquitectura (Backend <-> IA).
Estricto formato camelCase para integración con Spring Boot.
"""
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class AiCorrectionRequestDTO:
    originalText: str
    studentId: str

@dataclass
class AiCorrectionResponseDTO:
    studentId: str
    correctedText: str
    processingTimeMs: int
    suggestions: List[str] = field(default_factory=list)

@dataclass
class AiFeedbackRequestDTO:
    studentId: str
    originalText: str
    selectedSuggestion: Optional[str]
    accepted: bool