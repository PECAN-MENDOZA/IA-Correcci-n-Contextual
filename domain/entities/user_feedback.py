"""
domain/entities/user_feedback.py
Entidad de feedback: captura la aceptación de una corrección por parte del usuario.
"""
from dataclasses import dataclass


@dataclass
class UserFeedback:
    """Representa la señal de feedback que el usuario envía tras una corrección."""
    user_id: str
    original_text: str
    corrected_text: str
    accepted: bool
