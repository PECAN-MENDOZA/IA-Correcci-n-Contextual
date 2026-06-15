"""
domain/repositories/interfaces.py
Puertos (interfaces) del dominio. Definen los contratos que la infraestructura debe cumplir.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple

from domain.entities.user_feedback import UserFeedback


class IUserHistoryRepository(ABC):
    """Contrato para persistir y recuperar el historial de correcciones por usuario."""

    @abstractmethod
    def save_feedback(self, feedback: UserFeedback) -> None:
        """Persiste una corrección aceptada por el usuario."""

    @abstractmethod
    def get_user_pairs(self, user_id: str) -> List[Tuple[str, str]]:
        """Devuelve los pares (erróneo, correcto) del historial de un usuario."""

    @abstractmethod
    def get_user_vocabulary(self, user_id: str) -> dict:
        """Devuelve un mapa {palabra_errónea: palabra_correcta} del usuario."""
