"""
application/use_cases/save_feedback.py

El feedback aceptado por el alumno SOLO se persiste (CSV) para su curación
posterior como dato de entrenamiento del LoRA global. No alimenta ninguna
memoria por alumno ni encola entrenamientos: la siguiente corrección del
mismo alumno pasa por el mismo pipeline global que la anterior.
"""

from application.dtos import AiFeedbackRequestDTO
from domain.entities.user_feedback import UserFeedback
from domain.repositories.interfaces import IUserHistoryRepository


class SaveFeedbackUseCase:
    def __init__(self, user_repo: IUserHistoryRepository):
        self._user_repo = user_repo

    def execute(self, request: AiFeedbackRequestDTO) -> None:
        if not (request.accepted and request.selectedSuggestion):
            return

        self._user_repo.save_feedback(UserFeedback(
            user_id=request.studentId,
            original_text=request.originalText,
            corrected_text=request.selectedSuggestion,
            accepted=request.accepted,
        ))
