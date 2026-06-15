"""
application/use_cases/save_feedback.py
Caso de uso: registrar el feedback del usuario sobre una corrección.
"""
from application.dtos import FeedbackRequestDTO
from domain.entities.user_feedback import UserFeedback
from domain.repositories.interfaces import IUserHistoryRepository


class SaveFeedbackUseCase:
    def __init__(self, user_repo: IUserHistoryRepository):
        self._user_repo = user_repo

    def execute(self, request: FeedbackRequestDTO) -> dict:
        if not request.accepted or not request.original or not request.corrected:
            return {"status": "ignored"}

        feedback = UserFeedback(
            user_id=request.user_id,
            original_text=request.original,
            corrected_text=request.corrected,
            accepted=request.accepted,
        )
        self._user_repo.save_feedback(feedback)
        return {"status": "saved"}
