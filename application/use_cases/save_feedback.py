"""
application/use_cases/save_feedback.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Cambio respecto a la versión anterior:
  ANTES → lanzaba un threading.Thread descontrolado por cada feedback.
           Esto causaba que inferencia y training se ejecutaran
           simultáneamente en GPU → device-side assert triggered.

  AHORA → solo llama a TrainingWorker.enqueue(user_id).
           El worker ya existe (arrancó con el servidor) y procesa
           las tareas una por una (cola con un solo consumidor),
           usando una instancia de modelo separada de la de inferencia,
           por lo que YA NO bloquea las correcciones de otros usuarios
           mientras entrena (ver application/training_queue.py v5).
           La API responde 204 en microsegundos, sin importar cuánto
           tarde el fine-tuning.
"""

from application.dtos import AiFeedbackRequestDTO
from domain.entities.user_feedback import UserFeedback
from domain.repositories.interfaces import IUserHistoryRepository
from application.training_queue import TrainingWorker


class SaveFeedbackUseCase:
    def __init__(
        self,
        user_repo: IUserHistoryRepository,
        training_worker: TrainingWorker = None,   # ← reemplaza train_user_use_case
        pipeline=None,
        correct_use_case=None,
        user_memory=None,
    ):
        self._user_repo       = user_repo
        self._training_worker = training_worker
        self._pipeline        = pipeline
        self._correct_use_case = correct_use_case
        self._user_memory     = user_memory   # memoria por-usuario (Redis)

    def execute(self, request: AiFeedbackRequestDTO) -> None:
        if not (request.accepted and request.selectedSuggestion):
            return

        # 1. Memoria del usuario (Redis, por-usuario, compartida entre réplicas):
        #    instantánea y con precedencia en la próxima corrección de este alumno.
        if self._user_memory is not None:
            self._user_memory.remember(
                request.studentId, request.originalText, request.selectedSuggestion
            )
        elif self._pipeline is not None:
            self._pipeline.remember(request.originalText, request.selectedSuggestion)

        # 2. Persistencia en CSV
        self._user_repo.save_feedback(UserFeedback(
            user_id=request.studentId,
            original_text=request.originalText,
            corrected_text=request.selectedSuggestion,
            accepted=request.accepted,
        ))

        # 3. Encolar tarea de fine-tuning (no bloquea, no crashea)
        if self._training_worker:
            self._training_worker.enqueue(request.studentId)
