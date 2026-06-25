"""
application/use_cases/correct_text.py

v2 → v3: _get_pipeline_for_user() ahora llama a T5CorrectionModel.validate_dir()
antes de intentar cargar el modelo de usuario. Si el directorio está corrupto
(pesos truncados, config.json inválido, etc.) lo elimina y cae al modelo base
en lugar de crashear la GPU con `input[0] != 0`.
"""
import shutil
import time
from pathlib import Path

from application.dtos import AiCorrectionRequestDTO, AiCorrectionResponseDTO
from domain.repositories.interfaces import IUserHistoryRepository
from infrastructure.nlp.correction_pipeline import CorrectionPipeline


class CorrectTextUseCase:
    def __init__(
        self,
        pipeline: CorrectionPipeline,
        user_repo: IUserHistoryRepository,
        user_models_dir: str = "./models/users",
    ):
        self._pipeline        = pipeline
        self._user_repo       = user_repo
        self._user_models_dir = Path(user_models_dir)
        self._user_pipelines: dict[str, CorrectionPipeline] = {}

    def execute(self, request: AiCorrectionRequestDTO) -> AiCorrectionResponseDTO:
        start_time = time.time()
        user_vocab = self._user_repo.get_user_vocabulary(request.studentId)
        pipeline   = self._get_pipeline_for_user(request.studentId)
        suggestions = pipeline.correct(request.originalText, user_vocab)
        processing_time = int((time.time() - start_time) * 1000)

        return AiCorrectionResponseDTO(
            studentId=request.studentId,
            correctedText=suggestions[0],
            processingTimeMs=processing_time,
            suggestions=suggestions,
        )

    def invalidate_user_cache(self, student_id: str) -> None:
        """
        Descarta el pipeline cacheado de un usuario.
        Llamar tras fine-tuning para que la próxima request cargue
        el modelo recién entrenado desde disco.
        """
        self._user_pipelines.pop(student_id, None)

    def _get_pipeline_for_user(self, student_id: str) -> CorrectionPipeline:
        """
        Devuelve el pipeline personalizado del usuario si existe y es válido.
        Si el directorio de modelo está corrupto, lo elimina y usa el base.
        """
        if student_id in self._user_pipelines:
            return self._user_pipelines[student_id]

        user_model_dir = self._user_models_dir / student_id

        if not user_model_dir.exists():
            return self._pipeline

        # ── FIX BUG #5: Validar integridad antes de cargar ──────────────────
        from infrastructure.ml.t5_model import T5CorrectionModel
        if not T5CorrectionModel.validate_dir(user_model_dir):
            print(f"[WARN] Modelo de '{student_id}' corrupto o incompleto. "
                  f"Eliminando y usando modelo base.")
            shutil.rmtree(user_model_dir, ignore_errors=True)
            return self._pipeline

        try:
            from infrastructure.ml.t5_model import T5SpanishTokenizer

            print(f"[INFO] Cargando modelo personalizado para '{student_id}'...")
            user_tokenizer = T5SpanishTokenizer(model_name=str(user_model_dir))
            user_model     = T5CorrectionModel(save_dir=str(user_model_dir))

            user_pipeline = CorrectionPipeline(
                phonetic=self._pipeline._phonetic,
                judge=self._pipeline._judge,
                seq2seq=user_model,
                tokenizer=user_tokenizer,
            )
            user_pipeline.user_memory = self._pipeline.user_memory

            self._user_pipelines[student_id] = user_pipeline
            print(f"[OK] Pipeline personalizado listo para '{student_id}'.")
            return user_pipeline

        except Exception as e:
            print(f"[WARN] No se pudo cargar modelo de '{student_id}': {e}. "
                  f"Eliminando directorio y usando modelo base.")
            shutil.rmtree(user_model_dir, ignore_errors=True)
            return self._pipeline
