"""
application/use_cases/correct_text.py
Caso de uso: corregir un texto recibido del usuario.
Orquesta el pipeline de corrección sin conocer detalles de infraestructura.
"""
from application.dtos import CorrectionRequestDTO, CorrectionResponseDTO, WordErrorDTO
from domain.entities.correction import Correction
from domain.repositories.interfaces import IUserHistoryRepository
from domain.services.error_analyzer import ErrorAnalyzerService
from infrastructure.nlp.correction_pipeline import CorrectionPipeline


class CorrectTextUseCase:
    def __init__(
        self,
        pipeline: CorrectionPipeline,
        user_repo: IUserHistoryRepository,
        error_analyzer: ErrorAnalyzerService,
    ):
        self._pipeline = pipeline
        self._user_repo = user_repo
        self._error_analyzer = error_analyzer

    def execute(self, request: CorrectionRequestDTO) -> CorrectionResponseDTO:
        user_vocab = self._user_repo.get_user_vocabulary(request.user_id)
        corrected_text = self._pipeline.correct(request.text, user_vocab)

        errors = self._error_analyzer.analyze(request.text, corrected_text)

        return CorrectionResponseDTO(
            input=request.text,
            original=request.text,
            corrected=corrected_text,
            errors=[
                WordErrorDTO(word=e.original_word, correction=e.corrected_word, type=e.error_type)
                for e in errors
            ]
        )
