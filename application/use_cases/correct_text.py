"""
application/use_cases/correct_text.py  v4 (LoRA por alumno) → v5 (LoRA global)

Un único pipeline global (reglas + BETO + LoRA gramatical) atiende a todos los
alumnos. Ya no existe memoria por alumno, vocabulario personalizado ni
adaptadores LoRA por alumno: la ruta de inferencia es idéntica para cualquier
`studentId`, y el identificador solo se devuelve en la respuesta.

`modelVersion` identifica el modelo global que produjo la corrección para que
el backend pueda trazarlo por ejecución. El valor por defecto y su resolución
viven en `infrastructure/versioning.py` (compartido con `evaluate.py`); aquí
solo se reexporta `DEFAULT_MODEL_VERSION`.
"""
import time
from typing import TYPE_CHECKING

from application.dtos import AiCorrectionRequestDTO, AiCorrectionResponseDTO
from infrastructure.versioning import DEFAULT_MODEL_VERSION  # noqa: F401 (reexportado)

if TYPE_CHECKING:  # solo para tipado: evita cargar torch/transformers al importar
    from infrastructure.nlp.correction_pipeline import CorrectionPipeline


class CorrectTextUseCase:
    def __init__(
        self,
        pipeline: "CorrectionPipeline",
        model_version: str = DEFAULT_MODEL_VERSION,
    ):
        self._pipeline      = pipeline
        self._model_version = model_version

    def execute(self, request: AiCorrectionRequestDTO) -> AiCorrectionResponseDTO:
        start = time.time()
        # Vocabulario vacío: el pipeline global no recibe datos del alumno.
        suggestions = self._pipeline.correct(request.originalText, {})
        return AiCorrectionResponseDTO(
            studentId=request.studentId,
            correctedText=suggestions[0],
            processingTimeMs=int((time.time() - start) * 1000),
            suggestions=suggestions,
            modelVersion=self._model_version,
        )
