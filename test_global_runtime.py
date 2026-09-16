"""
test_global_runtime.py

Verifica que el runtime de corrección es GLOBAL: un único pipeline (reglas +
BETO + LoRA gramatical) para todos los alumnos, sin memoria, inferencia ni
entrenamiento por alumno.

Corre SIN torch/transformers/GPU: los colaboradores son fakes puros que solo
exponen `correct` y `save_feedback` (ningún método de memoria ni de
entrenamiento). Si un caso de uso intentara llamar algo más, el fake falla.

Uso:
    .venv\\Scripts\\python.exe -m unittest -v test_global_runtime
"""
import sys
import unittest

from application.dtos import (
    AiCorrectionRequestDTO,
    AiCorrectionResponseDTO,
    AiFeedbackRequestDTO,
)
from application.use_cases.correct_text import CorrectTextUseCase
from application.use_cases.save_feedback import SaveFeedbackUseCase


class FakePipeline:
    """Pipeline global mínimo: registra las llamadas y devuelve sugerencias fijas."""

    def __init__(self, suggestions):
        self._suggestions = list(suggestions)
        self.calls = []

    def correct(self, text, user_vocab):
        self.calls.append((text, dict(user_vocab)))
        return list(self._suggestions)


class FakeRepository:
    """Repositorio mínimo: solo persiste feedback aceptado."""

    def __init__(self):
        self.saved = []

    def save_feedback(self, feedback):
        self.saved.append(feedback)


class GlobalRuntimeTests(unittest.TestCase):

    def test_correction_uses_empty_global_vocabulary(self):
        pipeline = FakePipeline(["Los niños juegan."])
        result = CorrectTextUseCase(pipeline).execute(
            AiCorrectionRequestDTO("los niño juega.", "student-1")
        )
        self.assertEqual(pipeline.calls, [("los niño juega.", {})])
        self.assertEqual(result.correctedText, "Los niños juegan.")
        self.assertEqual(result.suggestions, ["Los niños juegan."])
        self.assertEqual(result.studentId, "student-1")
        self.assertIsInstance(result, AiCorrectionResponseDTO)

    def test_correction_defaults_model_version(self):
        pipeline = FakePipeline(["hola"])
        result = CorrectTextUseCase(pipeline).execute(
            AiCorrectionRequestDTO("ola", "student-1")
        )
        self.assertEqual(result.modelVersion, "global-lora-unversioned")

    def test_correction_carries_explicit_model_version(self):
        pipeline = FakePipeline(["hola"])
        result = CorrectTextUseCase(pipeline, model_version="t5@abc12345+grammar@def67890").execute(
            AiCorrectionRequestDTO("ola", "student-1")
        )
        self.assertEqual(result.modelVersion, "t5@abc12345+grammar@def67890")

    def test_same_text_two_students_same_call(self):
        """Ningún dato del alumno altera la llamada al pipeline global."""
        pipeline = FakePipeline(["hola"])
        use_case = CorrectTextUseCase(pipeline)
        use_case.execute(AiCorrectionRequestDTO("ola", "student-1"))
        use_case.execute(AiCorrectionRequestDTO("ola", "student-2"))
        self.assertEqual(pipeline.calls, [("ola", {}), ("ola", {})])

    def test_feedback_only_persists_for_later_curation(self):
        repo = FakeRepository()
        SaveFeedbackUseCase(repo).execute(
            AiFeedbackRequestDTO("student-1", "iva", "iba", True)
        )
        self.assertEqual(len(repo.saved), 1)
        saved = repo.saved[0]
        self.assertEqual(saved.user_id, "student-1")
        self.assertEqual(saved.original_text, "iva")
        self.assertEqual(saved.corrected_text, "iba")
        self.assertTrue(saved.accepted)

    def test_rejected_or_empty_feedback_is_ignored(self):
        repo = FakeRepository()
        use_case = SaveFeedbackUseCase(repo)
        use_case.execute(AiFeedbackRequestDTO("student-1", "iva", "iba", False))
        use_case.execute(AiFeedbackRequestDTO("student-1", "iva", None, True))
        use_case.execute(AiFeedbackRequestDTO("student-1", "iva", "", True))
        self.assertEqual(repo.saved, [])

    def test_feedback_does_not_change_next_correction(self):
        """El feedback no alimenta ninguna memoria: la siguiente corrección es idéntica."""
        pipeline = FakePipeline(["hola"])
        repo = FakeRepository()
        correct = CorrectTextUseCase(pipeline)
        SaveFeedbackUseCase(repo).execute(
            AiFeedbackRequestDTO("student-1", "ola", "holaaa", True)
        )
        result = correct.execute(AiCorrectionRequestDTO("ola", "student-1"))
        self.assertEqual(result.correctedText, "hola")
        self.assertEqual(pipeline.calls, [("ola", {})])

    def test_use_cases_do_not_load_ml_frameworks(self):
        """Los casos de uso no deben arrastrar torch/transformers al importarse."""
        for heavy in ("torch", "transformers"):
            self.assertFalse(heavy in sys.modules, f"'{heavy}' se cargó al importar los casos de uso")


if __name__ == "__main__":
    unittest.main()
