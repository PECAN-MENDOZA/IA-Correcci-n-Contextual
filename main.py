"""
main.py - Composition Root
Ensambla todas las capas DDD y arranca el servidor Flask.
"""
from infrastructure.persistence.csv_user_repository import CsvUserHistoryRepository
from infrastructure.nlp.phonetic_engine import PhoneticEngine
from infrastructure.nlp.context_judge import ContextJudge
from infrastructure.nlp.correction_pipeline import CorrectionPipeline
from domain.services.error_analyzer import ErrorAnalyzerService
from application.use_cases.correct_text import CorrectTextUseCase
from application.use_cases.save_feedback import SaveFeedbackUseCase
from interfaces.api.routes import create_app


def build_app():
    # Infraestructura
    user_repo        = CsvUserHistoryRepository()
    phonetic_engine  = PhoneticEngine()
    context_judge    = ContextJudge()
    pipeline         = CorrectionPipeline(phonetic=phonetic_engine, judge=context_judge)

    # Dominio
    error_analyzer   = ErrorAnalyzerService()

    # Casos de uso
    correct_uc       = CorrectTextUseCase(pipeline, user_repo, error_analyzer)
    feedback_uc      = SaveFeedbackUseCase(user_repo)

    return create_app(correct_uc, feedback_uc)


if __name__ == "__main__":
    app = build_app()
    app.run(host="0.0.0.0", port=5000, debug=False)
