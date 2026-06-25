"""
interfaces/api/routes.py
Expone los endpoints estrictamente como dicta la arquitectura para la red interna.
"""
import json
from flask import Flask, request, Response

from application.dtos import AiCorrectionRequestDTO, AiFeedbackRequestDTO
from application.use_cases.correct_text import CorrectTextUseCase
from application.use_cases.save_feedback import SaveFeedbackUseCase

def create_app(
    correct_use_case: CorrectTextUseCase,
    feedback_use_case: SaveFeedbackUseCase,
) -> Flask:
    app = Flask(__name__)
    app.config["JSON_AS_ASCII"] = False

    # Endpoint interno para corrección
    @app.route("/interno/corregir", methods=["POST"])
    def correct():
        data = request.json or {}
        
        original_text = data.get("originalText", "")
        student_id = data.get("studentId", "")

        if not original_text or not student_id:
            return Response(
                json.dumps({"error": "Faltan campos obligatorios (originalText, studentId)"}, ensure_ascii=False),
                status=400, mimetype="application/json; charset=utf-8"
            )

        dto = AiCorrectionRequestDTO(
            originalText=original_text, 
            studentId=student_id
        )
        
        result = correct_use_case.execute(dto)

        payload = {
            "studentId": result.studentId,
            "correctedText": result.correctedText,
            "processingTimeMs": result.processingTimeMs,
            "suggestions": result.suggestions
        }
        
        return Response(
            json.dumps(payload, ensure_ascii=False),
            status=200, mimetype="application/json; charset=utf-8"
        )
        
    # Endpoint interno para feedback (entrenamiento)
    @app.route("/interno/feedback", methods=["POST"])
    def feedback():
        data = request.json or {}
        
        student_id = data.get("studentId")
        original_text = data.get("originalText")
        
        if not student_id or not original_text:
            return Response(
                json.dumps({"error": "Faltan campos obligatorios (studentId, originalText)"}, ensure_ascii=False),
                status=400, mimetype="application/json; charset=utf-8"
            )
            
        dto = AiFeedbackRequestDTO(
            studentId=student_id,
            originalText=original_text,
            selectedSuggestion=data.get("selectedSuggestion"),
            accepted=data.get("accepted", False)
        )
        
        feedback_use_case.execute(dto)
        return Response(status=204)

    return app