"""
interfaces/api/routes.py
Capa de interfaz HTTP: recibe peticiones, delega a casos de uso, devuelve JSON.
No contiene lógica de negocio.
"""
import json
from flask import Flask, request, Response

from application.dtos import CorrectionRequestDTO, FeedbackRequestDTO
from application.use_cases.correct_text import CorrectTextUseCase
from application.use_cases.save_feedback import SaveFeedbackUseCase


def create_app(
    correct_use_case: CorrectTextUseCase,
    feedback_use_case: SaveFeedbackUseCase,
) -> Flask:
    app = Flask(__name__)
    app.config["JSON_AS_ASCII"] = False

    @app.route("/correct", methods=["POST"])
    def correct():
        data    = request.json or {}
        text    = data.get("text", data.get("input", ""))
        user_id = data.get("user_id", "default_user")

        if not text:
            return Response(
                json.dumps({"error": "Texto vacío"}, ensure_ascii=False),
                status=400, mimetype="application/json; charset=utf-8"
            )

        dto    = CorrectionRequestDTO(text=text, user_id=user_id)
        result = correct_use_case.execute(dto)

        payload = {
            "input":     result.input,
            "original":  result.original,
            "corrected": result.corrected,
            "analysis":  {
                "details": [
                    {"word": e.word, "correction": e.correction, "type": e.type}
                    for e in result.errors
                ]
            },
        }
        return Response(
            json.dumps(payload, ensure_ascii=False),
            status=200, mimetype="application/json; charset=utf-8"
        )

    @app.route("/feedback", methods=["POST"])
    def feedback():
        data     = request.json or {}
        accepted_raw = data.get("accepted", False)
        accepted = str(accepted_raw).lower() in ("true", "1", "t", "y", "yes")

        dto    = FeedbackRequestDTO(
            user_id   = data.get("user_id", "default_user"),
            original  = data.get("original", ""),
            corrected = data.get("corrected", ""),
            accepted  = accepted,
        )
        result = feedback_use_case.execute(dto)
        return Response(
            json.dumps(result, ensure_ascii=False),
            status=200, mimetype="application/json; charset=utf-8"
        )

    return app
