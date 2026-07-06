"""
application/use_cases/correct_text.py  v3 → v4 (LoRA)

_get_pipeline_for_user() ahora busca adaptador LoRA en models/loras/<user_id>/
en vez de modelo completo en models/users/<user_id>/.

Si existe adaptador válido → usa _LoraProxyPipeline (genera con LoRA).
Si no existe             → usa pipeline base compartido.
Compatibilidad legacy con models/users/ mantenida.
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
        loras_dir: str = "./models/loras",
        user_memory=None,
        enable_user_lora: bool = False,
    ):
        self._pipeline        = pipeline
        self._user_repo       = user_repo
        self._user_models_dir = Path(user_models_dir)
        self._loras_dir       = Path(loras_dir)
        self._user_pipelines: dict[str, CorrectionPipeline] = {}
        self._base_model    = None
        self._base_tokenizer = None
        self._user_memory   = user_memory   # memoria por-usuario (Redis), tiene precedencia
        # El LoRA por-usuario reemplaza TODO el pipeline por una generación T5
        # estocástica que alucina y no es determinista (baja la calidad de 37/38
        # a ~17/38 para alumnos con feedback). Desactivado: la personalización
        # útil ya vive en la memoria Capa 0 (Redis). Reactivar solo si se corrige
        # esa ruta (generación determinista + refinamiento sobre reglas+BETO).
        self._enable_user_lora = enable_user_lora

    def set_base_model(self, model, tokenizer) -> None:
        """Inyecta el modelo base desde main.py para uso en LoRA."""
        self._base_model     = model
        self._base_tokenizer = tokenizer

    def execute(self, request: AiCorrectionRequestDTO) -> AiCorrectionResponseDTO:
        start_time  = time.time()

        # CAPA 0 — Memoria del usuario (Redis, por-usuario, compartida entre
        # réplicas). Si este alumno ya validó esta frase (o una muy similar),
        # devolvemos su corrección directamente. Tiene precedencia sobre todo el
        # pipeline (incluido el LoRA por usuario) y es instantánea (~0 ms).
        if self._user_memory is not None:
            try:
                remembered = self._user_memory.lookup(request.studentId, request.originalText)
            except Exception as exc:
                remembered = None
                print(f"[WARN] Lookup de memoria de usuario falló: {exc}")
            if remembered:
                return AiCorrectionResponseDTO(
                    studentId=request.studentId,
                    correctedText=remembered,
                    processingTimeMs=int((time.time() - start_time) * 1000),
                    suggestions=[remembered],
                )

        user_vocab  = self._user_repo.get_user_vocabulary(request.studentId)
        pipeline    = self._get_pipeline_for_user(request.studentId)
        suggestions = pipeline.correct(request.originalText, user_vocab)
        processing_time = int((time.time() - start_time) * 1000)

        return AiCorrectionResponseDTO(
            studentId=request.studentId,
            correctedText=suggestions[0],
            processingTimeMs=processing_time,
            suggestions=suggestions,
        )

    def invalidate_user_cache(self, student_id: str) -> None:
        self._user_pipelines.pop(student_id, None)
        print(f"[Cache] Pipeline de '{student_id}' invalidado.")

    def _get_pipeline_for_user(self, student_id: str) -> CorrectionPipeline:
        # FIX (reporte degradación LoRA): el LoRA/legacy por-usuario degrada la
        # calidad y no es determinista. Salvo que se reactive explícitamente,
        # todos los alumnos usan el pipeline base completo (reglas+BETO+gramática+T5).
        if not self._enable_user_lora:
            return self._pipeline

        if student_id in self._user_pipelines:
            return self._user_pipelines[student_id]

        # ── Ruta LoRA (nueva) ────────────────────────────────────────────────
        from infrastructure.ml.t5_model import T5CorrectionModel, generate_with_lora
        lora_dir = self._loras_dir / student_id

        if T5CorrectionModel.validate_lora_dir(lora_dir):
            try:
                base_model    = self._base_model or self._pipeline._seq2seq
                base_tokenizer = self._base_tokenizer or self._pipeline._tokenizer
                if base_model is None:
                    return self._pipeline

                lora_pipeline = _LoraProxyPipeline(
                    base_pipeline=self._pipeline,
                    base_model=base_model,
                    base_tokenizer=base_tokenizer,
                    lora_dir=lora_dir,
                )
                self._user_pipelines[student_id] = lora_pipeline
                print(f"[OK] Pipeline LoRA listo para '{student_id}'.")
                return lora_pipeline

            except Exception as e:
                print(f"[WARN] Error cargando LoRA de '{student_id}': {e}. Usando base.")
                shutil.rmtree(lora_dir, ignore_errors=True)
                return self._pipeline

        # ── Ruta legacy (full fine-tuning, compatibilidad) ───────────────────
        user_model_dir = self._user_models_dir / student_id
        if user_model_dir.exists():
            if not T5CorrectionModel.validate_dir(user_model_dir):
                print(f"[WARN] Modelo legacy de '{student_id}' corrupto. Eliminando.")
                shutil.rmtree(user_model_dir, ignore_errors=True)
                return self._pipeline
            try:
                from infrastructure.ml.t5_model import T5SpanishTokenizer
                user_tokenizer = T5SpanishTokenizer(model_name=str(user_model_dir))
                user_model     = T5CorrectionModel(save_dir=str(user_model_dir))
                user_pipeline  = CorrectionPipeline(
                    phonetic=self._pipeline._phonetic,
                    judge=self._pipeline._judge,
                    seq2seq=user_model,
                    tokenizer=user_tokenizer,
                )
                user_pipeline.user_memory = self._pipeline.user_memory
                self._user_pipelines[student_id] = user_pipeline
                print(f"[OK] Pipeline legacy listo para '{student_id}'.")
                return user_pipeline
            except Exception as e:
                print(f"[WARN] Error cargando modelo legacy de '{student_id}': {e}.")
                shutil.rmtree(user_model_dir, ignore_errors=True)
                return self._pipeline

        return self._pipeline


class _LoraProxyPipeline:
    """Pipeline ligero que usa el adaptador LoRA del usuario para inferencia."""

    def __init__(self, base_pipeline, base_model, base_tokenizer, lora_dir: Path):
        self._base       = base_pipeline
        self._base_model = base_model
        self._base_tok   = base_tokenizer
        self._lora_dir   = lora_dir
        self.user_memory = getattr(base_pipeline, "user_memory", {})

    def correct(self, text: str, user_vocab: dict = None) -> list:
        from infrastructure.ml.t5_model import generate_with_lora
        try:
            suggestions = generate_with_lora(
                base_model=self._base_model,
                tokenizer=self._base_tok,
                lora_dir=self._lora_dir,
                text=text,
                num_returns=2,
            )
            if suggestions:
                return suggestions
        except Exception as e:
            print(f"[WARN] Inferencia LoRA falló: {e}. Usando pipeline base.")
        return self._base.correct(text, user_vocab)
