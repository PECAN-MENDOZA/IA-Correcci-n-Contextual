"""
application/use_cases/train_model.py  v3 → v4 (LoRA)

TrainUserModelUseCase ahora usa LoRA en vez de copytree.
- ANTES: shutil.copytree(base → users/user_id/)  ~1 GB por usuario
- AHORA: base_model.train_lora(loras/user_id/)   ~10-50 MB por usuario

El modelo base se carga una sola vez (lazy singleton) y se reutiliza.
"""
import shutil
from typing import Optional, Callable
from pathlib import Path
import torch

from domain.repositories.interfaces import IUserHistoryRepository
from infrastructure.ml.t5_model import T5SpanishTokenizer, T5CorrectionModel
from infrastructure.ml.dataset_loader import DatasetLoaderService


class TrainBaseModelUseCase:
    """Entrena el modelo base con datos del corpus. Sin cambios."""

    def __init__(self, loader: DatasetLoaderService, tokenizer: T5SpanishTokenizer, model: T5CorrectionModel):
        self._loader    = loader
        self._tokenizer = tokenizer
        self._model     = model

    def execute(self, epochs: int = 3, batch_size: int = 4) -> None:
        print("=== INICIANDO ENTRENAMIENTO BASE (T5) ===")
        pairs = self._loader.load_pairs()
        if not pairs:
            print("[ERROR] Sin pares de entrenamiento.")
            return
        self._loader.save_csv(pairs, "./data/training_pairs.csv")
        train_dataset = self._tokenizer.build_hf_dataset(pairs)
        self._model.train(
            train_dataset=train_dataset,
            eval_dataset=None,
            tokenizer=self._tokenizer,
            epochs=epochs,
            batch_size=batch_size,
        )
        print("=== ENTRENAMIENTO COMPLETADO ===")


class TrainUserModelUseCase:
    """
    Fine-tuning LoRA personalizado por usuario.
    Solo guarda el adaptador (~10-50 MB) en models/loras/<user_id>/.
    El modelo base NO se copia ni se modifica.
    Requiere: pip install peft
    """

    MIN_PAIRS = 1

    def __init__(
        self,
        user_repo: IUserHistoryRepository,
        tokenizer: T5SpanishTokenizer,
        base_model_dir: str = "./models/t5_correction",
        loras_dir: str = "./models/loras",
        on_train_end: Optional[Callable[[str], None]] = None,
    ):
        self._user_repo      = user_repo
        self._tokenizer      = tokenizer
        self._base_model_dir = base_model_dir
        self._loras_dir      = Path(loras_dir)
        self._on_train_end   = on_train_end
        self._base_model: Optional[T5CorrectionModel] = None

    def _get_base_model(self) -> T5CorrectionModel:
        """Carga el modelo base solo la primera vez (lazy singleton)."""
        if self._base_model is None:
            base = Path(self._base_model_dir)
            if not T5CorrectionModel.validate_dir(base):
                raise RuntimeError(
                    f"Modelo base no válido en {self._base_model_dir}. "
                    "Ejecuta train.py primero."
                )
            print(f"[LoRA] Cargando modelo base desde {self._base_model_dir}...")
            self._base_model = T5CorrectionModel(save_dir=self._base_model_dir)
            print("[LoRA] Modelo base listo.")
        return self._base_model

    def execute(self, user_id: str, epochs: int = 10) -> None:
        pairs = self._user_repo.get_user_pairs(user_id)
        pairs = [p for p in pairs if isinstance(p[0], str) and isinstance(p[1], str)
                 and len(p[0]) > 2 and len(p[1]) > 2]

        if len(pairs) < self.MIN_PAIRS:
            print(f"[WARN] Sin pares válidos para '{user_id}'.")
            return

        lora_dir = self._loras_dir / user_id
        print(f"[LoRA] Entrenando adaptador para '{user_id}' con {len(pairs)} par(es).")
        print(f"[LoRA] Destino: {lora_dir}  (~10-50 MB, modelo base intacto)")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            base_model    = self._get_base_model()
            train_dataset = self._tokenizer.build_hf_dataset(pairs)
            base_model.train_lora(
                train_dataset=train_dataset,
                tokenizer=self._tokenizer,
                lora_save_dir=lora_dir,
                epochs=epochs,
                batch_size=1,
                learning_rate=3e-4,
            )
            print(f"[OK] Adaptador LoRA guardado en: {lora_dir}")
        except Exception as exc:
            tmp_dir = lora_dir.parent / f"{lora_dir.name}_tmp"
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            raise exc
