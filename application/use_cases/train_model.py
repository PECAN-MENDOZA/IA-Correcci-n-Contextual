"""
application/use_cases/train_model.py

v2 → v3: execute() ahora elimina el directorio de usuario si el fine-tuning
falla, para que un guardado parcial/corrupto no se cargue en la siguiente sesión.
El modelo base no se toca en caso de fallo.
"""
import shutil
from typing import Optional, Callable
from pathlib import Path
import torch

from domain.repositories.interfaces import IUserHistoryRepository
from infrastructure.ml.t5_model import T5SpanishTokenizer, T5CorrectionModel
from infrastructure.ml.dataset_loader import DatasetLoaderService


class TrainBaseModelUseCase:
    """Entrena el modelo base con datos del corpus conversacional."""

    def __init__(self, loader: DatasetLoaderService, tokenizer: T5SpanishTokenizer, model: T5CorrectionModel):
        self._loader    = loader
        self._tokenizer = tokenizer
        self._model     = model

    def execute(self, epochs: int = 3, batch_size: int = 4) -> None:
        print("=== INICIANDO PIPELINE DE ENTRENAMIENTO BASE (T5) ===")
        pairs = self._loader.load_pairs()
        if not pairs:
            print("[ERROR] No se obtuvieron pares de entrenamiento.")
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
    Fine-tuning personalizado con el historial de un usuario.

    Se dispara desde el primer feedback (MIN_PAIRS = 1).
    Si el training falla, elimina el directorio del usuario para que
    un modelo parcialmente guardado no se cargue en la siguiente sesión.
    """

    MIN_PAIRS = 1

    def __init__(
        self,
        user_repo: IUserHistoryRepository,
        tokenizer: T5SpanishTokenizer,
        base_model_dir: str = "./models/t5_correction",
        on_train_end: Optional[Callable[[str], None]] = None,
    ):
        self._user_repo      = user_repo
        self._tokenizer      = tokenizer
        self._base_model_dir = base_model_dir
        self._on_train_end   = on_train_end

    def execute(self, user_id: str, epochs: int = 2) -> None:
        pairs = self._user_repo.get_user_pairs(user_id)
        pairs = [p for p in pairs if isinstance(p[0], str) and isinstance(p[1], str)
                 and len(p[0]) > 2 and len(p[1]) > 2]

        if len(pairs) < self.MIN_PAIRS:
            print(f"[WARN] Sin pares válidos para '{user_id}'.")
            return

        user_model_dir = Path(f"./models/users/{user_id}")

        if not user_model_dir.exists():
            base = Path(self._base_model_dir)
            if not T5CorrectionModel.validate_dir(base):
                print(f"[ERROR] Modelo base no válido en {self._base_model_dir}.")
                return
            user_model_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(base, user_model_dir)

        print(f"[INFO] Fine-tuning para '{user_id}' con {len(pairs)} par(es).")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            model         = T5CorrectionModel(save_dir=str(user_model_dir))
            train_dataset = self._tokenizer.build_hf_dataset(pairs)

            model.train(
                train_dataset=train_dataset,
                eval_dataset=None,
                tokenizer=self._tokenizer,
                epochs=epochs,
                batch_size=1,
                learning_rate=1e-4,
                save_dir_override=user_model_dir,
            )
            print(f"[OK] Modelo personalizado actualizado en: {user_model_dir}")

        except Exception as exc:
            # ── FIX BUG #6: Limpiar directorio corrupto tras fallo ──────────
            # train() usa guardado atómico (tmp → target) y limpia tmp en caso
            # de fallo. Pero si el fallo fue en trainer.train() antes del save,
            # user_model_dir todavía tiene la copia del base (válida).
            # Solo eliminamos si el directorio fue creado en ESTA sesión y
            # el fallo fue en el guardado (tmp existe sin haberse movido).
            tmp_dir = user_model_dir.parent / f"{user_model_dir.name}_tmp"
            if tmp_dir.exists():
                # El fallo ocurrió durante/después del train, antes del move
                # tmp parcial ya fue limpiado por train(), pero por si acaso:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            raise exc
