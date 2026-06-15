"""
application/use_cases/train_model.py
Casos de uso de entrenamiento: base y personalizado por usuario.
"""
from typing import Optional

from domain.repositories.interfaces import IUserHistoryRepository
from infrastructure.ml.beto_model import BetoTokenizer, BETOModel
from infrastructure.ml.dataset_loader import DatasetLoaderService


class TrainBaseModelUseCase:
    """Entrena el modelo base con datos del corpus conversacional."""

    def __init__(self, loader: DatasetLoaderService, tokenizer: BetoTokenizer, model: BETOModel):
        self._loader = loader
        self._tokenizer = tokenizer
        self._model = model

    def execute(self, epochs: int = 3, batch_size: int = 2) -> None:
        print("=== INICIANDO PIPELINE DE ENTRENAMIENTO BASE ===")
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
    """Fine-tuning personalizado con el historial de un usuario específico."""

    def __init__(
        self,
        user_repo: IUserHistoryRepository,
        tokenizer: BetoTokenizer,
        base_model_dir: str = "./models/beto_correction",
    ):
        self._user_repo = user_repo
        self._tokenizer = tokenizer
        self._base_model_dir = base_model_dir

    def execute(self, user_id: str, epochs: int = 5) -> None:
        import shutil
        from pathlib import Path

        pairs = self._user_repo.get_user_pairs(user_id)
        if len(pairs) < 5:
            print(f"[WARN] El usuario '{user_id}' no tiene suficientes datos ({len(pairs)} pares).")
            return

        user_model_dir = f"./models/users/{user_id}"
        if not Path(user_model_dir).exists():
            Path(user_model_dir).parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(self._base_model_dir, user_model_dir)

        print(f"[INFO] Fine-tuning para '{user_id}' con {len(pairs)} pares.")
        model = BETOModel(save_dir=user_model_dir)
        train_dataset = self._tokenizer.build_hf_dataset(pairs)
        model.train(
            train_dataset=train_dataset,
            eval_dataset=None,
            tokenizer=self._tokenizer,
            epochs=epochs,
            batch_size=2,
            learning_rate=1e-5,
        )
        print(f"[OK] Modelo personalizado guardado en: {user_model_dir}")
