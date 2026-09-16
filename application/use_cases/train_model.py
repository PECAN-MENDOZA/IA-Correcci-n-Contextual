"""
application/use_cases/train_model.py

Caso de uso de entrenamiento del modelo base T5 (CLI: train.py).

El fine-tuning por alumno (TrainUserModelUseCase, adaptadores en
models/loras/<user_id>/) se eliminó: el servicio usa un único LoRA gramatical
global entrenado offline con train_grammar_lora.py.
"""
from infrastructure.ml.t5_model import T5SpanishTokenizer, T5CorrectionModel
from infrastructure.ml.dataset_loader import DatasetLoaderService


class TrainBaseModelUseCase:
    """Entrena el modelo base con datos del corpus."""

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
