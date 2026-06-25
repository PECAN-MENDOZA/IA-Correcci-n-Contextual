"""
train.py - CLI de Entrenamiento Base (T5)
Uso: python train.py --epochs 3 --batch_size 4
"""
import argparse

from domain.services.noise_injector import NoiseInjectorService
from infrastructure.ml.dataset_loader import DatasetLoaderService
from infrastructure.ml.t5_model import T5SpanishTokenizer, T5CorrectionModel
from application.use_cases.train_model import TrainBaseModelUseCase


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento base del modelo T5 para corrección en español")
    parser.add_argument("--epochs",     type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    noise_injector = NoiseInjectorService()
    loader         = DatasetLoaderService(noise_injector=noise_injector)
    tokenizer      = T5SpanishTokenizer()
    model          = T5CorrectionModel(save_dir="./models/t5_correction")

    use_case = TrainBaseModelUseCase(loader=loader, tokenizer=tokenizer, model=model)
    use_case.execute(epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
