"""
train.py - CLI de Entrenamiento Base
Uso: python train.py --epochs 3 --batch_size 2
"""
import argparse

from domain.services.noise_injector import NoiseInjectorService
from infrastructure.ml.dataset_loader import DatasetLoaderService
from infrastructure.ml.beto_model import BetoTokenizer, BETOModel
from application.use_cases.train_model import TrainBaseModelUseCase


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento base del modelo BETO")
    parser.add_argument("--epochs",     type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    noise_injector = NoiseInjectorService()
    loader         = DatasetLoaderService(noise_injector=noise_injector)
    tokenizer      = BetoTokenizer()
    model          = BETOModel(save_dir="./models/beto_correction")

    use_case = TrainBaseModelUseCase(loader=loader, tokenizer=tokenizer, model=model)
    use_case.execute(epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
