"""
train_user.py - CLI de Fine-Tuning por Usuario
Uso: python train_user.py --user_id usuario_dislexia_visual --epochs 5
"""
import argparse

from infrastructure.persistence.csv_user_repository import CsvUserHistoryRepository
from infrastructure.ml.beto_model import BetoTokenizer
from application.use_cases.train_model import TrainUserModelUseCase


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning personalizado por usuario")
    parser.add_argument("--user_id", type=str, required=True)
    parser.add_argument("--epochs",  type=int, default=5)
    args = parser.parse_args()

    user_repo = CsvUserHistoryRepository()
    tokenizer = BetoTokenizer()

    use_case = TrainUserModelUseCase(
        user_repo=user_repo,
        tokenizer=tokenizer,
        base_model_dir="./models/beto_correction",
    )
    use_case.execute(user_id=args.user_id, epochs=args.epochs)


if __name__ == "__main__":
    main()
