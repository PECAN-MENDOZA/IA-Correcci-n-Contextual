"""
infrastructure/ml/dataset_loader.py
Adaptador de datos: descarga y prepara el corpus conversacional OPUS-100.
"""
import pandas as pd
from typing import List, Tuple
from pathlib import Path
from datasets import load_dataset

from domain.services.noise_injector import NoiseInjectorService


class DatasetLoaderService:
    """
    Descarga frases en español de OPUS-100 e inyecta ruido sintético
    para generar pares (texto_erróneo, texto_correcto) de entrenamiento.
    """

    def __init__(
        self,
        cache_dir: str = "./data/conversational",
        noise_injector: NoiseInjectorService = None,
    ):
        self._cache_dir     = Path(cache_dir)
        self._noise_injector = noise_injector or NoiseInjectorService()
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def load_pairs(self) -> List[Tuple[str, str]]:
        print("[INFO] Conectando a Hugging Face para descargar OPUS-100...")
        try:
            dataset = load_dataset("Helsinki-NLP/opus-100", "en-es", split="train", trust_remote_code=False)
            raw_sentences = set()
            for item in dataset:
                es_text = item["translation"]["es"].strip()
                if 15 <= len(es_text) <= 80:
                    raw_sentences.add(es_text)
                    if len(raw_sentences) >= 15_000:
                        break

            clean = list(raw_sentences)
            print(f"[OK] {len(clean)} frases conversacionales únicas obtenidas.")
        except Exception as exc:
            print(f"[ERROR] No se pudo descargar el dataset: {exc}")
            return []

        pairs: List[Tuple[str, str]] = []
        print("[INFO] Inyectando ruido sintético...")
        for text in clean:
            noisy = self._noise_injector.inject(text)
            if noisy != text:
                pairs.append((noisy, text))
            else:
                pairs.append((text.replace("v", "b").replace("s", "z").replace("h", ""), text))

        print(f"[OK] {len(pairs)} pares listos para entrenamiento.")
        return pairs

    def save_csv(self, pairs: List[Tuple[str, str]], output_path: str) -> None:
        df = pd.DataFrame(pairs, columns=["erronea", "corregida"])
        df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"[OK] CSV guardado en: {output_path}")
