"""
infrastructure/nlp/context_judge.py
Juez de contexto: usa BETO como Masked Language Model para desambiguar homófonos.
Extraído de inference_api.py — responsabilidad única de decisión contextual.
"""
from typing import List
import torch
from transformers import BertTokenizerFast, BertForMaskedLM

MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"


class ContextJudge:
    """
    Dado un contexto (lista de palabras) y un índice, selecciona los candidatos
    más probables evaluando la coherencia de toda la oración usando BETO.
    """

    def __init__(self):
        print("[INFO] Inicializando Juez de Contexto (BETO MLM)...")
        self._tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
        self._model     = BertForMaskedLM.from_pretrained(MODEL_NAME, use_safetensors=False)
        self._device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._model.eval()
        print(f"[OK] BETO listo en: {self._device}")

    def best_candidate(
            self,
            context_words: List[str],
            target_index: int,
            candidates: List[str],
    ) -> str:
        """Mantiene compatibilidad con el pipeline original usando rank_top_k_candidates."""
        return self.rank_top_k_candidates(context_words, target_index, candidates, k=1)[0]

    def rank_top_k_candidates(
            self,
            context_words: List[str],
            target_index: int,
            candidates: List[str],
            k: int = 3
    ) -> List[str]:
        """Devuelve los k mejores candidatos evaluando la coherencia de toda la oración."""
        if len(candidates) <= 1:
            return candidates

        scored_candidates = []
        for cand in candidates:
            test_sentence = context_words.copy()
            test_sentence[target_index] = cand
            text = " ".join(test_sentence)

            inputs = self._tokenizer(text, return_tensors="pt").to(self._device)

            with torch.no_grad():
                outputs = self._model(**inputs, labels=inputs.input_ids)
                loss = outputs.loss.item()

            scored_candidates.append((cand, loss))

        scored_candidates.sort(key=lambda x: x[1])
        return [cand for cand, score in scored_candidates][:k]