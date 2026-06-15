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
    Dado un contexto (lista de palabras) y un índice, selecciona el candidato
    más probable según la distribución de probabilidad del MLM de BETO.
    """

    def __init__(self):
        print("[INFO] Inicializando Juez de Contexto (BETO MLM)...")
        self._tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
        self._model     = BertForMaskedLM.from_pretrained(MODEL_NAME)
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
        """Devuelve el candidato con mayor score en la posición enmascarada."""
        if len(candidates) == 1:
            return candidates[0]

        masked = context_words.copy()
        masked[target_index] = self._tokenizer.mask_token
        inputs = self._tokenizer(" ".join(masked), return_tensors="pt").to(self._device)

        with torch.no_grad():
            logits = self._model(**inputs).logits

        try:
            mask_idx = (
                inputs.input_ids[0] == self._tokenizer.mask_token_id
            ).nonzero(as_tuple=True)[0].item()
        except IndexError:
            return candidates[0]

        best, best_score = candidates[0], float("-inf")
        for cand in candidates:
            token_ids = self._tokenizer.encode(cand, add_special_tokens=False)
            if not token_ids:
                continue
            score = logits[0, mask_idx, token_ids[0]].item()
            if score > best_score:
                best_score, best = score, cand

        return best
