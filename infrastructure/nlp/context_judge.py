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

    def score_candidates(
            self,
            context_words: List[str],
            target_index: int,
            candidates: List[str],
    ) -> List[tuple]:
        """
        Puntúa cada candidato por su log-probabilidad MEDIA bajo BETO (MLM) en
        la ranura `target_index`, enmascarando sus subtokens uno a uno
        (pseudo-log-likelihood). Devuelve [(candidato, logprob)] de mejor a peor.

        A diferencia de rank_top_k_candidates (que compara la loss de la frase
        COMPLETA), este método evalúa SOLO los tokens del candidato. Eso lo hace
        robusto ante typos raros / [UNK]: una palabra basura obtiene log-prob
        baja en vez de bajar artificialmente la loss promedio de la frase.
        """
        mask_id = self._tokenizer.mask_token_id
        cls_id  = self._tokenizer.cls_token_id
        sep_id  = self._tokenizer.sep_token_id
        scored  = []

        for cand in candidates:
            ids  = [cls_id]
            span = None
            for j, word in enumerate(context_words):
                piece = cand if j == target_index else word
                sub = self._tokenizer.encode(piece, add_special_tokens=False)
                if not sub:
                    sub = [self._tokenizer.unk_token_id]
                if j == target_index:
                    span = (len(ids), len(ids) + len(sub))
                ids.extend(sub)
            ids.append(sep_id)

            tensor = torch.tensor([ids], device=self._device)
            total  = 0.0
            for pos in range(span[0], span[1]):
                masked = tensor.clone()
                target = tensor[0, pos].item()
                masked[0, pos] = mask_id
                with torch.no_grad():
                    logits = self._model(masked).logits[0, pos]
                total += torch.log_softmax(logits, dim=-1)[target].item()

            scored.append((cand, total / max(span[1] - span[0], 1)))

        scored.sort(key=lambda cs: -cs[1])
        return scored

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