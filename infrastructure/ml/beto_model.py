"""
infrastructure/ml/beto_model.py
Adaptador ML: encapsula el modelo EncoderDecoder BETO y su tokenizador.
Depende de HuggingFace Transformers — detalle de infraestructura, no de dominio.
"""
import torch
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from transformers import (
    BertTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EncoderDecoderModel,
)
from datasets import Dataset

PRETRAINED_MODEL  = "dccuchile/bert-base-spanish-wwm-cased"
DEFAULT_SAVE_DIR  = "./models/beto_correction"
MAX_INPUT_LEN     = 128
MAX_TARGET_LEN    = 128


class BetoTokenizer:
    def __init__(self, model_name: str = PRETRAINED_MODEL):
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.tokenizer.bos_token = self.tokenizer.cls_token
        self.tokenizer.eos_token = self.tokenizer.sep_token

    def encode_single(self, text: str, max_length: int = MAX_INPUT_LEN) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            text, max_length=max_length, padding="max_length",
            truncation=True, return_tensors="pt"
        )

    def decode(self, token_ids: torch.Tensor, skip_special: bool = True) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special)

    def build_hf_dataset(self, pairs: List[Tuple[str, str]]) -> Dataset:
        sources = [p[0] for p in pairs]
        targets = [p[1] for p in pairs]

        def tokenize_fn(batch):
            model_inputs = self.tokenizer(
                batch["source"], max_length=MAX_INPUT_LEN, truncation=True, padding="max_length"
            )
            labels = self.tokenizer(
                text_target=batch["target"], max_length=MAX_TARGET_LEN,
                truncation=True, padding="max_length"
            )
            model_inputs["labels"] = [
                [-100 if t == self.tokenizer.pad_token_id else t for t in lbl]
                for lbl in labels["input_ids"]
            ]
            return model_inputs

        raw = Dataset.from_dict({"source": sources, "target": targets})
        return raw.map(tokenize_fn, batched=True, remove_columns=["source", "target"])


class BETOModel:
    def __init__(self, save_dir: Optional[str] = None):
        self.save_dir = Path(save_dir) if save_dir else Path(DEFAULT_SAVE_DIR)
        self.device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.save_dir.exists() and (self.save_dir / "config.json").exists():
            self.model = EncoderDecoderModel.from_pretrained(str(self.save_dir))
        else:
            self._init_fresh_model()

        self.model.to(self.device)

    def _init_fresh_model(self) -> None:
        self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(
            PRETRAINED_MODEL, PRETRAINED_MODEL, tie_encoder_decoder=True
        )
        self.model.config.decoder_start_token_id = 4
        self.model.config.pad_token_id            = 1
        self.model.config.eos_token_id            = 5
        self.model.config.vocab_size              = self.model.config.encoder.vocab_size

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset],
        tokenizer: BetoTokenizer,
        epochs: int = 3,
        batch_size: int = 2,
        learning_rate: float = 5e-5,
    ) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)

        training_args = Seq2SeqTrainingArguments(
            output_dir=str(self.save_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=100,
            weight_decay=0.01,
            save_strategy="epoch",
            predict_with_generate=True,
            fp16=torch.cuda.is_available(),
            report_to="none",
            logging_steps=50,
        )
        collator = DataCollatorForSeq2Seq(tokenizer.tokenizer, model=self.model, padding=True)
        trainer  = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=tokenizer.tokenizer,
            data_collator=collator,
        )
        trainer.train()
        trainer.save_model(str(self.save_dir))

    def quantize_and_save(self, save_path: str) -> None:
        """Cuantización dinámica: reduce el modelo de ~450MB a ~120MB (float32 → int8)."""
        quantized = torch.quantization.quantize_dynamic(
            self.model, {torch.nn.Linear}, dtype=torch.qint8
        )
        torch.save(quantized.state_dict(), save_path)
        print(f"[OK] Modelo cuantizado guardado en: {save_path}")
