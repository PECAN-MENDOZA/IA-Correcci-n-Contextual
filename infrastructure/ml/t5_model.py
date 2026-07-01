"""
infrastructure/ml/t5_model.py

CORRECCIONES ACUMULADAS:

  v1 → v2 (sesión anterior):
    - evaluation_strategy='no' + predict_with_generate=False  [BUG #1]
    - DataCollatorForSeq2Seq padding=False                    [BUG #2]
    - bf16=False, fp16=False                                  [BUG #3]
    - gpu_lock en generate_corrections()                      [concurrencia]

  v2 → v3 (esta sesión):
    - BUG #5: T5CorrectionModel.__init__ ahora sobrescribe generation_config
      con valores conocidos-buenos DESPUÉS de cargar los pesos.
      Un modelo guardado por un Trainer crasheado puede tener
      generation_config.json con forced_bos_token_id=0 o
      decoder_start_token_id=None, lo que hace que el encoder
      reciba input_ids[0]=0 (pad) y dispare:
        TensorCompare.cu:110 `input[0] != 0`
      Esto explicaba por qué el crash ocurría en /corregir ANTES
      de cualquier fine-tuning, al cargar el modelo de usuario
      corrompido de una sesión anterior.

    - BUG #6 (preventivo): train() envuelve trainer.save_model() en
      try/finally + limpieza de directorio parcial si falla el guardado.
      Un save interrumpido dejaba modelos corruptos en disco que se
      cargaban en la siguiente sesión.
"""
import json
import shutil
import torch
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    GenerationConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset

try:
    from application.training_queue import gpu_lock
except ImportError:
    import threading
    gpu_lock = threading.Lock()

PRETRAINED_MODEL = "vgaraujov/t5-base-spanish"
DEFAULT_SAVE_DIR  = "./models/t5_correction"
MAX_INPUT_LEN     = 128
MAX_TARGET_LEN    = 128
TASK_PREFIX       = "corrige: "

# generation_config conocida-buena para T5.
# Se aplica SIEMPRE después de cargar pesos, sobreescribiendo
# cualquier generation_config.json corrupto que haya en el directorio.
_SAFE_GENERATION_CONFIG = {
    "decoder_start_token_id": 0,
    "eos_token_id": 1,
    "pad_token_id": 0,
    "max_new_tokens": MAX_TARGET_LEN,
}


class T5SpanishTokenizer:
    def __init__(self, model_name: str = PRETRAINED_MODEL):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode_single(self, text: str, max_length: int = MAX_INPUT_LEN) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            TASK_PREFIX + text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def decode(self, token_ids: torch.Tensor, skip_special: bool = True) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special)

    def build_hf_dataset(self, pairs: List[Tuple[str, str]]) -> Dataset:
        vocab_size = self.tokenizer.vocab_size
        sources    = [TASK_PREFIX + str(p[0]) for p in pairs]
        targets    = [str(p[1]) for p in pairs]

        def tokenize_fn(batch):
            model_inputs = self.tokenizer(
                batch["source"],
                max_length=MAX_INPUT_LEN,
                truncation=True,
                padding="max_length",
            )
            labels = self.tokenizer(
                text_target=batch["target"],
                max_length=MAX_TARGET_LEN,
                truncation=True,
                padding="max_length",
            )
            masked_labels = []
            for lbl in labels["input_ids"]:
                row = []
                for t in lbl:
                    if t == self.tokenizer.pad_token_id or t >= vocab_size:
                        row.append(-100)
                    else:
                        row.append(t)
                masked_labels.append(row)
            model_inputs["labels"] = masked_labels
            return model_inputs

        raw = Dataset.from_dict({"source": sources, "target": targets})
        return raw.map(tokenize_fn, batched=True, remove_columns=["source", "target"])


class T5CorrectionModel:
    def __init__(self, save_dir: Optional[str] = None):
        self.save_dir = Path(save_dir) if save_dir else Path(DEFAULT_SAVE_DIR)
        self.device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        load_kwargs = {"use_safetensors": True}

        if self.save_dir.exists() and (self.save_dir / "config.json").exists():
            self.model = AutoModelForSeq2SeqLM.from_pretrained(str(self.save_dir), **load_kwargs)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(PRETRAINED_MODEL, **load_kwargs)
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # ── FIX BUG #5: Sobreescribir generation_config con valores seguros ──
        # Un modelo guardado por un Trainer crasheado puede tener un
        # generation_config.json corrupto (forced_bos_token_id=0, etc.)
        # que hace que el encoder reciba input_ids[0]=0 y dispare
        # TensorCompare.cu `input[0] != 0`. Aplicamos la config segura
        # SIEMPRE, independientemente de lo que haya en disco.
        self.model.generation_config = GenerationConfig(**_SAFE_GENERATION_CONFIG)

        self.model.to(self.device)

    @staticmethod
    def validate_dir(model_dir: Path) -> bool:
        """
        Valida mínimamente que un directorio de modelo sea usable.
        Devuelve False si faltan archivos críticos o la config está corrupta.
        Usado por CorrectTextUseCase antes de cargar modelos de usuario.
        """
        required = ["config.json", "tokenizer_config.json"]
        if not all((model_dir / f).exists() for f in required):
            return False

        # Verificar que config.json sea JSON válido y tenga vocab_size
        try:
            with open(model_dir / "config.json") as f:
                cfg = json.load(f)
            if "vocab_size" not in cfg:
                return False
        except Exception:
            return False

        # Verificar que haya pesos (safetensors o pytorch_model.bin)
        has_weights = (
            list(model_dir.glob("*.safetensors")) or
            (model_dir / "pytorch_model.bin").exists()
        )
        return bool(has_weights)

    def generate_corrections(
        self,
        text: str,
        tokenizer: T5SpanishTokenizer,
        num_returns: int = 2,
    ) -> List[str]:
        """
        Genera versiones corregidas. Adquiere gpu_lock para exclusión
        mutua con el fine-tuning (nunca concurrencia en la misma GPU).
        """
        inputs         = tokenizer.encode_single(text)
        input_ids      = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Guardia: si el primer token es pad (0), el encoder fallará.
        # En lugar de crashear la GPU, devolvemos lista vacía para
        # que el pipeline caiga al fallback simbólico.
        if input_ids[0, 0].item() == 0:
            raise ValueError(
                f"input_ids[0][0]==0 (pad token): texto vacío o tokenizador corrupto. "
                f"Texto recibido: {repr(text[:80])}"
            )

        with gpu_lock:
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=MAX_TARGET_LEN,
                    num_return_sequences=num_returns,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                )

        return [tokenizer.decode(out, skip_special=True).strip() for out in outputs]

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset],
        tokenizer: T5SpanishTokenizer,
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 3e-4,
        save_dir_override: Optional[Path] = None,
    ) -> None:
        """
        Fine-tuning del modelo. El gpu_lock ya fue adquirido por TrainingWorker.
        Guarda en un directorio temporal y solo mueve al final si todo fue bien.
        Así un crash a mitad del guardado NO deja pesos corruptos en disco.
        """
        # NOTA: CUDA_LAUNCH_BLOCKING=1 fuerza el modo sincrono de CUDA (util solo para
        # depurar stacktraces). En produccion serializa todos los kernels y agrava el
        # bloqueo del worker durante el fine-tuning, por lo que se deja desactivado.
        # import os
        # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

        target_dir = save_dir_override or self.save_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        # Directorio temporal para el guardado seguro
        tmp_dir = target_dir.parent / f"{target_dir.name}_tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

        training_args = Seq2SeqTrainingArguments(
            output_dir=str(tmp_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=max(1, 16 // batch_size),
            learning_rate=learning_rate,
            warmup_steps=min(10, max(1, len(train_dataset) // 2)),
            weight_decay=0.01,
            eval_strategy="no",
            save_strategy="no",
            predict_with_generate=False,
            fp16=False,
            bf16=False,
            dataloader_pin_memory=True,
            report_to="none",
            logging_steps=5,
        )

        collator = DataCollatorForSeq2Seq(
            tokenizer.tokenizer,
            model=self.model,
            padding=False,
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=None,
            # `tokenizer=` es compatible con transformers 4.45.0 (el pin de requirements.txt).
            # El argumento `processing_class=` solo existe desde transformers 4.46.0 y provocaba
            # `Seq2SeqTrainer.__init__() got an unexpected keyword argument 'processing_class'`,
            # haciendo fallar TODO fine-tuning y colgando el servicio.
            tokenizer=tokenizer.tokenizer,
            data_collator=collator,
        )

        try:
            trainer.train()
            # Guardado atómico: save en tmp, luego reemplazar target
            trainer.save_model(str(tmp_dir))
            # Sobreescribir la generation_config antes de mover
            gen_cfg = GenerationConfig(**_SAFE_GENERATION_CONFIG)
            gen_cfg.save_pretrained(str(tmp_dir))

            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.move(str(tmp_dir), str(target_dir))
            print(f"[OK] Modelo guardado en: {target_dir}")

        except Exception:
            # Limpiar el directorio temporal corrupto
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            raise   # re-lanzar para que TrainingWorker lo registre

    def quantize_and_save(self, save_path: str) -> None:
        quantized = torch.quantization.quantize_dynamic(
            self.model, {torch.nn.Linear}, dtype=torch.qint8
        )
        torch.save(quantized.state_dict(), save_path)
        print(f"[OK] Modelo cuantizado guardado en: {save_path}")
