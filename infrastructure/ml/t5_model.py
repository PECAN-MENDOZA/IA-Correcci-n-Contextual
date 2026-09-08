"""
infrastructure/ml/t5_model.py  v3 → v4 (LoRA)

Cambios:
  - Rutas absolutas configuradas para evitar errores de directorio en Windows.
  - Lógica de safetensors adaptada: True para modelo local, False para descargas HF.
  - train_lora(): fine-tuning por usuario con PEFT/LoRA
    Guarda solo el adaptador (~10-50 MB) en models/loras/<user_id>/
    El modelo base NO se modifica ni se copia.
  - generate_with_lora(): inferencia con adaptador LoRA cargado en memoria
  - validate_lora_dir(): valida que el adaptador sea usable
  - train() intacto para el modelo base (train.py)
  - Fix Windows: USE_LIBUV=0 + no_cuda logic + ddp deshabilitado

Comparativa:
  Full fine-tuning: ~1 GB por usuario x 100 = ~100 GB
  LoRA:             ~10-50 MB por usuario x 100 = ~1-5 GB
"""
import torch.distributed.tensor
import os
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

# Fix Windows: NCCL no está disponible, forzar gloo o deshabilitar distributed
os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "12355")
os.environ["NCCL_DEBUG"] = "OFF"

# Esto es lo clave: deshabilitar NCCL completamente
import torch.distributed as dist
if not dist.is_initialized():
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"

try:
    # gpu_lock ahora solo serializa inferencias concurrentes entre sí
    # (generate_corrections / generate_with_lora). El TrainingWorker
    # (application/training_queue.py) ya NO lo usa: el entrenamiento
    # corre en su propia instancia de modelo, en paralelo con esto.
    from application.training_queue import gpu_lock
except ImportError:
    import threading
    gpu_lock = threading.Lock()

PRETRAINED_MODEL = "vgaraujov/t5-base-spanish"

BASE_DIR         = Path(__file__).resolve().parent.parent.parent
DEFAULT_SAVE_DIR = BASE_DIR / "models" / "t5_correction"

MAX_INPUT_LEN  = 128
MAX_TARGET_LEN = 128
TASK_PREFIX    = "corrige: "

_SAFE_GENERATION_CONFIG = {
    "decoder_start_token_id": 0,
    "eos_token_id": 1,
    "pad_token_id": 0,
    "max_new_tokens": MAX_TARGET_LEN,
}

LORA_CONFIG = {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "bias": "none",
    "task_type": "SEQ_2_SEQ_LM",
    "target_modules": ["q", "v"],
}

# Learning rate conservador para fine-tuning por-usuario. 3e-4 (el valor
# anterior) es agresivo para datasets de 20-100 pares por alumno; con esto
# el adaptador converge más lento pero no destruye la fluidez del modelo
# base. Ver train_lora() y TrainUserModelUseCase.execute().
DEFAULT_LORA_LR = 1e-4

# Guarda anti-alucinación compartida (T5 base y LoRA). Vive en guards.py,
# sin dependencias de torch, para poder testearla sin GPU/modelo real.
from infrastructure.ml.guards import is_safe_refinement  # noqa: E402,F401

# ── Detectar si hay GPU real disponible ──────────────────────────────────────
_HAS_GPU = torch.cuda.is_available()


def _training_args_base(output_dir: str, epochs: int, batch_size: int,
                         learning_rate: float, warmup_steps: int) -> Seq2SeqTrainingArguments:
    """
    Crea TrainingArguments comunes para train() y train_lora().
    no_cuda=False cuando hay GPU real, True cuando no hay.
    use_cpu evita que accelerate intente iniciar torch.distributed en Windows.
    """
    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        eval_strategy="no",
        save_strategy="no",
        predict_with_generate=False,
        fp16=False,
        bf16=False,
        dataloader_pin_memory=False,
        report_to="none",
        logging_steps=5,
        # ── Fix Windows distributed ──────────────────────────────────────────
        no_cuda=False,           # dejar que use GPU si hay
        use_cpu=not _HAS_GPU,   # fuerza CPU solo si no hay GPU
        local_rank=-1,           # deshabilita DDP completamente
        ddp_backend=None,        # sin backend distributed
        gradient_accumulation_steps=1,
    )


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
    """Modelo base compartido. Se carga UNA vez para todos los usuarios."""

    def __init__(self, save_dir: Optional[str] = None):
        self.save_dir = Path(save_dir) if save_dir else DEFAULT_SAVE_DIR
        self.device   = torch.device("cuda" if _HAS_GPU else "cpu")

        if self.save_dir.exists() and (self.save_dir / "config.json").exists():
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                str(self.save_dir), use_safetensors=True
            )
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                PRETRAINED_MODEL, use_safetensors=False
            )
            self.save_dir.mkdir(parents=True, exist_ok=True)

        self.model.generation_config = GenerationConfig(**_SAFE_GENERATION_CONFIG)
        self.model.to(self.device)

    @staticmethod
    def validate_dir(model_dir: Path) -> bool:
        required = ["config.json", "tokenizer_config.json"]
        if not all((model_dir / f).exists() for f in required):
            return False
        try:
            with open(model_dir / "config.json") as f:
                cfg = json.load(f)
            if "vocab_size" not in cfg:
                return False
        except Exception:
            return False
        has_weights = (
            list(model_dir.glob("*.safetensors")) or
            (model_dir / "pytorch_model.bin").exists()
        )
        return bool(has_weights)

    @staticmethod
    def validate_lora_dir(lora_dir: Path) -> bool:
        if not lora_dir.exists():
            return False
        if not (lora_dir / "adapter_config.json").exists():
            return False
        has_weights = (
            (lora_dir / "adapter_model.safetensors").exists() or
            (lora_dir / "adapter_model.bin").exists()
        )
        return has_weights

    def _clean_peft_if_attached(self) -> None:
        """Limpia cualquier adaptador LoRA que haya quedado pegado al modelo base."""
        try:
            if hasattr(self.model, "peft_config"):
                self.model = self.model.get_base_model()
                self.model.generation_config = GenerationConfig(**_SAFE_GENERATION_CONFIG)
                self.model.to(self.device)
        except Exception:
            pass

    def generate_corrections(self, text: str, tokenizer: T5SpanishTokenizer,
                              num_returns: int = 2) -> List[str]:
        inputs         = tokenizer.encode_single(text)
        input_ids      = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        if input_ids[0, 0].item() == 0:
            raise ValueError(f"input_ids[0][0]==0: texto vacío. Texto: {repr(text[:80])}")

        with gpu_lock:
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=MAX_TARGET_LEN,
                    num_beams=4,
                    num_return_sequences=min(num_returns, 4),
                    do_sample=False,
                    early_stopping=True,
                )
        return [tokenizer.decode(out, skip_special=True).strip() for out in outputs]

    def train(self, train_dataset, eval_dataset, tokenizer, epochs=3, batch_size=4,
              learning_rate=3e-4, save_dir_override=None):
        """Fine-tuning completo del modelo base. Solo para train.py."""
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

        target_dir = save_dir_override or self.save_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir = target_dir.parent / f"{target_dir.name}_tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

        warmup = min(10, max(1, len(train_dataset) // 2))
        training_args = _training_args_base(
            str(tmp_dir), epochs, batch_size, learning_rate, warmup
        )
        # Para el base usamos gradient_accumulation más alto
        training_args.gradient_accumulation_steps = max(1, 16 // batch_size)

        collator = DataCollatorForSeq2Seq(tokenizer.tokenizer, model=self.model, padding=False)
        trainer  = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=None,
            tokenizer=tokenizer.tokenizer,
            data_collator=collator,
        )
        try:
            trainer.train()
            trainer.save_model(str(tmp_dir))
            GenerationConfig(**_SAFE_GENERATION_CONFIG).save_pretrained(str(tmp_dir))
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.move(str(tmp_dir), str(target_dir))
            print(f"[OK] Modelo base guardado en: {target_dir}")
        except Exception:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

    def train_lora(self, train_dataset, tokenizer, lora_save_dir: Path,
                   epochs=3, batch_size=1, learning_rate=DEFAULT_LORA_LR):
        """
        Fine-tuning LoRA por usuario SIN Seq2SeqTrainer.
        Loop manual con PyTorch puro para evitar torch.distributed/NCCL en Windows.
        Solo guarda el adaptador (~10-50 MB) en lora_save_dir.
        """
        from torch.utils.data import DataLoader
        from torch.optim import AdamW

        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError:
            raise ImportError("PEFT no instalado. Ejecuta: pip install peft")

        # Limpiar adaptador previo si quedó pegado
        self._clean_peft_if_attached()

        tmp_dir = lora_save_dir.parent / f"{lora_save_dir.name}_tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=LORA_CONFIG["r"],
            lora_alpha=LORA_CONFIG["lora_alpha"],
            lora_dropout=LORA_CONFIG["lora_dropout"],
            bias=LORA_CONFIG["bias"],
            target_modules=LORA_CONFIG["target_modules"],
        )
        peft_model = get_peft_model(self.model, peft_config)
        trainable, total = peft_model.get_nb_trainable_parameters()
        print(f"[LoRA] Params entrenables: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

        peft_model.to(self.device)
        peft_model.train()

        # Convertir dataset HuggingFace a formato PyTorch
        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        optimizer = AdamW(peft_model.parameters(), lr=learning_rate, weight_decay=0.01)

        try:
            for epoch in range(epochs):
                total_loss = 0.0
                for step, batch in enumerate(loader):
                    input_ids      = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels         = batch["labels"].to(self.device)

                    optimizer.zero_grad()
                    outputs = peft_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                avg_loss = total_loss / max(len(loader), 1)
                print(f"[LoRA] Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

            # Guardar solo el adaptador
            peft_model.save_pretrained(str(tmp_dir))
            if lora_save_dir.exists():
                shutil.rmtree(lora_save_dir)
            shutil.move(str(tmp_dir), str(lora_save_dir))
            print(f"[LoRA] Adaptador guardado en: {lora_save_dir}")

        except Exception:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

        finally:
            # Desacoplar LoRA del modelo base siempre
            try:
                peft_model.unload()
            except Exception:
                pass
            self.model = peft_model.get_base_model()
            self.model.generation_config = GenerationConfig(**_SAFE_GENERATION_CONFIG)
            self.model.to(self.device)

    def quantize_and_save(self, save_path: str) -> None:
        quantized = torch.quantization.quantize_dynamic(
            self.model, {torch.nn.Linear}, dtype=torch.qint8
        )
        torch.save(quantized.state_dict(), save_path)
        print(f"[OK] Modelo cuantizado guardado en: {save_path}")


def generate_with_lora(base_model: T5CorrectionModel, tokenizer: T5SpanishTokenizer,
                        lora_dir: Path, text: str, num_returns: int = 2) -> List[str]:
    try:
        from peft import PeftModel
    except ImportError:
        raise ImportError("PEFT no instalado. Ejecuta: pip install peft")

    # Limpiar adaptador previo si quedó pegado
    base_model._clean_peft_if_attached()

    device         = base_model.device
    inputs         = tokenizer.encode_single(text)
    input_ids      = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    if input_ids[0, 0].item() == 0:
        raise ValueError(f"input_ids[0][0]==0: texto vacío.")

    with gpu_lock:
        peft_model = PeftModel.from_pretrained(
            base_model.model, str(lora_dir), is_trainable=False
        )
        peft_model.to(device)
        peft_model.eval()

        with torch.no_grad():
            # Determinista (beam search), igual que generate_corrections() para
            # el T5 base. El muestreo estocástico (do_sample=True + temperature/
            # top_p) que había antes aquí era la principal fuente de respuestas
            # no reproducibles y "inventadas": con un adaptador entrenado sobre
            # pocos ejemplos por usuario, samplear con temperatura alta amplifica
            # cualquier ruido del fine-tuning en vez de suavizarlo.
            outputs = peft_model.generate(
                input_ids=input_ids,          # ← keyword argument, no posicional
                attention_mask=attention_mask,
                max_new_tokens=MAX_TARGET_LEN,
                num_beams=max(4, num_returns),
                num_return_sequences=num_returns,
                do_sample=False,
                early_stopping=True,
            )

        # Desacoplar adaptador y restaurar modelo base limpio
        try:
            peft_model.unload()
        except Exception:
            pass
        base_model.model = peft_model.get_base_model()
        base_model.model.generation_config = GenerationConfig(**_SAFE_GENERATION_CONFIG)
        base_model.model.to(device)

    return [tokenizer.decode(out, skip_special=True).strip() for out in outputs]