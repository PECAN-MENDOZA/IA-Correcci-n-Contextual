"""
test_train_gating.py

Verifica que TrainUserModelUseCase.execute() NO intenta entrenar (ni toca el
modelo) cuando un usuario tiene menos de MIN_PAIRS pares de feedback.

Este test inyecta stubs mínimos de torch/transformers/datasets en
sys.modules ANTES de importar el paquete, porque este sandbox no tiene esas
librerías instaladas (ni acceso a Hugging Face / GPU). Los stubs solo cubren
los símbolos que se tocan a nivel de import de infrastructure/ml/t5_model.py
— no permiten entrenar de verdad. Para validar entrenamiento/inferencia
reales, corre esto en tu entorno con GPU (ver README de train_user.py).

Uso:
    python3 test_train_gating.py
"""
import sys
import types


def _install_stubs() -> None:
    if "torch" in sys.modules:
        return  # ya instalado de verdad; no pisar

    torch_stub = types.ModuleType("torch")
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
    torch_stub.device = lambda *a, **k: "cpu"
    torch_stub.no_grad = lambda: _NullCtx()
    torch_stub.Tensor = type("Tensor", (), {})
    distributed_stub = types.ModuleType("torch.distributed")
    distributed_stub.is_initialized = lambda: True
    torch_stub.distributed = distributed_stub
    sys.modules["torch"] = torch_stub
    sys.modules["torch.distributed"] = distributed_stub

    transformers_stub = types.ModuleType("transformers")
    for name in (
        "AutoModelForSeq2SeqLM", "AutoTokenizer", "GenerationConfig",
        "Seq2SeqTrainer", "Seq2SeqTrainingArguments", "DataCollatorForSeq2Seq",
    ):
        setattr(transformers_stub, name, type(name, (), {}))
    sys.modules["transformers"] = transformers_stub

    datasets_stub = types.ModuleType("datasets")
    datasets_stub.Dataset = type("Dataset", (), {})
    datasets_stub.load_dataset = lambda *a, **k: None
    sys.modules["datasets"] = datasets_stub


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def check(label: str, condition: bool) -> bool:
    status = "OK  " if condition else "FAIL"
    print(f"[{status}] {label}")
    return condition


def main() -> int:
    _install_stubs()

    from application.use_cases.train_model import TrainUserModelUseCase

    results = []
    results.append(check(
        "MIN_PAIRS se subió de 1 a un umbral que evita memorizar (>=20)",
        TrainUserModelUseCase.MIN_PAIRS >= 20,
    ))

    class _FakeUserRepo:
        """Simula un usuario con muy pocos pares de feedback."""
        def get_user_pairs(self, user_id):
            return [("fui al mercado ayer", "fui al mercado ayer")] * 3  # 3 < MIN_PAIRS

    class _ExplodingTokenizer:
        """Si execute() llega a tocar el modelo, esto revienta el test
        (en vez de fallar silenciosamente o de intentar cargar un modelo
        real que no existe en este sandbox)."""
        def build_hf_dataset(self, pairs):
            raise AssertionError("No debería llegar a tokenizar: hay menos de MIN_PAIRS pares")

    use_case = TrainUserModelUseCase(
        user_repo=_FakeUserRepo(),
        tokenizer=_ExplodingTokenizer(),
        base_model_dir="./models/no_existe",
    )

    try:
        use_case.execute(user_id="alumno_de_prueba", epochs=1)
        gate_ok = True
    except AssertionError:
        gate_ok = False

    results.append(check(
        "con menos de MIN_PAIRS, execute() no intenta cargar/entrenar el modelo",
        gate_ok,
    ))
    results.append(check(
        "el modelo base sigue sin cargarse tras el intento (lazy singleton nunca se disparó)",
        use_case._base_model is None,
    ))

    total, passed = len(results), sum(results)
    print(f"\n{passed}/{total} tests pasaron.")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
