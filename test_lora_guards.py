"""
test_lora_guards.py

Tests de la guarda anti-alucinación compartida (infrastructure/ml/guards.py).
Corren SIN torch/transformers/peft/GPU porque guards.py es puro Python — a
propósito, para poder validar esta lógica en cualquier máquina (incluido
este sandbox, que no tiene acceso a Hugging Face ni GPU).

Uso:
    python3 test_lora_guards.py

NOTA: esto NO reemplaza probar el entrenamiento/inferencia real con el
modelo T5. Para eso, en un entorno con GPU y el modelo descargado, corre
train_user.py sobre un usuario de prueba y compara antes/después con
evaluate.py.
"""
import sys

from infrastructure.ml.guards import is_safe_refinement


def check(label: str, condition: bool) -> bool:
    status = "OK  " if condition else "FAIL"
    print(f"[{status}] {label}")
    return condition


def main() -> int:
    results = []

    # ── Casos que DEBEN aceptarse (variación conservadora) ──────────────
    results.append(check(
        "acepta un cambio de concordancia menor",
        is_safe_refinement("los niño juega en el parque", "los niños juegan en el parque"),
    ))
    results.append(check(
        "acepta corrección idéntica (sin cambios)",
        is_safe_refinement("hoy fui al mercado", "hoy fui al mercado"),
    ))
    results.append(check(
        "acepta una palabra corregida de 6",
        is_safe_refinement("el perro corre rapido por el parque", "el perro corre rápido por el parque"),
    ))

    # ── Casos que DEBEN rechazarse (alucinación / colapso) ───────────────
    results.append(check(
        "rechaza salida vacía",
        not is_safe_refinement("hoy fui al mercado", ""),
    ))
    results.append(check(
        "rechaza salida absurdamente larga (>1.5x)",
        not is_safe_refinement(
            "hoy fui al mercado",
            "hoy fui al mercado y luego pase por la casa de mi abuela y compre pan y leche",
        ),
    ))
    results.append(check(
        "rechaza salida absurdamente corta (<0.6x)",
        not is_safe_refinement("hoy fui al mercado a comprar pan y leche", "fui"),
    ))
    results.append(check(
        "rechaza cuando cambia casi todas las palabras (posible invención)",
        not is_safe_refinement(
            "el niño juega en el parque con su perro",
            "la señora compra fruta en el mercado central",
        ),
    ))
    results.append(check(
        "rechaza repetición degenerada (colapso típico de un LoRA roto)",
        not is_safe_refinement("hoy fui al mercado", "que que que que que que que que"),
    ))

    total, passed = len(results), sum(results)
    print(f"\n{passed}/{total} tests pasaron.")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
