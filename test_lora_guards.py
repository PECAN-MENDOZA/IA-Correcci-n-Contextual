"""
test_lora_guards.py

Tests de la guarda anti-alucinación compartida (infrastructure/ml/guards.py).
Corren SIN torch/transformers/peft/GPU porque guards.py es puro Python — a
propósito, para poder validar esta lógica en cualquier máquina (incluido
este sandbox, que no tiene acceso a Hugging Face ni GPU).

Uso:
    python3 test_lora_guards.py

NOTA: esto NO reemplaza probar el entrenamiento/inferencia real con el
modelo T5. Para eso, en un entorno con GPU y el modelo descargado, entrena el
LoRA gramatical global con train_grammar_lora.py y compara antes/después con
evaluate.py.
"""
import sys

from infrastructure.ml.guards import is_lexically_plausible_refinement, is_safe_refinement


def check(label: str, condition: bool) -> bool:
    status = "OK  " if condition else "FAIL"
    print(f"[{status}] {label}")
    return condition


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # consola cp1252 de Windows
    except Exception:
        pass
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

    # ── Guarda léxica del beam recomendado (solo reemplazos 1:1) ─────────
    # Una sustitución léxica (pasto→maíz 0.22, voy→iré 0.0) no es una
    # corrección; una flexión (es→son 0.40, hizo→hicieron 0.50,
    # viene→vengan 0.55) sí. Umbral 0.3 sobre la similitud sin tildes.
    results.append(check(
        "léxica: bloquea la sustitución léxica pasto→maíz (0.22)",
        not is_lexically_plausible_refinement("la vaca comió pasto verde", "la vaca comió maíz verde"),
    ))
    results.append(check(
        "léxica: bloquea la paráfrasis voy→iré (0.0)",
        not is_lexically_plausible_refinement("mañana voy al parque", "mañana iré al parque"),
    ))
    results.append(check(
        "léxica: conserva la concordancia irregular es→son (0.40)",
        is_lexically_plausible_refinement("la gente es muy amable", "la gente son muy amables"),
    ))
    results.append(check(
        "léxica: conserva hizo→hicieron (0.50) y viene→vengan (0.55)",
        is_lexically_plausible_refinement("ellos hizo la tarea", "ellos hicieron la tarea")
        and is_lexically_plausible_refinement("espero que ellos viene", "espero que ellos vengan"),
    ))
    results.append(check(
        "léxica: las tildes no cuentan (esta→está = 1.0) ni la puntuación pegada",
        is_lexically_plausible_refinement("esta bien, nos vemos", "Está bien, nos vemos.")
        and is_lexically_plausible_refinement("hola como estas", "Hola, ¿cómo estás?"),
    ))
    results.append(check(
        "léxica: inserciones, borrados y bloques 1:n / n:1 no se juzgan",
        is_lexically_plausible_refinement("llego tarde a clase", "llegué tarde a la clase")
        and is_lexically_plausible_refinement("muy muy bien", "muy bien")
        and is_lexically_plausible_refinement("voy a el parque", "voy al parque")
        and is_lexically_plausible_refinement("voy a ir mañana", "iré mañana"),
    ))
    results.append(check(
        "léxica: un bloque n:n se juzga palabra a palabra (niño→niños, juega→juegan)",
        is_lexically_plausible_refinement("los niño juega en el parque", "los niños juegan en el parque")
        and not is_lexically_plausible_refinement("los niño juega en el parque", "los niños comen en el parque"),
    ))
    results.append(check(
        "léxica: el umbral es configurable (pasto→maíz pasa con 0.2)",
        is_lexically_plausible_refinement("la vaca comió pasto verde", "la vaca comió maíz verde", min_similarity=0.2),
    ))
    results.append(check(
        "léxica: identidad y candidato vacío",
        is_lexically_plausible_refinement("hoy fui al mercado", "hoy fui al mercado")
        and is_lexically_plausible_refinement("hoy fui al mercado", ""),
    ))

    total, passed = len(results), sum(results)
    print(f"\n{passed}/{total} tests pasaron.")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
