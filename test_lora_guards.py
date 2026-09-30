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

from infrastructure.ml.guards import (
    is_lexically_plausible_refinement, is_safe_refinement, is_same_word_variant,
    revert_lexical_substitutions,
)

# Clave fonética mínima para los tests (la real es phonetic_engine.to_phonetic).
_SOUND = lambda w: w.replace("h", "").replace("v", "b").replace("z", "s")
_KNOWN = set("la vaca comio pasto verde agarro alcancia que estaba arriba ver tele y leer un "
             "libro otra aula me empujo tiene madre ellos fue es muy grande vino ayer".split()).__contains__


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
    # ── Inserciones, borrados y bloques 1:n / n:1: cerrados salvo lista blanca ──
    results.append(check(
        "léxica: borrar un negador invierte el sentido (no quiero ir → quiero ir) y se bloquea",
        not is_lexically_plausible_refinement("no quiero ir", "quiero ir")
        and not is_lexically_plausible_refinement("nunca llego tarde", "llego tarde")
        and not is_lexically_plausible_refinement("no hay nadie en casa", "no hay en casa"),
    ))
    results.append(check(
        "léxica: insertar un negador o cuantificador se bloquea (quiero ir → no quiero ir)",
        not is_lexically_plausible_refinement("quiero ir", "no quiero ir")
        and not is_lexically_plausible_refinement("los niños juegan", "todos los niños juegan")
        and not is_lexically_plausible_refinement("ellos llegan tarde", "ellos siempre llegan tarde"),
    ))
    results.append(check(
        "léxica: insertar un artículo o preposición se permite (llego tarde a clase → llegué tarde a la clase)",
        is_lexically_plausible_refinement("llego tarde a clase", "llegué tarde a la clase")
        and is_lexically_plausible_refinement("fui parque", "fui al parque")
        and is_lexically_plausible_refinement("dijo vendría", "dijo que vendría"),
    ))
    results.append(check(
        "léxica: la unión / división con las mismas letras se permite (ala → a la, por que → porque)",
        is_lexically_plausible_refinement("voy ala escuela", "voy a la escuela")
        and is_lexically_plausible_refinement("no sé por que vino", "no sé porque vino")
        and is_lexically_plausible_refinement("voy a el parque", "voy al parque"),
    ))
    results.append(check(
        "léxica: borrar un duplicado adyacente se permite (muy muy bien → muy bien)",
        is_lexically_plausible_refinement("muy muy bien", "muy bien")
        and is_lexically_plausible_refinement("fui a a la casa", "fui a la casa"),
    ))
    results.append(check(
        "léxica: la puntuación suelta se puede insertar o borrar",
        is_lexically_plausible_refinement("hola como estas", "hola , como estas ?")
        and is_lexically_plausible_refinement("hola - como estas", "hola como estas"),
    ))
    results.append(check(
        "léxica: insertar o borrar una palabra de contenido se bloquea (fui parque → fui al gran parque)",
        not is_lexically_plausible_refinement("fui parque", "fui al gran parque")
        and not is_lexically_plausible_refinement("la vaca comió pasto verde", "la vaca comió pasto")
        and not is_lexically_plausible_refinement("voy a ir mañana", "iré mañana"),
    ))
    results.append(check(
        "léxica: la lista blanca no rescata un negador dentro de un bloque permitido",
        not is_lexically_plausible_refinement("quiero ir a clase", "no quiero ir a la clase")
        and not is_lexically_plausible_refinement("no voy a la casa", "voy a la casa"),
    ))
    # ── Negadores/cuantificadores también en reemplazos 1:1 (cierre) ────
    # Un reemplazo "cercano" que mete o saca una palabra protegida cambia el
    # sentido igual que insertarla o borrarla: lo/no (0.5), yo/no (0.5),
    # ni/mi (0.5) pasan la similitud y deben bloquearse igual.
    results.append(check(
        "léxica: sustituir 1:1 un negador por otra palabra se bloquea (no quiero → lo quiero)",
        not is_lexically_plausible_refinement("no quiero ir", "lo quiero ir")
        and not is_lexically_plausible_refinement("no quiero ir", "yo quiero ir")
        and not is_lexically_plausible_refinement("ni quiero ir", "mi quiero ir")
        and not is_lexically_plausible_refinement("nada me gusta", "cada me gusta")
        and not is_lexically_plausible_refinement("nunca voy", "nuca voy"),
    ))
    results.append(check(
        "léxica: sustituir 1:1 otra palabra por un negador se bloquea (lo quiero → no quiero)",
        not is_lexically_plausible_refinement("lo quiero ir", "no quiero ir")
        and not is_lexically_plausible_refinement("yo quiero ir", "no quiero ir")
        and not is_lexically_plausible_refinement("mi quiero ir", "ni quiero ir"),
    ))
    results.append(check(
        "léxica: dos protegidas distintas tampoco se intercambian (nunca→jamás, nada→nadie, no→ni)",
        not is_lexically_plausible_refinement("nunca voy", "jamás voy")
        and not is_lexically_plausible_refinement("nada vino", "nadie vino")
        and not is_lexically_plausible_refinement("no quiero", "ni quiero"),
    ))
    results.append(check(
        "léxica: la flexión de número/género de la misma protegida se permite (todos→todas, ningún→ninguna)",
        is_lexically_plausible_refinement("todos las niñas", "todas las niñas")
        and is_lexically_plausible_refinement("ningún niña vino", "ninguna niña vino")
        and is_lexically_plausible_refinement("ninguno niño vino", "ningún niño vino")
        and is_lexically_plausible_refinement("no hay ninguna", "no hay ninguno"),
    ))
    results.append(check(
        "léxica: la misma protegida a ambos lados (con tilde o puntuación) sigue pasando",
        is_lexically_plausible_refinement("no quiero ir", "No, quiero ir")
        and is_lexically_plausible_refinement("jamas voy", "jamás voy"),
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
        "léxica: la identidad pasa; el candidato vacío borra palabras de contenido y se bloquea",
        is_lexically_plausible_refinement("hoy fui al mercado", "hoy fui al mercado")
        and not is_lexically_plausible_refinement("hoy fui al mercado", ""),
    ))

    results.append(check(
        "léxica: perder la marca de 2.ª persona del pretérito se bloquea",
        not is_lexically_plausible_refinement("tú viniste conmigo al parque", "tú vino conmigo al parque")
        and not is_lexically_plausible_refinement("por qué no viniste ayer", "por qué no vino ayer")
        and not is_lexically_plausible_refinement("ayer jugaste muy bien", "ayer jugó muy bien"),
    ))
    results.append(check(
        "léxica: ganar la marca de 2.ª persona (concordancia) sigue pasando",
        is_lexically_plausible_refinement("ayer tú comio mucho", "ayer tú comiste mucho")
        and is_lexically_plausible_refinement("tú viniste conmigo", "tú viniste conmigo"),
    ))
    results.append(check(
        "léxica: las palabras en -ste que no son pretérito de 2.ª persona no se bloquean",
        is_lexically_plausible_refinement("los niños estan triste", "los niños están tristes")
        and is_lexically_plausible_refinement("los problemas existe", "los problemas existen"),
    ))

    # ── Reversión de sustituciones léxicas (2026-09-30) ─────────────────
    results.append(check(
        "variante: flexiones, homófonos e irregulares de ser/ir/haber son la misma palabra",
        all(is_same_word_variant(a, b, _SOUND) for a, b in [
            ("fue", "fueron"), ("vino", "vinieron"), ("sabe", "sepa"), ("feliz", "felices"),
            ("ves", "vez"), ("a", "ha"), ("tubo", "tuvo"), ("es", "son"), ("eres", "seas"),
            ("va", "vaya"), ("special", "especial"), ("los", "las"), ("confidencia", "confianza")]),
    ))
    results.append(check(
        "variante: sinónimos, paráfrasis y expansiones NO son la misma palabra",
        not any(is_same_word_variant(a, b, _SOUND) for a, b in [
            ("alcancia", "caja"), ("aula", "clase"), ("madre", "padre"), ("ropero", "armario"),
            ("tele", "television"), ("verdaderamente", "realmente"), ("se", "le")]),
    ))
    results.append(check(
        "variante (auditoría B): otra palabra con la misma inicial o cambio de tiempo NO es variante",
        not any(is_same_word_variant(a, b, _SOUND) for a, b in [
            ("casa", "cosa"), ("es", "fue"), ("madre", "padre")])
        and all(is_same_word_variant(a, b, _SOUND) for a, b in [
            ("vienes", "vengas"), ("das", "des"), ("tenia", "tuviera"), ("sabe", "sepa")]),
    ))
    results.append(check(
        "reversión: devuelve solo la palabra mal cambiada y conserva el resto del beam",
        revert_lexical_substitutions("agarro la alcancia que estaba arriba",
                                     "agarró la caja que estaba arriba", _KNOWN, _SOUND)
        == "agarró la alcancia que estaba arriba"
        and revert_lexical_substitutions("ver la tele y leer un libro",
                                         "ver la televisión y leer un libro", _KNOWN, _SOUND)
        == "ver la tele y leer un libro",
    ))
    results.append(check(
        "reversión: conserva concordancia y la puntuación/mayúsculas de la palabra devuelta",
        revert_lexical_substitutions("ellos fue muy grande", "ellos fueron muy grandes", _KNOWN, _SOUND)
        == "ellos fueron muy grandes"
        and revert_lexical_substitutions("Tiene madre, y vino ayer.", "Tiene padre, y vinieron ayer.",
                                         _KNOWN, _SOUND)
        == "Tiene madre, y vinieron ayer.",
    ))
    results.append(check(
        "reversión: palabra fuera del diccionario (falta) acepta un arreglo parecido y no uno lejano",
        revert_lexical_substitutions("me dicieron que", "me dijeron que", _KNOWN, _SOUND) == "me dijeron que"
        and revert_lexical_substitutions("un sanguche", "un viaje", _KNOWN, _SOUND) == "un sanguche",
    ))

    total, passed = len(results), sum(results)
    print(f"\n{passed}/{total} tests pasaron.")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
