"""
test_alternatives.py

Tests puros del selector de alternativas (infrastructure/nlp/alternatives.py)
y de la señal de ambigüedad en homófonos de `_disambiguate_context`.

Corren SIN torch/transformers/GPU: el selector solo usa difflib/re y la
guarda `is_safe_refinement`; para el pipeline se sustituyen los módulos
pesados (juez BETO y T5) por stubs, acotados a la clase de test con
`unittest.mock.patch.dict(sys.modules, ...)` para no contaminar otros módulos
de test del mismo proceso (test_modelo.py usa el juez y el T5 reales).

Uso:
    .venv\\Scripts\\python.exe -m unittest test_alternatives -v
"""
import importlib
import json
import os
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from infrastructure.nlp import alternatives
from infrastructure.nlp.alternatives import THRESHOLDS, select_alternatives

_REPO_DIR = Path(__file__).resolve().parent
_NO_TORCH_CHECK_ENV = "TESIS_NO_TORCH_CHECK_CHILD"


def run_class_in_clean_process(module_name: str, class_name: str) -> dict:
    """
    Ejecuta la clase de test indicada en un intérprete nuevo y devuelve
    {"ok": bool, "loaded": [módulos pesados presentes en sys.modules]}.
    Así la comprobación "no se cargó torch" no depende del orden de los tests
    del proceso actual (p. ej. `discover` con test_modelo.py antes).
    """
    code = (
        "import os, sys, unittest, importlib, json\n"
        f"mod = importlib.import_module({module_name!r})\n"
        f"suite = unittest.defaultTestLoader.loadTestsFromTestCase(getattr(mod, {class_name!r}))\n"
        "result = unittest.TextTestRunner(stream=open(os.devnull, 'w')).run(suite)\n"
        "loaded = [h for h in ('torch', 'transformers', 'peft') if h in sys.modules]\n"
        "print(json.dumps({'ok': result.wasSuccessful(), 'loaded': loaded}))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=str(_REPO_DIR),
        env={**os.environ, _NO_TORCH_CHECK_ENV: "1"},
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        raise AssertionError(f"el proceso hijo falló ({proc.returncode}):\n{proc.stderr[-2000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


class SelectAlternativesTests(unittest.TestCase):
    # ---- tests del plan original -------------------------------------------

    def test_returns_single_option_when_one_candidate_dominates(self):
        out = select_alternatives("el niño juega", "el niño juega",
                                  [("El niño juega.", -0.1), ("El niño jugaba.", -3.0)], score_margin=1.0)
        self.assertEqual(out, ["El niño juega."])

    def test_offers_distinct_interpretations_up_to_three(self):
        cands = [("Está bien.", -0.2), ("Esta bien.", -0.4), ("Está bien!", -0.5),
                 ("Estaba bien.", -0.6), ("Esta bien!", -0.7)]
        out = select_alternatives("esta bien", "esta bien", cands, max_options=3)
        self.assertEqual(out[0], "Está bien.")
        self.assertLessEqual(len(out), 3)
        self.assertNotIn("Está bien!", out)   # misma edición que la primera; solo cambia la puntuación final

    def test_drops_unsafe_and_identical_candidates(self):
        # El primer beam es la propia frase: la recomendada es la base (como
        # antes de la Task 5) y no se duplica; el candidato inseguro se descarta;
        # la lectura con tildes y signos se ofrece como alternativa.
        out = select_alternatives("hola como estas", "hola como estas",
                                  [("hola como estas", -0.1), ("Hola, ¿cómo estás?", -0.3),
                                   ("Adiós, hasta luego, nos vemos mañana temprano", -0.4)])
        self.assertEqual(out, ["hola como estas", "Hola, ¿cómo estás?"])

    # ---- recomendada estable (item 1 del fix brief) ------------------------

    def test_recommended_is_first_safe_beam_even_if_rule_e_would_reject_it(self):
        # Las capas 1-3 corrigieron "xq→porque" (similitud 0.25) y T5 heredó ese
        # tramo: la recomendada sigue siendo el mejor beam seguro (juega→juegan).
        # La base (juega, la forma del original) no es otra lectura: sin señal.
        out = select_alternatives("xq ellos juega mucho", "porque ellos juega mucho",
                                  [("porque ellos juegan mucho", -0.04)], score_margin=0.3)
        self.assertEqual(out, ["porque ellos juegan mucho"])

    def test_recommended_keeps_short_irregular_verb_corrections(self):
        # es/son tiene similitud 0.4 < 0.6: (e) no se aplica a la recomendada.
        out = select_alternatives("la gente son muy amables", "la gente son muy amables",
                                  [("la gente es muy amable", -0.05)], score_margin=0.3)
        self.assertEqual(out, ["la gente es muy amable"])
        # Comportamiento previo a la Task 5: el primer beam seguro se recomienda
        # aunque sea una lectura lejana; el segundo queda fuera por (d).
        out = select_alternatives("ellos fue al mercado", "ellos fue al mercado",
                                  [("él fue al mercado", -0.05), ("ellos fueron al mercado", -0.5)],
                                  score_margin=0.3)
        self.assertEqual(out, ["él fue al mercado"])

    def test_duplicate_token_deletion_is_a_valid_edit(self):
        # "muy muy bien → muy bien" como recomendada.
        out = select_alternatives("muy muy bien", "muy muy bien", [("muy bien", -0.1)], score_margin=0.3)
        self.assertEqual(out, ["muy bien"])
        # Como alternativa no: un borrado (duplicado o no) no es otra lectura
        # de la recomendada (señal (i): solo flexiones 1:1).
        out = select_alternatives("hoy muy muy bien", "hoy muy muy bien",
                                  [("Hoy muy muy bien.", -0.05), ("hoy muy bien", -0.1),
                                   ("muy muy bien", -0.12)], score_margin=0.3)
        self.assertEqual(out, ["Hoy muy muy bien."])

    def test_beto_variant_is_never_recommended_without_t5(self):
        # Sin T5 (o con excepción de T5) la recomendada es la base aunque sea
        # igual al original; la variante empatada de BETO va detrás.
        out = select_alternatives("esta bien", "esta bien", [],
                                  variants=[("está bien", -0.13)],
                                  score_margin=0.3, recommended="esta bien")
        self.assertEqual(out, ["esta bien", "está bien"])
        # Sin candidatos (juez ausente) solo queda la base.
        out = select_alternatives("esta bien", "esta bien", [], score_margin=0.3, recommended="esta bien")
        self.assertEqual(out, ["esta bien"])
        out = select_alternatives("esta bien", "esta bien", [], score_margin=0.3)
        self.assertEqual(out, ["esta bien"])

    def test_variants_are_never_deduced_as_recommended(self):
        # Sin `recommended` explícito, las segundas lecturas de BETO
        # (`variants=`) no pueden ocupar la posición 0 aunque sean seguras y
        # las únicas candidatas: la recomendada es la base.
        out = select_alternatives("esta bien", "esta bien", [],
                                  variants=[("está bien", -0.13)], score_margin=0.3)
        self.assertEqual(out, ["esta bien", "está bien"])
        # Con un beam, la recomendada es el beam; la variante sigue siendo una
        # extra ordenada por su score anclado.
        out = select_alternatives("esta bien", "esta bien", [("Esta bien.", -0.2)],
                                  variants=[("está bien", -0.1)], score_margin=0.3)
        self.assertEqual(out, ["Esta bien.", "está bien"])

    def test_deduced_recommended_is_beam_one_if_safe_else_base(self):
        # Sin `recommended`: la regla es "beam 1 si es seguro, si no la base";
        # nunca se salta al beam 2 para recomendar. El beam 2 seguro solo
        # podría ir detrás como alternativa, y aquí no lo hace: corregir la
        # concordancia (niño→niños, juega→juegan) no es otra lectura.
        unsafe = "los niño juega y corre y salta y canta todos los días en el parque"
        out = select_alternatives("los niño juega", "los niño juega",
                                  [(unsafe, -0.1), ("los niños juegan", -0.2)], score_margin=0.3)
        self.assertEqual(out[0], "los niño juega")
        self.assertNotIn(unsafe, out)
        self.assertEqual(out, ["los niño juega"])
        out = select_alternatives("los niño juega", "los niño juega",
                                  [("", -0.1), ("los niños juegan", -0.2)], score_margin=0.3)
        self.assertEqual(out[0], "los niño juega")

    def test_explicit_recommended_that_is_unsafe_falls_back_to_base(self):
        out = select_alternatives("hoy fui al mercado", "hoy fui al mercado",
                                  [("fui", -0.1)], score_margin=0.3, recommended="fui")
        self.assertEqual(out, ["hoy fui al mercado"])

    def test_identity_best_beam_recommends_base_not_next_beam(self):
        # Beams reales sobre una frase correcta: el mejor es la identidad, el
        # siguiente una paráfrasis. La recomendada es la base (no "iré") y la
        # paráfrasis no se ofrece ni con margen amplio (regla (e)).
        original = "mañana voy al parque con mis amigos"
        out = select_alternatives(original, original,
                                  [(original, -0.01), ("mañana iré al parque con mis amigos", -0.38),
                                   ("Mañana voy al parque con mis amigos", -0.41)], score_margin=1.0)
        self.assertEqual(out, [original])

    # ---- regla (c) por resultado de edición (item 2 del fix brief) ----------

    def test_indicative_and_subjunctive_are_two_readings(self):
        # Beams reales: juegan (recomendada) y jueguen son un contraste de modo
        # sobre la misma palabra corregida → dos lecturas. La base (juega, la
        # forma del original) no tiene señal y además es el indicativo tras
        # "me gusta que".
        original = "a mi me gusta que ellos juega mucho"
        base     = "a mí me gusta que ellos juega mucho"
        cands = [("a mí me gusta que ellos juegan mucho", -0.04),
                 ("a mí me gusta que ellos jueguen mucho", -0.18),
                 ("A mí me gusta que ellos juegan mucho", -0.372)]
        out = select_alternatives(original, base, cands, score_margin=0.3, lexicon={"jugar"})
        self.assertEqual(out, ["a mí me gusta que ellos juegan mucho",
                               "a mí me gusta que ellos jueguen mucho"])

    def test_same_edit_only_differing_in_final_punctuation_is_one_reading(self):
        # "Está bien!" es la misma edición que "Está bien." (regla (c));
        # "Esta bien." es la forma del original (sin señal); "Estaba bien." no
        # es una flexión de está (ni tilde ni modo).
        cands = [("Está bien.", -0.2), ("Esta bien.", -0.4), ("Está bien!", -0.5),
                 ("Estaba bien.", -0.6), ("Esta bien!", -0.7)]
        out = select_alternatives("esta bien", "esta bien", cands, max_options=3, score_margin=0.3)
        self.assertEqual(out, ["Está bien."])
        # Con dos ediciones distintas de verdad (tilde y modo), (c) sigue
        # colapsando solo la puntuación.
        cands = [("Está bien.", -0.2), ("Esté bien.", -0.25), ("Está bien!", -0.3), ("Esté bien!", -0.35)]
        out = select_alternatives("esta bien", "esta bien", cands, max_options=3, score_margin=0.3)
        self.assertEqual(out, ["Está bien.", "Esté bien."])

    # ---- señal positiva de ambigüedad (ola final, item C) --------------------

    def test_agreement_corrections_are_not_alternatives(self):
        # Número/persona los fija el sujeto: fue→fueron, juega→juegan,
        # lleguen→llegue nunca son "otra lectura" aunque sean flexiones cercanas.
        out = select_alternatives("ellos fue al mercado", "ellos fue al mercado",
                                  [("ellos fue al mercado", -0.06), ("ellos fueron al mercado", -0.306)],
                                  score_margin=0.3)
        self.assertEqual(out, ["ellos fue al mercado"])
        out = select_alternatives("es posible que ellos llega tarde", "es posible que ellos llega tarde",
                                  [("es posible que ellos lleguen tarde", -0.123),
                                   ("es posible que ellos llegue tarde", -0.2)],
                                  score_margin=0.3, lexicon={"llegar"})
        self.assertEqual(out, ["es posible que ellos lleguen tarde"])

    def test_accent_only_contrast_on_a_corrected_word_is_a_signal(self):
        # callo/calló, papa/papá, como/cómo: la otra lectura corrige una
        # palabra que la recomendada dejó (o cambió) y solo difiere en la tilde.
        out = select_alternatives("se callo de repente", "se callo de repente",
                                  [("se callo de repente", -0.114), ("se calló de repente", -0.234)],
                                  score_margin=0.3)
        self.assertEqual(out, ["se callo de repente", "se calló de repente"])
        out = select_alternatives("el papa llego temprano", "el papa llegó temprano",
                                  [("el papa llegó temprano", -0.084), ("el papá llegó temprano", -0.35)],
                                  score_margin=0.3)
        self.assertEqual(out, ["el papa llegó temprano", "el papá llegó temprano"])

    def test_variant_equal_to_the_base_carries_no_signal(self):
        # Caso real: BETO empató mi/mí y generó la variante con "mí", pero la
        # capa 3 (gustar) también puso "mí" en la base: la variante ES la base
        # (juega, forma del original) y no es una segunda lectura.
        original = "a mi me gusta que ellos juega mucho"
        base     = "a mí me gusta que ellos juega mucho"
        out = select_alternatives(original, base,
                                  [("a mí me gusta que ellos juegan mucho", -0.04)],
                                  variants=[(base, -0.04)], score_margin=0.3, lexicon={"jugar"},
                                  recommended="a mí me gusta que ellos juegan mucho")
        self.assertEqual(out, ["a mí me gusta que ellos juegan mucho"])

    def test_reverting_to_the_original_form_is_not_a_reading(self):
        # La base (o un beam) que solo vuelve a la forma del original en el
        # tramo que la recomendada corrigió no es otra lectura: el original
        # nunca es una corrección (regla (b), por tramo).
        out = select_alternatives("se que no vendra hoy", "se que no vendrá hoy",
                                  [("sé que no vendrá hoy", -0.015), ("se que no vendrá hoy", -0.1)],
                                  score_margin=0.3)
        self.assertEqual(out, ["sé que no vendrá hoy"])

    def test_insertion_or_deletion_against_the_recommendation_is_not_a_reading(self):
        out = select_alternatives("llego tarde a clase", "llego tarde a clase",
                                  [("llegué tarde a la clase", -0.164), ("llegué tarde a clase", -0.197)],
                                  score_margin=0.3)
        self.assertEqual(out, ["llegué tarde a la clase"])

    def test_mood_contrast_without_trigger_offers_both_moods(self):
        out = select_alternatives("quizás ellos llega tarde", "quizás ellos llega tarde",
                                  [("quizás ellos llegan tarde", -0.1), ("quizás ellos lleguen tarde", -0.2)],
                                  score_margin=0.3, lexicon={"llegar"})
        self.assertEqual(out, ["quizás ellos llegan tarde", "quizás ellos lleguen tarde"])
        # Verbos en -er/-ir: el subjuntivo lleva "a" (comen/coman).
        out = select_alternatives("mientras ellos come", "mientras ellos come",
                                  [("mientras ellos comen", -0.1), ("mientras ellos coman", -0.2)],
                                  score_margin=0.3, lexicon={"comer"})
        self.assertEqual(out, ["mientras ellos comen", "mientras ellos coman"])

    def test_obligatory_subjunctive_trigger_filters_the_indicative_variant(self):
        # es posible que → solo el subjuntivo; llegan no se ofrece (llegue tampoco: número).
        cands = [("es posible que ellos lleguen tarde", -0.123), ("es posible que ellos llegan tarde", -0.158),
                 ("es posible que ellos llegue tarde", -0.409)]
        out = select_alternatives("es posible que ellos llega tarde", "es posible que ellos llega tarde",
                                  cands, score_margin=0.3, lexicon={"llegar"})
        self.assertEqual(out, ["es posible que ellos lleguen tarde"])
        # ojalá (con o sin que): la base con el indicativo tampoco.
        out = select_alternatives("ojala que nosotros gana el partido", "ojalá que nosotros gana el partido",
                                  [("ojalá que nosotros gane el partido", -0.09),
                                   ("ojalá que nosotros gana el partido", -0.244)],
                                  score_margin=0.3, lexicon={"ganar"})
        self.assertEqual(out, ["ojalá que nosotros gane el partido"])
        # Recomendada en indicativo (error de T5): el subjuntivo sí puede ir detrás.
        out = select_alternatives("me alegra que ustedes esta aqui", "me alegra que ustedes esta aquí",
                                  [("me alegra que ustedes están aquí", -0.122),
                                   ("me alegra que ustedes estén aquí", -0.315)],
                                  score_margin=0.3, lexicon={"estar"})
        self.assertEqual(out, ["me alegra que ustedes están aquí", "me alegra que ustedes estén aquí"])
        # Verbo desconocido para el léxico: no se puede saber cuál es el
        # indicativo → tras el disparador no se ofrece el par de modo (fail-closed).
        out = select_alternatives("me alegra que ustedes esta aqui", "me alegra que ustedes esta aquí",
                                  [("me alegra que ustedes están aquí", -0.122),
                                   ("me alegra que ustedes estén aquí", -0.315)],
                                  score_margin=0.3, lexicon=set())
        self.assertEqual(out, ["me alegra que ustedes están aquí"])
        # El disparador solo actúa sobre el verbo que gobierna (después de él).
        out = select_alternatives("ellos llega antes de que salgamos", "ellos llega antes de que salgamos",
                                  [("ellos llegan antes de que salgamos", -0.1),
                                   ("ellos lleguen antes de que salgamos", -0.2)],
                                  score_margin=0.3, lexicon={"llegar"})
        self.assertEqual(out, ["ellos llegan antes de que salgamos", "ellos lleguen antes de que salgamos"])

    def test_subjunctive_triggers_are_the_documented_word_list(self):
        self.assertIn(("es", "posible", "que"), alternatives.SUBJUNCTIVE_TRIGGERS)
        self.assertIn(("ojala",), alternatives.SUBJUNCTIVE_TRIGGERS)
        self.assertIn(("no", "creo", "que"), alternatives.SUBJUNCTIVE_TRIGGERS)
        self.assertEqual(len(alternatives.SUBJUNCTIVE_TRIGGERS), 11)

    def test_mood_pair_detection_handles_spelling_changes(self):
        pair = alternatives._mood_pair
        self.assertEqual(pair("llegan", "lleguen"), ("a", "e"))
        self.assertEqual(pair("jueguen", "juegan"), ("e", "a"))
        self.assertEqual(pair("busca", "busque"), ("a", "e"))
        self.assertEqual(pair("cruza", "cruce"), ("a", "e"))
        self.assertEqual(pair("coge", "coja"), ("e", "a"))
        self.assertEqual(pair("estan", "esten"), ("a", "e"))
        self.assertIsNone(pair("juega", "juegan"))        # número
        self.assertIsNone(pair("vienen", "vengan"))       # irregular: raíz distinta
        self.assertIsNone(pair("va", "ve"))               # raíz demasiado corta
        self.assertIsNone(pair("como", "cómo"))
        self.assertEqual(alternatives._verb_class("lleg", "llegu", {"llegar"}), "ar")
        self.assertEqual(alternatives._verb_class("jueg", "juegu", {"jugar"}), "ar")
        self.assertEqual(alternatives._verb_class("com", "com", {"comer"}), "er")
        self.assertEqual(alternatives._verb_class("sig", "sigu", {"seguir"}), "er")
        self.assertIsNone(alternatives._verb_class("com", "com", {"comar", "comer"}))
        self.assertIsNone(alternatives._verb_class("xyz", "xyz", {"llegar"}))

    def test_paraphrase_with_tiny_gap_is_still_not_an_alternative(self):
        original = "mañana voy al parque con mis amigos"
        out = select_alternatives(original, original,
                                  [(original, -0.012), ("mañana iré al parque con mis amigos", -0.05)],
                                  score_margin=0.3)
        self.assertEqual(out, [original])

    def test_single_word_text_with_two_readings(self):
        out = select_alternatives("juega", "juega", [("juegan", -0.05), ("jueguen", -0.2)], score_margin=0.3)
        self.assertEqual(out, ["juegan", "jueguen"])

    # ---- resto de reglas ----------------------------------------------------

    def test_homophone_variant_offered_only_within_score_margin(self):
        original = "esta bien, nos vemos luego"
        base     = "esta bien, nos vemos luego"
        # T5 solo pone mayúscula y punto (no cambia la lectura de "esta").
        refined  = ("Esta bien, nos vemos luego.", -0.2)
        # Variante BETO "segunda lectura" empatada (score cerca de 0): se ofrece.
        out = select_alternatives(original, base,
                                  [refined, ("está bien, nos vemos luego", -0.3)], score_margin=1.0)
        self.assertEqual(out, ["Esta bien, nos vemos luego.", "está bien, nos vemos luego"])
        # La misma variante fuera del margen de score: no se ofrece.
        out = select_alternatives(original, base,
                                  [refined, ("está bien, nos vemos luego", -4.0)], score_margin=1.0)
        self.assertEqual(out, ["Esta bien, nos vemos luego."])
        # Si T5 ya eligió la lectura "está", la variante BETO es la misma edición
        # y no se duplica; aunque tenga mejor score, nunca es la recomendada.
        out = select_alternatives(original, base,
                                  [("Está bien, nos vemos luego.", -0.2),
                                   ("está bien, nos vemos luego", -0.1)], score_margin=1.0,
                                  recommended="Está bien, nos vemos luego.")
        self.assertEqual(out, ["Está bien, nos vemos luego."])

    def test_max_options_is_respected_with_five_valid_candidates(self):
        original = "el nino come pan y toma agua"
        cands = [
            ("el niño come pan y toma agua", -0.1),   # recomendada (ñ)
            ("el niño coma pan y toma agua", -0.2),   # modo en come
            ("el niño come pan y tome agua", -0.3),   # modo en toma
            ("el niño coma pan y tome agua", -0.4),   # modo en ambas
            ("el niño come pan y tomá agua", -0.5),   # tilde en toma
            ("el niño come pan y toma aguas", -0.6),  # número: sin señal
        ]
        out = select_alternatives(original, original, cands, max_options=3, score_margin=1.0,
                                  lexicon={"comer", "tomar"})
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0], "el niño come pan y toma agua")
        self.assertNotIn("el niño come pan y toma aguas", out)
        out = select_alternatives(original, original, cands, max_options=2, score_margin=1.0,
                                  lexicon={"comer", "tomar"})
        self.assertEqual(len(out), 2)
        out = select_alternatives(original, original, cands, max_options=5, score_margin=1.0,
                                  lexicon={"comer", "tomar"})
        self.assertEqual(len(out), 5)

    def test_base_is_kept_when_refined_differs_even_with_same_edits(self):
        # Refinado (T5) y base (reglas+BETO) corrigen la misma palabra con lecturas
        # distintas: la base nunca se elimina si el refinado difiere de ella.
        original = "ellos juega mucho"
        base     = "ellos juegan mucho"
        out = select_alternatives(original, base,
                                  [("ellos jueguen mucho", -0.2), (base, -0.2)], score_margin=1.0)
        self.assertEqual(out, ["ellos jueguen mucho", "ellos juegan mucho"])

    def test_base_identical_to_original_is_the_only_option(self):
        out = select_alternatives("el niño juega", "el niño juega",
                                  [("el niño juega", 0.0)])
        self.assertEqual(out, ["el niño juega"])

    def test_paraphrases_are_not_alternatives(self):
        out = select_alternatives("esta bien, nos vemos luego", "esta bien, nos vemos luego",
                                  [("está bien, nos vemos luego", -0.02),
                                   ("está bien, nos vemos después", -0.42)], score_margin=1.0)
        self.assertEqual(out, ["está bien, nos vemos luego"])
        # Una lectura cercana sí se ofrece (fue→fueron), una lejana no (ellos→él).
        out = select_alternatives("ellos fue al mercado", "ellos fue al mercado",
                                  [("ellos fueron al mercado", -0.3), ("él fue al mercado", -0.4)], score_margin=1.0)
        self.assertEqual(out, ["ellos fueron al mercado"])
        # Dos ediciones contiguas se juzgan palabra a palabra: "vendra→vendrá"
        # (1.0) no rescata a "hoy→ayer" (0.0); la base solo vuelve a "se".
        out = select_alternatives("se que no vendra hoy", "se que no vendrá hoy",
                                  [("sé que no vendrá hoy", -0.01), ("se que no vendrá ayer", -0.08)], score_margin=1.0)
        self.assertEqual(out, ["sé que no vendrá hoy"])

    def test_rule_e_is_measured_against_the_base_not_the_original(self):
        # La base ya corrigió "tb→también" (similitud 0.44 respecto al original);
        # una alternativa que solo cambia "juegan→jueguen" sobre la base pasa (e).
        out = select_alternatives("tb ellos juegan", "también ellos juegan",
                                  [("También ellos juegan.", -0.05), ("también ellos jueguen", -0.1)],
                                  score_margin=0.3)
        self.assertEqual(out, ["También ellos juegan.", "también ellos jueguen"])

    def test_thresholds_are_exposed_as_code_constants(self):
        self.assertEqual(THRESHOLDS, {"scoreMargin": 0.3, "minSpanSimilarity": 0.6,
                                      "maxFreqRatio": 20, "betoTieMargin": 0.3,
                                      "recommendedMinSimilarity": 0.3, "inflectionMinPrefix": 0.6})
        self.assertEqual(alternatives.MIN_SPAN_SIMILARITY, THRESHOLDS["minSpanSimilarity"])
        self.assertEqual(alternatives.SCORE_MARGIN, THRESHOLDS["scoreMargin"])
        self.assertEqual(alternatives.RECOMMENDED_MIN_SIMILARITY, THRESHOLDS["recommendedMinSimilarity"])
        self.assertEqual(alternatives.INFLECTION_MIN_PREFIX, THRESHOLDS["inflectionMinPrefix"])

    def test_edit_signature_keeps_accents_and_marks_insertions_and_deletions(self):
        sig = alternatives._edit_signature
        words = alternatives._words_for_edits
        self.assertEqual(sig(words("esta bien"), words("Está bien.")), frozenset({(0, 1, "está")}))
        self.assertEqual(sig(words("esta bien"), words("Esta bien!")), frozenset())
        deletion = sig(words("muy muy bien"), words("muy bien"))
        self.assertEqual(len(deletion), 1)
        self.assertEqual(next(iter(deletion))[2], "-")   # borrado marcado, sin palabra resultante
        self.assertEqual(sig(words("voy parque"), words("voy al parque")), frozenset({(1, 1, "+al")}))
        self.assertNotEqual(sig(words("ellos juega"), words("ellos juegan")),
                            sig(words("ellos juega"), words("ellos jueguen")))


# ---------------------------------------------------------------------------
# Prueba de aceptación de la calibración (scripts/calibrate_alternatives.py
# --check): lógica pura sobre resultados ya calculados, sin modelo.
# ---------------------------------------------------------------------------

class CalibrationCheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from scripts import calibrate_alternatives
        cls.calib = calibrate_alternatives

    def _case(self, text, expected, forbidden="", suggestions=()):
        case = {"text": text, "expected": expected, "expected_ambiguous": int(expected.split("-")[0]) >= 2,
                "forbidden": [f for f in forbidden.split(";") if f], "notes": ""}
        return self.calib.evaluate_case(case, list(suggestions))

    def test_csv_has_forbidden_readings_and_labels(self):
        cases = self.calib.load_cases(_REPO_DIR / "data" / "ambiguity_calibration.csv")
        self.assertEqual(len(cases), 39)
        by_text = {c["text"]: c for c in cases}
        self.assertIn("llegan", by_text["es posible que ellos llega tarde"]["forbidden"])
        self.assertIn("quizas", by_text["quizás ellos llega tarde"]["forbidden"])
        self.assertFalse(by_text["es posible que ellos llega tarde"]["expected_ambiguous"])
        self.assertEqual(sum(1 for c in cases if not c["expected_ambiguous"]), 20)

    def test_forbidden_readings_are_matched_on_alternatives_only(self):
        # La recomendada no se juzga (regla del beam 1); las alternativas sí,
        # por palabra, con tildes y sin puntuación exterior.
        case = self._case("es posible que ellos llega tarde", "1", "llega;llegan",
                          ["es posible que ellos lleguen tarde", "es posible que ellos llegan tarde."])
        self.assertEqual(case["forbidden_offered"], [("es posible que ellos llegan tarde.", "llegan")])
        case = self._case("es posible que ellos llega tarde", "1", "llega;llegan",
                          ["es posible que ellos llegan tarde"])
        self.assertEqual(case["forbidden_offered"], [])
        case = self._case("quizás ellos llega tarde", "2", "quizas",
                          ["quizás ellos llega tarde", "Quizas ellos llega tarde"])
        self.assertEqual(case["forbidden_offered"], [("Quizas ellos llega tarde", "quizas")])

    def test_check_fails_on_more_than_two_clear_fp_or_any_forbidden_reading(self):
        clear = [self._case(f"clara {i}", "1", "", [f"clara {i}."]) for i in range(20)]
        ambiguous = [self._case(f"ambigua {i}", "2", "", [f"ambigua {i}.", f"ambigua {i}!"]) for i in range(5)]
        summary = self.calib.summarize(clear + ambiguous)
        self.assertEqual((summary["tp"], summary["fp"], summary["fn"], summary["tn"]), (5, 0, 0, 20))
        self.assertTrue(summary["check"]["passed"])
        for i in range(2):
            clear[i]["offered_ambiguous"] = True
        summary = self.calib.summarize(clear + ambiguous)
        self.assertEqual(summary["fp"], 2)
        self.assertTrue(summary["check"]["passed"])            # FP ≤ 2/20 pasa
        clear[2]["offered_ambiguous"] = True
        summary = self.calib.summarize(clear + ambiguous)
        self.assertFalse(summary["check"]["passed"])           # FP 3/20 falla
        self.assertIn("FP", summary["check"]["reasons"][0])
        for i in range(3):
            clear[i]["offered_ambiguous"] = False
        ambiguous[0]["forbidden_offered"] = [("ambigua 0!", "ambigua")]
        summary = self.calib.summarize(clear + ambiguous)
        self.assertFalse(summary["check"]["passed"])
        self.assertIn("ambigua", summary["check"]["reasons"][0])
        self.assertEqual(self.calib.MAX_CLEAR_FP, 2)
        # El recall se informa pero no se exige.
        for case in ambiguous:
            case["offered_ambiguous"] = False
            case["forbidden_offered"] = []
        summary = self.calib.summarize(clear + ambiguous)
        self.assertEqual(summary["recall"], 0.0)
        self.assertTrue(summary["check"]["passed"])


# ---------------------------------------------------------------------------
# Integración ligera de _disambiguate_context (sin torch): stubs de los módulos
# pesados acotados a la clase + juez falso con puntuaciones fijas.
# ---------------------------------------------------------------------------

_MISSING = object()


def _stub_heavy_modules() -> dict:
    """Stubs de context_judge y t5_model (solo los nombres que importa el pipeline)."""
    judge_mod = types.ModuleType("infrastructure.nlp.context_judge")
    judge_mod.ContextJudge = type("ContextJudge", (), {})
    t5_mod = types.ModuleType("infrastructure.ml.t5_model")
    t5_mod.T5CorrectionModel = type("T5CorrectionModel", (), {})
    t5_mod.T5SpanishTokenizer = type("T5SpanishTokenizer", (), {})
    return {"infrastructure.nlp.context_judge": judge_mod,
            "infrastructure.ml.t5_model": t5_mod}


def import_isolated(test_class, module_name: str, stubs: dict):
    """
    Importa `module_name` de cero con `stubs` registrados en sys.modules y
    deja sys.modules y el atributo del paquete padre como estaban al terminar
    la clase de test (patch.dict + cleanups), para que otros módulos de test
    del mismo proceso vean los módulos reales.
    """
    patcher = patch.dict(sys.modules, stubs)
    patcher.start()
    test_class.addClassCleanup(patcher.stop)
    sys.modules.pop(module_name, None)
    parent_name, _, child = module_name.rpartition(".")
    parent = importlib.import_module(parent_name)
    previous = parent.__dict__.get(child, _MISSING)

    def restore_parent_attr():
        if previous is _MISSING:
            parent.__dict__.pop(child, None)
        else:
            setattr(parent, child, previous)

    test_class.addClassCleanup(restore_parent_attr)
    return importlib.import_module(module_name)


class FakePhonetic:
    def __init__(self, homophones, word_freqs=None):
        self._homophones = homophones
        self.word_freqs  = dict(word_freqs or {})

    def homophone_candidates(self, word, max_candidates=6):
        return list(self._homophones.get(word, []))


class FakeJudge:
    """Devuelve log-probs fijas por candidato (ordenadas de mejor a peor)."""

    def __init__(self, scores):
        self._scores = scores

    def score_candidates(self, context_words, target_index, candidates):
        scored = [(c, self._scores.get(c, -10.0)) for c in candidates]
        scored.sort(key=lambda cs: -cs[1])
        return scored


class DisambiguateContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = import_isolated(cls, "infrastructure.nlp.correction_pipeline", _stub_heavy_modules())

    def _pipeline(self, homophones, scores, word_freqs=None):
        pipe = self.mod.CorrectionPipeline.__new__(self.mod.CorrectionPipeline)
        pipe._phonetic = FakePhonetic(homophones, word_freqs)
        pipe._judge    = FakeJudge(scores)
        return pipe

    def test_no_torch_loaded(self):
        # Independiente del orden: esta clase entera (stubs + pipeline con
        # juez falso) corre en un intérprete limpio y no debe cargar torch.
        if os.environ.get(_NO_TORCH_CHECK_ENV):
            self.skipTest("ya se está comprobando desde el proceso padre")
        outcome = run_class_in_clean_process("test_alternatives", "DisambiguateContextTests")
        self.assertTrue(outcome["ok"], "la clase falló en el proceso limpio")
        self.assertEqual(outcome["loaded"], [], "módulos pesados cargados en un test puro")

    def test_pipeline_reads_thresholds_from_alternatives(self):
        self.assertEqual(self.mod._BETO_TIE_MARGIN, THRESHOLDS["betoTieMargin"])
        self.assertEqual(self.mod._SCORE_MARGIN, THRESHOLDS["scoreMargin"])
        self.assertEqual(self.mod._AMBIGUITY_MAX_FREQ_RATIO, THRESHOLDS["maxFreqRatio"])
        self.assertFalse(hasattr(self.mod, "HOMOPHONE_AMBIGUITY_RATIO"))

    def test_confident_reading_overwrites_without_variants(self):
        pipe = self._pipeline({"esta": ["está"]}, {"esta": -3.0, "está": -1.0})
        text, variants = pipe._disambiguate_context("esta bien")
        self.assertEqual(text, "está bien")
        self.assertEqual(variants, [])

    def test_tied_reading_keeps_text_and_offers_variant(self):
        # Diferencia 0.2 < _BETO_TIE_MARGIN 0.3: no se sobrescribe, pero se ofrece
        # la segunda lectura con score = -|diferencia| (más cerca de 0 = más empate).
        pipe = self._pipeline({"esta": ["está"]}, {"esta": -1.2, "está": -1.0})
        text, variants = pipe._disambiguate_context("esta bien")
        self.assertEqual(text, "esta bien")
        self.assertEqual(len(variants), 1)
        self.assertEqual(variants[0][0], "está bien")
        self.assertAlmostEqual(variants[0][1], -0.2, places=6)

    def test_current_best_but_second_within_tie_margin_offers_variant(self):
        pipe = self._pipeline({"esta": ["está"]}, {"esta": -1.0, "está": -1.2})
        text, variants = pipe._disambiguate_context("esta bien")
        self.assertEqual(text, "esta bien")
        self.assertEqual([v[0] for v in variants], ["está bien"])
        self.assertAlmostEqual(variants[0][1], -0.2, places=6)

    def test_below_overwrite_margin_but_above_tie_margin_offers_nothing(self):
        # tubo/tuvo dif 1.0: por debajo del margen de sobrescritura (5.0) pero
        # por encima del empate (0.3): ni se sobrescribe ni se ofrece.
        pipe = self._pipeline({"tubo": ["tuvo"]}, {"tubo": -3.0, "tuvo": -2.0})
        text, variants = pipe._disambiguate_context("el tubo un accidente")
        self.assertEqual(text, "el tubo un accidente")
        self.assertEqual(variants, [])
        # se/sé con los scores reales (dif 1.4): tampoco.
        pipe = self._pipeline({"se": ["sé"]}, {"se": -7.2, "sé": -5.8})
        text, variants = pipe._disambiguate_context("se que no vendrá hoy")
        self.assertEqual(text, "se que no vendrá hoy")
        self.assertEqual(variants, [])

    def test_clearly_current_reading_offers_nothing(self):
        pipe = self._pipeline({"esta": ["está"]}, {"esta": -1.0, "está": -4.0})
        text, variants = pipe._disambiguate_context("esta bien")
        self.assertEqual(text, "esta bien")
        self.assertEqual(variants, [])

    def test_at_most_three_positions_closest_ties_first(self):
        homophones = {"esta": ["está"], "tubo": ["tuvo"], "boy": ["voy"], "se": ["sé"]}
        scores = {"esta": -1.0, "está": -1.2,    # dif 0.2
                  "tubo": -1.0, "tuvo": -1.05,   # dif 0.05
                  "boy": -1.0, "voy": -1.25,     # dif 0.25
                  "se": -1.0, "sé": -1.1}        # dif 0.1
        pipe = self._pipeline(homophones, scores)
        text, variants = pipe._disambiguate_context("esta tubo boy se")
        self.assertEqual(text, "esta tubo boy se")
        self.assertEqual(len(variants), 3)
        self.assertEqual([v[0] for v in variants],
                         ["esta tuvo boy se", "esta tubo boy sé", "está tubo boy se"])

    def test_much_rarer_reading_is_not_an_ambiguity(self):
        # Empate de BETO (dif 0.07) entre "hoy" y "holly": la segunda es 33x más
        # rara que la escrita, no se ofrece. "sé" (3.4x más rara que "se") sí.
        freqs = {"hoy": 202178, "holly": 6071, "se": 2578296, "sé": 761934}
        pipe = self._pipeline({"hoy": ["holly"], "se": ["sé"]},
                              {"hoy": -8.16, "holly": -8.09, "se": -7.2, "sé": -7.0}, freqs)
        text, variants = pipe._disambiguate_context("se que no vendrá hoy")
        self.assertEqual(text, "se que no vendrá hoy")
        self.assertEqual([v[0] for v in variants], ["sé que no vendrá hoy"])
        # Sin datos de frecuencia no se descarta nada.
        pipe = self._pipeline({"hoy": ["holly"]}, {"hoy": -8.16, "holly": -8.09})
        _, variants = pipe._disambiguate_context("se que no vendrá hoy")
        self.assertEqual([v[0] for v in variants], ["se que no vendrá holly"])

    def test_punctuation_is_preserved_in_variants(self):
        pipe = self._pipeline({"esta": ["está"]}, {"esta,": -1.0, "está,": -1.2})
        text, variants = pipe._disambiguate_context("esta, bien")
        self.assertEqual(text, "esta, bien")
        self.assertEqual([v[0] for v in variants], ["está, bien"])

    def test_written_accent_is_trusted_never_overwritten_nor_offered_without_it(self):
        # quizás → quizas: BETO prefiere la variante sin tilde por más del
        # margen (0.5), pero una tilde ESCRITA por el alumno se respeta (como
        # en la capa 1): ni se sobrescribe ni se ofrece la variante sin tilde.
        pipe = self._pipeline({"quizás": ["quizas"]}, {"quizás": -3.0, "quizas": -1.0})
        text, variants = pipe._disambiguate_context("quizás ellos llegan tarde", trusted_accents={"quizás"})
        self.assertEqual(text, "quizás ellos llegan tarde")
        self.assertEqual(variants, [])
        pipe = self._pipeline({"quizás": ["quizas"]}, {"quizás": -1.0, "quizas": -1.1})
        text, variants = pipe._disambiguate_context("quizás ellos llegan tarde", trusted_accents={"quizás"})
        self.assertEqual(text, "quizás ellos llegan tarde")
        self.assertEqual(variants, [])
        # Una tilde que puso la capa 2 (restore_accent: papa → papá) no es del
        # alumno: BETO sigue pudiendo decidir por contexto (el papa = el Papa).
        pipe = self._pipeline({"papá": ["papa"]}, {"papá": -7.5, "papa": -5.3})
        text, variants = pipe._disambiguate_context("el papá llegó temprano", trusted_accents=set())
        self.assertEqual(text, "el papa llegó temprano")
        # Sin el argumento (llamada antigua) no se protege nada.
        self.assertEqual(pipe._disambiguate_context("el papá llegó temprano")[0], "el papa llegó temprano")
        # La palabra sin tilde sí puede recibirla (esta → está), como siempre.
        pipe = self._pipeline({"esta": ["está"]}, {"esta": -3.0, "está": -1.0})
        self.assertEqual(pipe._disambiguate_context("esta bien", trusted_accents=set())[0], "está bien")

    def test_mixed_case_token_does_not_produce_a_case_only_variant(self):
        # "eSta" no está entre los candidatos puntuados (match_case lo normaliza
        # a "esta"): antes salía la variante espuria "esta bien" con score -0.0.
        pipe = self._pipeline({"esta": ["está"]}, {"esta": -1.0, "está": -1.2})
        text, variants = pipe._disambiguate_context("eSta bien")
        self.assertEqual(text, "eSta bien")
        self.assertEqual(variants, [])


if __name__ == "__main__":
    unittest.main()
