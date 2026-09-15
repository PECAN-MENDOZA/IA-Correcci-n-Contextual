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
        # tramo: la recomendada sigue siendo el mejor beam seguro (juega→juegan)
        # y la base queda como alternativa. Antes del fix (e) tiraba el beam.
        out = select_alternatives("xq ellos juega mucho", "porque ellos juega mucho",
                                  [("porque ellos juegan mucho", -0.04)], score_margin=0.3)
        self.assertEqual(out, ["porque ellos juegan mucho", "porque ellos juega mucho"])

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
        # Como alternativa, borrar un token igual a su vecino pasa (e)
        # (similitud 1.0); borrar cualquier otra palabra no.
        out = select_alternatives("hoy muy muy bien", "hoy muy muy bien",
                                  [("Hoy muy muy bien.", -0.05), ("hoy muy bien", -0.1),
                                   ("muy muy bien", -0.12)], score_margin=0.3)
        self.assertEqual(out, ["Hoy muy muy bien.", "hoy muy bien"])

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
        # nunca se salta al beam 2 para recomendar. El beam 2 seguro puede
        # ofrecerse detrás de la base como alternativa.
        unsafe = "los niño juega y corre y salta y canta todos los días en el parque"
        out = select_alternatives("los niño juega", "los niño juega",
                                  [(unsafe, -0.1), ("los niños juegan", -0.2)], score_margin=0.3)
        self.assertEqual(out[0], "los niño juega")
        self.assertNotIn(unsafe, out)
        self.assertEqual(out, ["los niño juega", "los niños juegan"])
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
        original = "a mi me gusta que ellos juega mucho"
        base     = "a mí me gusta que ellos juega mucho"
        cands = [("a mí me gusta que ellos juegan mucho", -0.04),
                 ("a mí me gusta que ellos jueguen mucho", -0.18),
                 ("A mí me gusta que ellos juegan mucho", -0.372)]
        out = select_alternatives(original, base, cands, score_margin=0.3)
        self.assertEqual(out, ["a mí me gusta que ellos juegan mucho", base,
                               "a mí me gusta que ellos jueguen mucho"])

    def test_same_edit_only_differing_in_final_punctuation_is_one_reading(self):
        cands = [("Está bien.", -0.2), ("Esta bien.", -0.4), ("Está bien!", -0.5),
                 ("Estaba bien.", -0.6), ("Esta bien!", -0.7)]
        out = select_alternatives("esta bien", "esta bien", cands, max_options=3, score_margin=0.3)
        self.assertEqual(out, ["Está bien.", "Esta bien."])

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
            ("el niño come pan y toma agua", -0.1),   # edita 1
            ("el nino comió pan y toma agua", -0.2),  # edita 2
            ("el nino come pan y tomó agua", -0.3),   # edita 5
            ("el nino come pan y toma aguas", -0.4),  # edita 6
            ("el nino come pan, y toma agua", -0.5),  # edita 3
        ]
        out = select_alternatives(original, original, cands, max_options=3, score_margin=1.0)
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0], "el niño come pan y toma agua")
        out = select_alternatives(original, original, cands, max_options=2, score_margin=1.0)
        self.assertEqual(len(out), 2)

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
        # (1.0) no rescata a "hoy→ayer" (0.0).
        out = select_alternatives("se que no vendra hoy", "se que no vendrá hoy",
                                  [("sé que no vendrá hoy", -0.01), ("se que no vendrá ayer", -0.08)], score_margin=1.0)
        self.assertEqual(out, ["sé que no vendrá hoy", "se que no vendrá hoy"])

    def test_rule_e_is_measured_against_the_base_not_the_original(self):
        # La base ya corrigió "tb→también" (similitud 0.44 respecto al original);
        # una alternativa que solo cambia "juega→juegan" sobre la base pasa (e).
        out = select_alternatives("tb ellos juega", "también ellos juega",
                                  [("También ellos juega.", -0.05), ("también ellos juegan", -0.1)],
                                  score_margin=0.3)
        self.assertEqual(out, ["También ellos juega.", "también ellos juegan"])

    def test_thresholds_are_exposed_as_code_constants(self):
        self.assertEqual(THRESHOLDS, {"scoreMargin": 0.3, "minSpanSimilarity": 0.6,
                                      "maxFreqRatio": 20, "betoTieMargin": 0.3,
                                      "recommendedMinSimilarity": 0.3})
        self.assertEqual(alternatives.MIN_SPAN_SIMILARITY, THRESHOLDS["minSpanSimilarity"])
        self.assertEqual(alternatives.SCORE_MARGIN, THRESHOLDS["scoreMargin"])
        self.assertEqual(alternatives.RECOMMENDED_MIN_SIMILARITY, THRESHOLDS["recommendedMinSimilarity"])

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

    def test_mixed_case_token_does_not_produce_a_case_only_variant(self):
        # "eSta" no está entre los candidatos puntuados (match_case lo normaliza
        # a "esta"): antes salía la variante espuria "esta bien" con score -0.0.
        pipe = self._pipeline({"esta": ["está"]}, {"esta": -1.0, "está": -1.2})
        text, variants = pipe._disambiguate_context("eSta bien")
        self.assertEqual(text, "eSta bien")
        self.assertEqual(variants, [])


if __name__ == "__main__":
    unittest.main()
