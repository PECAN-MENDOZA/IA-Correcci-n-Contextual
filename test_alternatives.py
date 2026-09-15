"""
test_alternatives.py

Tests puros del selector de alternativas (infrastructure/nlp/alternatives.py)
y de la señal de ambigüedad en homófonos de `_disambiguate_context`.

Corren SIN torch/transformers/GPU: el selector solo usa difflib/re y la
guarda `is_safe_refinement`; para el pipeline se sustituyen los módulos
pesados (juez BETO y T5) por stubs antes de importarlo y se inyectan fakes
que devuelven puntuaciones fijas.

Uso:
    .venv\\Scripts\\python.exe -m unittest test_alternatives -v
"""
import sys
import types
import unittest

from infrastructure.nlp.alternatives import select_alternatives


class SelectAlternativesTests(unittest.TestCase):
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
        out = select_alternatives("hola como estas", "hola como estas",
                                  [("hola como estas", -0.1), ("Hola, ¿cómo estás?", -0.3),
                                   ("Adiós, hasta luego, nos vemos mañana temprano", -0.4)])
        self.assertEqual(out, ["Hola, ¿cómo estás?"])

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
        # Si T5 ya eligió la lectura "está", la variante BETO es la misma edición: no se duplica.
        out = select_alternatives(original, base,
                                  [("Está bien, nos vemos luego.", -0.2),
                                   ("está bien, nos vemos luego", -0.1)], score_margin=1.0)
        self.assertEqual(out, ["está bien, nos vemos luego"])

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

    def test_base_identical_to_original_is_not_offered(self):
        out = select_alternatives("el niño juega", "el niño juega",
                                  [("el niño juega", 0.0)])
        self.assertEqual(out, [])

    def test_paraphrases_are_not_alternatives(self):
        # Beams reales de T5 sobre frases ya correctas: el mejor beam es la propia
        # frase y los siguientes son paráfrasis (voy→iré, luego→después). Ni con
        # un margen de score amplio se ofrecen: no son otra lectura de la palabra.
        original = "mañana voy al parque con mis amigos"
        out = select_alternatives(original, original,
                                  [(original, -0.01), ("mañana iré al parque con mis amigos", -0.38),
                                   ("Mañana voy al parque con mis amigos", -0.41)], score_margin=1.0)
        self.assertEqual(out, ["Mañana voy al parque con mis amigos"])
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

    def test_base_goes_right_after_the_best_candidate(self):
        original = "a mi me gusta que ellos juega mucho"
        base     = "a mí me gusta que ellos juega mucho"
        cands = [("a mí me gusta que ellos juegan mucho", -0.04),
                 ("a mí me gusta que ellos jueguen mucho", -0.18),   # misma edición que "juegan"
                 ("a mí me gusta que ellos juegas mucho", -0.20)]
        out = select_alternatives(original, base, cands, score_margin=1.0)
        self.assertEqual(out, ["a mí me gusta que ellos juegan mucho", base])


# ---------------------------------------------------------------------------
# Integración ligera de _disambiguate_context (sin torch): stubs de los módulos
# pesados + juez falso con puntuaciones fijas.
# ---------------------------------------------------------------------------

def _stub_heavy_modules():
    """Registra stubs de context_judge y t5_model si aún no están importados."""
    if "infrastructure.nlp.context_judge" not in sys.modules:
        judge_mod = types.ModuleType("infrastructure.nlp.context_judge")
        judge_mod.ContextJudge = type("ContextJudge", (), {})
        sys.modules["infrastructure.nlp.context_judge"] = judge_mod
    if "infrastructure.ml.t5_model" not in sys.modules:
        t5_mod = types.ModuleType("infrastructure.ml.t5_model")
        t5_mod.T5CorrectionModel = type("T5CorrectionModel", (), {})
        t5_mod.T5SpanishTokenizer = type("T5SpanishTokenizer", (), {})
        sys.modules["infrastructure.ml.t5_model"] = t5_mod


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
        _stub_heavy_modules()
        from infrastructure.nlp import correction_pipeline
        cls.mod = correction_pipeline

    def _pipeline(self, homophones, scores, word_freqs=None):
        pipe = self.mod.CorrectionPipeline.__new__(self.mod.CorrectionPipeline)
        pipe._phonetic = FakePhonetic(homophones, word_freqs)
        pipe._judge    = FakeJudge(scores)
        return pipe

    def test_no_torch_loaded(self):
        for heavy in ("torch", "transformers"):
            self.assertFalse(heavy in sys.modules, f"'{heavy}' se cargó en un test puro")

    def test_confident_reading_overwrites_without_variants(self):
        pipe = self._pipeline({"esta": ["está"]}, {"esta": -3.0, "está": -1.0})
        text, variants = pipe._disambiguate_context("esta bien")
        self.assertEqual(text, "está bien")
        self.assertEqual(variants, [])

    def test_tied_reading_keeps_text_and_offers_variant(self):
        # Diferencia 0.2 < margen de acento 0.5: no se sobrescribe, pero se ofrece
        # la segunda lectura con score = -|diferencia| (más cerca de 0 = más empate).
        pipe = self._pipeline({"esta": ["está"]}, {"esta": -1.2, "está": -1.0})
        text, variants = pipe._disambiguate_context("esta bien")
        self.assertEqual(text, "esta bien")
        self.assertEqual(len(variants), 1)
        self.assertEqual(variants[0][0], "está bien")
        self.assertAlmostEqual(variants[0][1], -0.2, places=6)

    def test_current_best_but_second_within_margin_offers_variant(self):
        pipe = self._pipeline({"esta": ["está"]}, {"esta": -1.0, "está": -1.3})
        text, variants = pipe._disambiguate_context("esta bien")
        self.assertEqual(text, "esta bien")
        self.assertEqual([v[0] for v in variants], ["está bien"])
        self.assertAlmostEqual(variants[0][1], -0.3, places=6)

    def test_clearly_current_reading_offers_nothing(self):
        pipe = self._pipeline({"esta": ["está"]}, {"esta": -1.0, "está": -4.0})
        text, variants = pipe._disambiguate_context("esta bien")
        self.assertEqual(text, "esta bien")
        self.assertEqual(variants, [])

    def test_at_most_three_positions_closest_ties_first(self):
        homophones = {"esta": ["está"], "tubo": ["tuvo"], "boy": ["voy"], "se": ["sé"]}
        scores = {"esta": -1.0, "está": -1.3,    # dif 0.3
                  "tubo": -1.0, "tuvo": -1.1,    # dif 0.1
                  "boy": -1.0, "voy": -1.4,      # dif 0.4
                  "se": -1.0, "sé": -1.2}        # dif 0.2
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
                              {"hoy": -8.16, "holly": -8.09, "se": -7.2, "sé": -5.8}, freqs)
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


if __name__ == "__main__":
    unittest.main()
