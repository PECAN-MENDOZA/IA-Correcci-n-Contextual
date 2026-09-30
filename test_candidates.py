"""Tests de infrastructure/nlp/candidates.py (capa 2: coste disléxico y
formas verbales regulares). Puros: sin diccionario, sin torch."""
import unittest

from infrastructure.nlp.candidates import dyslexic_cost, is_regular_verb_form, pick_candidate

# Candidatos tal como los devuelve SymSpell (término, frecuencia) para cada
# palabra, copiados de es_50k el 2026-09-19.
JUGAN = [("jugar", 52990), ("juan", 8241), ("juegan", 3507), ("juzgan", 359), ("dugan", 349)]
ARUZ = [("cruz", 9455), ("arun", 209), ("agua", 99571), ("arroz", 7162), ("aros", 944), ("luz", 63192)]
CAMERA = [("cadera", 2248), ("ramera", 2208), ("casera", 2057), ("camara", 1410), ("cámara", 33907)]
VACATION = [("action", 227), ("vacación", 186)]
PENSAN = [("pensar", 85868), ("piensan", 14421), ("pesan", 790)]
PODEN = [("poder", 95665), ("ponen", 9636), ("piden", 3455), ("pueden", 124700)]
GAOT = [("got", 2593), ("gato", 19159), ("good", 2433)]
KASA = [("kisha", 287), ("casa", 496977), ("kara", 1696)]
LEXICON = {"nadar": 8545, "pintar": 4929, "dibujar": 2323, "comer": 50000, "saltar": 3000,
           "estar": 900000, "escribir": 20000, "pedir": 15000, "jugar": 52990}


class TestCosteDislexico(unittest.TestCase):
    def test_tilde_es_lo_mas_barato(self):
        self.assertLess(dyslexic_cost("esta", "está"), dyslexic_cost("esta", "esa"))

    def test_vocal_y_confundibles_mas_baratas_que_consonante(self):
        self.assertLess(dyslexic_cost("baca", "vaca"), dyslexic_cost("baca", "boca") + 0.01)
        self.assertLess(dyslexic_cost("camera", "camara"), dyslexic_cost("camera", "cadera"))

    def test_primera_letra_penaliza_salvo_confundible(self):
        self.assertGreater(dyslexic_cost("aruz", "cruz"), dyslexic_cost("aruz", "arroz"))
        self.assertLess(dyslexic_cost("kasa", "casa"), dyslexic_cost("kasa", "kisha"))


class TestPickCandidate(unittest.TestCase):
    def test_conserva_la_flexion(self):
        self.assertEqual(pick_candidate("jugan", JUGAN), "juegan")
        self.assertEqual(pick_candidate("pensan", PENSAN), "piensan")

    def test_empate_lo_decide_la_frecuencia(self):
        # piden (vocal<->vocal) y pueden (inserción de vocal) cuestan igual.
        self.assertEqual(pick_candidate("poden", PODEN), "pueden")

    def test_no_cambia_la_primera_letra_si_hay_alternativa(self):
        self.assertEqual(pick_candidate("aruz", ARUZ), "arroz")

    def test_prefiere_tilde_y_vocal_a_consonante(self):
        self.assertIn(pick_candidate("camera", CAMERA), ("camara", "cámara"))
        self.assertEqual(pick_candidate("vacation", VACATION), "vacación")

    def test_transposicion_y_ruido_corto(self):
        self.assertEqual(pick_candidate("gaot", GAOT), "gato")
        # La penalización a candidatos cortos no aplica a los de igual longitud.
        self.assertEqual(pick_candidate("mys", [("mays", 400), ("mis", 500000), ("mas", 300000)]), "mis")

    def test_piso_de_frecuencia(self):
        self.assertIsNone(pick_candidate("dugan", [("dugan", 349)]))         # es la misma palabra
        self.assertIsNone(pick_candidate("xyz", [("xyzz", 20)]))            # por debajo del piso


class TestFormaVerbalRegular(unittest.TestCase):
    def test_futuro_e_imperfecto_regulares(self):
        for w in ("nadaremos", "dibujaremos", "pintaremos", "saltando", "dibujamos", "estaremos"):
            self.assertTrue(is_regular_verb_form(w, LEXICON), w)

    def test_no_protege_cambios_de_raiz_ni_participios(self):
        for w in ("jugan", "escribido", "pediendo", "komimos", "yebaremos", "jugar"):
            self.assertFalse(is_regular_verb_form(w, LEXICON), w)


class TestYeismo(unittest.TestCase):
    def test_ll_y_omitida_o_cambiada_cuesta_como_confusion(self):
        self.assertEqual(dyslexic_cost("rodia", "rodilla"), 0.5)
        self.assertEqual(dyslexic_cost("cabayo", "caballo"), 0.5)
        self.assertEqual(dyslexic_cost("yuvia", "lluvia"), 0.5)
        self.assertLess(dyslexic_cost("tiyo", "tío"), 1.0)


if __name__ == "__main__":
    unittest.main()
