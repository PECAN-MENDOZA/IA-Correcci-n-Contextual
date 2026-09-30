"""
test_generate_agreement_data.py

Tests de `scripts/generate_agreement_data.py`: la frase CORRECTA de cada par debe
serlo de verdad (género del predicado, clase del sujeto) y la minería solo debe
corromper verbos con sujeto plural explícito. Puros: sin modelo ni red.

Uso:
    .venv\\Scripts\\python.exe -m unittest test_generate_agreement_data -v
"""
import random
import re
import unittest

from scripts import generate_agreement_data as g


def _tokens(text: str) -> list:
    return g._TOKEN.findall(text.lower())


class TestTemplatePairs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        random.seed(0)
        cls.pairs = g.template_pairs(5000)

    def test_predicado_concuerda_en_genero_con_el_sujeto(self):
        masc = {m for _, _, preds in g.PRED_SER for m, f, _ in preds if m != f}
        fem = {f for _, _, preds in g.PRED_SER for m, f, _ in preds if m != f}
        for _, correct in self.pairs:
            subj = correct.split()[0] if correct.split()[0] in g.PRONS else correct.split()[1]
            gender = g.PRONS.get(subj) or g.NOUNS[subj][0]
            wrong = fem if gender == "m" else masc
            self.assertFalse(any(correct.endswith(" " + p) for p in wrong), correct)

    def test_sin_predicados_absurdos_para_la_clase(self):
        for _, correct in self.pairs:
            words = correct.split()
            if words[0] in g.PRONS or g.NOUNS[words[1]][1] == "ser":
                self.assertNotRegex(correct, r"(caros|caras|de madera|precio|rotos|rotas)$")
            else:
                self.assertNotRegex(correct, r"(hambre|cansados|cansadas|rápidos|rápidas)$")

    def test_casos_del_bug_de_julio_ya_no_aparecen(self):
        corrects = {c for _, c in self.pairs}
        for bad in [r"^\w+ (casas|amigas|vecinas|puertas|camisas)\b.*\b(buenos|caros|mojados|sucios|bonitos)$",
                    r"\bcamisas\b.*\bhambre$", r"^ellas .*\b(cansados|buenos|sucios)$",
                    r"^ell[oa]s que "]:
            self.assertFalse(any(re.search(bad, c) for c in corrects), bad)

    def test_error_solo_cambia_el_verbo_a_singular(self):
        sg_of = {pl: sg for sg, pl, _ in g.PRED_SER}
        for err, correct in self.pairs:
            diff = [(a, b) for a, b in zip(err.split(), correct.split()) if a != b]
            self.assertEqual(len(diff), 1, (err, correct))
            self.assertEqual(sg_of[diff[0][1]], diff[0][0])


class TestExplicitSubject(unittest.TestCase):
    def check(self, sentence: str, verb: str) -> bool:
        toks = _tokens(sentence)
        return g.has_explicit_plural_subject(toks, toks.index(verb))

    def test_acepta_sujeto_plural_explicito(self):
        self.assertTrue(self.check("Los padres no pueden pasar de aquí.", "pueden"))
        self.assertTrue(self.check("Las cosas nunca me salen bien.", "salen"))
        self.assertTrue(self.check("Los resultados previstos en esta esfera programática son los siguientes:", "son"))

    def test_acepta_sujeto_pospuesto_a_copulativo_al_abrir_clausula(self):
        self.assertTrue(self.check("¿Qué son las funciones ejecutivas?", "son"))
        self.assertTrue(self.check("-Son estos asesinatos.", "son"))

    def test_rechaza_cuando_el_singular_tambien_es_correcto(self):
        self.assertFalse(self.check("Tienen más cosas en común que los patos.", "tienen"))
        self.assertFalse(self.check("Le dije a las niñas que podían tener un gato.", "podían"))
        self.assertFalse(self.check("21% de los niños viven en la pobreza.", "viven"))
        self.assertFalse(self.check("¿Qué me dicen de Bill Rusell de los Celtics?", "dicen"))
        self.assertFalse(self.check("La edad mínima legal para el matrimonio de ambos sexos son los 16 años.", "son"))

    def test_la_puntuacion_corta_la_busqueda_del_sujeto(self):
        self.assertFalse(self.check("Vi a los perros, pusieron sus cosas en la carreta.", "pusieron"))


class TestExclusiones(unittest.TestCase):
    def test_haber_impersonal_e_imperativo_fuera(self):
        self.assertNotIn("habían", g.PL2SG)
        self.assertNotIn("ven", g.PL2SG)

    def test_muestra_general_sin_habia_a_habian(self):
        random.seed(0)
        for err, correct in g.general_sample(20000):
            self.assertFalse(re.search(r"(?i)\bhabía\b", err) and re.search(r"(?i)\bhabían\b", correct),
                             correct)


if __name__ == "__main__":
    unittest.main()
