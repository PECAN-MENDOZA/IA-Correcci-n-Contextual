"""
test_segmentation.py

Tests de `infrastructure/nlp/segmentation.py` (palabras partidas y pegadas de la
escritura infantil). Puros: diccionario de juguete y un juez BETO simulado.

Uso:
    .venv\\Scripts\\python.exe -m unittest test_segmentation -v
"""
import unittest

from infrastructure.nlp import segmentation as seg
from infrastructure.nlp.phonetic_engine import to_phonetic

FREQS = {
    # palabras frecuentes (>= 10 000 para las de 3 letras)
    "que": 12_000_000, "no": 5_000_000, "por": 3_700_000, "eso": 1_700_000, "es": 3_000_000,
    "con": 2_500_000, "las": 1_700_000, "la": 5_000_000, "se": 2_500_000, "de": 6_000_000,
    "en": 5_700_000, "el": 5_000_000, "a": 8_000_000, "y": 4_000_000, "o": 900_000,
    "yo": 2_000_000, "su": 1_000_000, "ir": 300_000, "tan": 480_000, "bien": 1_700_000,
    "todo": 1_090_000, "familia": 90_000, "estaba": 200_000, "preparando": 6_800,
    "encontró": 20_000, "contenta": 10_000, "abajo": 77_000, "bajo": 95_000,
    "aunque": 87_000, "aun": 31_000, "también": 700_000, "precio": 20_000, "cama": 64_000,
    "partir": 20_000, "tele": 8_400, "ley": 36_000, "voy": 530_000, "boya": 400,
    "favor": 300_000, "porfavor": 2_000, "almorzar": 6_600, "alcancía": 186,
    # ruido de subtítulos: presentes pero no cuentan como español
    "to": 17_000, "do": 8_400, "der": 1_222, "he": 90_000, "air": 1_800, "yyo": 800,
    "ato": 324, "ise": 195, "ira": 11_000,
}
ACCENTS = {"encontro": "encontró", "tambien": "también", "alcancia": "alcancía"}


class FakeJudge:
    """Puntúa cada palabra de contexto con +bonus si la frase contiene alguna de
    las cadenas de `prefer` (así la variante preferida sube su score de contexto)."""

    def __init__(self, *prefer, bonus=1.0):
        self.prefer, self.bonus, self.calls = prefer, bonus, 0

    def score_candidates(self, context_words, target_index, candidates):
        self.calls += 1
        text = " ".join(context_words)
        score = -5.0 + (self.bonus if any(p in text for p in self.prefer) else 0.0)
        return [(c, score) for c in candidates]


def join(text, judge=None, phonetic=None):
    return seg.join_split_words(text, FREQS, ACCENTS, judge=judge, phonetic_dict=phonetic)


class TestJoin(unittest.TestCase):
    def test_une_si_un_trozo_no_es_palabra(self):
        self.assertEqual(join("En contró a Ignacio"), "Encontró a Ignacio")
        self.assertEqual(join("Susana estaba con tenta"), "Susana estaba contenta")
        self.assertEqual(join("la al cancía que"), "la alcancía que")

    def test_restaura_la_tilde_de_la_union(self):
        self.assertEqual(join("en contro a Ignacio"), "encontró a Ignacio")

    def test_ruido_ingles_de_dos_letras_cuenta_como_no_palabra(self):
        self.assertEqual(join("a matar a to do el mundo"), "a matar a todo el mundo")

    def test_trozo_de_una_letra_o_falta_de_ortografia_no_se_une(self):
        self.assertEqual(join("salió i se encontró"), "salió i se encontró")
        phonetic = {to_phonetic("voy"): "voy"}
        self.assertEqual(join("un día boy a ir", phonetic=phonetic), "un día boy a ir")

    def test_puntuacion_entre_trozos_impide_unir(self):
        self.assertEqual(join("en, contró"), "en, contró")

    def test_dos_palabras_reales_decide_beto(self):
        self.assertEqual(join("lo metió a bajo de la cama", FakeJudge("abajo", bonus=2.0)),
                         "lo metió abajo de la cama")
        self.assertEqual(join("lo vendió a bajo precio", FakeJudge("a bajo", bonus=2.0)),
                         "lo vendió a bajo precio")

    def test_dos_palabras_reales_bajo_el_margen_no_se_unen(self):
        judge = FakeJudge("abajo", bonus=seg.JOIN_MARGIN / (2 * seg.CONTEXT_WINDOW) / 2)
        self.assertEqual(join("lo metió a bajo de la cama", judge), "lo metió a bajo de la cama")

    def test_dos_palabras_reales_sin_beto_no_se_unen(self):
        self.assertEqual(join("lo metió a bajo de la cama"), "lo metió a bajo de la cama")

    def test_uniones_prohibidas_o_cortas_ni_consultan_a_beto(self):
        judge = FakeJudge("también", "air", "yyo", "porfavor", bonus=9.0)
        for text in ["lo hizo tan bien", "voy a ir", "halla y yo te", "vas a venir por favor"]:
            self.assertEqual(join(text, judge), text)
        self.assertEqual(judge.calls, 0)


class TestSplit(unittest.TestCase):
    def split(self, word):
        return seg.split_candidates(word, FREQS, ACCENTS)

    def test_particiones_en_palabras_espanolas(self):
        self.assertEqual(self.split("queno")[0], ["que", "no"])
        self.assertEqual(self.split("estabapreparando")[0], ["estaba", "preparando"])
        self.assertEqual(self.split("asufamilia")[0], ["a", "su", "familia"])
        self.assertIn(["tele", "y"], self.split("teley"))

    def test_solo_las_de_menos_trozos(self):
        self.assertEqual(self.split("poreso"), [["por", "eso"]])

    def test_ruido_corto_no_sirve_de_trozo(self):
        self.assertEqual(self.split("heladero"), [])      # he la der o

    def test_palabras_validas_fuera_del_diccionario_no_se_parten(self):
        self.assertFalse(seg._plausible_split(["imagina", "ti", "vos"]))
        self.assertFalse(seg._plausible_split(["confiar", "se"]))
        self.assertFalse(seg._plausible_split(["diciendo", "le"]))
        self.assertTrue(seg._looks_compound("sobreentrenamiento"))
        self.assertTrue(seg._looks_compound("latinoamericanos"))
        self.assertEqual(seg.split_candidates("latinoamericanos", {"latin": 500, "o": 1, "americanos": 5000}, {}), [])
        self.assertTrue(seg._plausible_split(["se", "las"]))
        self.assertTrue(seg._plausible_split(["a", "su", "familia"]))
        self.assertTrue(seg._plausible_split(["estaba", "preparando"]))

    def test_choose_split_sin_beto(self):
        self.assertEqual(seg.choose_split(["dijo", "queno", "importaba"], 1, [["que", "no"]], None),
                         "que no")
        self.assertIsNone(seg.choose_split(["dijo", "queno"], 1, [["que", "no"]], "quemo", pick_cost=0.3))
        self.assertEqual(seg.choose_split(["dijo", "queno"], 1, [["que", "no"]], "quemo", pick_cost=1.0),
                         "que no")

    def test_choose_split_con_beto_y_puntuacion(self):
        tokens = ["le", "dijo", "queno,", "importaba"]
        self.assertEqual(seg.choose_split(tokens, 2, [["que", "no"]], "quemo", FakeJudge("que no,")),
                         "que no")
        self.assertIsNone(seg.choose_split(tokens, 2, [["que", "no"]], "quemo", FakeJudge("quemo,")))


if __name__ == "__main__":
    unittest.main()
