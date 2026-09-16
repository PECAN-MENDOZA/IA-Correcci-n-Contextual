"""
test_amalgams.py

Capa de reglas cerradas (Task 2 del plan de mejora del modelo):

  - `expand_amalgams`: amalgamas que el alumno escribe pegadas y que el
    corrector léxico no puede arreglar porque el resultado son DOS palabras
    (`aver` -> `a ver`, `asique` -> `así que`, `sino vienes` -> `si no vienes`,
    `porque no vienes?` -> `por qué no vienes?`).
  - `correct_grammar`: concordancia con sustantivos colectivos singulares
    (`la gente son amables` -> `la gente es amable`).

Cada regla lleva su caso negativo: el uso legítimo NO se toca. Tests puros
(sin torch ni BETO).

Uso:
    .venv\\Scripts\\python.exe -m unittest test_amalgams -v
"""
import unittest

from infrastructure.nlp.amalgams import expand_amalgams
from infrastructure.nlp.grammar_rules import correct_grammar


class TestAmalgams(unittest.TestCase):
    """`aver`, `haber si`, `asique`, `sino`, `porque` interrogativo."""

    def test_aver_siempre_se_separa(self):
        # "aver" no existe en español: la separación es incondicional.
        self.assertEqual(expand_amalgams("aver si vienes a la fiesta"),
                         "a ver si vienes a la fiesta")

    def test_aver_conserva_puntuacion_y_mayuscula(self):
        self.assertEqual(expand_amalgams("Aver, ¿me ayudas?"), "A ver, ¿me ayudas?")

    def test_haber_si_es_a_ver_si(self):
        self.assertEqual(expand_amalgams("haber si vienes a la fiesta"),
                         "a ver si vienes a la fiesta")

    def test_haber_si_tras_verbo_modal_se_conserva(self):
        # "puede haber si..." es raro pero gramatical; la regla no dispara
        # cuando "haber" lleva delante un modal (uso de infinitivo legítimo).
        self.assertEqual(expand_amalgams("puede haber si tú quieres"),
                         "puede haber si tú quieres")

    def test_haber_con_participio_se_conserva(self):
        self.assertEqual(expand_amalgams("debe haber comido temprano"),
                         "debe haber comido temprano")

    def test_asique_se_separa(self):
        self.assertEqual(expand_amalgams("asique me fui a casa"),
                         "así que me fui a casa")

    def test_sino_ante_verbo_es_si_no(self):
        self.assertEqual(expand_amalgams("sino vienes me voy"),
                         "si no vienes me voy")

    def test_sino_ante_clitico_es_si_no(self):
        self.assertEqual(expand_amalgams("sino te apuras llegamos tarde"),
                         "si no te apuras llegamos tarde")

    def test_sino_adversativo_se_conserva(self):
        # "no quiero esto sino aquello": conjunción legítima ante determinante.
        self.assertEqual(expand_amalgams("no quiero esto sino aquello"),
                         "no quiero esto sino aquello")

    def test_sino_que_se_conserva(self):
        self.assertEqual(expand_amalgams("no vino sino que llamó"),
                         "no vino sino que llamó")

    def test_porque_interrogativo_se_separa(self):
        self.assertEqual(expand_amalgams("porque no vienes?"), "por qué no vienes?")

    def test_porque_tras_apertura_de_interrogacion_se_separa(self):
        self.assertEqual(expand_amalgams("¿porque llegaste tarde?"),
                         "¿por qué llegaste tarde?")

    def test_porque_causal_dentro_de_pregunta_se_conserva(self):
        # Hay signo de interrogación, pero "porque" no abre la pregunta.
        self.assertEqual(expand_amalgams("¿no viniste porque estabas enfermo?"),
                         "¿no viniste porque estabas enfermo?")

    def test_porque_causal_sin_interrogacion_se_conserva(self):
        self.assertEqual(expand_amalgams("no vine porque estaba enfermo"),
                         "no vine porque estaba enfermo")

    def test_texto_sin_amalgamas_no_cambia(self):
        for frase in ("mañana voy al parque con mis amigos",
                      "a ver si vienes a la fiesta",
                      "quiero que tú vengas conmigo"):
            self.assertEqual(expand_amalgams(frase), frase)

    def test_texto_vacio(self):
        self.assertEqual(expand_amalgams(""), "")
        self.assertEqual(expand_amalgams("   "), "   ")


class TestColectivos(unittest.TestCase):
    """Concordancia con colectivos singulares dentro de `correct_grammar`."""

    def test_la_gente_son_amables(self):
        self.assertEqual(correct_grammar("la gente son muy amables"),
                         "la gente es muy amable")

    def test_la_gente_estan(self):
        self.assertEqual(correct_grammar("la gente están cansados"),
                         "la gente está cansado")

    def test_la_familia_vienen(self):
        self.assertEqual(correct_grammar("mi familia vienen mañana"),
                         "mi familia viene mañana")

    def test_el_equipo_ganaron(self):
        self.assertEqual(correct_grammar("el equipo ganaron el partido"),
                         "el equipo ganó el partido")

    def test_colectivo_con_verbo_singular_se_conserva(self):
        self.assertEqual(correct_grammar("la gente es muy amable"),
                         "la gente es muy amable")

    def test_mayoria_se_conserva(self):
        # Ambigüedad aceptada por la norma (concordancia ad sensum).
        self.assertEqual(correct_grammar("la mayoría llegaron tarde"),
                         "la mayoría llegaron tarde")

    def test_gente_no_inmediata_se_conserva(self):
        # El verbo plural no es el del colectivo: no se toca.
        self.assertEqual(correct_grammar("la gente que conocí son mis amigos"),
                         "la gente que conocí son mis amigos")

    def test_plural_real_se_conserva(self):
        self.assertEqual(correct_grammar("los equipos ganaron el partido"),
                         "los equipos ganaron el partido")

    def test_adjetivo_fuera_de_la_lista_blanca_se_conserva(self):
        # El verbo se corrige; el predicado solo se singulariza si está en la
        # lista blanca de adjetivos (evita tocar sintagmas nominales).
        self.assertEqual(correct_grammar("la gente son mis vecinos"),
                         "la gente es mis vecinos")


if __name__ == "__main__":
    unittest.main()
