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
from infrastructure.nlp.grammar_rules import correct_grammar, normalize_modern_spelling


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

    def test_sino_abriendo_oracion_es_si_no(self):
        # La adversativa nunca abre frase: "Sino llueve, ..." es "si no".
        self.assertEqual(expand_amalgams("sino llueve mañana, iremos al parque"),
                         "si no llueve mañana, iremos al parque")
        self.assertEqual(expand_amalgams("es tarde. Sino corres, lo pierdes"),
                         "es tarde. Si no corres, lo pierdes")

    def test_sino_que_abriendo_oracion_se_conserva(self):
        self.assertEqual(expand_amalgams("sino que me quedé en casa"),
                         "sino que me quedé en casa")

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
        # "gente" es femenino: el predicado concuerda en género ("cansada").
        self.assertEqual(correct_grammar("la gente están cansados"),
                         "la gente está cansada")

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

    def test_verbo_regular_por_sufijo(self):
        # Formas fuera de la tabla: pretérito, futuro, imperfecto y presente.
        self.assertEqual(correct_grammar("el equipo perdieron el partido"),
                         "el equipo perdió el partido")
        self.assertEqual(correct_grammar("la gente aplaudieron mucho"),
                         "la gente aplaudió mucho")
        self.assertEqual(correct_grammar("mi familia leyeron el libro"),
                         "mi familia leyó el libro")
        self.assertEqual(correct_grammar("el equipo ganaran si entrenan"),
                         "el equipo ganará si entrenan")
        self.assertEqual(correct_grammar("mi familia vendran el domingo"),
                         "mi familia vendrá el domingo")
        self.assertEqual(correct_grammar("la gente tenian miedo"),
                         "la gente tenía miedo")
        self.assertEqual(correct_grammar("la gente cambian de opinión"),
                         "la gente cambia de opinión")
        self.assertEqual(correct_grammar("el grupo cantan muy bien"),
                         "el grupo canta muy bien")

    def test_presente_solo_si_el_singular_esta_en_el_lexico(self):
        # Sustantivos en -an/-en tras el colectivo no se tocan.
        for frase in ("el grupo examen", "el equipo alemán ganó",
                      "la gente joven baila", "el equipo tren", "el grupo tan grande"):
            self.assertEqual(correct_grammar(frase), frase)

    def test_haber_impersonal_futuro_y_subjuntivo(self):
        self.assertEqual(correct_grammar("mañana habran muchas nubes"),
                         "mañana habrá muchas nubes")
        self.assertEqual(correct_grammar("espero que hayan muchas personas"),
                         "espero que haya muchas personas")
        # Auxiliar legítimo: no se toca.
        self.assertEqual(correct_grammar("habrán llegado ya"), "habrán llegado ya")
        self.assertEqual(correct_grammar("ojalá hayan comido"), "ojalá hayan comido")

    def test_adjetivo_fuera_de_la_lista_blanca_se_conserva(self):
        # El verbo se corrige; el predicado solo se singulariza si está en la
        # lista blanca de adjetivos (evita tocar sintagmas nominales).
        self.assertEqual(correct_grammar("la gente son mis vecinos"),
                         "la gente es mis vecinos")


class TestOrtografiaVigente(unittest.TestCase):
    """`normalize_modern_spelling`: tildes abolidas por la RAE en 2010."""

    def test_solo_sin_tilde(self):
        self.assertEqual(normalize_modern_spelling("sólo quiero agua"), "solo quiero agua")
        self.assertEqual(normalize_modern_spelling("Sólo quiero agua."), "Solo quiero agua.")
        self.assertEqual(normalize_modern_spelling("éste es mi guión"), "este es mi guion")

    def test_respeta_lo_que_escribio_el_alumno(self):
        self.assertEqual(normalize_modern_spelling("sólo quiero agua", written={"sólo"}),
                         "sólo quiero agua")

    def test_no_toca_otras_tildes(self):
        self.assertEqual(normalize_modern_spelling("él está solo en la habitación"),
                         "él está solo en la habitación")


class TestEscrituraInfantilReal(unittest.TestCase):
    """Reglas del 2026-09-30 (test_ninos_reales_dev)."""

    def test_aver_ante_participio_es_haber(self):
        self.assertEqual(expand_amalgams("algo interesante de aver pasado a otro grado"),
                         "algo interesante de haber pasado a otro grado")
        self.assertEqual(expand_amalgams("aver si vienes"), "a ver si vienes")

    def test_por_que_abriendo_oracion_es_interrogativo(self):
        self.assertEqual(expand_amalgams("y por que tienes la nariz tan grande"),
                         "y por qué tienes la nariz tan grande")
        self.assertEqual(expand_amalgams("Por que no vienes"), "Por qué no vienes")

    def test_por_que_en_mitad_de_frase_es_causal(self):
        self.assertEqual(expand_amalgams("no entendía nada por que yo no quería"),
                         "no entendía nada porque yo no quería")
        self.assertEqual(expand_amalgams("¿Por qué lloras? Por que me caí."),
                         "¿Por qué lloras? Porque me caí.")

    def test_por_que_interrogativo_indirecto_o_sustantivo_se_conserva(self):
        for frase in ["no sé por que lloras", "el por que de las cosas"]:
            self.assertEqual(expand_amalgams(frase), frase)

    def test_monosilabos_pierden_la_tilde_aunque_la_escriba_el_alumno(self):
        self.assertEqual(normalize_modern_spelling("se fué y me dió", {"fué", "dió"}), "se fue y me dio")
        self.assertEqual(normalize_modern_spelling("sólo quiero", {"sólo"}), "sólo quiero")


if __name__ == "__main__":
    unittest.main()
