"""
test_confusions.py

Reglas cerradas de la revisión del 2026-09-29 (errores de escritura infantil):

  - `fix_lexical_confusions` (capa 0.2): r/rr tras determinante
    (`el pero` -> `el perro`, `el caro` -> `el carro`, `el aros` -> `el arroz`).
  - `resolve_final_confusions` (capa 6, tras T5): caer/callar (`se callo` ->
    `se cayó` / `se calló`) y `yo` + pretérito de 3.ª persona (`yo comió` ->
    `yo comí`).
  - `expand_amalgams`: locuciones pegadas, `con migo`, `ir haber a`.
  - `correct_grammar`: `está hay` -> `está ahí`, `el canción` -> `la canción`.

Cada regla lleva su caso negativo: el uso legítimo NO se toca. Tests puros
(sin torch ni BETO: el juez es un doble).

Uso:
    .venv\\Scripts\\python.exe -m unittest test_confusions -v
"""
import unittest

from infrastructure.nlp.amalgams import expand_amalgams
from infrastructure.nlp.confusions import fix_lexical_confusions, resolve_final_confusions
from infrastructure.nlp.grammar_rules import correct_grammar


class FakeJudge:
    """Puntúa con una tabla fija {candidato: score} y registra las llamadas."""

    def __init__(self, scores):
        self._scores = scores
        self.calls = 0

    def score_candidates(self, context_words, target_index, candidates):
        self.calls += 1
        scored = [(c, self._scores.get(c, -10.0)) for c in candidates]
        scored.sort(key=lambda cs: -cs[1])
        return scored


class TestRR(unittest.TestCase):
    def test_pero_tras_determinante_es_perro(self):
        self.assertEqual(fix_lexical_confusions("el pero ladra mucho"), "el perro ladra mucho")
        self.assertEqual(fix_lexical_confusions("Mi pero se llama Toby"), "Mi perro se llama Toby")

    def test_pero_conjuncion_se_conserva(self):
        self.assertEqual(fix_lexical_confusions("quiero ir pero llueve"), "quiero ir pero llueve")
        self.assertEqual(fix_lexical_confusions("es bonito, pero caro"), "es bonito, pero caro")

    def test_caro_sustantivo_es_carro(self):
        self.assertEqual(fix_lexical_confusions("el caro de mi papá es rojo"), "el carro de mi papá es rojo")
        self.assertEqual(fix_lexical_confusions("los caros hacen ruido"), "los carros hacen ruido")

    def test_caro_adjetivo_se_conserva(self):
        self.assertEqual(fix_lexical_confusions("el helado es muy caro"), "el helado es muy caro")

    def test_aros_tras_singular_es_arroz(self):
        self.assertEqual(fix_lexical_confusions("me gusta el aros con pollo"), "me gusta el arroz con pollo")

    def test_aros_plural_se_conserva(self):
        self.assertEqual(fix_lexical_confusions("los aros de colores"), "los aros de colores")

    def test_determinante_con_signo_corta_el_marco(self):
        self.assertEqual(fix_lexical_confusions("vino él, pero tarde"), "vino él, pero tarde")


class TestCaerCallar(unittest.TestCase):
    def test_preposicion_de_lugar_decide_cayo(self):
        for frase, esperado in [
            ("el perro se calló en el río", "el perro se cayó en el río"),
            ("mi hermano se calló de la bicicleta", "mi hermano se cayó de la bicicleta"),
            ("la pelota se callo al suelo", "la pelota se cayó al suelo"),
        ]:
            self.assertEqual(resolve_final_confusions(frase), esperado)

    def test_dativo_decide_cayo(self):
        self.assertEqual(resolve_final_confusions("se me calló el helado"), "se me cayó el helado")

    def test_pista_de_silencio_decide_callo(self):
        self.assertEqual(resolve_final_confusions("mi mamá se cayó y no dijo nada"),
                         "mi mamá se calló y no dijo nada")

    def test_pista_de_caida_decide_cayo(self):
        self.assertEqual(resolve_final_confusions("se calló y se lastimó la rodilla"),
                         "se cayó y se lastimó la rodilla")

    def test_sin_marco_decide_beto(self):
        judge = FakeJudge({"calló": -0.2, "cayó": -2.6})
        self.assertEqual(resolve_final_confusions("el niño se cayó cuando entró la maestra", judge=judge),
                         "el niño se calló cuando entró la maestra")
        self.assertEqual(judge.calls, 1)

    def test_sin_marco_ni_juez_no_cambia(self):
        self.assertEqual(resolve_final_confusions("el niño se calló cuando entró la maestra"),
                         "el niño se calló cuando entró la maestra")

    def test_tilde_escrita_por_el_alumno_se_respeta(self):
        self.assertEqual(resolve_final_confusions("el niño se calló en clase", written={"calló"}),
                         "el niño se calló en clase")

    def test_sin_se_no_se_toca(self):
        for frase in ["tengo un callo en el pie", "yo me callo cuando habla la profesora",
                      "el rayo cayó en la casa"]:
            self.assertEqual(resolve_final_confusions(frase), frase)

    def test_puntuacion_y_mayuscula(self):
        self.assertEqual(resolve_final_confusions("Se calló de la cama."), "Se cayó de la cama.")


class TestYoPreterito(unittest.TestCase):
    def test_irregulares(self):
        self.assertEqual(resolve_final_confusions("yo fue al parque"), "yo fui al parque")
        self.assertEqual(resolve_final_confusions("yo no hizo la tarea"), "yo no hice la tarea")
        self.assertEqual(resolve_final_confusions("yo me durmió tarde"), "yo me dormí tarde")

    def test_regulares_con_tilde(self):
        self.assertEqual(resolve_final_confusions("ayer yo comió pan"), "ayer yo comí pan")
        self.assertEqual(resolve_final_confusions("yo jugó fútbol"), "yo jugué fútbol")
        self.assertEqual(resolve_final_confusions("yo estudió mucho"), "yo estudié mucho")

    def test_presente_sin_tilde_se_conserva(self):
        for frase in ["yo estudio mucho", "yo canto en el coro", "yo como pan"]:
            self.assertEqual(resolve_final_confusions(frase), frase)

    def test_sin_yo_no_se_toca(self):
        self.assertEqual(resolve_final_confusions("ella comió pan"), "ella comió pan")


class TestAmalgamasNuevas(unittest.TestCase):
    def test_locuciones_pegadas(self):
        self.assertEqual(expand_amalgams("porfavor dame agua"), "por favor dame agua")
        self.assertEqual(expand_amalgams("Enserio me gustó"), "En serio me gustó")
        self.assertEqual(expand_amalgams("aveces voy al parque"), "a veces voy al parque")
        self.assertEqual(expand_amalgams("yo nose que hacer"), "yo no sé que hacer")

    def test_con_migo(self):
        self.assertEqual(expand_amalgams("ven con migo al parque"), "ven conmigo al parque")
        self.assertEqual(expand_amalgams("juega con tigo, dijo"), "juega contigo, dijo")

    def test_con_seguido_de_otra_palabra_se_conserva(self):
        self.assertEqual(expand_amalgams("voy con mi amigo"), "voy con mi amigo")

    def test_ir_haber_a_es_a_ver(self):
        self.assertEqual(expand_amalgams("tengo que ir haber a mi abuela"),
                         "tengo que ir a ver a mi abuela")
        self.assertEqual(expand_amalgams("vamos haber la película"), "vamos a ver la película")

    def test_va_haber_existencial_es_a_haber(self):
        self.assertEqual(expand_amalgams("mañana va haber una fiesta"), "mañana va a haber una fiesta")

    def test_haber_infinitivo_legitimo_se_conserva(self):
        self.assertEqual(expand_amalgams("puede haber problemas"), "puede haber problemas")
        self.assertEqual(expand_amalgams("va haber si quieres"), "va haber si quieres")


class TestGramaticaNueva(unittest.TestCase):
    def test_esta_hay_es_ahi(self):
        self.assertEqual(correct_grammar("mi mochila está hay"), "mi mochila está ahí")
        self.assertEqual(correct_grammar("lo dejé hay en la mesa"), "lo dejé ahí en la mesa")

    def test_por_hay_final_es_ahi(self):
        self.assertEqual(correct_grammar("anda por hay"), "anda por ahí")

    def test_hay_existencial_se_conserva(self):
        for frase in ["hay muchos niños", "no hay de qué", "por hay muchos perros",
                      "está, hay que ir"]:
            self.assertEqual(correct_grammar(frase), frase)

    def test_articulo_femenino_por_sufijo(self):
        self.assertEqual(correct_grammar("el canción es bonita"), "la canción es bonita")
        self.assertEqual(correct_grammar("voy al estación"), "voy a la estación")
        self.assertEqual(correct_grammar("un ciudad grande"), "una ciudad grande")
        self.assertEqual(correct_grammar("el edad de mi abuelo"), "la edad de mi abuelo")

    def test_articulo_correcto_o_masculino_se_conserva(self):
        for frase in ["la canción es bonita", "el camión es grande", "el avión vuela"]:
            self.assertEqual(correct_grammar(frase), frase)

    def test_posesivo_ante_plural(self):
        self.assertEqual(correct_grammar("mi amigos vinieron"), "mis amigos vinieron")
        self.assertEqual(correct_grammar("su hermanas juegan"), "sus hermanas juegan")
        self.assertEqual(correct_grammar("Mi papás trabajan"), "Mis papás trabajan")

    def test_posesivo_singular_o_pronombre_se_conserva(self):
        for frase in ["mi dios", "mi amigo vino", "mi lunes favorito", "mi Carlos es alto"]:
            self.assertEqual(correct_grammar(frase), frase)
        # "tu" + verbo en -s sigue siendo el pronombre (regla 8), no "tus".
        self.assertEqual(correct_grammar("tu juegas mucho"), "tú juegas mucho")

    def test_colectivo_femenino_predicado_femenino(self):
        self.assertEqual(correct_grammar("la gente estan contentos"), "la gente está contenta")
        self.assertEqual(correct_grammar("mi familia son muy buenos"), "mi familia es muy buena")

    def test_colectivo_masculino_predicado_masculino(self):
        self.assertEqual(correct_grammar("el equipo estan contentos"), "el equipo está contento")


class TestAuxiliarHaber(unittest.TestCase):
    """Reglas del 2026-09-30 (test_ninos_reales_dev)."""

    def test_a_mas_participio_es_ha(self):
        self.assertEqual(correct_grammar("muchas cosas raras a habido muchos asaltos"),
                         "muchas cosas raras ha habido muchos asaltos")
        self.assertEqual(correct_grammar("Karol a ido de paseo"), "Karol ha ido de paseo")

    def test_a_preposicion_se_conserva(self):
        for frase in ["fue a cuidado de su tía", "a pedido de su madre", "voy a nado",
                      "le dio el libro a Conrado", "se fue a dormido"]:
            self.assertEqual(correct_grammar(frase), frase)

    def test_aya_mas_participio_es_haya(self):
        self.assertEqual(correct_grammar("no sea que le aya pasado algo"),
                         "no sea que le haya pasado algo")
        self.assertEqual(correct_grammar("la aya cuidaba al niño"), "la aya cuidaba al niño")

    def test_por_hay_ante_verbo_es_por_ahi(self):
        self.assertEqual(correct_grammar("paseaba por hay voy a entrar"), "paseaba por ahí voy a entrar")


if __name__ == "__main__":
    unittest.main()
