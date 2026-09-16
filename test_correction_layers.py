"""
test_correction_layers.py

Integración pura (sin torch/transformers/GPU) de las capas del pipeline:

  - `CorrectionPipeline.correct()` de punta a punta con juez BETO falso,
    T5 falso, sin T5 y con T5 que lanza excepción: la recomendada
    (`suggestions[0]`, de la que dependen `correctedText`, evaluate.py y TAS)
    es el beam 1 de T5 si es seguro y léxicamente plausible y, en cualquier
    otro caso, la base (nunca el beam 2); las segundas lecturas de BETO solo
    pueden ir detrás.
  - Contrato de `generate_with_lora` (mismo que `generate_corrections`):
    4 beams, `min(num_returns, 4)` secuencias, `[(texto, score)]` y modelo
    base restaurado aunque `generate()` falle.

Los módulos pesados se sustituyen por stubs acotados a cada clase de test con
`unittest.mock.patch.dict(sys.modules, ...)`: al terminar la clase, sys.modules
y los atributos de paquete quedan como estaban, así que test_modelo.py (modelo
real) puede correr en el mismo proceso.

Uso:
    .venv\\Scripts\\python.exe -m unittest test_correction_layers -v
"""
import contextlib
import importlib
import json
import os
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

_MISSING = object()
_REPO_DIR = Path(__file__).resolve().parent
_NO_TORCH_CHECK_ENV = "TESIS_NO_TORCH_CHECK_CHILD"


def run_class_in_clean_process(module_name: str, class_name: str) -> dict:
    """
    Ejecuta la clase de test indicada en un intérprete nuevo y devuelve
    {"ok": bool, "loaded": [módulos pesados presentes en sys.modules]}.
    Así "no se cargó torch" no depende del orden de los tests del proceso
    actual (p. ej. `discover` con test_modelo.py, que carga el modelo real).
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


def import_isolated(test_class, module_name: str, stubs: dict):
    """
    Importa `module_name` de cero con `stubs` registrados en sys.modules y
    restaura sys.modules y el atributo del paquete padre al terminar la clase.
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


# ---------------------------------------------------------------------------
# Capas 1-5: CorrectionPipeline.correct() con colaboradores falsos
# ---------------------------------------------------------------------------

def _pipeline_stubs() -> dict:
    judge_mod = types.ModuleType("infrastructure.nlp.context_judge")
    judge_mod.ContextJudge = type("ContextJudge", (), {})
    t5_mod = types.ModuleType("infrastructure.ml.t5_model")
    t5_mod.T5CorrectionModel = type("T5CorrectionModel", (), {})
    t5_mod.T5SpanishTokenizer = type("T5SpanishTokenizer", (), {})
    return {"infrastructure.nlp.context_judge": judge_mod,
            "infrastructure.ml.t5_model": t5_mod}


class FakePhonetic:
    """Capas 1-2 neutras: sin diccionario fonético, tildes y eñes se dejan como están."""

    def __init__(self, homophones=None, word_freqs=None):
        self._homophones   = dict(homophones or {})
        self.word_freqs    = dict(word_freqs or {})
        self.phonetic_dict = {}

    def homophone_candidates(self, word, max_candidates=6):
        return list(self._homophones.get(word, []))

    def restore_accent(self, word):
        return word

    def restore_enye(self, word):
        return word


class FakeSymSpell:
    """Toda palabra cuenta como bien escrita (lookup exacto truthy)."""

    def lookup(self, term, verbosity, max_edit_distance=2, **kwargs):
        return [term] if max_edit_distance == 0 else []


class FakeJudge:
    def __init__(self, scores):
        self._scores = scores

    def score_candidates(self, context_words, target_index, candidates):
        scored = [(c, self._scores.get(c, -10.0)) for c in candidates]
        scored.sort(key=lambda cs: -cs[1])
        return scored


class FakeSeq2Seq:
    """Devuelve beams fijos `[(texto, score)]` o lanza `error`."""

    def __init__(self, beams=(), error=None):
        self._beams = list(beams)
        self._error = error
        self.calls  = []

    def generate_corrections(self, text, tokenizer, num_returns=2):
        self.calls.append((text, num_returns))
        if self._error is not None:
            raise self._error
        return list(self._beams)


class CorrectPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = import_isolated(cls, "infrastructure.nlp.correction_pipeline", _pipeline_stubs())

    def _pipeline(self, *, judge=None, seq2seq=None, homophones=None, word_freqs=None):
        pipe = self.mod.CorrectionPipeline.__new__(self.mod.CorrectionPipeline)
        pipe._phonetic  = FakePhonetic(homophones, word_freqs)
        pipe._symspell  = FakeSymSpell()
        pipe._judge     = judge
        pipe._seq2seq   = seq2seq
        pipe._tokenizer = object() if seq2seq is not None else None
        return pipe

    def test_no_torch_loaded(self):
        # Independiente del orden: esta clase entera corre en un intérprete
        # limpio y no debe cargar torch/transformers/peft (aunque en el
        # proceso actual ya haya corrido test_modelo.py con el modelo real).
        if os.environ.get(_NO_TORCH_CHECK_ENV):
            self.skipTest("ya se está comprobando desde el proceso padre")
        outcome = run_class_in_clean_process("test_correction_layers", "CorrectPipelineTests")
        self.assertTrue(outcome["ok"], "la clase falló en el proceso limpio")
        self.assertEqual(outcome["loaded"], [], "módulos pesados cargados en un test puro")

    def test_real_beams_offer_indicative_and_subjunctive(self):
        # Beams reales del smoke sobre "a mi me gusta que ellos juega mucho"
        # (la capa 3 ya puso "mí"): juegan (recomendada) y jueguen son un
        # contraste de modo; la base (juega: forma del original e indicativo
        # tras "me gusta que") no se ofrece. El léxico de frecuencias del motor
        # fonético decide la conjugación (jugar → -ar).
        seq2seq = FakeSeq2Seq([("a mí me gusta que ellos juegan mucho", -0.044),
                               ("a mí me gusta que ellos jueguen mucho", -0.184),
                               ("A mí me gusta que ellos juegan mucho", -0.372)])
        pipe = self._pipeline(judge=FakeJudge({}), seq2seq=seq2seq, word_freqs={"jugar": 1000})
        out = pipe.correct("a mi me gusta que ellos juega mucho", {})
        self.assertEqual(out, ["a mí me gusta que ellos juegan mucho",
                               "a mí me gusta que ellos jueguen mucho"])
        self.assertEqual(seq2seq.calls, [("a mí me gusta que ellos juega mucho", self.mod._T5_NUM_RETURNS)])
        # Sin léxico no se puede saber cuál es el indicativo: fail-closed.
        pipe = self._pipeline(judge=FakeJudge({}), seq2seq=FakeSeq2Seq(seq2seq._beams))
        self.assertEqual(pipe.correct("a mi me gusta que ellos juega mucho", {}),
                         ["a mí me gusta que ellos juegan mucho"])

    def test_alternative_with_an_extra_edit_or_a_noun_reading_is_not_offered(self):
        # Cierre (N3): "jueguen muchos" cambia dos palabras respecto a la
        # recomendada → fuera; "la case" no es un verbo (casa ≫ casar en el
        # léxico de frecuencias) → fuera.
        seq2seq = FakeSeq2Seq([("ellos juegan mucho", -1.0), ("ellos jueguen muchos", -1.1)])
        pipe = self._pipeline(judge=FakeJudge({}), seq2seq=seq2seq, word_freqs={"jugar": 1000})
        self.assertEqual(pipe.correct("ellos juega mucho", {}), ["ellos juegan mucho"])
        seq2seq = FakeSeq2Seq([("la casa es bonita", -0.01), ("la case es bonita", -0.05)])
        pipe = self._pipeline(judge=FakeJudge({}), seq2seq=seq2seq,
                              word_freqs={"casa": 496977, "case": 4770, "casar": 10364})
        self.assertEqual(pipe.correct("la casa es bonita", {}), ["la casa es bonita"])

    def test_manual_correction_does_not_lose_t5_agreement(self):
        # "xq→porque" (capa 1) y T5 corrige juega→juegan: la recomendada es el
        # beam; la base (juega) no es otra lectura (concordancia, forma original).
        seq2seq = FakeSeq2Seq([("porque ellos juegan mucho", -0.04)])
        pipe = self._pipeline(judge=FakeJudge({}), seq2seq=seq2seq)
        self.assertEqual(pipe.correct("xq ellos juega mucho", {}), ["porque ellos juegan mucho"])

    def test_obligatory_subjunctive_filters_indicative_alternatives_end_to_end(self):
        # Beams reales de "es posible que ellos llega tarde": lleguen
        # (recomendada), llegan (indicativo tras es posible que: fuera) y
        # llegue (número: sin señal). Una sola opción.
        seq2seq = FakeSeq2Seq([("es posible que ellos lleguen tarde", -0.123),
                               ("es posible que ellos llegan tarde", -0.158),
                               ("es posible que ellos llegue tarde", -0.409)])
        pipe = self._pipeline(judge=FakeJudge({}), seq2seq=seq2seq, word_freqs={"llegar": 1000})
        self.assertEqual(pipe.correct("es posible que ellos llega tarde", {}),
                         ["es posible que ellos lleguen tarde"])

    def test_short_irregular_verb_correction_is_recommended(self):
        seq2seq = FakeSeq2Seq([("la gente es muy amable", -0.05)])
        pipe = self._pipeline(judge=FakeJudge({}), seq2seq=seq2seq)
        self.assertEqual(pipe.correct("la gente son muy amables", {}), ["la gente es muy amable"])

    def test_without_t5_tied_beto_reading_goes_after_the_base(self):
        # esta/está empatados (dif 0.13 < 0.3): _disambiguate_context no
        # sobrescribe y la capa 5 tampoco: la base (igual al original) sigue en [0].
        pipe = self._pipeline(judge=FakeJudge({"esta": -3.93, "está": -3.80}),
                              homophones={"esta": ["está"]})
        self.assertEqual(pipe.correct("esta bien", {}), ["esta bien", "está bien"])

    def test_t5_exception_falls_back_to_base_then_variants(self):
        pipe = self._pipeline(judge=FakeJudge({"esta": -3.93, "está": -3.80}),
                              seq2seq=FakeSeq2Seq(error=RuntimeError("CUDA out of memory")),
                              homophones={"esta": ["está"]})
        self.assertEqual(pipe.correct("esta bien", {}), ["esta bien", "está bien"])

    def test_all_beams_unsafe_returns_base(self):
        seq2seq = FakeSeq2Seq([("el niño juega y corre y salta y canta todos los días en el parque", -0.1),
                               ("", -0.2)])
        pipe = self._pipeline(judge=FakeJudge({}), seq2seq=seq2seq)
        self.assertEqual(pipe.correct("el niño juega", {}), ["el niño juega"])

    def test_unsafe_first_beam_recommends_base_not_second_beam(self):
        # Regla pre-Task 5: "beam 1 si es seguro, si no la base". Si el mejor
        # beam alucina, T5 no es fiable para esa frase: no se salta al beam 2
        # para recomendar (el beam 2 seguro solo podría ir detrás de la base).
        unsafe = "el niño juega y corre y salta y canta todos los días en el parque"
        seq2seq = FakeSeq2Seq([(unsafe, -0.1), ("el niño juega.", -0.2)])
        pipe = self._pipeline(judge=FakeJudge({}), seq2seq=seq2seq)
        out = pipe.correct("el niño juega", {})
        self.assertEqual(out[0], "el niño juega")
        self.assertNotIn(unsafe, out)
        seq2seq = FakeSeq2Seq([("", -0.1), ("los niños juegan", -0.2)])
        pipe = self._pipeline(judge=FakeJudge({}), seq2seq=seq2seq)
        out = pipe.correct("los niño juega", {})
        self.assertEqual(out[0], "los niño juega")
        self.assertEqual(out, ["los niño juega"])   # T5 no cuenta para esta frase

    def test_lexical_substitution_in_first_beam_recommends_base(self):
        # Beams reales sobre "la baca comio pasto verde" (la base ya dice
        # "la vaca comió pasto verde"): el beam 1 sustituye pasto→maíz
        # (similitud 0.22 < recommendedMinSimilarity 0.3). Pasa la guarda de
        # cantidad pero no la léxica: se recomienda la base y no se ofrece
        # ninguna sustitución léxica como alternativa.
        base = "la vaca comió pasto verde"
        seq2seq = FakeSeq2Seq([("la vaca comió maíz verde", -0.288),
                               ("la vaca comió carne verde", -0.405),
                               ("la vaca comió pasto rojo", -0.653)])
        pipe = self._pipeline(judge=FakeJudge({}), seq2seq=seq2seq)
        self.assertEqual(pipe.correct(base, {}), [base])
        self.assertEqual(self.mod._RECOMMENDED_MIN_SIMILARITY,
                         self.mod.THRESHOLDS["recommendedMinSimilarity"])

    def test_first_beam_that_drops_a_negation_recommends_base(self):
        # Un beam 1 que borra "no" (o inserta un cuantificador) pasa la guarda
        # de cantidad pero no la léxica: la recomendada es la base y el beam
        # no se ofrece ni como alternativa.
        seq2seq = FakeSeq2Seq([("quiero ir", -0.05), ("no quiero ir.", -0.1)])
        pipe = self._pipeline(judge=FakeJudge({}), seq2seq=seq2seq)
        self.assertEqual(pipe.correct("no quiero ir", {}), ["no quiero ir"])
        seq2seq = FakeSeq2Seq([("todos los niños juegan", -0.05)])
        pipe = self._pipeline(judge=FakeJudge({}), seq2seq=seq2seq)
        self.assertEqual(pipe.correct("los niños juegan", {}), ["los niños juegan"])
        # Insertar un artículo sí es una corrección recomendable.
        seq2seq = FakeSeq2Seq([("llegué tarde a la clase", -0.05)])
        pipe = self._pipeline(judge=FakeJudge({}), seq2seq=seq2seq)
        self.assertEqual(pipe.correct("llego tarde a clase", {})[0], "llegué tarde a la clase")

    def test_first_beam_that_swaps_a_negation_one_to_one_recommends_base(self):
        # Cierre: un beam 1 que sustituye 1:1 un negador por una palabra
        # parecida (yo→no, no→lo) pasa la guarda de cantidad y la similitud
        # (0.5 ≥ 0.3) pero invierte el sentido: se recomienda la base y el
        # beam no se ofrece como alternativa.
        for original, beam in [("yo quiero ir", "no quiero ir"), ("no quiero ir", "lo quiero ir"),
                               ("lo quiero ir", "no quiero ir"), ("ni quiero ir", "mi quiero ir")]:
            seq2seq = FakeSeq2Seq([(beam, -0.05), (original + ".", -0.1)])
            pipe = self._pipeline(judge=FakeJudge({}), seq2seq=seq2seq)
            self.assertEqual(pipe.correct(original, {}), [original], (original, beam))
        # La flexión de la misma protegida sí se recomienda (todos→todas).
        seq2seq = FakeSeq2Seq([("todas las niñas juegan", -0.05)])
        pipe = self._pipeline(judge=FakeJudge({}), seq2seq=seq2seq)
        self.assertEqual(pipe.correct("todos las niñas juegan", {})[0], "todas las niñas juegan")

    def test_lexical_guard_keeps_irregular_agreement_in_first_beam(self):
        # es→son (0.40), hizo→hicieron (0.50), viene→vengan (0.55) siguen
        # recomendándose; una flexión no es una sustitución léxica.
        for original, beam in [("la gente es muy amables", "la gente son muy amables"),
                               ("ellos hizo la tarea", "ellos hicieron la tarea"),
                               ("espero que ellos viene", "espero que ellos vengan")]:
            pipe = self._pipeline(judge=FakeJudge({}), seq2seq=FakeSeq2Seq([(beam, -0.05)]))
            self.assertEqual(pipe.correct(original, {})[0], beam)

    def test_lexical_guard_does_not_apply_to_later_beams(self):
        # El beam 1 es plausible y se recomienda; el beam 2 (sustitución
        # léxica juega→comen) solo lo frena la regla (e) de las alternativas,
        # y un beam 2 flexivo (jueguen; jugar en el léxico) sí se ofrece. La
        # base, igual al original, nunca se ofrece.
        seq2seq = FakeSeq2Seq([("los niños juegan en el parque", -0.05),
                               ("los niños comen en el parque", -0.1)])
        pipe = self._pipeline(judge=FakeJudge({}), seq2seq=seq2seq, word_freqs={"jugar": 1000})
        self.assertEqual(pipe.correct("los niño juega en el parque", {}),
                         ["los niños juegan en el parque"])
        seq2seq = FakeSeq2Seq([("los niños juegan en el parque", -0.05),
                               ("los niños jueguen en el parque", -0.1)])
        pipe = self._pipeline(judge=FakeJudge({}), seq2seq=seq2seq, word_freqs={"jugar": 1000})
        self.assertEqual(pipe.correct("los niño juega en el parque", {}),
                         ["los niños juegan en el parque", "los niños jueguen en el parque"])

    def test_without_judge_and_without_t5_returns_base_only(self):
        pipe = self._pipeline(judge=None, seq2seq=None, homophones={"esta": ["está"]})
        self.assertEqual(pipe.correct("esta bien", {}), ["esta bien"])

    def test_identity_best_beam_keeps_base_recommended_and_variant_behind(self):
        # T5 dice que la frase está bien (identidad −0.01) y BETO empata
        # está/esta: la recomendada es la base, la variante va detrás; el beam
        # a −0.5 queda fuera por (d) con el margen de 0.3.
        original = "esta bien, nos vemos luego"
        seq2seq = FakeSeq2Seq([(original, -0.01), ("Esta bien, nos vemos luego.", -0.5)])
        pipe = self._pipeline(judge=FakeJudge({"esta": -3.93, "está": -3.80}),
                              seq2seq=seq2seq, homophones={"esta": ["está"]})
        self.assertEqual(pipe.correct(original, {}), [original, "está bien, nos vemos luego"])

    def test_beto_variant_is_gated_by_the_tie_margin(self):
        # tubo/tuvo dif 1.0 (< margen de sobrescritura 5.0, > empate 0.3): ni
        # se sobrescribe ni se ofrece. Con dif 0.13 se ofrece detrás de la base.
        pipe = self._pipeline(judge=FakeJudge({"tubo": -3.0, "tuvo": -2.0}), homophones={"tubo": ["tuvo"]})
        self.assertEqual(pipe.correct("el tubo un accidente", {}), ["el tubo un accidente"])
        # La variante pasa por la capa 3 igual que la base ("el tuvo" → "él tuvo").
        pipe = self._pipeline(judge=FakeJudge({"tubo": -2.13, "tuvo": -2.0}), homophones={"tubo": ["tuvo"]})
        self.assertEqual(pipe.correct("el tubo un accidente", {}),
                         ["el tubo un accidente", "él tuvo un accidente"])

    def test_written_accent_survives_beto_and_is_not_offered_without_it(self):
        # "quizás" escrito con tilde: BETO prefiere "quizas" (dif > 0.5) pero
        # la tilde escrita se respeta y la base con "quizas" ya no aparece como
        # opción 2. T5 solo corrige la concordancia (llega→llegan: sin señal).
        seq2seq = FakeSeq2Seq([("quizás ellos llega tarde", -0.12), ("quizás ellos llegan tarde", -0.35)])
        pipe = self._pipeline(judge=FakeJudge({"quizás": -3.0, "quizas": -1.0}),
                              seq2seq=seq2seq, homophones={"quizás": ["quizas"]})
        self.assertEqual(pipe.correct("quizás ellos llega tarde", {}), ["quizás ellos llega tarde"])
        self.assertEqual(seq2seq.calls[0][0], "quizás ellos llega tarde")

    def test_beto_overwrite_still_wins_when_confident(self):
        pipe = self._pipeline(judge=FakeJudge({"esta": -3.0, "está": -1.0}), homophones={"esta": ["está"]})
        self.assertEqual(pipe.correct("esta bien", {}), ["está bien"])

    def test_refined_beam_keeps_base_capitalization(self):
        seq2seq = FakeSeq2Seq([("El niño juega.", -0.1)])
        pipe = self._pipeline(judge=FakeJudge({}), seq2seq=seq2seq)
        self.assertEqual(pipe.correct("el niño juega", {}), ["el niño juega."])


# ---------------------------------------------------------------------------
# Capa 4: contrato de generate_with_lora con modelo/tokenizer falsos
# ---------------------------------------------------------------------------

class _FakeTensor:
    def __init__(self, first=5):
        self._first = first

    def to(self, device):
        return self

    def __getitem__(self, idx):
        return self

    def item(self):
        return self._first


class FakeTokenizer:
    def encode_single(self, text, max_length=128):
        return {"input_ids": _FakeTensor(), "attention_mask": _FakeTensor()}

    def decode(self, token_ids, skip_special=True):
        return str(token_ids)


class _FakeScores:
    def __init__(self, values):
        self._values = list(values)

    def tolist(self):
        return list(self._values)


class _FakeOutputs:
    def __init__(self, sequences, scores):
        self.sequences        = list(sequences)
        self.sequences_scores = _FakeScores(scores)


class FakeRestoredModel:
    """Lo que devuelve get_base_model(): debe aceptar generation_config y .to()."""

    def to(self, device):
        return self


class FakeBaseModel:
    def __init__(self):
        self.model  = "modelo-base"
        self.device = "cpu"
        self.cleaned = 0

    def _clean_peft_if_attached(self):
        self.cleaned += 1


class FakePeft:
    """Adaptador falso: registra kwargs de generate() y el ciclo unload/base."""
    instances = []
    generate_error = None
    outputs = _FakeOutputs([" beam uno ", "beam dos"], [-0.1, -0.5])

    def __init__(self, base):
        self.base = base
        self.restored = FakeRestoredModel()
        self.generate_kwargs = None
        self.unloaded = False
        self.evaluated = False
        FakePeft.instances.append(self)

    @classmethod
    def from_pretrained(cls, base, lora_dir, is_trainable=False):
        assert is_trainable is False
        return cls(base)

    def to(self, device):
        return self

    def eval(self):
        self.evaluated = True

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        if FakePeft.generate_error is not None:
            raise FakePeft.generate_error
        return FakePeft.outputs

    def unload(self):
        self.unloaded = True

    def get_base_model(self):
        return self.restored


class FakeGenerationConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _torch_stubs() -> dict:
    torch_mod = types.ModuleType("torch")
    torch_mod.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch_mod.no_grad = contextlib.nullcontext
    torch_mod.Tensor = object
    torch_mod.device = lambda name: name
    dist_mod = types.ModuleType("torch.distributed")
    dist_mod.is_initialized = lambda: False
    tensor_mod = types.ModuleType("torch.distributed.tensor")
    torch_mod.distributed = dist_mod
    dist_mod.tensor = tensor_mod

    transformers_mod = types.ModuleType("transformers")
    for name in ("AutoModelForSeq2SeqLM", "AutoTokenizer", "Seq2SeqTrainer",
                 "Seq2SeqTrainingArguments", "DataCollatorForSeq2Seq"):
        setattr(transformers_mod, name, type(name, (), {}))
    transformers_mod.GenerationConfig = FakeGenerationConfig

    datasets_mod = types.ModuleType("datasets")
    datasets_mod.Dataset = type("Dataset", (), {})

    peft_mod = types.ModuleType("peft")
    peft_mod.PeftModel = FakePeft

    return {"torch": torch_mod, "torch.distributed": dist_mod, "torch.distributed.tensor": tensor_mod,
            "transformers": transformers_mod, "datasets": datasets_mod, "peft": peft_mod}


class GenerateWithLoraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = import_isolated(cls, "infrastructure.ml.t5_model", _torch_stubs())

    def setUp(self):
        FakePeft.instances = []
        FakePeft.generate_error = None
        self.base = FakeBaseModel()

    def _generate(self, num_returns):
        return self.mod.generate_with_lora(self.base, FakeTokenizer(), "models/grammar_lora",
                                           "los niño juega", num_returns=num_returns)

    def test_returns_sequences_aligned_with_scores(self):
        out = self._generate(2)
        self.assertEqual(out, [("beam uno", -0.1), ("beam dos", -0.5)])
        self.assertEqual(self.base.cleaned, 1)

    def test_generate_kwargs_share_the_base_generator_contract(self):
        self._generate(2)
        kwargs = FakePeft.instances[-1].generate_kwargs
        self.assertEqual(kwargs["num_beams"], 4)
        self.assertEqual(kwargs["num_return_sequences"], 2)
        self.assertFalse(kwargs["do_sample"])
        self.assertTrue(kwargs["output_scores"])
        self.assertTrue(kwargs["return_dict_in_generate"])
        self.assertEqual(kwargs["max_new_tokens"], self.mod.MAX_TARGET_LEN)
        self.assertIn("input_ids", kwargs)   # va como keyword, no posicional

    def test_num_returns_is_capped_at_four_beams(self):
        self._generate(7)
        kwargs = FakePeft.instances[-1].generate_kwargs
        self.assertEqual((kwargs["num_beams"], kwargs["num_return_sequences"]), (4, 4))
        with self.assertRaises(ValueError):
            self._generate(0)

    def test_base_model_is_restored_after_generation(self):
        self._generate(1)
        peft = FakePeft.instances[-1]
        self.assertTrue(peft.evaluated and peft.unloaded)
        self.assertIs(self.base.model, peft.restored)
        self.assertIsInstance(self.base.model.generation_config, FakeGenerationConfig)
        self.assertEqual(self.base.model.generation_config.kwargs, self.mod._SAFE_GENERATION_CONFIG)

    def test_base_model_is_restored_and_lock_released_when_generate_raises(self):
        FakePeft.generate_error = RuntimeError("CUDA error")
        with self.assertRaises(RuntimeError):
            self._generate(2)
        peft = FakePeft.instances[-1]
        self.assertTrue(peft.unloaded)
        self.assertIs(self.base.model, peft.restored)
        self.assertIsInstance(self.base.model.generation_config, FakeGenerationConfig)
        self.assertTrue(self.mod.gpu_lock.acquire(blocking=False), "gpu_lock quedó tomado")
        self.mod.gpu_lock.release()

    def test_no_real_torch_loaded(self):
        self.assertIs(sys.modules["torch"], self.mod.torch)
        self.assertFalse(hasattr(sys.modules["torch"], "__file__"))


if __name__ == "__main__":
    unittest.main()
