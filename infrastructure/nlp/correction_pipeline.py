"""
infrastructure/nlp/correction_pipeline.py
Pipeline neuro-simbólico-fonético de corrección.
Orquesta las capas de ortografía y finaliza con Seq2Seq para gramática de forma limpia.
"""
import difflib
import os
import re
import unicodedata
from symspellpy import SymSpell, Verbosity

from infrastructure.nlp.phonetic_engine import PhoneticEngine, match_case, to_phonetic, DICT_PATH, load_supplement
from infrastructure.nlp.context_judge import ContextJudge
from infrastructure.nlp.amalgams import expand_amalgams
from infrastructure.nlp.confusions import fix_lexical_confusions, resolve_final_confusions
from infrastructure.nlp.grammar_rules import correct_auxiliaries, correct_grammar, normalize_modern_spelling
from infrastructure.nlp.candidates import dyslexic_cost, is_derived_form, is_regular_verb_form, rank_candidates
from infrastructure.nlp.segmentation import (
    FUNCTION_WORDS, choose_split, is_spanish_word, join_split_words, split_candidates,
)
from infrastructure.nlp.alternatives import THRESHOLDS, select_alternatives
from infrastructure.ml.t5_model import T5CorrectionModel, T5SpanishTokenizer
from infrastructure.ml.guards import (
    _lexical_key, is_lexically_plausible_refinement, is_safe_refinement, revert_lexical_substitutions,
)

MANUAL_CORRECTIONS = {
    "uillos": "niños",  "ciubab": "ciudad",  "caíbas": "caídas",
    "bamo":   "vamos",  "bamos":  "vamos",   "oy":     "hoy",
    "ise":    "hice",   "iso":    "hizo",     "iva":    "iba",
    "aiga":   "haya",   "haiga":  "haya",     "ay":     "hay",
    "dondis": "donde",  "dodes":  "donde",    "dondes": "donde",
    "be":     "de",     "ala":    "a la",     "delo":   "de lo",
    "delos":  "de los", "dela":   "de la",    "delas":  "de las",
    # Abreviaturas de chat/disgrafía (se corrigen antes de la guarda de ≤2 letras)
    "qe":     "que",    "q":      "que",      "porqe":  "porque",
    "xq":     "porque", "pq":     "porque",   "tb":     "también",
    "tmb":    "también", "i":     "y",        "mui":    "muy",
}

# Palabras con acento que SymSpell no debe tocar
_ACCENTED_RE = re.compile(r'[áéíóúüñÁÉÍÓÚÜÑ]')
_PUNCT_RE    = re.compile(r'^([^\wáéíóúüñÁÉÍÓÚÜÑ]*)(.*?)([^\wáéíóúüñÁÉÍÓÚÜÑ]*)$')

# Márgenes (en log-prob de BETO) que el mejor candidato debe superar a la
# palabra original para que la pasada de desambiguación la sobrescriba.
# Calibrados contra la batería de test_modelo.py:
#   - Variante SOLO-ACENTO (esta/está, compre/compré): barata y segura → bajo.
#   - HOMÓFONO de palabras distintas (tubo/tuvo, boy/voy): arriesgado; solo se
#     acepta cuando BETO está segurísimo (evita el falso tuvo→tubo) → alto.
#   - MONOSÍLABO diacrítico (el/él, se/sé): altísima frecuencia base → muy
#     conservador para no corromper artículos/preposiciones correctos.
_MARGIN_ACCENT    = 0.5
_MARGIN_HOMOPHONE = 5.0
_MARGIN_MONO      = 3.0

# Señal de ambigüedad en homófonos: si |score(alternativa) − score(actual)|
# < _BETO_TIE_MARGIN (diferencia de PLL media de BETO, ver
# alternatives.THRESHOLDS["betoTieMargin"]) las dos lecturas están empatadas:
# el texto principal no cambia pero la segunda lectura se ofrece como
# candidato (ver _disambiguate_context). Es un umbral en unidades de BETO,
# independiente de los márgenes de sobrescritura de arriba.
_BETO_TIE_MARGIN = THRESHOLDS["betoTieMargin"]

# Como mucho tantas posiciones ambiguas por frase (las más empatadas) generan
# una variante; una alternativa por posición.
_MAX_AMBIGUOUS_POSITIONS = 3

# Una segunda lectura mucho más rara (en el corpus es_50k) que la palabra
# escrita no se ofrece como ambigüedad (ver alternatives.THRESHOLDS
# ["maxFreqRatio"]). La sobrescritura de la capa 2.5 no usa esta guarda: BETO
# puede seguir imponiendo la lectura rara si supera su margen.
_AMBIGUITY_MAX_FREQ_RATIO = THRESHOLDS["maxFreqRatio"]

# Cuántos beams de T5 se piden como candidatos (num_beams=4 en el modelo).
_T5_NUM_RETURNS = 3

# Margen de score respecto al mejor beam para que una alternativa se ofrezca
# (ver alternatives.THRESHOLDS["scoreMargin"]).
_SCORE_MARGIN = THRESHOLDS["scoreMargin"]

# Guarda léxica del beam RECOMENDADO (beam 1): similitud mínima, sin tildes,
# de cada reemplazo 1:1 de palabra respecto a la base (ver
# alternatives.THRESHOLDS["recommendedMinSimilarity"] y
# guards.is_lexically_plausible_refinement). Bloquea pasto→maíz, no es→son.
_RECOMMENDED_MIN_SIMILARITY = THRESHOLDS["recommendedMinSimilarity"]

# Capa 2 con contexto (2026-09-30): entre los candidatos de SymSpell a menos
# de _CANDIDATE_TIE_WINDOW del mejor coste disléxico (entrege -> entrega /
# entregué / entregó, utimos -> átomos / últimos) BETO elige, si supera al de
# menor coste por _CANDIDATE_BETO_MARGIN (log-prob media de la ranura); si
# no, gana el de menor coste (a igual coste, el más frecuente).
_CANDIDATE_TIE_WINDOW = 0.4        # tío (0,85) frente a tuyo (0,5) sí; cubra (1,0) frente a quiebra (0,5) no
_CANDIDATE_BETO_MARGIN = 1.0
_CANDIDATE_MAX = 4
# Palabra de es_50k demasiado rara para fiarse (ruido de subtítulos: quero
# 228, com 1 210; ver segmentation.is_spanish_word) con una rival a coste
# <= _RIVAL_MAX_COST y _RIVAL_MIN_RATIO veces más frecuente: BETO decide.
_RIVAL_MAX_COST = 0.5
_RIVAL_MIN_RATIO = 100
_RIVAL_MAX_LEN = 5            # quero, com; no "defino" ni "covers" (formas raras pero válidas)
_RIVAL_BETO_MARGIN = 3.0      # com -> con gana por 7,4; azur -> azar, covers -> cobres no
# Enclítico con "c" por "s": tragárcelo -> tragárselo, vercelo -> vérselo,
# diciéndocelo -> diciéndoselo. Delante debe quedar un infinitivo del
# diccionario (5+ letras o uno corto de la lista) o un gerundio: no "Marcelo".
_ENCLITIC_CE_RE = re.compile(r"^(.+?(?:[aáeéií]r|ndo))ce(l[oa]s?)$")
_SHORT_INFINITIVES = {"ver", "dar", "ir", "leer", "oir", "ser", "reir", "freir", "traer", "caer"}
_STRESS = str.maketrans("aei", "áéí")
# Palabra en mayúscula al abrir la frase y fuera del diccionario: puede ser un
# nombre (Cacachi, Toquepala). Solo se corrige si el mejor candidato es un error
# barato (Entonses, Binieron: 0,5) y solo se parte si empieza por una palabra
# funcional (Sefue); Cacachi -> Kakashi (1,5) y Toquepala -> "toque pala" no.
# Papel del T5 (experimento del 2026-09-30, ver README): "always" (por defecto,
# el de producción) o "verified": cada palabra que T5 cambia se acepta solo si
# BETO la puntúa en la frase al menos como la original + _T5_VERIFY_MARGIN
# (sin contar cambios solo de tilde/mayúscula, donde BETO tiene sesgo).
_T5_MODE = os.environ.get("T5_MODE", "always")
# Pasadas del pipeline completo (1 en producción): con 2, la recomendada se
# corrige otra vez (frases con varios errores; experimento del 2026-09-30).
_CORRECTION_PASSES = int(os.environ.get("CORRECTION_PASSES", "1"))
_T5_VERIFY_MARGIN = float(os.environ.get("T5_VERIFY_MARGIN", "0.0"))
# Guarda del T5: "strict" (producción) protege todas las palabras de la base;
# "child" protege solo las que escribió el niño y las capas dejaron igual, y deja
# al T5 reescribir las conjeturas de las capas (podio -> pidió, sefemó -> se
# enfermó). Experimento del 2026-09-30 para LoRA v5.
_T5_GUARD = os.environ.get("T5_GUARD", "strict")

_INITIAL_MAX_COST = 0.5
# Determinantes: tras ellos no se pone la tilde de pretérito por frecuencia
# ("al cerro" no es "al cerró").
_DETERMINERS = {"el", "la", "los", "las", "un", "una", "unos", "unas", "al", "del", "mi", "mis",
                "tu", "tus", "su", "sus", "este", "esta", "ese", "esa", "aquel", "aquella", "otro", "otra"}
# Un homófono fonético se acepta solo si también es un error barato: la clave
# fonética iguala "cacachi" y "kakashi" (coste 1,5), no así kaye/calle (1,0).
_PHONETIC_MAX_COST = 1.0
_ACCENT_VOWELS = str.maketrans("áéíóúü", "aeiouu")


def _has_accent(word: str) -> bool:
    return bool(_ACCENTED_RE.search(word))


def _strip_accents(s: str) -> str:
    """Quita tildes para comparación sin destruir eñes."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


class CorrectionPipeline:
    def __init__(
        self,
        phonetic: PhoneticEngine,
        judge: ContextJudge,
        seq2seq: T5CorrectionModel = None,
        tokenizer: T5SpanishTokenizer = None,
    ):
        self._phonetic  = phonetic
        self._judge     = judge  # Mantenemos la firma para compatibilidad con inyección de dependencias
        self._symspell  = SymSpell(max_dictionary_edit_distance=3, prefix_length=7)
        # encoding explícito: sin él, en Windows se lee en cp1252 y las palabras
        # con tilde entran como "dormirÃ¡n" (SymSpell nunca las proponía).
        self._symspell.load_dictionary(DICT_PATH, term_index=0, count_index=1, encoding="utf-8")
        for word, count in load_supplement().items():      # vocabulario escolar peruano
            self._symspell.create_dictionary_entry(word, count)
        self._seq2seq   = seq2seq
        self._tokenizer = tokenizer

    # ------------------------------------------------------------------
    # Desambiguación contextual (BETO)
    # ------------------------------------------------------------------

    def _disambiguate_context(self, sentence: str,
                              trusted_accents=frozenset()) -> tuple[str, list[tuple[str, float]]]:
        """
        Pasada palabra-a-palabra: para cada término con homófonos/variantes de
        acento reales, deja que BETO elija el más coherente con la frase.
        Solo sobrescribe si la mejora en log-prob supera el margen del tipo de
        cambio (acento / homófono / monosílabo). Es idempotente y greedy:
        cada decisión usa como contexto las decisiones ya tomadas a su izquierda.

        `trusted_accents` son las palabras (minúsculas) que el alumno escribió
        CON tilde: esas nunca se sobrescriben por su variante sin tilde ni la
        generan como segunda lectura (la tilde escrita se confía, como en la
        capa 1). Una tilde que puso la capa 2 (`restore_accent`) no está
        protegida: BETO puede seguir decidiéndola por contexto (papá / Papa).

        Devuelve `(texto, ambiguous_variants)`. Cuando las dos mejores lecturas
        de una posición quedan a menos de `_BETO_TIE_MARGIN` (el contexto no
        decide), el texto no cambia pero se genera una variante con la segunda
        lectura sustituida, con score = −|diferencia| (más cerca de 0 cuanto
        más empatado; solo sirve para ordenar: la variante nunca es la
        recomendada). Como mucho una alternativa por posición y
        `_MAX_AMBIGUOUS_POSITIONS` posiciones (las de menor diferencia).
        """
        tokens = sentence.split()
        cores, affixes = [], []
        for tok in tokens:
            m = _PUNCT_RE.match(tok)
            if m:
                affixes.append((m.group(1), m.group(3)))
                cores.append(m.group(2))
            else:
                affixes.append(("", ""))
                cores.append(tok)

        ambiguous = []   # (|diferencia|, posición, lectura alternativa)

        for i, core in enumerate(cores):
            if len(core) < 2:
                continue

            lower = core.lower()
            pref0, suff0 = affixes[i]
            if lower == "ay" and ("¡" in pref0 or suff0.startswith((",", "!"))):
                continue                                 # interjección: "¡Ay, me duele!"
            candidates = self._phonetic.homophone_candidates(lower)
            # La palabra original siempre compite consigo misma (permite "no tocar").
            candidates = list(dict.fromkeys([lower] + candidates))
            if len(candidates) < 2:
                continue

            pref, suff  = affixes[i]
            cand_tokens = [pref + match_case(core, c) + suff for c in candidates]
            scored      = self._judge.score_candidates(tokens, i, cand_tokens)

            best, best_score = scored[0]
            current       = tokens[i]
            current_score = dict(scored).get(current, best_score)

            # Lectura alternativa: la mejor si no es la actual; si la actual ya
            # es la mejor, la segunda (solo para medir si el contexto decide).
            if best != current:
                alt, alt_score = best, best_score
            elif len(scored) >= 2:
                alt, alt_score = scored[1]
            else:
                continue

            alt_core = _PUNCT_RE.match(alt).group(2).lower()
            if lower in trusted_accents and _strip_accents(alt_core) == _strip_accents(lower):
                # Tilde escrita por el alumno: se respeta (como en la capa 1).
                # Ni se sobrescribe "quizás" por "quizas" aunque BETO la
                # prefiera, ni se ofrece la variante sin tilde como segunda
                # lectura.
                continue
            if len(lower) <= 2:
                margin = _MARGIN_MONO
            elif _strip_accents(alt_core) == _strip_accents(lower):
                margin = _MARGIN_ACCENT
            else:
                margin = _MARGIN_HOMOPHONE

            # > 0: BETO prefiere la alternativa; < 0: prefiere la palabra actual.
            diff = alt_score - current_score
            if diff >= margin:
                tokens[i] = alt
            elif current not in dict(scored) or alt_core == lower:
                # Token con caso mixto ("eSta"): match_case lo normalizó y no
                # está entre los puntuados; la "alternativa" solo cambia el
                # caso. No es una segunda lectura.
                continue
            elif abs(diff) < _BETO_TIE_MARGIN and self._is_plausible_reading(lower, alt_core):
                ambiguous.append((abs(diff), i, alt))

        text = " ".join(tokens)

        # Variantes "segunda lectura" sobre el texto final, las más empatadas primero.
        ambiguous.sort(key=lambda entry: (entry[0], entry[1]))
        variants = []
        for diff, i, alt in ambiguous[:_MAX_AMBIGUOUS_POSITIONS]:
            alt_tokens    = list(tokens)
            alt_tokens[i] = alt
            variants.append((" ".join(alt_tokens), -diff))

        return text, variants

    def _beto_verify(self, base: str, beam: str) -> str:
        """Modo T5_MODE=verified: revierte cada reemplazo 1:1 de T5 que BETO no
        puntúa (en la frase base) al menos como la palabra original + margen.
        Los cambios solo de tilde o mayúscula no se verifican."""
        base_tokens, beam_tokens = base.split(), beam.split()
        key = lambda w: _strip_accents(w.strip(".,;:!?¡¿\"'()")).lower()
        out = list(beam_tokens)
        matcher = difflib.SequenceMatcher(a=[key(w) for w in base_tokens],
                                          b=[key(w) for w in beam_tokens], autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != "replace" or i2 - i1 != j2 - j1:
                continue
            for k in range(i2 - i1):
                old, new = base_tokens[i1 + k], beam_tokens[j1 + k]
                try:
                    scores = dict(self._judge.score_candidates(base_tokens, i1 + k, [old, new]))
                except Exception as e:
                    print(f"[WARN] Verificación de T5 (BETO) falló: {e}")
                    continue
                if scores.get(new, float("-inf")) < scores.get(old, float("-inf")) + _T5_VERIFY_MARGIN:
                    out[j1 + k] = old
        return " ".join(out)

    @staticmethod
    def _enclitic_host(base: str, freqs: dict) -> bool:
        """`base` (lo que precede a "celo") es un gerundio o un infinitivo del
        diccionario de 5+ letras o de la lista corta (ver, dar, leer...)."""
        if base.endswith("ndo"):
            return True
        plain = base.translate(_ACCENT_VOWELS)
        return plain in _SHORT_INFINITIVES or (len(plain) >= 5 and plain in freqs)

    @staticmethod
    def _stressed(base: str) -> str:
        """Tilde de la forma con dos enclíticos: vér-, tragár-, diciéndo-."""
        if _has_accent(base):
            return base
        if base.endswith("ndo"):
            return base[:-4] + base[-4].translate(_STRESS) + "ndo"
        return base[:-2] + base[-2].translate(_STRESS) + base[-1]

    def _contextual_pick(self, words: list, i: int, pref: str, suff: str, core: str,
                         ranked: list) -> str | None:
        """Capa 2: el candidato de `ranked` ([(coste, término)], de menor a
        mayor coste) que sustituye a words[i]. Sin BETO, o si BETO no supera
        al de menor coste por _CANDIDATE_BETO_MARGIN, el de menor coste."""
        if not ranked:
            return None
        default = self._phonetic.restore_accent(ranked[0][1])
        if self._judge is None:
            return default
        forms = []
        for cost, term in ranked:
            form = self._phonetic.restore_accent(term)
            if cost <= ranked[0][0] + _CANDIDATE_TIE_WINDOW and form not in forms:
                forms.append(form)
        # Si alguno conserva la última letra escrita (quebra -> quiebra, no
        # quebró; entrege -> entregué, no entrega), solo compiten esos: el niño
        # rara vez yerra la desinencia, y BETO cambiaría el tiempo verbal.
        last = core[-1:].lower().translate(_ACCENT_VOWELS)
        same_end = [f for f in forms if f[-1:].translate(_ACCENT_VOWELS) == last]
        if same_end:
            forms = same_end
            if default not in forms:
                default = forms[0]
        forms = forms[:_CANDIDATE_MAX]
        if len(forms) < 2:
            return default
        return self._beto_choice(words, i, pref, suff, core, forms, default)

    def _rescue_rare_word(self, words: list, i: int, pref: str, suff: str, core: str,
                          lower: str, current: str) -> str:
        """Palabra rara de es_50k (quero, com): si hay una rival cercana
        (coste <= _RIVAL_MAX_COST) y _RIVAL_MIN_RATIO veces más frecuente,
        BETO decide entre las dos; si no, se deja `current`."""
        if self._judge is None:
            return current
        freqs = getattr(self._phonetic, "word_freqs", None) or {}
        floor = max(freqs.get(lower, 0), 1) * _RIVAL_MIN_RATIO
        suggestions = self._symspell.lookup(lower, Verbosity.ALL, max_edit_distance=1)
        rivals = [self._phonetic.restore_accent(t)
                  for c, t in rank_candidates(lower, [(s.term, s.count) for s in suggestions])
                  if c <= _RIVAL_MAX_COST and freqs.get(t, 0) >= floor]
        rivals = list(dict.fromkeys(r for r in rivals if r != current.lower()))[:_CANDIDATE_MAX - 1]
        if not rivals:
            return current
        return match_case(core, self._beto_choice(words, i, pref, suff, core,
                                                  [current.lower()] + rivals, current.lower(),
                                                  margin=_RIVAL_BETO_MARGIN))

    def _beto_choice(self, words: list, i: int, pref: str, suff: str, core: str,
                     forms: list, default: str, margin: float = _CANDIDATE_BETO_MARGIN) -> str:
        """La forma de `forms` que BETO prefiere en la ranura i, si supera a
        `default` por `margin`; si no, `default`."""
        tokens = [pref + match_case(core, f) + suff for f in forms]
        try:
            scores = dict(self._judge.score_candidates(words, i, tokens))
        except Exception as e:
            print(f"[WARN] Desempate de candidatos (BETO) falló: {e}")
            return default
        best = max(forms, key=lambda f: scores[tokens[forms.index(f)]])
        if best != default and default in forms:
            gain = scores[tokens[forms.index(best)]] - scores[tokens[forms.index(default)]]
            if gain < margin:
                return default
        return best

    def _is_plausible_reading(self, current: str, alternative: str) -> bool:
        """
        La segunda lectura solo cuenta como ambigüedad si no es mucho más rara
        (_AMBIGUITY_MAX_FREQ_RATIO) que la palabra escrita. Sin datos de
        frecuencia para alguna de las dos, no se descarta.
        """
        freqs    = getattr(self._phonetic, "word_freqs", {}) or {}
        cur_freq = freqs.get(current, 0)
        alt_freq = freqs.get(alternative, 0)
        if not cur_freq or not alt_freq:
            return True
        return cur_freq <= alt_freq * _AMBIGUITY_MAX_FREQ_RATIO

    # ------------------------------------------------------------------
    # Pipeline principal
    # ------------------------------------------------------------------

    def correct(self, text: str, user_vocab: dict | None = None) -> list[str]:
        """Corrige `text` (ver `_correct_once`) en _CORRECTION_PASSES pasadas:
        cada una parte de la recomendada de la anterior y se para si no cambia."""
        result = self._correct_once(text, user_vocab)
        for _ in range(_CORRECTION_PASSES - 1):
            again = self._correct_once(result[0], user_vocab)
            if again[0] == result[0]:
                break
            result = again
        return result

    def _correct_once(self, text: str, user_vocab: dict | None = None) -> list[str]:
        """
        Corrige `text` con el pipeline global. `user_vocab` es un mapa opcional
        {palabra: reemplazo} de sobrescritura léxica; el runtime del servicio
        pasa siempre {} (no hay datos por alumno en la inferencia).

        Devuelve las sugerencias con la recomendada primero: normalmente una;
        hasta tres solo cuando la oración es ambigua (contraste de modo o de
        tilde en una sola palabra, o empate de BETO en un homófono; ver
        infrastructure/nlp/alternatives.py).
        """
        user_vocab = user_vocab or {}
        # Palabras tal como las escribió el niño (modo T5_GUARD=child, ver _refine_with_model).
        self._child_keys = frozenset(_lexical_key(w) for w in text.split())

        # --- CAPA 0: Amalgamas (cambian la tokenización, van antes de todo) ---
        # "aver si vienes" -> "a ver si vienes": SymSpell no puede arreglarlo
        # porque la corrección son dos palabras (elegiría "ver").
        text_expanded = expand_amalgams(text)

        # --- CAPA 0.1: palabras partidas ("en contró" -> "encontró", "a bajo de
        # la cama" -> "abajo"); antes de las capas léxicas, que corregirían cada
        # trozo por separado (contró -> contra). Ver segmentation.py.
        freqs   = getattr(self._phonetic, "word_freqs", None) or {}
        accents = getattr(self._phonetic, "accent_dict", None) or {}
        text_expanded = join_split_words(text_expanded, freqs, accents, judge=self._judge,
                                         phonetic_dict=getattr(self._phonetic, "phonetic_dict", None))

        # --- CAPA 0.2: r/rr entre palabras reales ("el pero ladra" -> "el
        # perro ladra"); antes de BETO, que con "el pero" haría "él pero".
        text_expanded = fix_lexical_confusions(text_expanded)

        # --- CAPAS 1-2: Corrección Ortográfica y Fonética ---
        words = text_expanded.split()
        pre_words    = []
        punctuations = []

        for w in words:
            m = _PUNCT_RE.match(w)
            if m:
                punctuations.append((m.group(1), m.group(3)))
                pre_words.append(m.group(2))
            else:
                punctuations.append(("", ""))
                pre_words.append(w)

        result = []

        for i, core in enumerate(pre_words):
            pref, suff = punctuations[i]
            if not core:
                result.append(pref + suff)
                continue

            lower    = core.lower()
            best     = core
            resolved = False
            keep_plain = False
            opens_sentence = i == 0 or words[i - 1].endswith((".", "!", "?", ":", ";"))
            proper_noun = core[:1].isupper() and not opens_sentence

            # Enclítico escrito con "c": "tragárcelo" -> "tragárselo" (no
            # "Marcelo": delante debe quedar un infinitivo de 4+ letras o un gerundio).
            m = _ENCLITIC_CE_RE.match(lower)
            if m and self._enclitic_host(m.group(1), freqs):
                best     = match_case(core, self._stressed(m.group(1)) + "se" + m.group(2))
                resolved = True

            # Tilde mal puesta que deja la palabra fuera del diccionario
            # ("vínieron"): si sin tildes es una palabra, se corrige como las demás.
            # Una tilde final ("encanté", "comerán") es una forma verbal válida
            # aunque falte en es_50k: se respeta.
            if (not resolved and _has_accent(lower) and not _has_accent(lower[-2:])
                    and not self._symspell.lookup(lower, Verbosity.TOP, max_edit_distance=0)):
                plain = lower.translate(_ACCENT_VOWELS)
                if self._symspell.lookup(plain, Verbosity.TOP, max_edit_distance=0):
                    lower, core = plain, match_case(core, plain)

            # Proteger palabras que ya tienen acento — no tocar
            if not resolved and _has_accent(lower):
                best     = self._phonetic.restore_accent(core)
                resolved = True

            # "¡Ay, me duele!": interjección, no "hay" (MANUAL_CORRECTIONS la cambiaría).
            if not resolved and lower == "ay" and ("¡" in pref or suff.startswith((",", "!"))):
                best     = core
                resolved = True

            # Correcciones manuales de alta prioridad
            if not resolved and lower in MANUAL_CORRECTIONS:
                best     = match_case(core, MANUAL_CORRECTIONS[lower])
                resolved = True

            # Sobrescritura léxica explícita (vacía en el runtime global)
            if not resolved and lower in user_vocab:
                best     = match_case(core, user_vocab[lower])
                resolved = True

            # Nombre propio: mayúscula en mitad de la oración y fuera del
            # diccionario ("Vamos a Cusco"): no se corrige (SymSpell lo
            # convertía en "Casco"). Abriendo oración la mayúscula no dice nada.
            if (not resolved and proper_noun
                    and not self._symspell.lookup(lower, Verbosity.TOP, max_edit_distance=0)):
                best     = core
                resolved = True

            # Palabras muy cortas — no tocar
            if not resolved and len(core) <= 2:
                best     = core
                resolved = True

            # Ya está bien escrita. Si es una palabra rara de es_50k (quero, com)
            # con una rival mucho más frecuente y cercana, decide BETO.
            if not resolved and self._symspell.lookup(lower, Verbosity.TOP, max_edit_distance=0):
                best     = self._phonetic.restore_accent(core)
                resolved = True
                # Tras un determinante va un sustantivo, no un pretérito: "al
                # cerro" no es "al cerró" (la tilde final en -ó/-é la ponía la
                # frecuencia). BETO no sirve aquí: prefiere las formas sin tilde
                # (pájaros, iré, éramos las perdía; medido el 2026-09-30).
                prev_word = words[i - 1].lower().strip(".,;:!?¡¿\"'()") if i > 0 else ""
                if (best.lower() != lower and best.lower().endswith(("ó", "é"))
                        and prev_word in _DETERMINERS):
                    best = core
                    keep_plain = True
                if (not proper_noun and len(lower) <= _RIVAL_MAX_LEN
                        and not is_spanish_word(best.lower(), freqs, accents)):
                    best = self._rescue_rare_word(words, i, pref, suff, core, lower, best)

            # Forma verbal regular fuera de las 50k palabras (nadaremos,
            # dibujaremos): se respeta; SymSpell la destrozaría (daremos).
            if not resolved and (is_regular_verb_form(lower, self._phonetic.word_freqs)
                                 or is_derived_form(lower, freqs, accents)):
                best     = core
                resolved = True

            # Búsqueda fonética
            if not resolved:
                word_sound = to_phonetic(lower)
                if word_sound in self._phonetic.phonetic_dict:
                    phonetic_best = self._phonetic.phonetic_dict[word_sound]
                    max_cost = _INITIAL_MAX_COST if (opens_sentence and core[:1].isupper()) else _PHONETIC_MAX_COST
                    # La "h" es muda: sin la penalización de primera letra (aser -> hacer, avia -> había).
                    h_diff = lower.startswith("h") != phonetic_best.startswith("h")
                    cost = dyslexic_cost(lower.removeprefix("h"), phonetic_best.removeprefix("h")) + (0.3 if h_diff else 0)
                    if phonetic_best != lower and cost <= max_cost:
                        best     = self._phonetic.restore_accent(match_case(core, phonetic_best))
                        resolved = True

            # SymSpell: todos los candidatos a distancia <= 2, elegidos por coste
            # disléxico (candidates.pick_candidate) y no por distancia entera +
            # frecuencia: jugan -> juegan (no jugar), aruz -> arroz (no cruz).
            # Palabras pegadas ("queno" -> "que no", "estabapreparando"): si la
            # palabra se parte en palabras españolas, BETO decide entre la
            # partición y la corrección de SymSpell ("quemo"). Ver segmentation.py.
            if not resolved:
                suggestions = self._symspell.lookup(lower, Verbosity.ALL, max_edit_distance=2)
                ranked = rank_candidates(lower, [(s.term, s.count) for s in suggestions]) if suggestions else []
                splits = split_candidates(lower, freqs, accents)
                if opens_sentence and core[:1].isupper() and not core.isupper():
                    if ranked and ranked[0][0] > _INITIAL_MAX_COST:
                        ranked = []
                    splits = [p for p in splits if p[0] in FUNCTION_WORDS]
                pick = self._contextual_pick(words, i, pref, suff, core, ranked)
                split = choose_split(
                    words, i, splits, pick, judge=self._judge,
                    pick_cost=dyslexic_cost(lower, pick) if pick else None,
                )
                if split:
                    result.append(pref + match_case(core, split) + suff)
                    continue
                if pick:
                    best = match_case(core, self._phonetic.restore_accent(pick))

            # Restauración de ñ (nino->niño, manana->mañana) y luego del acento
            # (compañia->compañía): ambas son seguras (solo actúan sobre entradas
            # de enye_dict/accent_dict, con guarda 5x). La ñ no en un nombre
            # propio ("soy Nino" no es "soy Niño").
            if not proper_noun:
                best = self._phonetic.restore_enye(best)
            if not keep_plain:                           # la tilde descartada no vuelve
                best = self._phonetic.restore_accent(best)
            result.append(pref + best + suff)

        base_corrected = " ".join(result)

        # --- CAPA 0.5: Desambiguación contextual con BETO (homófonos y tildes) ---
        # Las reglas anteriores no distinguen palabras válidas que solo el contexto
        # separa (esta/está, boy/voy, tubo/tuvo). BETO puntúa cada alternativa
        # según el resto de la frase y elige la más coherente.
        # Cuando el contexto no decide entre dos lecturas (esta/está, tubo/tuvo)
        # la segunda lectura se conserva como candidato para ofrecerla al alumno.
        # Las tildes que el alumno escribió (capa 1: "se respetan") no se
        # deshacen: BETO no puede quitar la tilde de "quizás" aunque prefiera
        # "quizas" (las que puso restore_accent sí siguen a su criterio).
        ambiguous_variants: list[tuple[str, float]] = []
        written_accents = {core.lower() for core in pre_words if core and _has_accent(core)}
        if self._judge is not None:
            try:
                base_corrected, ambiguous_variants = self._disambiguate_context(base_corrected, written_accents)
            except Exception as e:
                print(f"[WARN] Desambiguación contextual (BETO) falló: {e}")

        # --- CAPA 3: Corrección gramatical por reglas (haber impersonal, gustar, número) ---
        base_corrected = correct_grammar(base_corrected)
        ambiguous_variants = [(correct_grammar(v), s) for v, s in ambiguous_variants]

        # --- CAPA 4: Refinamiento gramatical con T5 (concordancia sujeto-verbo) ---
        # T5 (adaptador LoRA de concordancia) actúa sobre el texto ya corregido por
        # reglas+BETO y devuelve varios beams con su score. Si el beam 1 no es
        # seguro y léxicamente plausible, T5 no cuenta para esta frase
        # (`refined == []`); si lo es, `refined[0]` es el beam 1 y el resto
        # son los demás beams seguros (ver _refine_with_model).
        refined: list[tuple[str, float]] = []
        if self._seq2seq and self._tokenizer:
            try:
                refined = self._refine_with_model(base_corrected)
            except Exception as e:
                print(f"[WARN] Refinamiento T5 falló: {e}")

        # --- CAPA 5: Selección de alternativas (hasta 3, solo si hay ambigüedad) ---
        # La recomendada es el beam 1 de T5 si es seguro (como antes de la
        # Task 5) y en cualquier otro caso la base. Las segundas lecturas de
        # BETO (score = −|diferencia|) se anclan al mejor beam solo para
        # ordenarlas junto a los demás beams: van como `variants` (nunca son la
        # recomendada) y su puerta real es _BETO_TIE_MARGIN (≤ _SCORE_MARGIN,
        # así que (d) no las filtra). Un beam (o la base) solo se ofrece con
        # señal positiva de ambigüedad (contraste de modo o de tilde sobre una
        # palabra corregida); el léxico de frecuencias decide la conjugación
        # en el filtro de modo tras "es posible que", "ojalá", etc.
        # Ortografía vigente (RAE 2010): "sólo" -> "solo", salvo que el alumno
        # lo haya escrito con tilde. Va después de T5 porque el corpus de
        # entrenamiento es anterior a la reforma y la repone.
        base_corrected = normalize_modern_spelling(base_corrected, written_accents)
        # Los auxiliares de la capa 3 se reaplican a los beams: T5 no puede
        # deshacer una corrección determinista ("ha ido" -> "a ido"; auditoría B).
        refined = [(normalize_modern_spelling(correct_auxiliaries(t), written_accents), s) for t, s in refined]
        ambiguous_variants = [(normalize_modern_spelling(v, written_accents), s) for v, s in ambiguous_variants]

        # --- CAPA 6: Conjuntos de confusión que T5 deshace (confusions.py) ---
        # caer/callar ("se callo en el río" -> "se cayó", no el "se calló"
        # fijo de T5) y "yo" + pretérito de 3.ª persona ("yo comió" -> "yo
        # comí"). Sobre la base, los beams y las segundas lecturas.
        def _final(t: str) -> str:
            try:
                return resolve_final_confusions(t, written_accents, judge=self._judge)
            except Exception as e:
                print(f"[WARN] Conjuntos de confusión fallaron: {e}")
                return t
        base_corrected = _final(base_corrected)
        refined = [(_final(t), s) for t, s in refined]
        ambiguous_variants = [(_final(v), s) for v, s in ambiguous_variants]

        recommended = refined[0][0] if refined else base_corrected
        best_score  = max((s for _, s in refined), default=0.0)
        return select_alternatives(
            text, base_corrected, refined,
            score_margin=_SCORE_MARGIN, recommended=recommended,
            variants=[(v, best_score + s) for v, s in ambiguous_variants],
            lexicon=getattr(self._phonetic, "word_freqs", None) or None,
        )

    def _refine_with_model(self, text: str) -> list[tuple[str, float]]:
        """
        Pasa `text` (ya corregido por reglas+BETO) por T5 y devuelve los beams
        `[(texto, score)]` utilizables, para capturar la concordancia
        sujeto-verbo sin alucinar. Regla de la recomendada (la de antes de la
        Task 5): solo cuenta el beam 1. Si el beam 1 no pasa las guardas, el
        resultado es `[]` (la recomendada será la base) y no se salta al beam
        2: si el mejor beam alucina, T5 no es fiable para esta frase. Si pasa,
        `[0]` es el beam 1 y detrás van los beams 2/3 que sean seguros (solo
        candidatos a alternativa; la regla (e) de la capa 5 los filtra).
        Guardas del beam 1:
          - `is_safe_refinement`: longitud 0.6–1.5 del original y límite de
            palabras cambiadas (cantidad; anti-alucinación),
          - `is_lexically_plausible_refinement`: cada reemplazo 1:1 de
            palabra es una variante cercana (similitud sin tildes ≥
            _RECOMMENDED_MIN_SIMILARITY): bloquea sustituciones léxicas
            como pasto→maíz, conserva flexiones como es→son.
        Los beams 2/3 solo pasan por `is_safe_refinement`. Se preserva la
        capitalización inicial del texto base.
        Antes de las guardas, en cada beam se revierten las palabras que T5
        cambió por OTRA palabra (alcancía→caja, aula→clase, tele→televisión):
        solo se conservan flexiones, homófonos e irregulares de la misma
        palabra (guards.revert_lexical_substitutions).
        """
        generated = self._seq2seq.generate_corrections(
            text, self._tokenizer, num_returns=_T5_NUM_RETURNS
        )
        if not generated:
            return []
        freqs   = getattr(self._phonetic, "word_freqs", None) or {}
        accents = getattr(self._phonetic, "accent_dict", None) or {}
        known   = lambda key: key in freqs or key in accents
        free = frozenset()
        if _T5_GUARD == "child":
            free = frozenset(_lexical_key(w) for w in text.split()) - getattr(self, "_child_keys", frozenset())
        generated = [(revert_lexical_substitutions(text, str(g).strip(), known, to_phonetic, free), score)
                     for g, score in generated]
        if _T5_MODE == "verified" and self._judge is not None:
            generated = [(self._beto_verify(text, g), score) for g, score in generated]
        first = str(generated[0][0]).strip()
        if not is_safe_refinement(text, first) or not is_lexically_plausible_refinement(
            text, first, min_similarity=_RECOMMENDED_MIN_SIMILARITY, free_words=free
        ):
            return []

        beams = []
        for g, score in generated:
            g = str(g).strip()
            if not is_safe_refinement(text, g):
                continue
            if text[:1].islower() and g[:1].isupper():
                g = g[:1].lower() + g[1:]
            beams.append((g, float(score)))
        return beams