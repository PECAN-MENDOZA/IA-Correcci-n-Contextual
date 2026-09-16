"""
infrastructure/nlp/correction_pipeline.py
Pipeline neuro-simbólico-fonético de corrección.
Orquesta las capas de ortografía y finaliza con Seq2Seq para gramática de forma limpia.
"""
import re
import unicodedata
from symspellpy import SymSpell, Verbosity

from infrastructure.nlp.phonetic_engine import PhoneticEngine, match_case, to_phonetic, DICT_PATH
from infrastructure.nlp.context_judge import ContextJudge
from infrastructure.nlp.grammar_rules import correct_grammar
from infrastructure.nlp.alternatives import THRESHOLDS, select_alternatives
from infrastructure.ml.t5_model import T5CorrectionModel, T5SpanishTokenizer
from infrastructure.ml.guards import is_lexically_plausible_refinement, is_safe_refinement

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
        self._symspell.load_dictionary(DICT_PATH, term_index=0, count_index=1)
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
        """
        Corrige `text` con el pipeline global. `user_vocab` es un mapa opcional
        {palabra: reemplazo} de sobrescritura léxica; el runtime del servicio
        pasa siempre {} (no hay datos por alumno en la inferencia).

        Devuelve las sugerencias con la recomendada primero: normalmente una
        (o refinado + base); hasta tres solo cuando la oración es ambigua
        (ver infrastructure/nlp/alternatives.py).
        """
        user_vocab = user_vocab or {}

        # --- CAPAS 1-2: Corrección Ortográfica y Fonética ---
        words = text.split()
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

            # Proteger palabras que ya tienen acento — no tocar
            if _has_accent(lower):
                best     = self._phonetic.restore_accent(core)
                resolved = True

            # Correcciones manuales de alta prioridad
            if not resolved and lower in MANUAL_CORRECTIONS:
                best     = match_case(core, MANUAL_CORRECTIONS[lower])
                resolved = True

            # Sobrescritura léxica explícita (vacía en el runtime global)
            if not resolved and lower in user_vocab:
                best     = match_case(core, user_vocab[lower])
                resolved = True

            # Palabras muy cortas — no tocar
            if not resolved and len(core) <= 2:
                best     = core
                resolved = True

            # Ya está bien escrita
            if not resolved and self._symspell.lookup(lower, Verbosity.TOP, max_edit_distance=0):
                best     = self._phonetic.restore_accent(core)
                resolved = True

            # Búsqueda fonética
            if not resolved:
                word_sound = to_phonetic(lower)
                if word_sound in self._phonetic.phonetic_dict:
                    phonetic_best = self._phonetic.phonetic_dict[word_sound]
                    if phonetic_best != lower:
                        best     = self._phonetic.restore_accent(match_case(core, phonetic_best))
                        resolved = True

            # SymSpell sin destruir homófonos
            if not resolved:
                suggestions = self._symspell.lookup(lower, Verbosity.CLOSEST, max_edit_distance=2)
                if suggestions and suggestions[0].term != lower:
                    best = match_case(core, self._phonetic.restore_accent(suggestions[0].term))

            # Restauración de ñ (nino->niño, manana->mañana) y luego del acento
            # (compañia->compañía): ambas son seguras (solo actúan sobre entradas
            # de enye_dict/accent_dict, con guarda 5x).
            best = self._phonetic.restore_accent(self._phonetic.restore_enye(best))
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
        """
        generated = self._seq2seq.generate_corrections(
            text, self._tokenizer, num_returns=_T5_NUM_RETURNS
        )
        if not generated:
            return []
        first = str(generated[0][0]).strip()
        if not is_safe_refinement(text, first) or not is_lexically_plausible_refinement(
            text, first, min_similarity=_RECOMMENDED_MIN_SIMILARITY
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