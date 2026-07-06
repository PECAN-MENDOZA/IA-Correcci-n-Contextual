"""
infrastructure/nlp/correction_pipeline.py
Pipeline neuro-simbólico-fonético de corrección.
Orquesta las capas de ortografía y finaliza con Seq2Seq para gramática de forma limpia.
"""
import re
import unicodedata
from difflib import SequenceMatcher
from symspellpy import SymSpell, Verbosity

from infrastructure.nlp.phonetic_engine import PhoneticEngine, match_case, to_phonetic, DICT_PATH
from infrastructure.nlp.context_judge import ContextJudge
from infrastructure.nlp.grammar_rules import correct_grammar
from infrastructure.ml.t5_model import T5CorrectionModel, T5SpanishTokenizer

MANUAL_CORRECTIONS = {
    "uillos": "niños",  "ciubab": "ciudad",  "caíbas": "caídas",
    "bamo":   "vamos",  "bamos":  "vamos",   "oy":     "hoy",
    "ise":    "hice",   "iso":    "hizo",     "iva":    "iba",
    "aiga":   "haya",   "haiga":  "haya",     "ay":     "hay",
    "dondis": "donde",  "dodes":  "donde",    "dondes": "donde",
    "be":     "de",     "ala":    "a la",     "delo":   "de lo",
    "delos":  "de los", "dela":   "de la",    "delas":  "de las",
}

# Palabras con acento que SymSpell no debe tocar
_ACCENTED_RE = re.compile(r'[áéíóúüñÁÉÍÓÚÜÑ]')
_PUNCT_RE    = re.compile(r'^([^\wáéíóúüñÁÉÍÓÚÜÑ]*)(.*?)([^\wáéíóúüñÁÉÍÓÚÜÑ]*)$')

# Umbral mínimo de similitud para considerar un hit de memoria como válido.
# 0.82 tolera errores tipográficos menores y variaciones de puntuación
# sin confundir frases distintas.
_MEMORY_SIMILARITY_THRESHOLD = 0.82

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


def _has_accent(word: str) -> bool:
    return bool(_ACCENTED_RE.search(word))


def _strip_accents(s: str) -> str:
    """Quita tildes para comparación sin destruir eñes."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def _normalize_for_lookup(text: str) -> str:
    """
    Normalización ligera para comparar frases en memoria:
    minúsculas, sin tildes, sin puntuación extra, espacios colapsados.
    No destruye la estructura de la frase.
    """
    text = text.lower().strip()
    text = _strip_accents(text)
    text = re.sub(r"[^\w\s]", "", text)   # quita puntuación
    text = re.sub(r"\s+", " ", text)      # colapsa espacios
    return text


def _similarity(a: str, b: str) -> float:
    """Ratio de similitud entre dos strings (0.0 – 1.0)."""
    return SequenceMatcher(None, a, b).ratio()


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

        # Memoria de correcciones validadas por el usuario.
        # Estructura: { normalized_original: {"correction": str, "raw_original": str} }
        self.user_memory: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # API pública de memoria
    # ------------------------------------------------------------------

    def remember(self, original: str, correction: str) -> None:
        """
        Registra que el usuario prefiere `correction` cuando el input
        es `original`. Se llama desde el endpoint de feedback.

        Ejemplo:
            pipeline.remember("fuy al mercado", "fui al mercado")
        """
        key = _normalize_for_lookup(original)
        self.user_memory[key] = {
            "correction":    correction.strip(),
            "raw_original":  original.strip(),
        }

    def forget(self, original: str) -> bool:
        """
        Elimina una entrada de memoria. Devuelve True si existía.
        Útil para que el usuario pueda revertir un feedback incorrecto.
        """
        key = _normalize_for_lookup(original)
        if key in self.user_memory:
            del self.user_memory[key]
            return True
        return False

    def _lookup_memory(self, text: str) -> str | None:
        """
        Busca en user_memory si ya existe una corrección validada para
        `text` o una frase suficientemente similar.

        Primero intenta exact-match normalizado (O(1)).
        Si falla, hace fuzzy scan sobre toda la memoria (O(n), pero n
        es pequeño en uso real — correcciones de un solo usuario).

        Devuelve la corrección guardada o None si no hay hit.
        """
        normalized = _normalize_for_lookup(text)

        # 1. Exact match normalizado — el caso más común y más rápido
        if normalized in self.user_memory:
            return self.user_memory[normalized]["correction"]

        # 2. Fuzzy match — cubre variaciones tipográficas menores
        best_score  = 0.0
        best_result = None

        for key, entry in self.user_memory.items():
            score = _similarity(normalized, key)
            if score > best_score:
                best_score  = score
                best_result = entry["correction"]

        if best_score >= _MEMORY_SIMILARITY_THRESHOLD:
            return best_result

        return None

    # ------------------------------------------------------------------
    # Desambiguación contextual (BETO)
    # ------------------------------------------------------------------

    def _disambiguate_context(self, sentence: str) -> str:
        """
        Pasada palabra-a-palabra: para cada término con homófonos/variantes de
        acento reales, deja que BETO elija el más coherente con la frase.
        Solo sobrescribe si la mejora en log-prob supera el margen del tipo de
        cambio (acento / homófono / monosílabo). Es idempotente y greedy:
        cada decisión usa como contexto las decisiones ya tomadas a su izquierda.
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
            if best == current:
                continue

            best_core = _PUNCT_RE.match(best).group(2).lower()
            if len(lower) <= 2:
                margin = _MARGIN_MONO
            elif _strip_accents(best_core) == _strip_accents(lower):
                margin = _MARGIN_ACCENT
            else:
                margin = _MARGIN_HOMOPHONE

            if (best_score - current_score) >= margin:
                tokens[i] = best

        return " ".join(tokens)

    # ------------------------------------------------------------------
    # Pipeline principal
    # ------------------------------------------------------------------

    def correct(self, text: str, user_vocab: dict) -> list[str]:
        # --- CAPA 0: Memoria de correcciones del usuario ---
        # Si el usuario ya corrigió esta frase (o una muy similar) antes,
        # devolvemos esa corrección directamente sin pasar por T5 ni SymSpell.
        memory_hit = self._lookup_memory(text)
        if memory_hit is not None:
            return [memory_hit]

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

            # Vocabulario personalizado del usuario
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

            # Restauración de ñ (nino->niño, manana->mañana). Seguro para todas las
            # palabras: solo actúa sobre las que están en enye_dict (guarda 5x).
            best = self._phonetic.restore_enye(best)
            result.append(pref + best + suff)

        base_corrected = " ".join(result)

        # --- CAPA 0.5: Desambiguación contextual con BETO (homófonos y tildes) ---
        # Las reglas anteriores no distinguen palabras válidas que solo el contexto
        # separa (esta/está, boy/voy, tubo/tuvo). BETO puntúa cada alternativa
        # según el resto de la frase y elige la más coherente.
        if self._judge is not None:
            try:
                base_corrected = self._disambiguate_context(base_corrected)
            except Exception as e:
                print(f"[WARN] Desambiguación contextual (BETO) falló: {e}")

        # --- CAPA 3: Corrección gramatical por reglas (haber impersonal, gustar, número) ---
        base_corrected = correct_grammar(base_corrected)

        # --- CAPA 4: Corrección Gramatical con T5 (End-to-End) ---
        if self._seq2seq and self._tokenizer:
            try:
                generated = self._seq2seq.generate_corrections(
                    base_corrected, self._tokenizer, num_returns=3
                )
                final = []

                for g in generated:
                    if not g:
                        continue

                    # Filtro de longitud razonable
                    base_len = len(base_corrected.split())
                    cand_len = len(g.split())
                    if base_len == 0 or not (0.6 <= cand_len / base_len <= 1.5):
                        continue

                    if not any(g.lower().strip() == f.lower().strip() for f in final):
                        final.append(g)

                # Incluir la corrección simbólica base por si T5 falló por completo
                if not any(base_corrected.lower().strip() == f.lower().strip() for f in final):
                    final.append(base_corrected)

                return final[:2]

            except Exception as e:
                print(f"[WARN] Error en T5 Seq2Seq: {e}")

        # Fallback: solo corrección simbólica
        return [base_corrected]