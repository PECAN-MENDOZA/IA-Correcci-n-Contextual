"""
infrastructure/nlp/correction_pipeline.py
Pipeline neuro-simbólico-fonético de corrección.
Orquesta las 7 capas de corrección usando PhoneticEngine, ContextJudge y SymSpell.
"""
import re
from symspellpy import SymSpell, Verbosity

from infrastructure.nlp.phonetic_engine import PhoneticEngine, match_case, to_phonetic, DICT_PATH
from infrastructure.nlp.context_judge import ContextJudge

# ── Heurísticas manuales para errores severos de dislexia ─────────────────────
MANUAL_CORRECTIONS = {
    "uillos": "niños",  "ciubab": "ciudad",  "caíbas": "caídas",
    "bamo":   "vamos",  "bamos":  "vamos",   "oy":     "hoy",
    "ise":    "hice",   "iso":    "hizo",     "iva":    "iba",
    "aiga":   "haya",   "haiga":  "haya",     "ay":     "hay",
    "dondis": "donde",  "dodes":  "donde",    "dondes": "donde",
    "be":     "de",
}

# ── Homófonos que requieren decisión contextual ────────────────────────────────
HOMOPHONES_WATCHLIST = {
    "beses": ["beses", "veces"],
    "tubo":  ["tubo", "tuvo"],
    "asia":  ["asia", "hacia"],
    "baya":  ["baya", "vaya", "valla"],
    "valla": ["valla", "vaya", "baya"],
    "bello": ["bello", "vello"],
    "vello": ["vello", "bello"],
    "asta":  ["asta", "hasta"],
    "echo":  ["echo", "hecho"],
    "ola":   ["ola", "hola"],
    "olla":  ["olla", "hola"],
    "a":     ["a", "ha"],
    "e":     ["e", "he"],
}

_PUNCT_RE = re.compile(r'^([^\wáéíóúüñÁÉÍÓÚÜÑ]*)(.*?)([^\wáéíóúüñÁÉÍÓÚÜÑ]*)$')


class CorrectionPipeline:
    """
    Implementa el pipeline de 7 capas con control de flujo escalonado.
    """

    def __init__(self, phonetic: PhoneticEngine, judge: ContextJudge):
        self._phonetic = phonetic
        self._judge    = judge
        self._symspell = SymSpell(max_dictionary_edit_distance=3, prefix_length=7)
        self._symspell.load_dictionary(DICT_PATH, term_index=0, count_index=1)

    def correct(self, text: str, user_vocab: dict) -> str:
        words         = text.split()
        pre_words     = []
        punctuations  = []

        # ── Cirugía de puntuación ──────────────────────────────────────────────
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

            lower = core.lower()
            best  = core
            resolved = False

            # Capa -1: heurísticas manuales
            if not resolved and lower in MANUAL_CORRECTIONS:
                best = match_case(core, MANUAL_CORRECTIONS[lower])
                resolved = True

            # Capa -0.5: dislexia visual
            if not resolved and not self._symspell.lookup(lower, Verbosity.TOP, max_edit_distance=0):
                visual = lower.replace("q", "p").replace("w", "m").replace("b", "d")
                if self._symspell.lookup(visual, Verbosity.TOP, max_edit_distance=0):
                    best = match_case(core, visual)
                    resolved = True

            # Capa 0: vocabulario del usuario
            if not resolved and lower in user_vocab:
                best = match_case(core, user_vocab[lower])
                resolved = True

            # Capa 0.5: homófonos (contexto IA) - ANTES DE LAS PALABRAS CORTAS
            if not resolved and lower in HOMOPHONES_WATCHLIST:
                candidates = HOMOPHONES_WATCHLIST[lower]
                chosen     = self._judge.best_candidate(pre_words, i, candidates)
                best       = self._phonetic.restore_accent(match_case(core, chosen))
                resolved = True

            # Capa 0.1: escudo de palabras cortas
            if not resolved and len(core) <= 2:
                best = core
                resolved = True

            # Protección absoluta: palabra válida → solo restaurar tilde
            if not resolved and self._symspell.lookup(lower, Verbosity.TOP, max_edit_distance=0):
                best = self._phonetic.restore_accent(core)
                resolved = True

            # Capa 1: fonética dirigida a mayor frecuencia
            if not resolved:
                word_sound = to_phonetic(lower)
                if word_sound in self._phonetic.phonetic_dict:
                    phonetic_best = self._phonetic.phonetic_dict[word_sound]
                    if phonetic_best != lower:
                        best = self._phonetic.restore_accent(match_case(core, phonetic_best))
                        resolved = True

            # Capa 2: SymSpell + BETO
            if not resolved:
                suggestions = self._symspell.lookup(lower, Verbosity.CLOSEST, max_edit_distance=3)
                if not suggestions or suggestions[0].term == lower:
                    best = self._phonetic.restore_accent(core)
                else:
                    candidates = [s.term for s in suggestions[:3]]
                    chosen     = self._judge.best_candidate(pre_words, i, candidates)
                    chosen     = self._phonetic.restore_accent(chosen)
                    best       = match_case(core, chosen)
                resolved = True

            result.append(pref + best + suff)

        return " ".join(result)
