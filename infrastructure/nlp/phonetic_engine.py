"""
infrastructure/nlp/phonetic_engine.py
Motor fonético: normalización, conversión fonética y restauración de tildes.
Extraído de inference_api.py — responsabilidad única de transformación fonética.
"""
import os
import urllib.request


DICT_PATH = "./es_50k.txt"
DICT_URL   = (
    "https://raw.githubusercontent.com/hermitdave/FrequencyWords/"
    "master/content/2016/es/es_50k.txt"
)


def _ensure_dict() -> None:
    if not os.path.exists(DICT_PATH):
        print("[INFO] Descargando diccionario de frecuencias...")
        urllib.request.urlretrieve(DICT_URL, DICT_PATH)


def remove_accents(word: str) -> str:
    """Elimina tildes para comparación fonética."""
    for src, dst in {"á":"a","é":"e","í":"i","ó":"o","ú":"u","ü":"u"}.items():
        word = word.lower().replace(src, dst)
    return word


def to_phonetic(word: str) -> str:
    """Convierte una palabra española a su representación fonética canónica."""
    w = remove_accents(word)
    w = w.replace("q","p").replace("w","m").replace("h","").replace("v","b")
    w = w.replace("ll","y").replace("qu","k").replace("z","s")
    w = w.replace("ce","se").replace("ci","si")
    w = w.replace("ca","ka").replace("co","ko").replace("cu","ku")
    w = w.replace("ge","je").replace("gi","ji")

    # Eliminar consonantes dobles excepto 'rr'
    result = ""
    for i, ch in enumerate(w):
        if i > 0 and ch == w[i - 1] and ch != "r":
            continue
        result += ch
    return result


def match_case(original: str, corrected: str) -> str:
    """Preserva el estilo de capitalización del original en la corrección."""
    if not original:
        return corrected
    if original.isupper() and len(original) > 1:
        return corrected.upper()
    if original.istitle():
        return corrected.capitalize()
    return corrected.lower()


class PhoneticEngine:
    """
    Construye los índices fonético y de tildes a partir del diccionario de frecuencias.
    Provee métodos de lookup para el pipeline de corrección.
    """

    def __init__(self):
        _ensure_dict()
        self.word_freqs:         dict = {}
        self.phonetic_dict:      dict = {}
        self.accent_dict:        dict = {}
        self._build_indexes()

    def _build_indexes(self) -> None:
        with open(DICT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if parts:
                    self.word_freqs[parts[0].lower()] = int(parts[1])

        for word, freq in self.word_freqs.items():
            if len(word) <= 2:
                continue

            sound = to_phonetic(word)
            if sound not in self.phonetic_dict:
                self.phonetic_dict[sound] = word

            unaccented = remove_accents(word)
            if unaccented != word:
                freq_unaccented = self.word_freqs.get(unaccented, 0)
                if freq > freq_unaccented * 5:
                    existing = self.accent_dict.get(unaccented)
                    if not existing or freq > self.word_freqs.get(existing, 0):
                        self.accent_dict[unaccented] = word

    def restore_accent(self, word: str) -> str:
        lower    = word.lower()
        restored = self.accent_dict.get(lower, lower)
        return match_case(word, restored)

    def phonetic_lookup(self, word: str) -> str:
        """Devuelve la palabra de mayor frecuencia para el sonido dado (o la misma si no hay)."""
        return self.phonetic_dict.get(to_phonetic(word.lower()), word)
