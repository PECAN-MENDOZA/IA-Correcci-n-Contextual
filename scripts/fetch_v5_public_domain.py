"""
scripts/fetch_v5_public_domain.py
Literatura infantil y popular en español de DOMINIO PÚBLICO (Project Gutenberg) como fuente
de frases limpias para LoRA v5: descarga, quita cabecera/licencia de Gutenberg, parte en
oraciones, moderniza la ortografía anterior a 1952/2010 (á, é, ó, ú sueltas; fué, dió, vió;
sólo, éste) y descarta oraciones con palabras raras o dialectales (mismo filtro de léxico
que los subtítulos en build_v5_clean.py). Salida: data/v5/dominio_publico/<id>.txt

Libros (dominio público en EE. UU., Project Gutenberg):
  36558  Luis Coloma, "Ratón Pérez: cuento infantil" (1902)
  63424  Ramón A. Laval, "Cuentos populares en Chile" (1923)

Uso:  python scripts/fetch_v5_public_domain.py
"""
import re
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
OUT = REPO / "data" / "v5" / "dominio_publico"
BOOKS = {36558: "coloma_raton_perez", 63424: "laval_cuentos_populares_chile"}
URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"

_OLD_SPELLING = {
    "á": "a", "é": "e", "ó": "o", "ú": "u", "fué": "fue", "fuí": "fui", "dió": "dio", "vió": "vio",
    "dí": "di", "ví": "vi", "tí": "ti", "sólo": "solo", "éste": "este", "ésta": "esta", "éstos": "estos",
    "éstas": "estas", "ése": "ese", "ésa": "esa", "ésos": "esos", "ésas": "esas", "aquél": "aquel",
    "aquélla": "aquella", "guión": "guion",
}
_WORD = re.compile(r"\w+", re.UNICODE)


def body(text: str) -> str:
    start = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", text)
    end = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG", text)
    return text[start.end() if start else 0: end.start() if end else len(text)]


def modernize(sentence: str) -> str:
    def fix(m):
        w = m.group(0)
        low = w.lower()
        if low in _OLD_SPELLING:
            new = _OLD_SPELLING[low]
            return new.capitalize() if w[:1].isupper() else new
        return w
    return _WORD.sub(fix, sentence)


def sentences(text: str) -> list:
    text = re.sub(r"_([^_]+)_", r"\1", text)                     # cursivas de Gutenberg
    text = re.sub(r"\[.*?\]", " ", text, flags=re.S)             # notas e ilustraciones
    text = re.sub(r"\s+", " ", text).replace("--", "—")
    parts = re.split(r"(?<=[.!?»])\s+(?=[—¿¡«A-ZÁÉÍÓÚÑ])", text)
    return [p.strip(" -") for p in parts]


def valid(sentence: str, freqs: dict, accents: dict) -> bool:
    words = _WORD.findall(sentence)
    if not 4 <= len(words) <= 30 or any(ch.isdigit() for ch in sentence):
        return False
    if re.search(r"[A-Z]{3,}|http|www|Gutenberg|\*", sentence) or re.search(r"[A-Z]\.$", sentence):
        return False                                               # ... "el bueno de D." (cortada)
    for i, w in enumerate(words):
        low = w.lower()
        if w[:1].isupper() and i > 0:
            continue                                               # nombre propio
        if freqs.get(low, 0) < 300 or (low in accents and accents[low] != low):
            return False
    return True


def main() -> None:
    from infrastructure.nlp.phonetic_engine import PhoneticEngine
    pe = PhoneticEngine()
    OUT.mkdir(parents=True, exist_ok=True)
    for book_id, name in BOOKS.items():
        raw = urllib.request.urlopen(URL.format(id=book_id), timeout=60).read().decode("utf-8", "replace")
        kept = [s for s in (modernize(s) for s in sentences(body(raw))) if valid(s, pe.word_freqs, pe.accent_dict)]
        header = (f"# Project Gutenberg #{book_id} ({name}), dominio público; ortografía modernizada "
                  f"por scripts/fetch_v5_public_domain.py\n")
        (OUT / f"{name}.txt").write_text(header + "\n".join(kept) + "\n", encoding="utf-8")
        print(f"[{book_id}] {name}: {len(kept)} oraciones")


if __name__ == "__main__":
    main()
