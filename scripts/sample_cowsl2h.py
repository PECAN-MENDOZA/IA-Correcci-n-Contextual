"""Muestras fijas de COWS-L2H (dev) como set de regresión EXTERNO.

Toma `data/external/cowsl2h/pairs-dev.csv` (generado por fetch_external_data.py;
nunca el holdout) y escribe dos muestras de 200 oraciones reales con la
corrección del profesor como referencia:

  * data/test_cowsl2h_spelling200.csv — oraciones cuya corrección es solo
    ortográfica/morfológica: mismo número de tokens, 1-3 sustituciones 1:1 y
    cada una es un cambio de tilde/mayúscula o de forma cercana (ratio de
    SequenceMatcher >= 0,6). Es el terreno del corrector.
  * data/test_cowsl2h_random200.csv — oraciones al azar de 4-22 tokens (la
    mayoría son gramática/léxico de L2, fuera de alcance: mide el daño).

Semilla fija (2026): las muestras son reproducibles y quedan versionadas en
data/ (Apache-2.0, ver data/external/README.md). Uso:
    python scripts/sample_cowsl2h.py
"""
import difflib
import random
import re
import unicodedata

SRC = "data/external/cowsl2h/pairs-dev.csv"
SEED = 2026
N = 200
HEADER = (
    "# Oraciones REALES de COWS-L2H (github.com/ucdaviscl/cowsl2h, Apache-2.0), partición de\n"
    "# desarrollo (nunca el holdout), muestreadas por scripts/sample_cowsl2h.py con semilla 2026.\n"
    "# Referencia: corrección holística de un profesor (esperado_1; esperado_2 cuando hay dos).\n"
    "# Población: universitarios de español L2 — sirve para medir daño y comparar versiones,\n"
    "# no sustituye a la población objetivo. Los placeholders *FIRST_NAME* son del corpus.\n"
    "categoria|entrada|esperado_1|esperado_2\n"
)


def _tok(s: str) -> list:
    return re.findall(r"\w+|[^\w\s]", s)


def _strip(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def spelling_like(entrada: str, esperado: str) -> bool:
    te, tx = _tok(entrada), _tok(esperado)
    if len(te) != len(tx) or not (4 <= len(te) <= 22):
        return False
    diffs = [(a, b) for a, b in zip(te, tx) if a != b]
    if not diffs or len(diffs) > 3:
        return False
    for a, b in diffs:
        if _strip(a).lower() == _strip(b).lower():
            continue
        if difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() < 0.6:
            return False
    return True


def main() -> None:
    rows = []
    for line in open(SRC, encoding="utf-8"):
        if line.startswith("#") or line.startswith("categoria|") or not line.strip():
            continue
        rows.append(line.rstrip("\n").split("|"))
    short = [r for r in rows if 4 <= len(_tok(r[1])) <= 22]
    spell = [r for r in rows if spelling_like(r[1], r[2])]
    random.seed(SEED)
    samples = {
        "data/test_cowsl2h_random200.csv": random.sample(short, N),
        "data/test_cowsl2h_spelling200.csv": random.sample(spell, N),
    }
    for path, sample in samples.items():
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(HEADER)
            for r in sample:
                fh.write("|".join(["cowsl2h"] + r[1:]) + "\n")
        print(f"{path}: {len(sample)} oraciones")
    print(f"pares dev={len(rows)} cortas={len(short)} ortograficas={len(spell)}")


if __name__ == "__main__":
    main()
