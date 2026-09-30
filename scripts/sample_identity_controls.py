"""Muestra fija de oraciones CORRECTAS reales de COWS-L2H (partición de desarrollo).

El profesor dejó 13 772 oraciones sin cambios (pares de identidad; fetch_external_data.py
las cuenta pero no las escribe). Son el mejor texto correcto real disponible para medir
cuánto estropea el corrector lo que ya estaba bien (métrica de daño de evaluate.py,
auditoría A del 2026-09-30). Se reconstruyen desde `data/external/cowsl2h/raw/` con las
mismas funciones y la misma partición por AUTOR (semilla 42) que fetch_external_data.py,
y se toman **solo de autores de desarrollo**: el holdout bloqueado no se toca.

Salida: data/test_correcto_cowsl2h200.csv (entrada == esperado), semilla 2026.
Uso:  python scripts/sample_identity_controls.py
"""
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_external_data import (  # noqa: E402
    COWSL2H_FILES, HOLDOUT_FRACTION, SPLIT_SEED, align_essay, build_pairs_by_essay,
    parse_cowsl2h_csv, split_by_author, usable_essays,
)

RAW = Path(__file__).resolve().parent.parent / "data" / "external" / "cowsl2h" / "raw"
OUT = Path(__file__).resolve().parent.parent / "data" / "test_correcto_cowsl2h200.csv"
SEED = 2026
N = 200
HEADER = (
    "# Oraciones CORRECTAS reales de COWS-L2H (github.com/ucdaviscl/cowsl2h, Apache-2.0): el\n"
    "# profesor las dejó sin cambios. Solo autores de la partición de DESARROLLO (misma\n"
    "# partición por autor, semilla 42, que fetch_external_data.py); muestra de 200 con semilla\n"
    "# 2026 (scripts/sample_identity_controls.py). Mide el daño sobre texto correcto; población:\n"
    "# universitarios de español L2, no niños. Esperado = entrada.\n"
    "categoria|entrada|esperado\n"
)
_TOKENS = re.compile(r"\w+|[^\w\s]")


def main() -> None:
    essays = []
    for name in COWSL2H_FILES:
        essays.extend(usable_essays(parse_cowsl2h_csv((RAW / name).read_text(encoding="utf-8")), name))
    _, essay_authors, _ = build_pairs_by_essay(essays)
    dev_authors, holdout_authors = split_by_author(
        essay_authors.values(), seed=SPLIT_SEED, holdout_fraction=HOLDOUT_FRACTION)

    sentences = []
    for essay in essays:
        if essay["autor_id"] not in dev_authors:      # holdout y autores sin pares: fuera
            continue
        for entrada, _ in align_essay(essay["essay"], essay["corrected1"])["identity_pairs"]:
            if "*" in entrada or "|" in entrada or not 4 <= len(_TOKENS.findall(entrada)) <= 22:
                continue
            sentences.append(entrada)
    sentences = sorted(set(sentences))
    random.Random(SEED).shuffle(sentences)
    sample = sentences[:N]
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(HEADER)
        for s in sample:
            f.write(f"correcta_l2|{s}|{s}\n")
    print(f"[OK] {len(sentences)} oraciones correctas de autores de desarrollo; "
          f"{len(sample)} -> {OUT} (autores dev={len(dev_authors)}, holdout={len(holdout_authors)} excluidos)")


if __name__ == "__main__":
    main()
