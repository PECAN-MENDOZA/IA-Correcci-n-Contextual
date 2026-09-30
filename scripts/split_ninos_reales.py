"""
scripts/split_ninos_reales.py
Parte data/test_ninos_reales.csv en DESARROLLO (60 %) y RESERVA (40 %), estratificado
por categoría con semilla fija, conservando la cabecera y el comentario de fuente de
cada bloque. Las reglas y los reentrenamientos se ajustan mirando solo el de desarrollo;
la reserva se evalúa una única vez al cerrar cada ciclo (sin --show-fails).

Uso:  python scripts/split_ninos_reales.py
"""
import random
from collections import defaultdict

SRC = "data/test_ninos_reales.csv"
DEV = "data/test_ninos_reales_dev.csv"
RES = "data/test_ninos_reales_reserva.csv"
SEED = 2026
RESERVE_FRACTION = 0.4


def split(lines: list, seed: int = SEED) -> set:
    """Índices de `lines` que van a la reserva."""
    by_cat = defaultdict(list)
    for i, line in enumerate(lines):
        if line.strip() and not line.startswith("#") and not line.startswith("categoria|"):
            by_cat[line.split("|", 1)[0]].append(i)
    rng = random.Random(seed)
    reserve = set()
    for cat in sorted(by_cat):
        idx = by_cat[cat][:]
        rng.shuffle(idx)
        reserve.update(idx[:round(len(idx) * RESERVE_FRACTION)])
    return reserve


def main() -> None:
    lines = open(SRC, encoding="utf-8").read().splitlines()
    reserve = split(lines)
    header = ["# Partición de test_ninos_reales.csv (scripts/split_ninos_reales.py, semilla "
              f"{SEED}, {int(RESERVE_FRACTION * 100)} % reserva estratificada por categoría)."]
    for path, keep, note in [
        (DEV, lambda i: i not in reserve, "# DESARROLLO: se mira para diseñar reglas y datos."),
        (RES, lambda i: i in reserve, "# RESERVA: evaluar una vez por ciclo, sin mirar los fallos."),
    ]:
        out = header + [note]
        for i, line in enumerate(lines):
            is_row = line.strip() and not line.startswith("#") and not line.startswith("categoria|")
            if not is_row or keep(i):
                out.append(line)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(out) + "\n")
        rows = sum(1 for l in out if l.strip() and not l.startswith("#") and not l.startswith("categoria|"))
        print(f"[OK] {path}: {rows} filas")


if __name__ == "__main__":
    main()
