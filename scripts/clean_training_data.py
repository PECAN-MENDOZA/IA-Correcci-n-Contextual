"""
scripts/clean_training_data.py
Limpia el CSV de pares de entrenamiento (erronea,corregida) de la basura de
OPUS: URLs/dominios, caracteres no latinos, pares sin error real y duplicados.

Uso:  python scripts/clean_training_data.py [entrada.csv] [salida.csv]
Por defecto: data/training_pairs.csv -> data/training_pairs_clean.csv
"""
import csv
import re
import sys

IN_PATH  = sys.argv[1] if len(sys.argv) > 1 else "data/training_pairs.csv"
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "data/training_pairs_clean.csv"

_URL   = re.compile(r"https?://|www\.|\b\S+\.(?:com|net|org|ru|es|ar|co|io|info)\b|@", re.I)
_WEIRD = re.compile(r"[<>{}\\|~^]|[А-я]|[一-鿿]|[؀-ۿ]")   # símbolos, cirílico, CJK, árabe
_HASLETTER = re.compile(r"[a-záéíóúüñ]", re.I)


def is_good(err: str, cor: str) -> bool:
    err, cor = err.strip(), cor.strip()
    if not err or not cor:
        return False
    if err == cor:                      # sin error inyectado -> sin señal
        return False
    if not (10 <= len(cor) <= 90):
        return False
    if _URL.search(err) or _URL.search(cor):
        return False
    if _WEIRD.search(err) or _WEIRD.search(cor):
        return False
    if not _HASLETTER.search(cor):
        return False
    return True


def main() -> None:
    rows = list(csv.reader(open(IN_PATH, encoding="utf-8", errors="replace")))
    header, rows = rows[0], rows[1:]

    seen, clean = set(), []
    for r in rows:
        if len(r) < 2:
            continue
        err, cor = r[0], r[1]
        if not is_good(err, cor):
            continue
        key = (err.strip(), cor.strip())
        if key in seen:
            continue
        seen.add(key)
        clean.append((err.strip(), cor.strip()))

    with open(OUT_PATH, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["erronea", "corregida"])
        w.writerows(clean)

    print(f"[OK] {len(rows)} -> {len(clean)} pares limpios ({len(rows)-len(clean)} descartados)")
    print(f"[OK] guardado en {OUT_PATH}")


if __name__ == "__main__":
    main()
