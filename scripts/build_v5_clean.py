"""
scripts/build_v5_clean.py
Corpus LIMPIO para LoRA v5 (frases correctas; el corruptor infantil genera después las
entradas con errores). Fuentes, cada una en data/v5/clean_<fuente>.txt:

  * cowsl2h   — COWS-L2H, solo autores de DESARROLLO (misma partición por autor que
                fetch_external_data.py): oraciones corregidas por el profesor
                (esperado_1 de pairs-dev.csv) y oraciones que dejó sin cambios.
  * subtitulos — objetivos de training_pairs_clean.csv que pasan un filtro de calidad:
                todas las palabras válidas en el léxico (es_50k + léxico escolar),
                ninguna forma sin tilde de una palabra que la lleva ("vere", "oceano"),
                sin temas adultos.
  * claude     — data/v5/claude/*.txt (frases escritas por agentes Claude en registro
                escolar peruano; IA, declarado).
  * dominio_publico — data/v5/dominio_publico/*.txt si existe (ver fetch_v5_public_domain.py).

Ninguna frase que aparezca (normalizada) como entrada o referencia de un set de
evaluación data/test_*.csv o data/eval_gold.csv entra al corpus. La reserva se usa
solo para excluir y su contenido nunca se imprime.

Uso:  python scripts/build_v5_clean.py
"""
import csv
import glob
import re
import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))

OUT_DIR = REPO / "data" / "v5"
_TOKENS = re.compile(r"\w+|[^\w\s]")
_ADULT = {
    "sexo", "sexy", "puta", "puto", "mierda", "joder", "coño", "cabrón", "pene", "vagina", "droga",
    "drogas", "cocaína", "heroína", "alcohol", "cerveza", "whisky", "vodka", "borracho", "borracha",
    "pistola", "rifle", "disparo", "disparar", "matar", "maté", "mató", "asesinato", "asesino",
    "cadáver", "muerto", "muerta", "infierno", "diablo", "demonio", "tragaperras", "casino", "prostituta",
    "amante", "cama", "besar", "ligar", "novia", "novio", "follar", "cigarrillo", "cigarro", "fumar",
}


def norm(s: str) -> str:
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    return " ".join(re.findall(r"\w+", s.lower()))


def evaluation_sentences() -> set:
    """Entradas y referencias normalizadas de todos los sets de evaluación."""
    seen = set()
    for path in glob.glob(str(REPO / "data" / "test_*.csv")) + [str(REPO / "data" / "eval_gold.csv")]:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip() or line.startswith("#") or line.startswith("categoria|"):
                    continue
                for field in line.rstrip("\n").split("|")[1:]:
                    if field.strip():
                        seen.add(norm(field))
    return seen


def clean_ok(sentence: str, min_tokens: int = 4, max_tokens: int = 30) -> bool:
    n = len(_TOKENS.findall(sentence))
    return min_tokens <= n <= max_tokens and "*" not in sentence and "|" not in sentence


def from_cowsl2h() -> list:
    from fetch_external_data import (COWSL2H_FILES, HOLDOUT_FRACTION, SPLIT_SEED, align_essay,
                                     build_pairs_by_essay, parse_cowsl2h_csv, split_by_author,
                                     usable_essays)
    raw = REPO / "data" / "external" / "cowsl2h" / "raw"
    essays = []
    for name in COWSL2H_FILES:
        essays.extend(usable_essays(parse_cowsl2h_csv((raw / name).read_text(encoding="utf-8")), name))
    _, essay_authors, _ = build_pairs_by_essay(essays)
    dev_authors, _ = split_by_author(essay_authors.values(), seed=SPLIT_SEED, holdout_fraction=HOLDOUT_FRACTION)
    out = []
    for essay in essays:
        if essay["autor_id"] not in dev_authors:
            continue
        aligned = align_essay(essay["essay"], essay["corrected1"], essay["corrected2"])
        out.extend(target for _, target, _ in aligned["pairs"])
        out.extend(target for _, target in aligned["identity_pairs"])
    return out


def from_subtitles() -> list:
    from infrastructure.nlp.phonetic_engine import PhoneticEngine
    pe = PhoneticEngine()
    freqs, accents = pe.word_freqs, pe.accent_dict
    out = []
    with open(REPO / "data" / "training_pairs_clean.csv", encoding="utf-8") as fh:
        for row in list(csv.reader(fh))[1:]:
            if len(row) < 2:
                continue
            target = row[1].strip()
            words = re.findall(r"\w+", target)
            lows = [w.lower() for w in words]
            if any(w in _ADULT for w in lows):
                continue
            ok = True
            for w, low in zip(words, lows):
                if w[:1].isupper() and w is not words[0]:
                    continue                                   # nombre propio
                if low.isdigit():
                    continue
                if freqs.get(low, 0) < 300 or (low in accents and accents[low] != low):
                    ok = False                                  # rara o forma sin tilde ("vere")
                    break
            if ok:
                out.append(target)
    return out


def from_folder(folder: Path) -> list:
    out = []
    for path in sorted(folder.glob("*.txt")):
        with open(path, encoding="utf-8") as fh:
            out.extend(line.strip() for line in fh if line.strip() and not line.startswith("#"))
    return out


def main() -> None:
    excluded = evaluation_sentences()
    sources = {
        "cowsl2h": from_cowsl2h,
        "subtitulos": from_subtitles,
        "claude": lambda: from_folder(OUT_DIR / "claude"),
        "dominio_publico": lambda: from_folder(OUT_DIR / "dominio_publico"),
    }
    seen = set()
    for name, loader in sources.items():
        sentences = loader()
        kept, dup, test_hits, bad = [], 0, 0, 0
        for s in sentences:
            s = re.sub(r"\s+", " ", s).strip()
            key = norm(s)
            if not clean_ok(s):
                bad += 1
            elif key in excluded:
                test_hits += 1
            elif key in seen:
                dup += 1
            else:
                seen.add(key)
                kept.append(s)
        with open(OUT_DIR / f"clean_{name}.txt", "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(kept) + ("\n" if kept else ""))
        print(f"[{name}] {len(sentences)} -> {len(kept)} (fuera: {bad} por longitud/formato, "
              f"{test_hits} coinciden con sets de evaluación, {dup} duplicadas)")


if __name__ == "__main__":
    main()
