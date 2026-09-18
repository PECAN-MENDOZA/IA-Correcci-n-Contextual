"""Descarga y alinea el corpus público COWS-L2H para usarlo como sustituto
verificado de la recolección propia (Task 0 del plan `2026-09-16-mejora-modelo-ia`;
ver `.superpowers/sdd/task-0-brief.md`).

Qué hace `--cowsl2h`:
  1. Descarga los 28 CSV de COWS-L2H (commit fijado `COWSL2H_COMMIT`) a
     `data/external/cowsl2h/raw/` y verifica cada uno contra `FILE_HASHES_SHA256`
     (vacío hasta la primera descarga real; luego se fija en este archivo).
  2. Toma `essay` (entrada) y `corrected1` (esperado_1, y `corrected2` como
     esperado_2 cuando también exista), separa cada texto en oraciones con un
     separador consciente de abreviaturas y comillas (sin NLTK/spaCy) y alinea
     1:1 solo cuando ambos coinciden en número de oraciones.
  3. Descarta pares con distancia de edición por palabra > 40 % (reescrituras,
     no correcciones puntuales) y separa los pares de identidad (oración sin
     cambios): se cuentan pero no se entrenan con ellos.
  4. Reparte los **autores** (columna `id` = estudiante, no ensayo) 70/30
     (semilla 42) en `pairs-dev.csv` (dentro del repo, en la carpeta
     gitignored) y un holdout bloqueado fuera del repo, con su SHA-256
     registrado en `HOLDOUT-SHA256.txt`. El holdout ya escrito se protege: si
     el archivo existe, hace falta `--force-holdout` para sobrescribirlo.
  5. Escribe `data/external/cowsl2h-manifest.json` (sí versionado) con la
     procedencia y los conteos, para que la Task 1 y la Task 6 puedan auditar
     el dataset sin volver a tocar el holdout.

**Por qué la partición es por AUTOR y no por ensayo** (corrección de un bug
real encontrado en revisión): en el CSV de COWS-L2H la columna `id` identifica
al ESTUDIANTE, no al ensayo — el corpus es longitudinal, el mismo estudiante
escribe en varios prompts/quarters (28 archivos). De los 5 382 filas crudas
solo hay 1 935 `id` distintos. El plan original pedía partición "por ensayo...
para que no haya fugas de estilo"; la lectura que de verdad cumple esa
intención es particionar por autor, porque dejar ensayos del mismo autor en
dev y en holdout sería la fuga de estilo que el plan quiere evitar. Cada
ensayo se identifica de forma única con la clave `archivo#fila` (nunca se
usa `id` como clave de ensayo); `id` se conserva como `autor_id` para la
partición, junto con `prompt` y `quarter` como metadatos.

`--verify-only` solo revisa los CSV ya descargados contra los hashes fijados,
sin red ni reprocesar nada.

Uso:
    python scripts/fetch_external_data.py --cowsl2h
    python scripts/fetch_external_data.py --cowsl2h --force-holdout
    python scripts/fetch_external_data.py --verify-only
"""
import argparse
import csv
import difflib
import hashlib
import io
import json
import random
import re
import sys
import urllib.request
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent

# ───────────────────────────── COWS-L2H ─────────────────────────────

COWSL2H_COMMIT = "ebb11724f258f3ed377a27ed34f08897a8e5639c"
COWSL2H_RAW_BASE = f"https://raw.githubusercontent.com/ucdaviscl/cowsl2h/{COWSL2H_COMMIT}/csv"

COWSL2H_FILES = [
    "beautiful.F20.csv", "beautiful.S20.csv", "beautiful.W20.csv", "beautiful.W21.csv",
    "chaplin.S21.csv",
    "famous.F17.csv", "famous.S17.csv", "famous.S18.csv", "famous.SU17.csv", "famous.W18.csv",
    "place_you_dislike.S21.csv",
    "special.F18.csv", "special.F19.csv", "special.S19.csv", "special.W19.csv",
    "terrible.F18.csv", "terrible.F19.csv", "terrible.S19.csv", "terrible.W19.csv",
    "vacation.F17.csv", "vacation.S17.csv", "vacation.S18.csv", "vacation.SU17.csv", "vacation.W18.csv",
    "yourself.F20.csv", "yourself.S20.csv", "yourself.W20.csv", "yourself.W21.csv",
]

# SHA-256 (hex, minúsculas) de cada CSV en el commit fijado. Vacío hasta la
# primera descarga real; a partir de ahí queda fijado aquí (si el hash de un
# archivo cambiara con el mismo commit, algo andaría mal: la descarga falla).
FILE_HASHES_SHA256 = {
    "beautiful.F20.csv": "0b213eb17f5ab261bfc8fac60e8a323618f7aeedf41d3d66cc31b7f41f227aab",
    "beautiful.S20.csv": "433e5871f7855cc40010170069858be7ca034c2d2982e1ad857c758c37cbe395",
    "beautiful.W20.csv": "eb18e4645717aa7a7c6b4d13edb5bcdd7974181ace07ce424dbd37eb4e1f14ad",
    "beautiful.W21.csv": "335467bb41f5edd1694c3b102a091a964fe00968b79bf37960dc59a7801d169f",
    "chaplin.S21.csv": "71aa6e28d199b7317aa671494615454c7cb23f3bc213b8aebd5846af6ab621c3",
    "famous.F17.csv": "c3334ccb604222f52557e46a681ff35a759a5c3aa91cf3e5707ea72e9cb7031c",
    "famous.S17.csv": "1484a0f7e5e10b1fc8d0dab3b01938c82584947593a87fb0267fdd9cde1f6baf",
    "famous.S18.csv": "f27b6bb0b42a9aa3e47773bf9e46df21899088b466522be30843f32168167e73",
    "famous.SU17.csv": "e59e38f75e53a22ad70a2927fbacf11351509690d0d27b8d058287c2c8dd4fca",
    "famous.W18.csv": "db18616a163ba3184776b21306651fee43adba785c92cd889b5941b0f9b3ee7e",
    "place_you_dislike.S21.csv": "14a17a0c70ef1a61671f3a7e08c12b29c31f140a80b50f6b4fc6158425bc706e",
    "special.F18.csv": "c7d712cdcc31aebba41fe2c69af12794d163833ffad63b2d67ac2e9ff22065d8",
    "special.F19.csv": "222f84d008ef089e1a314438fa817dce98030c9fd57e3ba023680eb0b54b8a36",
    "special.S19.csv": "80c13ae0faf89362be460ef678ecaf4243a29ec3adbce330d7957b4911fbca38",
    "special.W19.csv": "d8564bd0eeea466838f729aa11b057f30b1d167d5b83fb6c8ac9ca2ada746aa4",
    "terrible.F18.csv": "468d2572625ed2d33c34bfa1313507f789c4a4ff9845a89ef84e48fab0990918",
    "terrible.F19.csv": "ea889a48af3da4eed3ddedfbf512f672f637e4fb9c336b92d6f5b2bdf2d66b29",
    "terrible.S19.csv": "2bb7ab1fb883e4c81a7fb2853a08df313750882032f435b55cdec2e147633076",
    "terrible.W19.csv": "ef532d43d2ab254c463dc66f62d85d4114242ac852de98d0ca4a64555b37145e",
    "vacation.F17.csv": "8136b1314443a88c8e7535430403477ad0ebd9b35f352558390e08c4f678ff22",
    "vacation.S17.csv": "4327ae3fe0d8106625408872fd197f7dd9a41ea061f5c8f594f2432810ce70a1",
    "vacation.S18.csv": "007abaecc2038c37172681581c03fc26b303a25d6c9f93be9363db9b6bde84b0",
    "vacation.SU17.csv": "ca04c829ba2f56b094cd27495d6761bc858ff0aa3ec0da62fc8ed5d6fb746933",
    "vacation.W18.csv": "c4c6418a6934569ca88ad76ed609c70c8e8d19023fd2debf6d10ffbb73f61216",
    "yourself.F20.csv": "b7e59144ac0c63fa2b7607ae9ef624f49fac4f92a7e9fc7d61784bbfffe02cf1",
    "yourself.S20.csv": "767682778207406136aaf4297d9c0f22a0dd3f14bb5ef61cb89efe44e18e3fd2",
    "yourself.W20.csv": "525ad1b073f9f7fcd1403c4f267b81c90bfbd9e958a65ebe78308148b7aaaa5c",
    "yourself.W21.csv": "9cc9bd086b22981fe5ebf739905c25f8e0359da16dcb426f9a600542458af9fc",
}

DEFAULT_OUT_DIR = REPO_DIR / "data" / "external" / "cowsl2h"
DEFAULT_HOLDOUT_PATH = Path(r"C:\Users\Dovamul\Desktop\TESIS\documentos\datos-reservados\holdout-cowsl2h.csv")
MANIFEST_PATH = REPO_DIR / "data" / "external" / "cowsl2h-manifest.json"

# ───────────────────────────── parámetros ─────────────────────────────

EDIT_RATIO_THRESHOLD = 0.4     # por encima: reescritura, se descarta
HOLDOUT_FRACTION = 0.3         # 30 % de los AUTORES (no de ensayos ni oraciones)
SPLIT_SEED = 42
CATEGORIA_SIN_ANOTAR = "sin_anotar"   # la asigna el anotador de la Task 1

_PAIRS_HEADER_COMMENT = (
    "# Pares generados desde COWS-L2H (https://github.com/ucdaviscl/cowsl2h,\n"
    f"# commit {COWSL2H_COMMIT}, licencia Apache-2.0) por scripts/fetch_external_data.py.\n"
    "# categoria='sin_anotar': la asigna el anotador de la Task 1 (ver README).\n"
    "# Formato: categoria|entrada|esperado_1|esperado_2 (esperado_2 opcional),\n"
    "# igual que evaluate.load_dataset.\n"
)


# ───────────────────────────── oraciones ─────────────────────────────

# Abreviaturas frecuentes en español (en minúsculas, sin el punto) tras las
# que NO se corta aunque siga una mayúscula. "EE. UU." y "p. ej." son
# abreviaturas de dos palabras: basta con incluir cada palabra suelta
# ("ee", "uu", "p", "ej") porque la palabra que sigue a "p." ya es minúscula
# (no dispara el corte por esa razón) y "EE."/"UU." quedan cubiertas cada una
# por su propia entrada.
_ABREVIATURAS = {
    "sr", "sra", "srta", "dr", "dra", "ud", "uds",
    "etc", "p", "ej", "ee", "uu", "núm", "pág", "av",
}

_WORD_CHARS_RE = re.compile(r"[A-Za-zÁÉÍÓÚÑÜáéíóúñü]+$")
_BOUNDARY_LOOKAHEAD_RE = re.compile(r"\s+(?=[A-ZÁÉÍÓÚÑÜ¿¡])")

# Comillas que protegen su contenido de un corte de oración: comilla de cierre
# esperada para cada comilla de apertura.
_CLOSING_QUOTE_FOR_OPENING = {'"': '"', "«": "»"}


def _preceding_word(text: str, punct_index: int) -> str:
    """Última palabra (en minúsculas) inmediatamente antes de
    `text[punct_index]`, para decidir si es una abreviatura conocida."""
    m = _WORD_CHARS_RE.search(text[:punct_index])
    return m.group(0).lower() if m else ""


def split_sentences(text: str) -> list:
    """Divide `text` en oraciones: corta tras una racha de '.', '!' o '?'
    cuando sigue un espacio y una mayúscula (o '¿'/'¡' de apertura), salvo que
    la palabra justo antes de la puntuación sea una abreviatura conocida
    (`_ABREVIATURAS`) o que el corte caiga dentro de una comilla abierta
    (`"..."` o `«...»`: no se corta dentro, cortar justo después de la
    comilla de cierre sí está permitido). No usa NLTK/spaCy; es una
    heurística suficiente para alinear ensayo/corrección oración a oración
    (no para lingüística fina)."""
    text = (text or "").strip()
    if not text:
        return []

    quote_stack = []   # comillas abiertas, en orden ('"' o '«')
    sentences = []
    start = 0
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == '"':
            if quote_stack and quote_stack[-1] == '"':
                quote_stack.pop()
            else:
                quote_stack.append('"')
            i += 1
            continue
        if ch == "«":
            quote_stack.append("«")
            i += 1
            continue
        if ch == "»":
            if quote_stack and quote_stack[-1] == "«":
                quote_stack.pop()
            i += 1
            continue
        if ch in ".!?":
            j = i
            while j < n and text[j] in ".!?":
                j += 1
            k = j
            closes_open_quote = (
                k < n and quote_stack and text[k] in _CLOSING_QUOTE_FOR_OPENING.values()
                and _CLOSING_QUOTE_FOR_OPENING.get(quote_stack[-1]) == text[k]
            )
            if closes_open_quote:
                k += 1
            m = _BOUNDARY_LOOKAHEAD_RE.match(text, k)
            still_inside_quote = quote_stack[:-1] if closes_open_quote else quote_stack
            if m and not still_inside_quote and _preceding_word(text, i) not in _ABREVIATURAS:
                if closes_open_quote:
                    quote_stack.pop()
                sentences.append(text[start:k].strip())
                start = m.end()
                i = start
                continue
            i = j
            continue
        i += 1

    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return [s for s in sentences if s]


def word_edit_ratio(entrada: str, esperado: str) -> float:
    """Distancia de edición a nivel de PALABRA (vía `difflib.SequenceMatcher`
    sobre listas de palabras) como proporción de las palabras de `entrada`.
    Se usa para descartar reescrituras (> 40 %) que no son correcciones
    puntuales, tal como pide el brief de la Task 0."""
    words_in = entrada.split()
    words_out = esperado.split()
    if not words_in:
        return 0.0 if not words_out else 1.0
    matcher = difflib.SequenceMatcher(a=words_in, b=words_out, autojunk=False)
    edits = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag != "equal")
    return edits / len(words_in)


def align_essay(essay: str, corrected1: str, corrected2: str = "") -> dict:
    """Alinea `essay` (entrada) con `corrected1` (esperado_1) oración a
    oración, y con `corrected2` (esperado_2) cuando también exista y alinee.

    Reglas (brief Task 0, Step 1):
      - alineación 1:1 solo si `essay` y `corrected1` tienen el mismo número
        de oraciones; si no, se descarta el ensayo completo (`aligned=False`);
      - oración idéntica -> par de identidad: se cuenta pero no se usa para
        entrenar (perjudican al T5, ver `generate_subjunctive_data.py`);
      - distancia de edición por palabra > 40 % -> se descarta (reescritura);
      - una oración con '|' (el delimitador del formato de `evaluate.py`) se
        descarta entera en vez de reescribirla, para no alterar el corpus;
      - `esperado_2` se añade solo si `corrected2` también alinea (mismo
        número de oraciones que `essay`); si su oración tiene '|', se omite
        únicamente `esperado_2` y se conserva el par con `esperado_1`.

    Devuelve un dict con `aligned`, `n_sentences`, `pairs`
    (`[(entrada, esperado_1, esperado_2_o_None), ...]`), `identity_pairs`
    (`[(entrada, esperado_1), ...]`), `discarded_filter` y `discarded_pipe`.
    """
    essay_sents = split_sentences(essay)
    c1_sents = split_sentences(corrected1)
    result = {
        "aligned": False,
        "n_sentences": len(essay_sents),
        "pairs": [],
        "identity_pairs": [],
        "discarded_filter": 0,
        "discarded_pipe": 0,
    }
    if not essay_sents or len(essay_sents) != len(c1_sents):
        return result
    result["aligned"] = True

    c2_sents = split_sentences(corrected2) if corrected2 else []
    c2_aligns = bool(c2_sents) and len(c2_sents) == len(essay_sents)

    for i, entrada in enumerate(essay_sents):
        esperado1 = c1_sents[i]
        esperado2 = c2_sents[i] if c2_aligns else ""

        if "|" in entrada or "|" in esperado1:
            result["discarded_pipe"] += 1
            continue
        if esperado2 and "|" in esperado2:
            esperado2 = ""

        if entrada == esperado1:
            result["identity_pairs"].append((entrada, esperado1))
            continue

        if word_edit_ratio(entrada, esperado1) > EDIT_RATIO_THRESHOLD:
            result["discarded_filter"] += 1
            continue

        result["pairs"].append((entrada, esperado1, esperado2 or None))

    return result


# ───────────────────────────── CSV COWS-L2H ─────────────────────────────

def parse_cowsl2h_csv(csv_text: str) -> list:
    """Parsea un CSV de COWS-L2H ya decodificado (utf-8) y devuelve las filas
    crudas tal cual las entrega `csv.DictReader` (sin filtrar)."""
    return list(csv.DictReader(io.StringIO(csv_text)))


def _normalize_whitespace(text: str) -> str:
    """Colapsa saltos de línea y espacios repetidos a un solo espacio. Varios
    ensayos de COWS-L2H traen '\\n' dentro del propio texto (párrafos); sin
    esto, una oración partida a mitad de un salto de línea produciría una fila
    con salto de línea real, rompiendo el formato de una-fila-por-línea que
    lee `evaluate.load_dataset`."""
    return re.sub(r"\s+", " ", text or "").strip()


def usable_essays(rows: list, filename: str = "") -> list:
    """Filtra filas con `essay` y `corrected1` no vacíos (muchas filas del
    corpus tienen `corrected1` vacío: el profesor no las corrigió; se
    descartan aquí, antes de alinear). Normaliza espacios en blanco (ver
    `_normalize_whitespace`).

    Cada fila resultante trae:
      - `essay_key`: clave ÚNICA de ensayo, `f"{filename}#{indice_de_fila}"`.
        NUNCA se usa la columna `id` como clave de ensayo: `id` es el
        estudiante, no el ensayo, y el mismo estudiante aparece en varias
        filas (mismo archivo o archivos distintos, ver docstring del módulo).
      - `autor_id`: la columna `id` cruda (estudiante), para particionar
        dev/holdout por autor.
      - `prompt`, `quarter`: metadatos del ensayo (columnas del CSV), no se
        usan para particionar, solo quedan disponibles para análisis futuro.
    """
    out = []
    for row_index, row in enumerate(rows):
        essay = _normalize_whitespace(row.get("essay"))
        c1 = _normalize_whitespace(row.get("corrected1"))
        if not essay or not c1:
            continue
        essay_key = f"{filename}#{row_index}"
        autor_id = (row.get("id") or "").strip() or f"_sin_id_{essay_key}"
        out.append({
            "essay_key": essay_key,
            "autor_id": autor_id,
            "prompt": (row.get("prompt") or "").strip(),
            "quarter": (row.get("quarter") or "").strip(),
            "essay": essay,
            "corrected1": c1,
            "corrected2": _normalize_whitespace(row.get("corrected2")),
        })
    return out


def build_pairs_by_essay(essays: list) -> tuple:
    """Alinea todos los `essays` y agrupa las filas resultantes por
    `essay_key` (nunca por `autor_id`: un autor puede tener varios ensayos).
    Devuelve `(pares_por_essay_key, essay_key_a_autor_id, stats)`; `stats`
    trae los conteos globales que va a llevar el manifiesto."""
    pairs_by_essay = {}
    essay_authors = {}
    stats = {
        "essays_con_corrected1": len(essays),
        "essays_con_corrected2": sum(1 for e in essays if e["corrected2"]),
        "essays_alineados": 0,
        "pares_alineados": 0,
        "pares_descartados_filtro": 0,
        "pares_descartados_pipe": 0,
        "pares_identidad": 0,
    }
    for essay in essays:
        r = align_essay(essay["essay"], essay["corrected1"], essay["corrected2"])
        stats["pares_descartados_filtro"] += r["discarded_filter"]
        stats["pares_descartados_pipe"] += r["discarded_pipe"]
        stats["pares_identidad"] += len(r["identity_pairs"])
        if r["aligned"]:
            stats["essays_alineados"] += 1
        if r["pairs"]:
            pairs_by_essay[essay["essay_key"]] = r["pairs"]
            essay_authors[essay["essay_key"]] = essay["autor_id"]
            stats["pares_alineados"] += len(r["pairs"])
    return pairs_by_essay, essay_authors, stats


# ───────────────────────────── partición ─────────────────────────────

def split_by_author(author_ids, seed: int = SPLIT_SEED, holdout_fraction: float = HOLDOUT_FRACTION) -> tuple:
    """Partición determinista por AUTOR (`autor_id`, columna `id` =
    estudiante), no por ensayo ni por oración. COWS-L2H es longitudinal (un
    mismo estudiante escribe en varios prompts/quarters); particionar por
    ensayo dejaría textos del mismo autor en dev y en holdout, que es
    justamente la fuga de estilo que se quiere evitar. Semilla 42, 30 % de
    los autores -> holdout."""
    ids = sorted({str(a) for a in author_ids})
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_holdout = round(len(ids) * holdout_fraction)
    holdout = set(ids[:n_holdout])
    dev = set(ids[n_holdout:])
    return dev, holdout


def _assert_partition_covers_pairs(pairs_by_essay: dict, dev_essay_keys, holdout_essay_keys,
                                    dev_rows: int, holdout_rows: int, pares_alineados: int) -> None:
    """Invariante que el manifiesto promete: todo ensayo con pares queda en
    exactamente un lado de la partición, y las filas escritas en dev+holdout
    suman exactamente `pares_alineados`. Falla ruidosamente si no (mejor un
    `AssertionError` claro aquí que un dataset silenciosamente incompleto)."""
    dev_essay_keys = set(dev_essay_keys)
    holdout_essay_keys = set(holdout_essay_keys)
    todos = set(pairs_by_essay.keys())
    if dev_essay_keys & holdout_essay_keys:
        raise AssertionError(
            "Invariante rota: hay ensayos en dev Y en holdout a la vez: "
            f"{sorted(dev_essay_keys & holdout_essay_keys)[:5]}..."
        )
    if (dev_essay_keys | holdout_essay_keys) != todos:
        faltan = todos - (dev_essay_keys | holdout_essay_keys)
        sobran = (dev_essay_keys | holdout_essay_keys) - todos
        raise AssertionError(
            "Invariante rota: la partición no cubre exactamente los ensayos con "
            f"pares (faltan {len(faltan)}, sobran {len(sobran)})."
        )
    if dev_rows + holdout_rows != pares_alineados:
        raise AssertionError(
            f"Invariante rota: dev_filas ({dev_rows}) + holdout_filas ({holdout_rows}) "
            f"= {dev_rows + holdout_rows} != pares_alineados ({pares_alineados})."
        )


# ───────────────────────────── salida ─────────────────────────────

def format_row(categoria: str, entrada: str, esperado1: str, esperado2: str = None) -> str:
    """Una fila en el formato `categoria|entrada|esperado_1[|esperado_2]` que
    lee `evaluate.load_dataset`."""
    campos = [categoria, entrada, esperado1]
    if esperado2:
        campos.append(esperado2)
    return "|".join(campos)


def write_pairs_csv(path, pairs_by_essay: dict, essay_ids, header_comment: str = "") -> int:
    """Escribe las filas de los ensayos en `essay_ids` (subconjunto de las
    claves de `pairs_by_essay`, es decir `essay_key`s) al archivo `path`, en
    el formato de `evaluate.py`. Devuelve el número de filas escritas."""
    path = Path(path)
    rows = []
    for essay_id in sorted(str(i) for i in essay_ids):
        for entrada, esperado1, esperado2 in pairs_by_essay.get(essay_id, []):
            rows.append(format_row(CATEGORIA_SIN_ANOTAR, entrada, esperado1, esperado2))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        if header_comment:
            fh.write(header_comment)
        fh.write("categoria|entrada|esperado_1|esperado_2\n")
        for row in rows:
            fh.write(row + "\n")
    return len(rows)


# ───────────────────────────── hashing / descarga ─────────────────────────────

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def download_file(url: str, dest_path, expected_sha256: str = None, opener=None) -> str:
    """Descarga `url` en `dest_path`. Si el archivo ya existe y coincide con
    `expected_sha256`, no vuelve a descargar (idempotente para re-ejecutar el
    script). Si se da `expected_sha256` y no coincide tras descargar, lanza
    `ValueError` (protege contra que el corpus se haya movido bajo el commit
    fijado). Devuelve el SHA-256 (hex, minúsculas) del contenido."""
    opener = opener or urllib.request.urlopen
    dest_path = Path(dest_path)
    if dest_path.exists() and expected_sha256:
        digest = sha256_file(dest_path)
        if digest == expected_sha256:
            return digest
    with opener(url, timeout=30) as resp:
        data = resp.read()
    digest = sha256_bytes(data)
    if expected_sha256 and digest != expected_sha256:
        raise ValueError(
            f"hash inesperado para {url}: esperado {expected_sha256}, obtenido {digest}"
        )
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_bytes(data)
    return digest


def verify_raw_files(raw_dir, hashes: dict) -> dict:
    """Verifica los CSV ya descargados en `raw_dir` contra `hashes`, sin red.
    Devuelve `{nombre: 'ok' | 'falta' | 'no_coincide'}`."""
    raw_dir = Path(raw_dir)
    out = {}
    for name, expected in hashes.items():
        p = raw_dir / name
        if not p.exists():
            out[name] = "falta"
            continue
        out[name] = "ok" if sha256_file(p) == expected else "no_coincide"
    return out


def ensure_holdout_writable(holdout_path, force: bool) -> None:
    """Guarda de congelamiento del holdout: si `holdout_path` ya existe, se
    niega a sobrescribirlo salvo que `force=True` (`--force-holdout`). Que el
    archivo exista puede significar que ya se usó para evaluar (Task 6); una
    reejecución silenciosa del script rompería esa garantía. No toca el
    archivo (no lo lee ni lo borra), solo decide si continuar."""
    holdout_path = Path(holdout_path)
    if holdout_path.exists() and not force:
        raise FileExistsError(
            f"El holdout ya existe en {holdout_path} y está bloqueado: no se "
            "sobrescribe salvo que pases --force-holdout explícitamente (solo "
            "si nadie lo ha leído/evaluado todavía)."
        )


# ───────────────────────────── manifiesto ─────────────────────────────

def build_manifest(*, source_commit, file_hashes, stats,
                    dev_author_ids, holdout_author_ids,
                    dev_essay_keys, holdout_essay_keys,
                    dev_rows, holdout_rows, seed, holdout_fraction,
                    holdout_sha256, holdout_file_name) -> dict:
    """Construye el manifiesto que se versiona en
    `data/external/cowsl2h-manifest.json` (el holdout en sí queda fuera del
    repo; aquí solo va su ruta y su hash). Reporta autores, ensayos y filas
    por lado, porque la partición es por autor pero lo que de verdad importa
    para entrenar/evaluar es cuántas filas (pares) caen en cada lado."""
    total_autores = len(dev_author_ids) + len(holdout_author_ids)
    total_essays = len(dev_essay_keys) + len(holdout_essay_keys)
    total_filas_particion = dev_rows + holdout_rows
    return {
        "fuente": "https://github.com/ucdaviscl/cowsl2h",
        "commit": source_commit,
        "licencia": "Apache-2.0",
        "hashes_csv": dict(file_hashes),
        "total_filas_csv": stats.get("total_filas_csv"),
        "total_essays_con_corrected1": stats["essays_con_corrected1"],
        "total_essays_con_corrected2": stats["essays_con_corrected2"],
        "essays_alineados": stats["essays_alineados"],
        "pares_alineados": stats["pares_alineados"],
        "pares_descartados_filtro_40pct": stats["pares_descartados_filtro"],
        "pares_descartados_delimitador_pipe": stats["pares_descartados_pipe"],
        "pares_identidad_no_incluidos": stats["pares_identidad"],
        "particion": {
            "unidad": "autor (columna `id` del CSV = estudiante, no ensayo)",
            "nota": (
                "El 30% es sobre AUTORES, no sobre ensayos ni oraciones: COWS-L2H "
                "es longitudinal (el mismo estudiante escribe en varios "
                "prompts/quarters, hasta 28 archivos). Particionar por ensayo "
                "dejaría ensayos del mismo autor en dev y en holdout, la fuga de "
                "estilo que la partición busca evitar. Por eso dev_essays/"
                "holdout_essays y dev_filas/holdout_filas no son exactamente "
                "70/30: el número de ensayos por autor no es uniforme."
            ),
            "semilla": seed,
            "fraccion_holdout_autores": holdout_fraction,
            "dev_autores": len(dev_author_ids),
            "holdout_autores": len(holdout_author_ids),
            "dev_essays": len(dev_essay_keys),
            "holdout_essays": len(holdout_essay_keys),
            "dev_filas": dev_rows,
            "holdout_filas": holdout_rows,
            "holdout_pct_autores": round(100 * len(holdout_author_ids) / total_autores, 2) if total_autores else None,
            "holdout_pct_essays": round(100 * len(holdout_essay_keys) / total_essays, 2) if total_essays else None,
            "holdout_pct_filas": round(100 * holdout_rows / total_filas_particion, 2) if total_filas_particion else None,
        },
        "holdout": {
            "ruta": str(holdout_file_name),
            "sha256": holdout_sha256,
            "nota": "fuera del repositorio; nunca releído salvo para recalcular el hash",
        },
    }


# ───────────────────────────── CLI ─────────────────────────────

def _download_all(raw_dir) -> dict:
    hashes = {}
    for name in COWSL2H_FILES:
        url = f"{COWSL2H_RAW_BASE}/{name}"
        expected = FILE_HASHES_SHA256.get(name)
        digest = download_file(url, Path(raw_dir) / name, expected_sha256=expected)
        hashes[name] = digest
    return hashes


def run_verify_only(out_dir) -> bool:
    raw_dir = Path(out_dir) / "raw"
    if not FILE_HASHES_SHA256:
        print("FILE_HASHES_SHA256 está vacío: ejecuta primero `--cowsl2h` para fijarlos.")
        return False
    results = verify_raw_files(raw_dir, FILE_HASHES_SHA256)
    ok = len(results) == len(COWSL2H_FILES) and all(v == "ok" for v in results.values())
    for name in COWSL2H_FILES:
        print(f"  {name}: {results.get(name, 'falta (no está en FILE_HASHES_SHA256)')}")
    print("VERIFICACION OK" if ok else "VERIFICACION FALLIDA")
    return ok


def run_cowsl2h(out_dir, holdout_path, force_holdout: bool = False) -> dict:
    out_dir = Path(out_dir)
    holdout_path = Path(holdout_path)
    raw_dir = out_dir / "raw"

    # Falla rápido, antes de descargar nada, si el holdout ya está congelado.
    ensure_holdout_writable(holdout_path, force_holdout)

    print(f"Descargando COWS-L2H (commit {COWSL2H_COMMIT}) en {raw_dir} ...")
    hashes = _download_all(raw_dir)
    missing_in_script = [n for n in COWSL2H_FILES if n not in FILE_HASHES_SHA256]
    if missing_in_script:
        print("AVISO: copia estos hashes en FILE_HASHES_SHA256 (primera descarga real):")
        for n in missing_in_script:
            print(f'    "{n}": "{hashes[n]}",')

    total_filas = 0
    all_essays = []
    for name in COWSL2H_FILES:
        text = (raw_dir / name).read_text(encoding="utf-8")
        rows = parse_cowsl2h_csv(text)
        total_filas += len(rows)
        all_essays.extend(usable_essays(rows, filename=name))

    pairs_by_essay, essay_authors, stats = build_pairs_by_essay(all_essays)
    stats["total_filas_csv"] = total_filas

    dev_authors, holdout_authors = split_by_author(
        essay_authors.values(), seed=SPLIT_SEED, holdout_fraction=HOLDOUT_FRACTION)
    dev_essay_keys = {k for k, a in essay_authors.items() if a in dev_authors}
    holdout_essay_keys = {k for k, a in essay_authors.items() if a in holdout_authors}

    dev_path = out_dir / "pairs-dev.csv"
    dev_rows = write_pairs_csv(dev_path, pairs_by_essay, dev_essay_keys, header_comment=_PAIRS_HEADER_COMMENT)
    holdout_rows = write_pairs_csv(holdout_path, pairs_by_essay, holdout_essay_keys,
                                    header_comment=_PAIRS_HEADER_COMMENT)

    # La garantía que el manifiesto promete: ni un ensayo con pares se pierde
    # ni se duplica entre dev y holdout. Falla ruidosamente si no se cumple.
    _assert_partition_covers_pairs(pairs_by_essay, dev_essay_keys, holdout_essay_keys,
                                    dev_rows, holdout_rows, stats["pares_alineados"])

    holdout_sha = sha256_file(holdout_path)   # única relectura permitida: para el hash
    hash_txt_path = holdout_path.parent / "HOLDOUT-SHA256.txt"
    hash_txt_path.write_text(
        f"{holdout_sha}  {holdout_path.name}\nfilas: {holdout_rows}\n",
        encoding="utf-8",
    )

    manifest = build_manifest(
        source_commit=COWSL2H_COMMIT, file_hashes=hashes, stats=stats,
        dev_author_ids=dev_authors, holdout_author_ids=holdout_authors,
        dev_essay_keys=dev_essay_keys, holdout_essay_keys=holdout_essay_keys,
        dev_rows=dev_rows, holdout_rows=holdout_rows,
        seed=SPLIT_SEED, holdout_fraction=HOLDOUT_FRACTION,
        holdout_sha256=holdout_sha, holdout_file_name=holdout_path,
    )
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"pairs-dev.csv: {dev_rows} filas ({len(dev_authors)} autores, {len(dev_essay_keys)} ensayos) -> {dev_path}")
    print(f"holdout: {holdout_rows} filas ({len(holdout_authors)} autores, {len(holdout_essay_keys)} ensayos) -> {holdout_path}")
    print(f"HOLDOUT-SHA256.txt -> {hash_txt_path}")
    print(f"manifiesto -> {MANIFEST_PATH}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cowsl2h", action="store_true", help="descarga, verifica y procesa COWS-L2H")
    parser.add_argument("--verify-only", action="store_true",
                         help="solo verifica los CSV ya descargados contra FILE_HASHES_SHA256, sin red")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                         help="carpeta de salida (por defecto data/external/cowsl2h)")
    parser.add_argument("--holdout-path", default=str(DEFAULT_HOLDOUT_PATH),
                         help="ruta del holdout bloqueado, fuera del repo")
    parser.add_argument("--force-holdout", action="store_true",
                         help="permite sobrescribir el holdout si ya existe (solo si nadie lo leyó/evaluó)")
    args = parser.parse_args()

    if args.verify_only:
        sys.exit(0 if run_verify_only(args.out_dir) else 1)
    if args.cowsl2h:
        try:
            run_cowsl2h(args.out_dir, args.holdout_path, force_holdout=args.force_holdout)
        except FileExistsError as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)
        return
    parser.print_help()


if __name__ == "__main__":
    main()
