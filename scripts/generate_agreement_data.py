"""
scripts/generate_agreement_data.py
Genera pares (erronea, corregida) enfocados en CONCORDANCIA sujeto-verbo, el
punto que el T5 no aprendió (fue->fueron). Combina:
  (a) Minería del corpus limpio: frases con verbo plural + sujeto plural claro,
      se corrompe el verbo a singular -> par de concordancia.
  (b) Plantillas del patrón difícil de largo alcance: "los N que <clausula> <ser>",
      donde el verbo está separado del sujeto por una oración de relativo.
El resultado se mezcla con una muestra de pares generales (para no olvidar la
ortografía) y se guarda en data/training_pairs_agreement_v5.csv.

Corrección del 2026-09-30 (la versión de julio generó data/training_pairs_agreement.csv,
que se conserva sin tocar porque v2-v4 se entrenaron con él):
  * Las plantillas usaban adjetivos solo en masculino con sujetos femeninos, y predicados
    sin filtrar por clase: la frase CORRECTA enseñaba "varias casas que encontramos eran
    buenos", "estas amigas fueron caros", "tres camisas tienen hambre" (~380 pares en v4).
    Ahora cada sustantivo lleva género y clase (ser/cosa) y cada predicado su forma
    femenina y las clases que admite.
  * La minería aceptaba cualquier frase con un determinante plural en algún sitio, aunque
    el singular también fuera correcto (sujeto omitido "tiene más cosas en común", o el
    plural era complemento: "le dije a las niñas que podía"). Ahora exige sujeto plural
    explícito antes del verbo, o pospuesto a un copulativo al abrir la cláusula.
  * "habían" (haber impersonal) enseñaba había -> habían, al revés de la norma; y "ven"
    -> "ve" convertía imperativos correctos en errores. Ambos quedan fuera.

Uso:  python scripts/generate_agreement_data.py [--out RUTA]
"""
import argparse
import csv
import random
import re

random.seed(7)

CLEAN_CSV = "data/training_pairs_clean.csv"
OUT_CSV   = "data/training_pairs_agreement_v5.csv"

# Verbo plural (3ª pers.) -> singular. Al corromper, el sujeto plural permanece,
# creando el error de concordancia que el modelo debe revertir.
PL2SG = {
    "son": "es", "eran": "era", "fueron": "fue", "están": "está",
    "estaban": "estaba", "estuvieron": "estuvo", "tienen": "tiene",
    "tenían": "tenía", "tuvieron": "tuvo", "hacen": "hace", "hacían": "hacía",
    "hicieron": "hizo", "van": "va", "iban": "iba", "vienen": "viene",
    "venían": "venía", "vinieron": "vino", "dicen": "dice", "decían": "decía",
    "dijeron": "dijo", "saben": "sabe", "sabían": "sabía", "quieren": "quiere",
    "querían": "quería", "quisieron": "quiso", "pueden": "puede",
    "podían": "podía", "pudieron": "pudo", "deben": "debe", "debían": "debía",
    "creen": "cree", "creían": "creía", "veían": "veía",
    "vieron": "vio", "dan": "da", "daban": "daba", "dieron": "dio",
    "llegan": "llega", "llegaron": "llegó", "comen": "come", "comían": "comía",
    "comieron": "comió", "corren": "corre", "corrieron": "corrió",
    "viven": "vive", "vivían": "vivía", "hablan": "habla", "hablaron": "habló",
    "juegan": "juega", "jugaron": "jugó", "trabajan": "trabaja", "ponen": "pone",
    "pusieron": "puso", "salen": "sale", "salieron": "salió", "entran": "entra",
    "entraron": "entró", "ganan": "gana", "ganaron": "ganó", "leen": "lee",
    "leyeron": "leyó", "parecen": "parece", "sienten": "siente",
    "sintieron": "sintió", "necesitan": "necesita", "buscan": "busca",
    "llevan": "lleva", "llevaron": "llevó", "traen": "trae", "trajeron": "trajo",
}
_PLURAL_CUE = re.compile(
    r"\b(los|las|unos|unas|muchos|muchas|varios|varias|pocos|pocas|algunos|"
    r"algunas|estos|estas|esos|esas|aquellos|aquellas|ellos|ellas|dos|tres|"
    r"cuatro|cinco|seis|mis|tus|sus)\b", re.I)
_TOKEN = re.compile(r"\w+|[^\w\s]")
# Un determinante plural tras una preposición es complemento ("a las niñas",
# "de los niños"), no sujeto.
_PREPOSITIONS = {"a", "al", "de", "del", "con", "para", "por", "en", "sin", "entre",
                 "sobre", "hacia", "desde", "hasta", "contra", "según", "tras"}
_SUBJECT_WINDOW = 8
# Sujeto pospuesto: solo con copulativos/verbos de presentación al abrir la cláusula
# ("¿Qué es las funciones…?" -> "son"; "Aquí viene sus amigos" -> "vienen").
_POSTPOSED_VERBS = {"son", "eran", "fueron", "están", "estaban", "vienen", "van"}
_CLAUSE_OPENERS = {"qué", "quiénes", "cuáles", "dónde", "cómo", "cuándo",
                   "aquí", "ahí", "allí", "allá", "ya"}


def has_explicit_plural_subject(tokens: list, vi: int) -> bool:
    """True si el verbo plural en tokens[vi] lleva un sujeto plural explícito, de modo
    que su forma singular es inequívocamente un error de concordancia."""
    for j in range(vi - 1, max(-1, vi - 1 - _SUBJECT_WINDOW), -1):
        tok = tokens[j]
        if not tok[0].isalnum():
            break                                    # la puntuación cierra la búsqueda
        if _PLURAL_CUE.fullmatch(tok):
            return j == 0 or tokens[j - 1] not in _PREPOSITIONS
    if tokens[vi] in _POSTPOSED_VERBS and vi + 1 < len(tokens) \
            and _PLURAL_CUE.fullmatch(tokens[vi + 1]):
        prev = tokens[vi - 1] if vi > 0 else ""
        return prev == "" or not prev[0].isalnum() or prev in _CLAUSE_OPENERS
    return False


def mine_pairs(limit: int) -> list:
    out = []
    with open(CLEAN_CSV, encoding="utf-8") as fh:
        rows = list(csv.reader(fh))[1:]
    for r in rows:
        if len(r) < 2:
            continue
        correct = r[1].strip()
        low = correct.lower()
        if not _PLURAL_CUE.search(low):
            continue
        tokens = _TOKEN.findall(low)
        # corrompe UN verbo plural presente en la frase
        for pl, sg in PL2SG.items():
            if pl in tokens:
                if has_explicit_plural_subject(tokens, tokens.index(pl)):
                    err = re.sub(rf"(?i)\b{pl}\b", sg, correct, count=1)
                    if err != correct:
                        out.append((err, correct))
                break
        if len(out) >= limit:
            break
    return out


# (b) Plantillas de largo alcance: sujeto plural + relativo + verbo lejano.
# Determinante y predicado concuerdan en género con el sustantivo, y la clase
# (ser vivo / cosa) filtra relativos y predicados absurdos ("las camisas tienen hambre").
# Se incluyen femeninos, cuantificadores y pronombres para que el modelo generalice
# y no memorice "los ... ".
NOUNS = {                                   # plural -> (género, clase)
    **{n: ("m", "ser") for n in ["perros", "gatos", "niños", "amigos", "chicos", "vecinos",
                                 "jugadores", "pájaros", "hermanos", "profesores"]},
    **{n: ("m", "cosa") for n in ["libros", "autos", "árboles", "regalos", "zapatos",
                                  "cuadros", "juguetes", "coches"]},
    **{n: ("f", "ser") for n in ["niñas", "amigas", "chicas", "vecinas", "gallinas",
                                 "profesoras"]},
    **{n: ("f", "cosa") for n in ["casas", "flores", "mesas", "sillas", "puertas", "ventanas",
                                  "plantas", "camisas", "manzanas", "historias", "canciones",
                                  "montañas"]},
}
DET = {"m": ["los", "unos", "muchos", "estos", "esos", "dos", "tres", "varios"],
       "f": ["las", "unas", "muchas", "estas", "esas", "dos", "tres", "varias"]}
REL = {  # relativo -> clases que lo admiten
    "que vi": ("ser", "cosa"), "que compré": ("cosa",), "que tenía": ("cosa",),
    "que había ahí": ("ser", "cosa"), "que llegaron": ("ser", "cosa"),
    "que estaban ahí": ("ser", "cosa"), "que me diste": ("cosa",),
    "que encontramos": ("ser", "cosa"), "que vimos ayer": ("ser", "cosa"),
    "de la casa": ("ser", "cosa"), "del parque": ("ser", "cosa"),
    "de mi tío": ("ser", "cosa"), "": ("ser", "cosa"),
}
# (verbo sg, verbo pl, [(complemento masc, complemento fem, clases)])
PRED_SER = [
    ("fue", "fueron", [("muy grandes", "muy grandes", ("ser", "cosa")),
                       ("muy bonitos", "muy bonitas", ("ser", "cosa")),
                       ("caros", "caras", ("cosa",)), ("nuevos", "nuevas", ("cosa",)),
                       ("azules", "azules", ("cosa",)),
                       ("pequeños", "pequeñas", ("ser", "cosa")),
                       ("geniales", "geniales", ("ser", "cosa")),
                       ("míos", "mías", ("ser", "cosa"))]),
    ("era", "eran", [("enormes", "enormes", ("ser", "cosa")),
                     ("divertidos", "divertidas", ("ser", "cosa")),
                     ("antiguos", "antiguas", ("cosa",)),
                     ("rápidos", "rápidas", ("ser",)),
                     ("buenos", "buenas", ("ser", "cosa"))]),
    ("estaba", "estaban", [("rotos", "rotas", ("cosa",)),
                           ("sucios", "sucias", ("ser", "cosa")),
                           ("listos", "listas", ("ser", "cosa")),
                           ("mojados", "mojadas", ("ser", "cosa")),
                           ("cerrados", "cerradas", ("cosa",)),
                           ("cansados", "cansadas", ("ser",))]),
    ("tiene", "tienen", [("dueño", "dueño", ("ser", "cosa")),
                         ("nombre", "nombre", ("ser", "cosa")),
                         ("precio", "precio", ("cosa",)),
                         ("hambre", "hambre", ("ser",))]),
    ("es", "son", [("de madera", "de madera", ("cosa",)),
                   ("de mi hermano", "de mi hermano", ("ser", "cosa")),
                   ("muy útiles", "muy útiles", ("cosa",)),
                   ("importantes", "importantes", ("ser", "cosa"))]),
]
PRONS = {"ellos": "m", "ellas": "f"}        # siempre seres; sin relativo ("ellos que vi" no es español)


def template_pairs(n: int) -> list:
    out = []
    for _ in range(n):
        sg, pl, preds = random.choice(PRED_SER)
        if random.random() < 0.15:                      # sujeto pronombre
            subj, rel, cls = random.choice(list(PRONS)), "", "ser"
            gender = PRONS[subj]
        else:
            noun = random.choice(list(NOUNS))
            gender, cls = NOUNS[noun]
            subj = f"{random.choice(DET[gender])} {noun}"
            rel = random.choice([r for r, classes in REL.items() if cls in classes])
        masc, fem, _ = random.choice([p for p in preds if cls in p[2]])
        pred = masc if gender == "m" else fem
        correct = f"{subj} {rel} {pl} {pred}".replace("  ", " ").strip()
        err     = f"{subj} {rel} {sg} {pred}".replace("  ", " ").strip()
        out.append((err, correct))
    return out


def general_sample(n: int) -> list:
    with open(CLEAN_CSV, encoding="utf-8") as fh:
        rows = list(csv.reader(fh))[1:]
    random.shuffle(rows)
    # fuera los pares que enseñan "había -> habían" (haber impersonal en plural)
    return [(r[0].strip(), r[1].strip()) for r in rows[:n] if len(r) >= 2
            and not (re.search(r"(?i)\bhabía\b", r[0]) and re.search(r"(?i)\bhabían\b", r[1]))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--out", default=OUT_CSV)
    args = parser.parse_args()
    mined = mine_pairs(limit=6000)
    templ = template_pairs(4000)
    general = general_sample(6000)
    allp = mined + templ + general
    random.shuffle(allp)
    # dedup
    seen, final = set(), []
    for e, c in allp:
        if (e, c) in seen or e == c:
            continue
        seen.add((e, c)); final.append((e, c))
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["erronea", "corregida"]); w.writerows(final)
    print(f"[OK] concordancia minada={len(mined)} plantillas={len(templ)} "
          f"general={len(general)} -> total limpio={len(final)}")
    print(f"[OK] guardado en {args.out}")
    print("=== muestra concordancia ===")
    for e, c in final[:6]:
        print(f"  ERR: {e}\n  OK : {c}")


if __name__ == "__main__":
    main()
