"""
scripts/generate_agreement_data.py
Genera pares (erronea, corregida) enfocados en CONCORDANCIA sujeto-verbo, el
punto que el T5 no aprendió (fue->fueron). Combina:
  (a) Minería del corpus limpio: frases con verbo plural + sujeto plural claro,
      se corrompe el verbo a singular -> par de concordancia.
  (b) Plantillas del patrón difícil de largo alcance: "los N que <clausula> <ser>",
      donde el verbo está separado del sujeto por una oración de relativo.
El resultado se mezcla con una muestra de pares generales (para no olvidar la
ortografía) y se guarda en data/training_pairs_agreement.csv.

Uso:  python scripts/generate_agreement_data.py
"""
import csv
import random
import re

random.seed(7)

CLEAN_CSV = "data/training_pairs_clean.csv"
OUT_CSV   = "data/training_pairs_agreement.csv"

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
    "creen": "cree", "creían": "creía", "ven": "ve", "veían": "veía",
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
    "habían": "había",   # haber impersonal (refuerza la regla)
}
_PLURAL_CUE = re.compile(
    r"\b(los|las|unos|unas|muchos|muchas|varios|varias|pocos|pocas|algunos|"
    r"algunas|estos|estas|esos|esas|aquellos|aquellas|ellos|ellas|dos|tres|"
    r"cuatro|cinco|seis|mis|tus|sus)\b", re.I)


def mine_pairs(limit: int) -> list:
    out = []
    rows = list(csv.reader(open(CLEAN_CSV, encoding="utf-8")))[1:]
    for r in rows:
        if len(r) < 2:
            continue
        correct = r[1].strip()
        low = correct.lower()
        if not _PLURAL_CUE.search(low):
            continue
        # corrompe UN verbo plural presente en la frase
        for pl, sg in PL2SG.items():
            if re.search(rf"\b{pl}\b", low):
                err = re.sub(rf"(?i)\b{pl}\b", sg, correct, count=1)
                if err != correct:
                    out.append((err, correct))
                break
        if len(out) >= limit:
            break
    return out


# (b) Plantillas de largo alcance: sujeto plural + relativo + verbo lejano.
# Determinante concuerda en género; se incluyen femeninos, cuantificadores y
# pronombres para que el modelo generalice y no memorice "los ... ".
NOUNS_M = ["perros", "gatos", "niños", "amigos", "libros", "autos", "árboles",
           "chicos", "vecinos", "jugadores", "pájaros", "regalos", "zapatos",
           "cuadros", "juguetes", "hermanos", "profesores", "coches"]
NOUNS_F = ["casas", "niñas", "amigas", "flores", "mesas", "sillas", "puertas",
           "ventanas", "plantas", "estrellas", "chicas", "vecinas", "camisas",
           "manzanas", "historias", "canciones", "montañas", "nubes"]
DET_M = ["los", "unos", "muchos", "estos", "esos", "dos", "tres", "varios"]
DET_F = ["las", "unas", "muchas", "estas", "esas", "dos", "tres", "varias"]
REL = ["que vi", "que compré", "que tenía", "que había ahí", "que llegaron",
       "que estaban ahí", "que me diste", "que encontramos", "que vimos ayer",
       "de la casa", "del parque", "de mi tío", ""]
PRED_SER = [("fue", "fueron", ["muy grandes", "muy bonitos", "caros", "nuevos",
                                "viejos", "azules", "pequeños", "geniales", "míos"]),
            ("era", "eran", ["enormes", "divertidos", "antiguos", "rápidos", "buenos"]),
            ("estaba", "estaban", ["rotos", "sucios", "listos", "mojados", "cerrados"]),
            ("tiene", "tienen", ["dueño", "nombre", "precio", "hambre"]),
            ("es", "son", ["de madera", "de mi hermano", "muy útiles", "importantes"])]
PRONS = ["ellos", "ellas"]


def template_pairs(n: int) -> list:
    out = []
    for _ in range(n):
        sg, pl, preds = random.choice(PRED_SER)
        pred = random.choice(preds)
        rel = random.choice(REL)
        if random.random() < 0.15:                      # sujeto pronombre
            subj = random.choice(PRONS)
            out.append((f"{subj} {rel} {sg} {pred}".replace("  ", " ").strip(),
                        f"{subj} {rel} {pl} {pred}".replace("  ", " ").strip()))
            continue
        if random.random() < 0.5:
            det, noun = random.choice(DET_M), random.choice(NOUNS_M)
        else:
            det, noun = random.choice(DET_F), random.choice(NOUNS_F)
        correct = f"{det} {noun} {rel} {pl} {pred}".replace("  ", " ").strip()
        err     = f"{det} {noun} {rel} {sg} {pred}".replace("  ", " ").strip()
        out.append((err, correct))
    return out


def general_sample(n: int) -> list:
    rows = list(csv.reader(open(CLEAN_CSV, encoding="utf-8")))[1:]
    random.shuffle(rows)
    return [(r[0].strip(), r[1].strip()) for r in rows[:n] if len(r) >= 2]


def main() -> None:
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
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["erronea", "corregida"]); w.writerows(final)
    print(f"[OK] concordancia minada={len(mined)} plantillas={len(templ)} "
          f"general={len(general)} -> total limpio={len(final)}")
    print(f"[OK] guardado en {OUT_CSV}")
    print("=== muestra concordancia ===")
    for e, c in final[:6]:
        print(f"  ERR: {e}\n  OK : {c}")


if __name__ == "__main__":
    main()
