"""Genera pares (erronea, corregida) para reforzar el SUBJUNTIVO IRREGULAR.

Mismo enfoque que `generate_agreement_data.py`: el LoRA global aprende sobre
`training_pairs_clean.csv`, donde el subjuntivo obligatorio con verbos
irregulares (quiero que tú *vienes* -> vengas) casi no aparece, así que el T5
no lo genera. Aquí se construyen pares dirigidos:

  * minados: frases correctas del corpus con disparador + subjuntivo, a las que
    se corrompe el verbo al indicativo;
  * plantillas: disparador + sujeto + verbo irregular en indicativo (error) ->
    subjuntivo (corrección), con complementos naturales por verbo;
  * controles: disparadores de INDICATIVO (creo que, sé que...) con el verbo
    correcto y solo tildes que restaurar en la entrada, para que el modelo no
    sobreaplique el cambio. Sin pares de identidad: el corpus base no tiene
    ninguno y añadirlos (7 % en un primer intento) inclinó casos al límite del
    T5 (se callo/calló) hacia "dejar como está".

Salida: data/training_pairs_subjunctive.csv (se mezcla con v2 -> v3).
Uso: python scripts/generate_subjunctive_data.py
"""
import csv
import random
import re
import unicodedata

random.seed(11)

CLEAN_CSV = "data/training_pairs_clean.csv"
OUT_CSV = "data/training_pairs_subjunctive.csv"

# Disparadores que exigen subjuntivo (alineados con SUBJUNCTIVE_TRIGGERS del
# módulo de alternativas, más algunos volitivos frecuentes en el aula).
TRIGGERS = [
    "quiero que", "espero que", "ojalá que", "ojalá", "es posible que",
    "es necesario que", "es importante que", "es mejor que", "me alegra que",
    "me gusta que", "para que", "antes de que", "dudo que", "no creo que",
    "te pido que", "necesito que", "prefiero que", "hace falta que",
    "es urgente que", "mi mamá quiere que", "el profesor quiere que",
    "mis padres esperan que", "no quiero que", "deseo que",
]

# Disparadores de indicativo: el verbo NO debe cambiar (controles).
IND_TRIGGERS = [
    "creo que", "sé que", "dice que", "veo que", "es verdad que",
    "pienso que", "me parece que", "sabemos que", "está claro que",
    "me dijo que", "es cierto que", "seguro que",
]

# lemma -> persona -> (indicativo, subjuntivo)
VERBS = {
    "venir":   {"tu": ("vienes", "vengas"), "el": ("viene", "venga"), "nos": ("venimos", "vengamos"), "ellos": ("vienen", "vengan")},
    "tener":   {"tu": ("tienes", "tengas"), "el": ("tiene", "tenga"), "nos": ("tenemos", "tengamos"), "ellos": ("tienen", "tengan")},
    "hacer":   {"tu": ("haces", "hagas"), "el": ("hace", "haga"), "nos": ("hacemos", "hagamos"), "ellos": ("hacen", "hagan")},
    "decir":   {"tu": ("dices", "digas"), "el": ("dice", "diga"), "nos": ("decimos", "digamos"), "ellos": ("dicen", "digan")},
    "poner":   {"tu": ("pones", "pongas"), "el": ("pone", "ponga"), "nos": ("ponemos", "pongamos"), "ellos": ("ponen", "pongan")},
    "salir":   {"tu": ("sales", "salgas"), "el": ("sale", "salga"), "nos": ("salimos", "salgamos"), "ellos": ("salen", "salgan")},
    "traer":   {"tu": ("traes", "traigas"), "el": ("trae", "traiga"), "nos": ("traemos", "traigamos"), "ellos": ("traen", "traigan")},
    "oír":     {"tu": ("oyes", "oigas"), "el": ("oye", "oiga"), "nos": ("oímos", "oigamos"), "ellos": ("oyen", "oigan")},
    "ir":      {"tu": ("vas", "vayas"), "el": ("va", "vaya"), "nos": ("vamos", "vayamos"), "ellos": ("van", "vayan")},
    "ser":     {"tu": ("eres", "seas"), "el": ("es", "sea"), "nos": ("somos", "seamos"), "ellos": ("son", "sean")},
    "estar":   {"tu": ("estás", "estés"), "el": ("está", "esté"), "nos": ("estamos", "estemos"), "ellos": ("están", "estén")},
    "saber":   {"tu": ("sabes", "sepas"), "el": ("sabe", "sepa"), "nos": ("sabemos", "sepamos"), "ellos": ("saben", "sepan")},
    "poder":   {"tu": ("puedes", "puedas"), "el": ("puede", "pueda"), "nos": ("podemos", "podamos"), "ellos": ("pueden", "puedan")},
    "querer":  {"tu": ("quieres", "quieras"), "el": ("quiere", "quiera"), "nos": ("queremos", "queramos"), "ellos": ("quieren", "quieran")},
    "dar":     {"tu": ("das", "des"), "el": ("da", "dé"), "nos": ("damos", "demos"), "ellos": ("dan", "den")},
    "ver":     {"tu": ("ves", "veas"), "el": ("ve", "vea"), "nos": ("vemos", "veamos"), "ellos": ("ven", "vean")},
    "llegar":  {"tu": ("llegas", "llegues"), "el": ("llega", "llegue"), "nos": ("llegamos", "lleguemos"), "ellos": ("llegan", "lleguen")},
    "jugar":   {"tu": ("juegas", "juegues"), "el": ("juega", "juegue"), "nos": ("jugamos", "juguemos"), "ellos": ("juegan", "jueguen")},
    "empezar": {"tu": ("empiezas", "empieces"), "el": ("empieza", "empiece"), "nos": ("empezamos", "empecemos"), "ellos": ("empiezan", "empiecen")},
    "pedir":   {"tu": ("pides", "pidas"), "el": ("pide", "pida"), "nos": ("pedimos", "pidamos"), "ellos": ("piden", "pidan")},
    "dormir":  {"tu": ("duermes", "duermas"), "el": ("duerme", "duerma"), "nos": ("dormimos", "durmamos"), "ellos": ("duermen", "duerman")},
    "conocer": {"tu": ("conoces", "conozcas"), "el": ("conoce", "conozca"), "nos": ("conocemos", "conozcamos"), "ellos": ("conocen", "conozcan")},
    "seguir":  {"tu": ("sigues", "sigas"), "el": ("sigue", "siga"), "nos": ("seguimos", "sigamos"), "ellos": ("siguen", "sigan")},
    "volver":  {"tu": ("vuelves", "vuelvas"), "el": ("vuelve", "vuelva"), "nos": ("volvemos", "volvamos"), "ellos": ("vuelven", "vuelvan")},
    "sentir":  {"tu": ("sientes", "sientas"), "el": ("siente", "sienta"), "nos": ("sentimos", "sintamos"), "ellos": ("sienten", "sientan")},
    "entender": {"tu": ("entiendes", "entiendas"), "el": ("entiende", "entienda"), "nos": ("entendemos", "entendamos"), "ellos": ("entienden", "entiendan")},
    "pensar":  {"tu": ("piensas", "pienses"), "el": ("piensa", "piense"), "nos": ("pensamos", "pensemos"), "ellos": ("piensan", "piensen")},
    "buscar":  {"tu": ("buscas", "busques"), "el": ("busca", "busque"), "nos": ("buscamos", "busquemos"), "ellos": ("buscan", "busquen")},
    # regulares, para que el patrón no dependa solo de la irregularidad
    "estudiar": {"tu": ("estudias", "estudies"), "el": ("estudia", "estudie"), "nos": ("estudiamos", "estudiemos"), "ellos": ("estudian", "estudien")},
    "comer":   {"tu": ("comes", "comas"), "el": ("come", "coma"), "nos": ("comemos", "comamos"), "ellos": ("comen", "coman")},
    "escribir": {"tu": ("escribes", "escribas"), "el": ("escribe", "escriba"), "nos": ("escribimos", "escribamos"), "ellos": ("escriben", "escriban")},
    "hablar":  {"tu": ("hablas", "hables"), "el": ("habla", "hable"), "nos": ("hablamos", "hablemos"), "ellos": ("hablan", "hablen")},
    "leer":    {"tu": ("lees", "leas"), "el": ("lee", "lea"), "nos": ("leemos", "leamos"), "ellos": ("leen", "lean")},
}

COMPLEMENTS = {
    "venir": ["conmigo", "a la fiesta", "temprano", "a mi casa", "mañana", "a la reunión", "con nosotros"],
    "tener": ["cuidado", "tiempo", "paciencia", "razón", "frío", "todo listo", "suerte"],
    "hacer": ["la tarea", "ejercicio", "silencio", "la cena", "un dibujo", "las cosas bien"],
    "decir": ["la verdad", "algo", "eso", "tu nombre", "lo que piensas", "mentiras"],
    "poner": ["la mesa", "atención", "los libros aquí", "el abrigo", "orden en el cuarto"],
    "salir": ["temprano", "de casa", "al recreo", "a jugar", "tan tarde", "con tus amigos"],
    "traer": ["el cuaderno", "la comida", "los libros", "agua", "el permiso firmado"],
    "oír": ["la música", "el ruido", "al profesor", "esto", "bien"],
    "ir": ["al colegio", "a la playa", "al médico", "con ellos", "solo", "a dormir"],
    "ser": ["puntual", "amable", "feliz", "honesto", "responsable", "así"],
    "estar": ["aquí", "bien", "en casa", "tranquilo", "listo a las ocho", "atento"],
    "saber": ["la respuesta", "la verdad", "eso", "nadar", "leer bien", "cuidarte"],
    "poder": ["venir", "ayudar", "descansar", "jugar", "terminar hoy", "entrar"],
    "querer": ["estudiar", "comer", "ayudar", "venir", "salir"],
    "dar": ["la respuesta", "un paseo", "las gracias", "el ejemplo", "la mano"],
    "ver": ["la película", "el partido", "a tu abuela", "esto", "el problema"],
    "llegar": ["temprano", "a tiempo", "tarde", "pronto", "antes que yo", "a la meta"],
    "jugar": ["en el parque", "fútbol", "con cuidado", "en la calle", "después de comer"],
    "empezar": ["la tarea", "temprano", "ahora", "la clase", "de nuevo"],
    "pedir": ["ayuda", "permiso", "perdón", "otra oportunidad", "más comida"],
    "dormir": ["temprano", "bien", "ocho horas", "un poco", "en clase"],
    "conocer": ["la ciudad", "a mi familia", "el museo", "a mis amigos", "la historia"],
    "seguir": ["estudiando", "las reglas", "adelante", "el camino", "así"],
    "volver": ["pronto", "a casa", "temprano", "mañana", "a intentarlo"],
    "sentir": ["miedo", "frío", "dolor", "vergüenza", "lo mismo"],
    "entender": ["la lección", "el problema", "esto", "la pregunta", "mis razones"],
    "pensar": ["en eso", "bien las cosas", "antes de hablar", "en el futuro", "así"],
    "buscar": ["la respuesta", "trabajo", "tus cosas", "otra solución", "el libro"],
    "estudiar": ["mucho", "para el examen", "en silencio", "por la tarde", "medicina"],
    "comer": ["verduras", "todo", "despacio", "en casa", "más frutas"],
    "escribir": ["la carta", "bien", "tu nombre", "el cuento", "con letra clara"],
    "hablar": ["despacio", "en clase", "con ella", "más fuerte", "en español"],
    "leer": ["el libro", "mucho", "en voz alta", "las instrucciones", "todos los días"],
}

# Sujetos por persona; "" = sujeto tácito. "usted" concuerda con la 3.ª singular.
SUBJECTS = {
    "tu": ["tú", "tú", "tú", ""],
    "el": ["él", "ella", "usted", "mi hermano", "la profesora", "el niño", "mi mamá", "tu amigo",
           "mi papá", "la niña", "el doctor", "tu hermana", ""],
    "nos": ["nosotros", "nosotras", ""],
    "ellos": ["ellos", "ellas", "ustedes", "los niños", "mis padres", "tus amigos", "los estudiantes",
              "las niñas", "mis hermanos", "los profesores", ""],
}

OPENERS = ["", "", "", "", "hola,", "mira,", "por favor,", "mamá,", "oye,", "profe,", "la verdad,", "hijo,"]
CLOSERS = ["", "", "", "por favor", "hoy", "mañana", "esta vez", "siempre", "también"]

# Inverso: forma subjuntiva -> indicativa (para minar el corpus).
SUBJ2IND = {}
for lemma, persons in VERBS.items():
    for ind, subj in persons.values():
        SUBJ2IND.setdefault(subj, ind)
SUBJ2IND["haya"] = "hay"
_TRIGGER_RE = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in sorted(TRIGGERS, key=len, reverse=True))
    + r")\s+((?:\w+\s+){0,3})(\w+)",
    re.IGNORECASE,
)


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _join(*parts: str) -> str:
    return re.sub(r"\s+", " ", " ".join(p for p in parts if p)).strip()


def _surface(err: str, ok: str) -> tuple:
    """Variantes de superficie: el alumno escribe a veces sin tildes ni mayúscula."""
    r = random.random()
    if r < 0.45:                                  # tal cual llega del teclado
        return err, ok
    if r < 0.75:                                  # sin tildes en la entrada
        return _strip_accents(err), ok
    return err[0].upper() + err[1:] + ".", ok[0].upper() + ok[1:] + "."   # oración formal


def mine_pairs(limit: int) -> list:
    """Frases correctas del corpus con disparador + subjuntivo -> se corrompe al indicativo."""
    out = []
    for row in csv.reader(open(CLEAN_CSV, encoding="utf-8")):
        if len(row) < 2:
            continue
        ok = row[1].strip()
        m = _TRIGGER_RE.search(ok)
        if not m:
            continue
        verb = m.group(3)
        low = verb.lower()
        if low not in SUBJ2IND:
            continue
        ind = SUBJ2IND[low]
        if verb[0].isupper():
            ind = ind.capitalize()
        err = ok[: m.start(3)] + ind + ok[m.end(3):]
        if err != ok:
            out.append((err, ok))
        if len(out) >= limit:
            break
    return out


def template_pairs(n: int) -> list:
    out = []
    lemmas = list(VERBS)
    for _ in range(n):
        lemma = random.choice(lemmas)
        person = random.choice(list(VERBS[lemma]))
        ind, subj = VERBS[lemma][person]
        subject = random.choice(SUBJECTS[person])
        trig = random.choice(TRIGGERS)
        comp = random.choice(COMPLEMENTS[lemma])
        opener, closer = random.choice(OPENERS), random.choice(CLOSERS)
        neg = "no " if random.random() < 0.12 else ""
        err = _join(opener, trig, subject, neg + ind, comp, closer)
        ok = _join(opener, trig, subject, neg + subj, comp, closer)
        out.append(_surface(err, ok))
    # "hay" -> "haya" (impersonal)
    for _ in range(n // 25):
        trig = random.choice(TRIGGERS)
        comp = random.choice(["clases mañana", "tiempo", "comida para todos", "un examen", "problemas", "sitio"])
        out.append(_surface(_join(trig, "hay", comp), _join(trig, "haya", comp)))
    return out


# Pretérito tras el clítico "se": la forma sin tilde (callo, paso, quedo...) es
# una 1.ª persona del presente, imposible con "se", así que va la 3.ª del
# pretérito. Sin este bloque el T5 dejaba "se callo" tal cual (empate de beams).
PRET_SE = {
    "callo": "calló", "paso": "pasó", "quedo": "quedó", "llamo": "llamó", "levanto": "levantó",
    "acabo": "acabó", "caso": "casó", "enojo": "enojó", "canso": "cansó", "sento": "sentó",
    "lastimo": "lastimó", "olvido": "olvidó", "escapo": "escapó", "perdio": "perdió",
    "rompio": "rompió", "durmio": "durmió", "rio": "rió", "cayo": "cayó", "fue": "fue",
    "asusto": "asustó", "peino": "peinó", "baño": "bañó", "lavo": "lavó", "porto": "portó",
}
PRET_SUBJ = ["", "mi hermano", "la niña", "el perro", "ella", "él", "mi mamá", "el profesor", "tu primo"]
PRET_COMP = ["de repente", "en la clase", "ayer", "en el recreo", "muy rápido", "sin querer",
             "en el patio", "toda la tarde", "por la mañana", "al final"]


def preterite_pairs(n: int) -> list:
    out = []
    words = [w for w in PRET_SE if w != PRET_SE[w]]
    for _ in range(n):
        w = random.choice(words)
        subj = random.choice(PRET_SUBJ)
        comp = random.choice(PRET_COMP)
        opener = random.choice(OPENERS)
        err = _join(opener, subj, "se", w, comp)
        ok = _join(opener, subj, "se", PRET_SE[w], comp)
        out.append(_surface(err, ok))
    return out


def control_pairs(n_ind: int) -> list:
    """Indicativo legítimo tras disparador de indicativo: la entrada solo pierde
    tildes y el verbo NO cambia. Se descartan las frases sin tilde (identidad)."""
    out = []
    lemmas = list(VERBS)
    while len(out) < n_ind:
        lemma = random.choice(lemmas)
        person = random.choice(list(VERBS[lemma]))
        ind, _ = VERBS[lemma][person]
        ok = _join(random.choice(OPENERS), random.choice(IND_TRIGGERS), random.choice(SUBJECTS[person]),
                   ind, random.choice(COMPLEMENTS[lemma]), random.choice(CLOSERS))
        err = _strip_accents(ok)
        if err == ok:
            continue
        if random.random() < 0.3:
            err, ok = err[0].upper() + err[1:] + ".", ok[0].upper() + ok[1:] + "."
        out.append((err, ok))
    return out


def main() -> None:
    mined = mine_pairs(limit=2000)
    templ = template_pairs(5000)
    ctrl = control_pairs(900)
    pret = preterite_pairs(500)
    rows = mined + templ + ctrl + pret
    random.shuffle(rows)
    seen, uniq = set(), []
    for err, ok in rows:
        if (err, ok) in seen or err == ok:
            continue
        seen.add((err, ok))
        uniq.append((err, ok))
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["erronea", "corregida"])
        w.writerows(uniq)
    print(f"minados={len(mined)} plantillas={len(templ)} controles={len(ctrl)} preterito={len(pret)} -> {len(uniq)} unicos en {OUT_CSV}")
    for err, ok in uniq[:12]:
        print(f"  {err!r:60} -> {ok!r}")


if __name__ == "__main__":
    main()
