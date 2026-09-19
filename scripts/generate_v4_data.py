"""Genera pares (erronea, corregida) dirigidos a los fallos que quedaron tras el
LoRA v3 y las reglas del 2026-09-18 (`data/test_tiempos.csv`):

  * controles de INDICATIVO tras subordinantes (aunque, cuando, si, porque,
    mientras, como, ya que, apenas): v3 aprendió "subordinante + indicativo ->
    subjuntivo" con los disparadores volitivos y lo sobreaplica ("aunque llueve"
    -> "llueva"). Aquí el verbo NO cambia; la entrada solo pierde tildes;
  * contrapeso: subjuntivo PROSPECTIVO tras cuando/apenas/en cuanto/hasta que
    ("cuando llegue a casa, te llamaré"), como control y como corrección
    ("cuando llegas, avísame" -> "llegues") SOLO con principal en futuro o
    imperativo; con principal en presente ("cuando llegue a casa, te llamo")
    solo control, porque es indistinguible del habitual ("cuando llego a
    casa, mi mamá prepara el café"). Sin el bloque, un primer v4 pasaba
    "cuando llegue a casa, te llamo" a pretérito; con correcciones en
    presente, un segundo v4 pasaba el habitual a subjuntivo;
  * concordancia del PREDICADO: sujeto plural (+ relativa) + verbo y adjetivo
    en singular -> ambos en plural ("los perros que vi estaba muy sucio" ->
    "estaban muy sucios"); v3 corregía el verbo y dejaba el adjetivo. Con
    controles en singular (solo tildes);
  * "té"/"te" y "él"/"el" en contexto ("el te esta caliente" -> "el té está
    caliente"; "el te ayuda" -> "él te ayuda").

Sin pares de identidad (misma lección que v3). Salida:
data/training_pairs_v4_extra.csv (se mezcla con v3 -> v4).
Uso: python scripts/generate_v4_data.py
"""
import csv
import random
import re
import unicodedata

random.seed(13)

OUT_CSV = "data/training_pairs_v4_extra.csv"


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _join(*parts: str) -> str:
    return re.sub(r"\s+", " ", " ".join(p for p in parts if p)).strip()


def _control(ok: str) -> tuple | None:
    """Par de control: la entrada solo pierde tildes (30 % además con mayúscula
    y punto). None si la frase no tiene tildes (sería identidad)."""
    err = _strip_accents(ok)
    if err == ok:
        return None
    if random.random() < 0.3:
        err, ok = err[0].upper() + err[1:] + ".", ok[0].upper() + ok[1:] + "."
    return err, ok


def _surface(err: str, ok: str) -> tuple:
    r = random.random()
    if r < 0.45:
        return err, ok
    if r < 0.75:
        return _strip_accents(err), ok
    return err[0].upper() + err[1:] + ".", ok[0].upper() + ok[1:] + "."


# ── 1. Indicativo legítimo tras subordinantes ────────────────────────────────

# Subordinada por tiempo (todas en indicativo, con alguna tilde cuando se puede).
SUB_CLAUSES = {
    "presente": [
        "llueve mucho", "hace frío", "hace calor", "llego a casa", "termino la tarea",
        "mi mamá cocina", "los niños juegan en el patio", "tengo tiempo", "estudias mucho",
        "el profesor explica la lección", "hay sol", "estamos cansados", "vienes temprano",
        "salimos del colegio", "mi papá trabaja", "no tengo dinero", "está oscuro",
        "el bebé duerme", "es tarde", "mi hermana canta", "tú no quieres", "nieva en la sierra",
        "ella está enferma", "no hay clases", "el perro ladra", "me duele la cabeza",
        "mis amigos vienen", "el bus llega tarde", "todos están aquí", "yo cocino",
    ],
    "pasado": [
        "llovió mucho", "hacía frío", "hacía calor", "llegué a casa", "terminé la tarea",
        "mi mamá cocinó", "los niños jugaban en el patio", "tuve tiempo", "estudiaste mucho",
        "el profesor explicó la lección", "había sol", "estábamos cansados", "viniste temprano",
        "salimos del colegio", "mi papá trabajó", "no tenía dinero", "estaba oscuro",
        "el bebé dormía", "era tarde", "mi hermana cantó", "tú no querías", "nevó en la sierra",
        "ella estaba enferma", "no hubo clases", "el perro ladró", "me dolía la cabeza",
        "mis amigos vinieron", "el bus llegó tarde", "todos estaban allí", "yo cociné",
    ],
    "futuro": [
        "lloverá mucho", "hará frío", "hará calor", "llegaré tarde", "terminaré la tarea",
        "mi mamá cocinará", "los niños jugarán en el patio", "tendré tiempo", "estudiarás mucho",
        "el profesor explicará la lección", "habrá sol", "estaremos cansados", "vendrás temprano",
        "saldremos del colegio", "mi papá trabajará", "no tendré dinero", "estará oscuro",
        "será tarde", "mi hermana cantará", "nevará en la sierra", "no habrá clases",
        "mis amigos vendrán", "el bus llegará tarde", "todos estarán aquí",
    ],
}

MAIN_CLAUSES = {
    "presente": [
        "los niños juegan en el patio", "mi mamá prepara el café", "vamos al parque",
        "yo me quedo en casa", "el perro duerme en el sofá", "mi papá está feliz",
        "todos comen juntos", "salgo a caminar", "no podemos salir", "nos quedamos en casa",
        "mi hermano ve televisión", "yo hago la tarea", "ella lee un libro", "tomamos té",
        "mi abuela cose", "los vecinos hacen ruido", "mi mamá está tranquila", "yo estudio",
    ],
    "pasado": [
        "los niños jugaron en el patio", "mi mamá preparó el café", "fuimos al parque",
        "yo me quedé en casa", "el perro durmió en el sofá", "mi papá estaba feliz",
        "todos comieron juntos", "salí a caminar", "no pudimos salir", "nos quedamos en casa",
        "mi hermano vio televisión", "yo hice la tarea", "ella leyó un libro", "tomamos té",
        "mi abuela cosió", "los vecinos hicieron ruido", "mi mamá estaba tranquila", "yo estudié",
    ],
    "futuro": [
        "los niños jugarán en el patio", "mi mamá preparará el café", "iremos al parque",
        "yo me quedaré en casa", "el perro dormirá en el sofá", "mi papá estará feliz",
        "todos comerán juntos", "saldré a caminar", "no podremos salir", "nos quedaremos en casa",
        "mi hermano verá televisión", "yo haré la tarea", "ella leerá un libro", "tomaremos té",
        "jugaré con mis amigos", "iremos a la piscina", "sacaré buena nota", "mi mamá estará feliz",
    ],
}

# subordinante -> {tiempo de la subordinada: tiempos válidos de la principal}
SUBORDINATORS = {
    "aunque":   {"presente": ["presente", "futuro"], "pasado": ["pasado"], "futuro": ["futuro"]},
    "cuando":   {"presente": ["presente"], "pasado": ["pasado"]},
    "si":       {"presente": ["presente", "futuro"]},
    "porque":   {"presente": ["presente"], "pasado": ["pasado"], "futuro": ["futuro"]},
    "mientras": {"presente": ["presente"], "pasado": ["pasado"]},
    "como":     {"presente": ["presente"], "pasado": ["pasado"]},
    "ya que":   {"presente": ["presente"], "pasado": ["pasado"], "futuro": ["futuro"]},
    "apenas":   {"pasado": ["pasado"]},          # solo con subordinadas afirmativas
}
OPENERS = ["", "", "", "", "hoy", "ayer", "mañana", "el sábado", "en la tarde", "esta semana"]
OPENER_TENSE = {"hoy": None, "ayer": "pasado", "mañana": "futuro", "el sábado": None,
                "en la tarde": None, "esta semana": None, "": None}


def indicative_controls(n: int) -> list:
    out = []
    subs = list(SUBORDINATORS)
    while len(out) < n:
        sub = random.choice(subs)
        sub_tense = random.choice(list(SUBORDINATORS[sub]))
        main_tense = random.choice(SUBORDINATORS[sub][sub_tense])
        clause = random.choice(SUB_CLAUSES[sub_tense])
        if sub == "apenas" and clause.startswith("no "):
            continue
        main = random.choice(MAIN_CLAUSES[main_tense])
        opener = random.choice(OPENERS)
        if OPENER_TENSE.get(opener) not in (None, main_tense):
            opener = ""
        if sub == "como" or random.random() < 0.7:          # "como" causal solo abre frase
            ok = _join(opener, sub, clause + ",", main)
        else:
            ok = _join(opener, main, sub, clause)
        pair = _control(ok)
        if pair:
            out.append(pair)
    return out


# ── 1b. Subjuntivo PROSPECTIVO tras cuando/apenas/en cuanto/hasta que ────────
# Contrapeso del bloque anterior: "cuando" + acción futura exige subjuntivo
# ("cuando llegue a casa, te llamaré"). Sin esto el modelo aprende "cuando =>
# indicativo" y pasa "cuando llegue a casa, te llamo" a pretérito. Se generan
# controles (solo tildes) y correcciones indicativo -> subjuntivo.

TEMPORAL_SUBS = ["cuando", "cuando", "cuando", "apenas", "en cuanto", "hasta que"]
# (indicativo, subjuntivo) + complemento
PROSPECTIVE = [
    ("llego", "llegue", "a casa"), ("llegas", "llegues", "a casa"), ("llegamos", "lleguemos", "al colegio"),
    ("termino", "termine", "la tarea"), ("terminas", "termines", "de comer"), ("terminamos", "terminemos", "el examen"),
    ("salgo", "salga", "del colegio"), ("sales", "salgas", "de clase"), ("salimos", "salgamos", "de casa"),
    ("vengo", "venga", "de la escuela"), ("vienes", "vengas", "a mi casa"), ("venimos", "vengamos", "del parque"),
    ("tengo", "tenga", "tiempo"), ("tienes", "tengas", "un rato"), ("tenemos", "tengamos", "dinero"),
    ("puedo", "pueda", ""), ("puedes", "puedas", ""), ("podemos", "podamos", ""),
    ("hago", "haga", "la tarea"), ("haces", "hagas", "la cama"), ("hacemos", "hagamos", "la compra"),
    ("vuelvo", "vuelva", "del trabajo"), ("vuelves", "vuelvas", "de viaje"), ("volvemos", "volvamos", "a casa"),
    ("empiezo", "empiece", "las clases"), ("empiezas", "empieces", "a estudiar"),
    ("estoy", "esté", "listo"), ("estás", "estés", "mejor"), ("estamos", "estemos", "en casa"),
    ("veo", "vea", "a mi abuela"), ("ves", "veas", "la película"),
    ("es", "sea", "de noche"), ("crezco", "crezca", ""), ("crezcas", "crezcas", ""),
]
# Principales en futuro/imperativo: con ellas "cuando + indicativo" es un error
# corregible. Las de presente ("te llamo") son indistinguibles del habitual
# ("cuando llego a casa, mi mamá prepara el café"), así que solo van en controles.
PROSPECTIVE_MAIN = [
    "te llamaré", "te avisaré", "avísame", "llámame", "iré al parque", "comeremos juntos",
    "jugaré con mis amigos", "me acostaré", "haré la cena", "saldremos a pasear", "te lo contaré",
    "veremos la película", "iremos a la piscina", "podrás salir", "descansaré", "mi mamá estará feliz",
    "escríbeme", "espérame", "no te preocupes", "vendré a verte", "te lo diré", "empezaremos a comer",
]
PROSPECTIVE_MAIN_PRESENT = ["te llamo", "te aviso", "vamos a jugar", "te lo cuento", "salimos a pasear",
                            "nos vamos", "te escribo", "hacemos la tarea"]
HASTA_QUE_MAIN = ["no saldré", "espérame", "no te vayas", "no comeremos", "seguiré estudiando",
                  "no me acostaré", "no podrás salir", "esperaremos aquí"]
HASTA_QUE_MAIN_PRESENT = ["no salgo", "me quedo aquí", "no como", "sigo estudiando"]
PROSPECTIVE_OPENERS = ["", "", "", "mañana", "esta tarde", "el sábado", "hoy", "más tarde"]


def prospective_pairs(n_ctrl: int, n_err: int) -> list:
    ctrl, err = [], []
    while len(ctrl) < n_ctrl or len(err) < n_err:
        sub = random.choice(TEMPORAL_SUBS)
        ind, subj, comp = random.choice(PROSPECTIVE)
        opener = random.choice(PROSPECTIVE_OPENERS)
        present_main = random.random() < 0.35                # solo control
        if sub == "hasta que":
            main = random.choice(HASTA_QUE_MAIN_PRESENT if present_main else HASTA_QUE_MAIN)
        else:
            main = random.choice(PROSPECTIVE_MAIN_PRESENT if present_main else PROSPECTIVE_MAIN)
        if random.random() < 0.7:
            ok = _join(opener, sub, _join(subj, comp) + ",", main)
            bad = _join(opener, sub, _join(ind, comp) + ",", main)
        else:
            ok = _join(opener, main, sub, subj, comp)
            bad = _join(opener, main, sub, ind, comp)
        if len(ctrl) < n_ctrl:
            pair = _control(ok)
            if pair:
                ctrl.append(pair)
        if len(err) < n_err and ind != subj and not present_main:
            err.append(_surface(bad, ok))
    return ctrl + err


# ── 1c. Presente HABITUAL tras cuando/siempre que/cada vez que ───────────────
# Refuerzo del control más frecuente en la escritura infantil ("cuando llego a
# casa hago la tarea"): subordinada y principal en presente, ambas se respetan.
# Sin este bloque las correcciones prospectivas (1b) superaban en número a los
# controles con "cuando" y el modelo pasaba el habitual a subjuntivo.
HABITUAL_SUBS = ["cuando", "cuando", "cuando", "siempre que", "cada vez que", "apenas"]
HABITUAL_SUB_CLAUSES = [
    "llego a casa", "llego al colegio", "termino la tarea", "salgo del colegio", "me levanto",
    "tengo hambre", "hace frío", "llueve", "mi mamá cocina", "mi papá llega del trabajo",
    "vamos al parque", "estoy cansado", "mi hermana estudia", "suena el timbre", "es sábado",
    "veo a mi abuela", "tengo tiempo", "hay sol", "juego fútbol", "leo un libro",
]
HABITUAL_MAIN = [
    "hago la tarea", "saludo a mis amigos", "mi mamá prepara el café", "me lavo las manos",
    "ayudo a mi mamá", "como fruta", "me pongo el abrigo", "nos quedamos en casa",
    "todos comen juntos", "mi papá está feliz", "me pongo contento", "descanso un rato",
    "ella lee un libro", "salimos a jugar", "veo televisión", "me acuesto temprano",
    "tomo té con leche", "mi perro se pone feliz", "me duele la cabeza", "estoy tranquilo",
]
HABITUAL_OPENERS = ["", "", "", "siempre", "todos los días", "los sábados", "en la tarde", "casi siempre"]


def habitual_controls(n: int) -> list:
    out = []
    while len(out) < n:
        sub = random.choice(HABITUAL_SUBS)
        clause = random.choice(HABITUAL_SUB_CLAUSES)
        main = random.choice(HABITUAL_MAIN)
        opener = random.choice(HABITUAL_OPENERS)
        if random.random() < 0.7:
            ok = _join(opener, sub, clause + ",", main)
        else:
            ok = _join(opener, main, sub, clause)
        pair = _control(ok)
        if pair:
            out.append(pair)
    return out


# ── 2. Concordancia del predicado ────────────────────────────────────────────

# (singular, plural, género, clase): la clase elige adjetivos plausibles.
SUBJECTS_PL = [
    ("el perro", "los perros", "m", "ser"), ("la casa", "las casas", "f", "cosa"), ("el niño", "los niños", "m", "ser"),
    ("la niña", "las niñas", "f", "ser"), ("mi amigo", "mis amigos", "m", "ser"), ("el libro", "los libros", "m", "cosa"),
    ("la flor", "las flores", "f", "cosa"), ("el zapato", "los zapatos", "m", "cosa"), ("la tarea", "las tareas", "f", "cosa"),
    ("el cuarto", "los cuartos", "m", "cosa"), ("la ventana", "las ventanas", "f", "cosa"), ("mi primo", "mis primos", "m", "ser"),
    ("el plato", "los platos", "m", "cosa"), ("la mesa", "las mesas", "f", "cosa"), ("el gato", "los gatos", "m", "ser"),
    ("la calle", "las calles", "f", "cosa"), ("el cuaderno", "los cuadernos", "m", "cosa"), ("la manzana", "las manzanas", "f", "cosa"),
    ("mi hermana", "mis hermanas", "f", "ser"), ("el árbol", "los árboles", "m", "cosa"),
]
RELATIVES = ["", "", "", "que vi ayer", "que compré", "que trajo mi mamá", "del jardín",
             "de la escuela", "de mi tía", "que me regalaron", "que vimos el sábado", "de la cocina"]
# singular -> plural (mismo tiempo)
VERBS_SG_PL = {
    "está": "están", "estaba": "estaban", "estuvo": "estuvieron", "estará": "estarán",
    "es": "son", "era": "eran", "fue": "fueron", "será": "serán",
    "parece": "parecen", "parecía": "parecían", "quedó": "quedaron",
}
INTENSIFIERS = ["", "", "muy", "muy", "bastante", "tan", "demasiado"]
# (masc sg, fem sg, masc pl, fem pl, clases)
ADJECTIVES = [
    ("sucio", "sucia", "sucios", "sucias", ("ser", "cosa",)), ("limpio", "limpia", "limpios", "limpias", ("ser", "cosa",)),
    ("grande", "grande", "grandes", "grandes", ("ser", "cosa",)), ("bonito", "bonita", "bonitos", "bonitas", ("ser", "cosa",)),
    ("feliz", "feliz", "felices", "felices", ("ser",)), ("cansado", "cansada", "cansados", "cansadas", ("ser",)),
    ("listo", "lista", "listos", "listas", ("ser",)), ("roto", "rota", "rotos", "rotas", ("cosa",)),
    ("viejo", "vieja", "viejos", "viejas", ("ser", "cosa",)), ("nuevo", "nueva", "nuevos", "nuevas", ("cosa",)),
    ("caro", "cara", "caros", "caras", ("cosa",)), ("pequeño", "pequeña", "pequeños", "pequeñas", ("ser", "cosa",)),
    ("contento", "contenta", "contentos", "contentas", ("ser",)), ("tranquilo", "tranquila", "tranquilos", "tranquilas", ("ser",)),
    ("ordenado", "ordenada", "ordenados", "ordenadas", ("cosa",)), ("mojado", "mojada", "mojados", "mojadas", ("ser", "cosa",)),
    ("lleno", "llena", "llenos", "llenas", ("cosa",)), ("vacío", "vacía", "vacíos", "vacías", ("cosa",)),
    ("difícil", "difícil", "difíciles", "difíciles", ("cosa",)), ("fácil", "fácil", "fáciles", "fáciles", ("cosa",)),
    ("triste", "triste", "tristes", "tristes", ("ser",)), ("alegre", "alegre", "alegres", "alegres", ("ser",)),
    ("rico", "rica", "ricos", "ricas", ("cosa",)), ("frío", "fría", "fríos", "frías", ("cosa",)),
]
CLOSERS = ["", "", "", "hoy", "ayer", "otra vez", "todavía", "esta mañana"]


def predicate_pairs(n_err: int, n_ctrl: int) -> list:
    out = []
    while len(out) < n_err:
        sg, pl, g, cls = random.choice(SUBJECTS_PL)
        rel = random.choice(RELATIVES)
        v_sg = random.choice(list(VERBS_SG_PL))
        v_pl = VERBS_SG_PL[v_sg]
        inten = random.choice(INTENSIFIERS)
        adj = random.choice([a for a in ADJECTIVES if cls in a[4]])
        a_sg = adj[0] if g == "m" else adj[1]
        a_pl = adj[2] if g == "m" else adj[3]
        closer = random.choice(CLOSERS)
        ok = _join(pl, rel, v_pl, inten, a_pl, closer)
        mode = random.random()
        if mode < 0.6:                                   # verbo y adjetivo en singular
            err = _join(pl, rel, v_sg, inten, a_sg, closer)
        elif mode < 0.85:                                # solo el adjetivo
            err = _join(pl, rel, v_pl, inten, a_sg, closer)
        else:                                            # solo el verbo
            err = _join(pl, rel, v_sg, inten, a_pl, closer)
        out.append(_surface(err, ok))
    ctrl = []
    while len(ctrl) < n_ctrl:                            # singular legítimo: solo tildes
        sg, pl, g, cls = random.choice(SUBJECTS_PL)
        rel = random.choice(RELATIVES)
        v_sg = random.choice(list(VERBS_SG_PL))
        adj = random.choice([a for a in ADJECTIVES if cls in a[4]])
        ok = _join(sg, rel, v_sg, random.choice(INTENSIFIERS), adj[0] if g == "m" else adj[1],
                   random.choice(CLOSERS))
        pair = _control(ok)
        if pair:
            ctrl.append(pair)
    return out + ctrl


# ── 3. té/te y él/el en contexto ─────────────────────────────────────────────

TEA = [
    "el té está caliente", "el té está frío", "el té está listo", "el té está muy rico",
    "el té está en la mesa", "el té está muy dulce", "quiero un té con leche",
    "mi mamá toma té por la tarde", "el té de mi abuela está delicioso", "prefiero el té al café",
    "el té se enfrió", "ese té está amargo", "el té ya está servido",
]
TEA_FRAMES = ["", "", "no sé si", "creo que", "dice que", "me parece que", "pregunta si", "mamá dice que"]
HE = [
    "él te quiere mucho", "él te ayuda con la tarea", "él te espera afuera", "él te llamó ayer",
    "él te dijo la verdad", "él te lo explicó", "él te presta el libro", "él te cuida bien",
    "él te va a acompañar", "él te escucha siempre", "él te invitó a la fiesta", "él te conoce",
    "él te mira", "él te escribió una carta", "él te dará una sorpresa",
]
HE_FRAMES = ["", "", "creo que", "sé que", "mamá dice que", "estoy seguro de que", "parece que"]


def te_el_pairs(n: int) -> list:
    out = []
    while len(out) < n:
        if random.random() < 0.5:
            ok = _join(random.choice(TEA_FRAMES), random.choice(TEA))
        else:
            ok = _join(random.choice(HE_FRAMES), random.choice(HE))
        pair = _control(ok)
        if pair:
            out.append(pair)
    return out


def main() -> None:
    ctrl = indicative_controls(2400)
    prosp = prospective_pairs(600, 400)
    habit = habitual_controls(800)
    pred = predicate_pairs(1500, 400)
    te = te_el_pairs(400)
    rows = ctrl + prosp + habit + pred + te
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
    print(f"indicativo={len(ctrl)} prospectivo={len(prosp)} habitual={len(habit)} predicado={len(pred)} te/el={len(te)} -> {len(uniq)} unicos en {OUT_CSV}")
    for err, ok in uniq[:12]:
        print(f"  {err!r:70} -> {ok!r}")


if __name__ == "__main__":
    main()
