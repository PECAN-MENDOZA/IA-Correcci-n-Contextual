"""
test_modelo.py
Prueba el pipeline en dos fases bien diferenciadas:

  FASE 1 — Primera corrección (sin feedback previo)
  FASE 2 — Retest inmediato tras feedback (prueba de memoria simbólica)

El fine-tuning con T5 es un proceso separado que ocurre en segundo plano
y requiere acumular >= 10 pares por usuario. Este script NO lo mide porque
tarda minutos y necesita muchas más rondas de feedback para activarse.
"""
import requests
import json

URL_CORREGIR = "http://localhost:8080/interno/corregir"
URL_FEEDBACK  = "http://localhost:8080/interno/feedback"
STUDENT_ID    = "test_001"

VERDE  = "\033[92m"
ROJO   = "\033[91m"
AMARIL = "\033[93m"
RESET  = "\033[0m"
NEGRITA = "\033[1m"

casos = [
    {
        "original": "mi mama dijo que coma despues tarea pero yo ya estaba tarde para dormir",
        "esperado": "mi mamá dijo que comiera después de la tarea, pero ya se me hacía tarde para dormir",
    },
    {
        "original": "yo queria leer el cuento de animales que mordio la niña en la biblioteca",
        "esperado": "yo quería leer el cuento de animales que mordió a la niña en la biblioteca",
    },
    {
        "original": "la maestra me puso bien porque estaba mal la palabra pero igual saque rojo",
        "esperado": "la maestra me corrigió bien porque estaba mal la palabra pero igual saqué rojo",
    },
    {
        "original": "el sabado que viene voy a ir al medico porque me duele el estomago",
        "esperado": "el sábado que viene voy a ir al médico porque me duele el estómago",
    },
    {
        "original": "los chicos que estaban ahí fue muy groseros conmigo",
        "esperado": "los chicos que estaban ahí fueron muy groseros conmigo",
    },
    {
        "original": "habían muchas personas esperando el bus desde temprano",
        "esperado": "había muchas personas esperando el bus desde temprano",
    },
    {
        "original": "voy a ver si puedo ir a ver la obra de teatro",
        "esperado": "voy a ver si puedo ir a ver la obra de teatro",
    },
    {
        "original": "el tubo que tuvo el accidente esta en el hospital",
        "esperado": "el tubo que tuvo el accidente está en el hospital",
    },
    {
        "original": "mi mama dise que tengo que estugiar mas sino no voy a pasar el año",
        "esperado": "mi mamá dice que tengo que estudiar más si no, no voy a pasar el año",
    },
    {
        "original": "ayer fui ala tienda y compre pan con mantequiya",
        "esperado": "ayer fui a la tienda y compré pan con mantequilla",
    },
]


def corregir(texto: str) -> dict | None:
    try:
        r = requests.post(URL_CORREGIR, json={"originalText": texto, "studentId": STUDENT_ID}, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  {ROJO}[ERROR /corregir]{RESET} {e}")
        return None


def enviar_feedback(original: str, correccion: str) -> bool:
    try:
        r = requests.post(
            URL_FEEDBACK,
            json={
                "studentId": STUDENT_ID,
                "originalText": original,
                "selectedSuggestion": correccion,
                "accepted": True,
            },
            timeout=10,
        )
        return r.status_code == 204
    except Exception as e:
        print(f"  {ROJO}[ERROR /feedback]{RESET} {e}")
        return False


def acierto(resultado: dict, esperado: str) -> bool:
    """True si el texto esperado aparece en correctedText o suggestions."""
    if not resultado:
        return False
    if resultado.get("correctedText") == esperado:
        return True
    if esperado in resultado.get("suggestions", []):
        return True
    return False


# ─────────────────────────────────────────────
# EJECUCIÓN
# ─────────────────────────────────────────────
print(f"\n{NEGRITA}{'═'*60}")
print("  TEST DE CORRECCIÓN + MEMORIA SIMBÓLICA")
print(f"{'═'*60}{RESET}\n")

stats = {"acierto_1": 0, "acierto_memoria": 0, "total": len(casos)}

for i, caso in enumerate(casos, 1):
    original = caso["original"]
    esperado = caso["esperado"]

    print(f"{NEGRITA}[CASO {i}/{len(casos)}]{RESET}")
    print(f"  ORIGINAL : {original}")
    print(f"  ESPERADO : {esperado}")

    # ── FASE 1: Primera corrección ──────────────────────────
    res1 = corregir(original)
    if res1:
        corr1 = res1.get("correctedText", "")
        sugs1 = res1.get("suggestions", [])
        print(f"  {NEGRITA}1er intento{RESET}: {corr1}")
        if sugs1:
            print(f"  Sugerencias: {sugs1}")

        if acierto(res1, esperado):
            stats["acierto_1"] += 1
            stats["acierto_memoria"] += 1   # ya acertó, cuenta para ambos
            print(f"  {VERDE}✓ Correcto a la primera{RESET}")
        else:
            print(f"  {AMARIL}⚠ Incorrecto — enviando feedback...{RESET}")

            # ── FASE 2: Feedback + retest inmediato ────────────────
            ok = enviar_feedback(original, esperado)
            if ok:
                print(f"  {VERDE}✓ Feedback registrado en memoria{RESET}")

                # El retest es INMEDIATO: remember() es síncrono,
                # no hay que esperar ningún entrenamiento.
                res2 = corregir(original)
                if res2:
                    corr2 = res2.get("correctedText", "")
                    print(f"  {NEGRITA}2do intento{RESET}: {corr2}")

                    if acierto(res2, esperado):
                        stats["acierto_memoria"] += 1
                        print(f"  {VERDE}✓ Memoria simbólica funcionó{RESET}")
                    else:
                        print(f"  {ROJO}✗ La memoria no devolvió el resultado esperado{RESET}")
            else:
                print(f"  {ROJO}✗ El feedback no fue aceptado por la API{RESET}")
    else:
        print(f"  {ROJO}✗ Sin respuesta de la API{RESET}")

    print(f"  {'─'*56}")

# ── Resumen ─────────────────────────────────────────────────
print(f"\n{NEGRITA}{'═'*60}")
print("  RESUMEN")
print(f"{'═'*60}{RESET}")
print(f"  Casos totales           : {stats['total']}")
print(f"  Aciertos a la 1ra       : {stats['acierto_1']} / {stats['total']}")
print(f"  Aciertos tras memoria   : {stats['acierto_memoria']} / {stats['total']}")
tasa = stats["acierto_memoria"] / stats["total"] * 100
color = VERDE if tasa >= 80 else AMARIL if tasa >= 50 else ROJO
print(f"  Tasa final              : {color}{tasa:.0f}%{RESET}")
print(f"{NEGRITA}{'═'*60}{RESET}\n")

print("NOTA: El fine-tuning con T5 ocurre en segundo plano y requiere")
print(f"      acumular >= 10 pares por usuario (actualmente {stats['total']} en este test).")
print("      Para verlo en acción, ejecuta el script varias veces.")