#!/bin/bash
# Vigilante de la IA en la VM (systemd: ia-watchdog.timer lo lanza cada minuto).
# El 2026-10-01 la IA quedó colgada sin responder ni por localhost y con el
# contenedor «Up»: gunicorn (gthread) no mata hilos trabados, así que no se
# recupera solo. Cada minuto se pide una corrección real; con dos fallos
# seguidos se guarda un diagnóstico (GPU, procesos, pila de todos los hilos vía
# SIGUSR2 -> faulthandler en main.py, últimos logs) en /var/log/ia-watchdog/ y
# se reinicia el contenedor. No actúa en los 10 min tras un arranque (carga de
# BETO y T5) ni si el contenedor no está corriendo (despliegue en curso).
set -u
CONTAINER=ia-tesis-api-1
URL=http://127.0.0.1:80/interno/corregir
STATE=/var/lib/ia-watchdog
LOG=/var/log/ia-watchdog
GRACE=600
mkdir -p "$STATE" "$LOG"

[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" = "true" ] || exit 0
started=$(date -d "$(docker inspect -f '{{.State.StartedAt}}' "$CONTAINER")" +%s)
[ $(( $(date +%s) - started )) -lt "$GRACE" ] && exit 0

if curl -s -m 45 -X POST "$URL" -H 'Content-Type: application/json' \
     -d '{"originalText":"el perro corre","studentId":"watchdog"}' | grep -q correctedText; then
  rm -f "$STATE/fails"
  exit 0
fi

fails=$(( $(cat "$STATE/fails" 2>/dev/null || echo 0) + 1 ))
echo "$fails" > "$STATE/fails"
logger -t ia-watchdog "sonda fallida ($fails seguidas)"
[ "$fails" -lt 2 ] && exit 0

ts=$(date -u +%Y%m%dT%H%M%SZ)
f="$LOG/cuelgue-$ts.log"
{
  echo "== $ts: la sonda falló $fails veces seguidas"
  nvidia-smi
  docker top "$CONTAINER"
  echo "== pila de los hilos (SIGUSR2)"
  docker exec "$CONTAINER" sh -c 'kill -USR2 $(cat /proc/1/task/1/children)'
  sleep 3
  docker logs --since 20m "$CONTAINER" 2>&1 | tail -600
  echo "== reinicio"
  docker restart "$CONTAINER"
} > "$f" 2>&1
rm -f "$STATE/fails"
logger -t ia-watchdog "IA reiniciada; diagnóstico en $f"
