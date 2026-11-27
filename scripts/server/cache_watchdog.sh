#!/bin/bash
# Vigía de la caché de la calculadora (regla del usuario: "no se debe llenar por parte del entrenamiento más de
# 250 GB ... si es que sucede otra vez, eliminar el archivo de caché y empezar otra vez").
#
# Cada $EVERY segundos mide $CACHE (más sus -wal/-shm). Si supera $LIMIT_GB, detiene la cola (toda la sesión del
# lanzador: cola, entrenamientos, evaluaciones y sus servidores), borra la caché y relanza el lanzador, que es
# reanudable (salta lo terminado; el entrenamiento en curso empieza de nuevo). Termina cuando la última línea de
# $QLOG es la marca $MARK.
#
#   setsid nohup bash scripts/server/cache_watchdog.sh <caché> <lanzador.sh> <queue.log> <MARCA> [LIMIT_GB] &
set -u
CACHE=$1; LAUNCHER=$2; QLOG=$3; MARK=$4; LIMIT_GB=${5:-250}; EVERY=${EVERY:-300}
log() { echo "$(date '+%F %T') [vigía caché] $*" >> "$QLOG"; }
size_gb() { du -cb "$CACHE" "$CACHE-wal" "$CACHE-shm" 2>/dev/null | tail -1 | awk '{printf "%.0f", $1 / 1e9}'; }
log "activo: $CACHE, límite $LIMIT_GB GB, cada $EVERY s"
while :; do
  # la marca debe ser la última línea (en el log puede haber marcas de intentos anteriores)
  tail -n 1 "$QLOG" 2>/dev/null | grep -q " $MARK\$" && { log "la cola terminó; fin del vigía"; exit 0; }
  gb=$(size_gb)
  if [ "${gb:-0}" -gt "$LIMIT_GB" ]; then
    log "caché en $gb GB > $LIMIT_GB GB: detengo la cola, borro la caché y la relanzo"
    pid=$(pgrep -f "bash $LAUNCHER" | head -1)
    if [ -n "$pid" ]; then
      sid=$(ps -o sid= -p "$pid" | tr -d ' ')
      pkill -TERM -s "$sid"; sleep "${KILL_WAIT:-20}"; pkill -KILL -s "$sid" 2>/dev/null
    fi
    if [ -z "${WATCHDOG_TEST:-}" ]; then  # WATCHDOG_TEST=1: prueba sin la limpieza global de servidores
      pkill -f "[p]okemon-showdown.*start" 2>/dev/null; pkill -f "calc_serve[r].js" 2>/dev/null; sleep 5
    fi
    rm -f "$CACHE" "$CACHE-wal" "$CACHE-shm"
    log "caché borrada (quedan $(size_gb) GB); relanzo $LAUNCHER"
    (setsid nohup bash "$LAUNCHER" </dev/null >/dev/null 2>&1 &)
    sleep 120
  fi
  sleep "$EVERY"
done
