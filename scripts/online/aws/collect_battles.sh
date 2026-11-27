#!/usr/bin/env bash
# Trae a la laptop las batallas jugadas en la web de retos BO3 (EC2) y regenera el resumen.
#
#   bash scripts/online/aws/collect_battles.sh            # desde la raíz del repo (Git Bash)
#
# Copia ~/online_web/ de la EC2 (fuera del repo allí) (replays HTML + bo3_<usuario>_<fecha>.json por serie) sobre la carpeta
# local: no borra nada local y es incremental en la práctica (sobrescribe con lo mismo). Se conecta por el
# nombre DuckDNS, así que funciona aunque la IP cambie en cada sesión del Lab (la EC2 debe estar encendida).
set -euo pipefail

HOST="${HOST:-teamofabrexuxa.duckdns.org}"
KEY="${KEY:-$HOME/.ssh/labsuser.pem}"
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"

mkdir -p "$REPO/logs/online_web"
before=$(find "$REPO/logs/online_web" -type f | wc -l)
ssh -i "$KEY" -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new "ubuntu@$HOST" \
  "tar -C ~ -cf - online_web" | tar -C "$REPO/logs" -xf -
after=$(find "$REPO/logs/online_web" -type f | wc -l)
echo "== $((after - before)) archivos nuevos (total $after) en logs/online_web/"

python "$REPO/scripts/online/summarize_web.py"
