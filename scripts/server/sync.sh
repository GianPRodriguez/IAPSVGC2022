#!/usr/bin/env bash
# Sincroniza el servidor con la laptop (fuente principal) sin perder resultados.
#
# Los entrenamientos escriben en archivos versionados de logs/ (reportes,
# ledgers de tune...). Eso haría fallar `git pull`. Este script:
#   1) copia los archivos de logs/ modificados o nuevos a
#      ~/iapsvgc_runs/_logs_backup/<fecha>/ (no se pierde nada),
#   2) restaura logs/ al estado del commit,
#   3) hace `git pull --ff-only`.
# Si hay cambios fuera de logs/ se detiene sin tocar nada: en el servidor no se
# edita código (se edita en la laptop y se hace commit allí).
#
# Uso: bash scripts/server/sync.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

outside=$(git status --porcelain | grep -v ' logs/' | grep -v '^?? logs/' || true)
if [ -n "$outside" ]; then
  echo "Hay cambios fuera de logs/ en el servidor; revisa antes de sincronizar:"
  echo "$outside"
  exit 1
fi

changed=$(git status --porcelain -- logs | awk '{print $2}')
if [ -n "$changed" ]; then
  stamp=$(date +%Y%m%d_%H%M%S)
  dest="${IAPSVGC_RUNS_DIR:-$HOME/iapsvgc_runs}/_logs_backup/$stamp"
  mkdir -p "$dest"
  while IFS= read -r f; do
    [ -e "$f" ] || continue
    mkdir -p "$dest/$(dirname "$f")"
    cp -a "$f" "$dest/$f"
  done <<< "$changed"
  echo "Respaldo de logs/: $dest"
  git checkout -- logs
  git clean -fdq -- logs
fi

git pull --ff-only
echo "Sincronizado en $(git log --oneline -1)"
