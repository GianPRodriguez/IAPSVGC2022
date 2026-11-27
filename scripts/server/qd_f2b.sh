#!/bin/bash
# F2b DE LA FAMILIA QD 2.0 (protocolo §4.15, exploración, decisión del usuario del 26-11-2026): el selector de
# élite por rival entrenado con equipos NO vistos (val, 30 equipos). En F2 los datos venían de fit (equipos
# de entrenamiento, memorizados: todas las élites ganaban 95–97 %), así que no había señal.
#   Mismo repertorio de F2 (10 élites). 2000 enfrentamientos val por heurístico (bloque 50002); la prueba
#   es la parte eval de F2 (mismas partidas, mismo control), así la comparación con F2 es directa.
# Espera a QD_F4_DONE (un método a la vez). Marca: QD_F2B_DONE. Resumen: $OUT/selector_analysis.txt.
#   bash scripts/server/qd_f2b.sh ~/iapsvgc_runs/qd_f2b
set -u
OUT=${1:-$HOME/iapsvgc_runs/qd_f2b}; MARK=QD_F2B_DONE
mkdir -p "$OUT"
REPO=$(cd "$(dirname "$0")/../.." && pwd); cd "$REPO"
source scripts/server/qd_f_common.sh
export PREVIEW_KIND=team_preview_ppo
wait_for "$RUNS/qd_f4" QD_F4_DONE
F2=$RUNS/qd_f2
need "$F1/deploy/deploy.json" "$F2/results.json" "$QI/qd/archive.pt" "$QJK/j2/archive.pt" "$QJK/k2/archive.pt" "$CTRL" "$PV"
log "inicio commit $(git rev-parse --short HEAD)"; clean_ps
OMP_NUM_THREADS=4 python -m pytest -q -p no:cacheprovider tests/test_qd_f234.py > "$OUT/tests.txt" 2>&1
rc=$?; log "tests rc=$rc ($(tail -1 "$OUT/tests.txt"))"; [ $rc = 0 ] || { log "ERROR: tests fallan"; log "$MARK"; exit 1; }
if [ ! -f "$OUT/results.json" ]; then
  python -m showdown.rl.qd_ppo.selector_f2 play --deploy "$F1/deploy/deploy.json" \
    --archive I="$QI/qd/archive.pt" --archive J="$QJK/j2/archive.pt" --archive K="$QJK/k2/archive.pt" \
    --control "$CTRL" --init "$INIT" --cfg-checkpoint "$CTRL" --out "$OUT" --r 10 \
    --fit-split val --fit-seed-block 50002 --per-foe-fit 2000 --eval-from "$F2/results.json" \
    --num-envs 32 --envs-per-server 4 --agent-team $TEAM --preview-checkpoint "$PV" --require-preview \
    > "$OUT/play.txt" 2>&1
  log "play rc=$?"; clean_ps
fi
python -m showdown.rl.qd_ppo.selector_f2 analyze --results "$OUT/results.json" > "$OUT/analyze.txt" 2>&1
log "analyze rc=$?"; disk
log "resumen en $OUT/selector_analysis.txt"
log "$MARK"
