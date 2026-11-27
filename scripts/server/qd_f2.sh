#!/bin/bash
# F2 DE LA FAMILIA QD 2.0 (protocolo §4.12, exploración): selector de élite por rival.
#   Repertorio = 10 élites con más victorias en el despliegue de F1 (I, J, K). Juegan 1000 enfrentamientos
#   fit por heurístico (datos del selector); en eval (bloque 60001, 1000 por heurístico) juegan las 10, el
#   control y PPO v2 s1. El selector (logística por élite sobre el equipo rival, λ por VC agrupada) elige.
# Espera a QD_F1_DONE. Marca: QD_F2_DONE. Resumen: $OUT/selector_analysis.txt.
#   bash scripts/server/qd_f2.sh ~/iapsvgc_runs/qd_f2
set -u
OUT=${1:-$HOME/iapsvgc_runs/qd_f2}; MARK=QD_F2_DONE
mkdir -p "$OUT"
REPO=$(cd "$(dirname "$0")/../.." && pwd); cd "$REPO"
source scripts/server/qd_f_common.sh
export PREVIEW_KIND=team_preview_ppo
wait_for "$F1" QD_F1_DONE
need "$F1/deploy/deploy.json" "$QI/qd/archive.pt" "$QJK/j2/archive.pt" "$QJK/k2/archive.pt" "$CTRL" "$INIT" "$PV"
log "inicio commit $(git rev-parse --short HEAD)"; clean_ps
OMP_NUM_THREADS=4 python -m pytest -q -p no:cacheprovider tests/test_qd_f234.py tests/test_qd_confirm_f1.py > "$OUT/tests.txt" 2>&1
rc=$?; log "tests rc=$rc ($(tail -1 "$OUT/tests.txt"))"; [ $rc = 0 ] || { log "ERROR: tests fallan"; log "$MARK"; exit 1; }
if [ ! -f "$OUT/results.json" ]; then
  python -m showdown.rl.qd_ppo.selector_f2 play --deploy "$F1/deploy/deploy.json" \
    --archive I="$QI/qd/archive.pt" --archive J="$QJK/j2/archive.pt" --archive K="$QJK/k2/archive.pt" \
    --control "$CTRL" --init "$INIT" --cfg-checkpoint "$CTRL" --out "$OUT" --r 10 \
    --per-foe-fit 1000 --per-foe-eval 1000 --num-envs 32 --envs-per-server 4 --agent-team $TEAM \
    --preview-checkpoint "$PV" --require-preview > "$OUT/play.txt" 2>&1
  log "play rc=$?"; clean_ps
fi
python -m showdown.rl.qd_ppo.selector_f2 analyze --results "$OUT/results.json" > "$OUT/analyze.txt" 2>&1
log "analyze rc=$?"; disk
log "resumen en $OUT/selector_analysis.txt"
log "$MARK"
