#!/bin/bash
# F4 DE LA FAMILIA QD 2.0 (protocolo §4.14, exploración).
#   4a. Preview QD-PGA v1.7 con descriptores por roles (equipo fijo), piloto = control (PPO v2 s1 + 1M),
#       64 iteraciones externas. El control juega con ese preview los mismos enfrentamientos eval de F1
#       (bloque 40001, 4000 por heurístico) y se parea con el control de F1 (preview v2.1b).
#   4b. El archivo como pool de rivales: PPO v2 s1 + 1M pasos con el mosaico 'pool' (20 % contra las 5
#       mejores élites de K en espejo). Juega los mismos enfrentamientos eval de F1 y se parea con el control.
# Espera a QD_F3_DONE. Marca: QD_F4_DONE. Resúmenes: $OUT/preview_roles.txt, $OUT/pool_vs_control.txt.
#   bash scripts/server/qd_f4.sh ~/iapsvgc_runs/qd_f4
set -u
OUT=${1:-$HOME/iapsvgc_runs/qd_f4}; MARK=QD_F4_DONE
mkdir -p "$OUT"
REPO=$(cd "$(dirname "$0")/../.." && pwd); cd "$REPO"
source scripts/server/qd_f_common.sh
wait_for "$RUNS/qd_f3" QD_F3_DONE
need "$INIT" "$CTRL" "$PV" "$F1/deploy/deploy.json" "$F1/confirm/results.json" "$QJK/k2/archive.pt"
log "inicio commit $(git rev-parse --short HEAD)"; clean_ps
OMP_NUM_THREADS=4 python -m pytest -q -p no:cacheprovider tests/test_qd_f234.py tests/test_qd_confirm_f1.py \
  tests/test_team_preview_qd_pga.py tests/test_frozen_preview.py tests/test_qd_ppo2.py > "$OUT/tests.txt" 2>&1
rc=$?; log "tests rc=$rc ($(tail -1 "$OUT/tests.txt"))"; [ $rc = 0 ] || { log "ERROR: tests fallan"; log "$MARK"; exit 1; }
CONF="--cfg-checkpoint $CTRL --num-envs 32 --envs-per-server 4 --agent-team $TEAM --require-preview --per-foe 4000 --seed-block 40001"

# ---------------------------------------------------------------- 4a. preview por roles
PVQD="$OUT/preview/best_elite.pt"
if [ ! -f "$PVQD" ]; then
  PREVIEW_PILOT_CHECKPOINT="$CTRL" PREVIEW_FOE_PILOT=greedy_joint python -m showdown.rl.team_preview_qd_pga.train \
    --seed 0 --device cuda --backend live --num-workers 28 --envs-per-server 4 --start-ps --pilot ppo \
    --own-team $TEAM --obs-abilities 1 --bd roles --n-outer 64 --checkpoint-dir "$OUT/preview" > "$OUT/preview.txt" 2>&1
  log "preview rc=$?"; clean_ps; disk
fi
if [ -f "$PVQD" ]; then
  log "preview: $(grep -E 'calibrated|outer=63 ' "$OUT/preview.txt" | tr '\n' ' ')"
  if [ ! -f "$OUT/confirm_pvqd/results.json" ]; then
    PREVIEW_KIND=team_preview_qd_pga python -m showdown.rl.qd_ppo.confirm_f1 confirm $CONF --preview-checkpoint "$PVQD" \
      --out "$OUT/confirm_pvqd" --agent control_pvqd="$CTRL" > "$OUT/confirm_pvqd.txt" 2>&1
    log "confirm preview rc=$?"; clean_ps
  fi
  python -m showdown.rl.qd_ppo.confirm_f1 stats --results "$F1/confirm/results.json" --results "$OUT/confirm_pvqd/results.json" \
    --control control --ni control_pvqd --sup "" --init init --out "$OUT/preview_roles" \
    --title "F4a: preview QD-PGA por roles frente a preview v2.1b, mismo combate (protocolo §4.14)" > /dev/null 2>&1
  log "stats preview rc=$?"
else
  log "ERROR: el preview por roles no produjo best_elite.pt; se sigue con 4b"
fi

# ---------------------------------------------------------------- 4b. archivo como pool de rivales
export PREVIEW_KIND=team_preview_ppo
if [ ! -f "$OUT/pool/final.pt" ]; then
  FOES=$(python -m showdown.rl.qd_ppo.confirm_f1 export --archive "$QJK/k2/archive.pt" --deploy "$F1/deploy/deploy.json" \
    --agent K --top 5 --cfg-checkpoint "$CTRL" --out "$OUT/pool_foes" 2>>"$OUT/pool_export.err" | tail -1)
  log "rivales del pool: $FOES"
  [ -n "$FOES" ] || { log "ERROR: no se exportaron élites de K"; log "$MARK"; exit 1; }
  PPO_FOE_CHECKPOINTS="$FOES" PPO_FOE_PREVIEW="$PV" python -m showdown.rl.qd_ppo.train2 --seed 0 --device cuda \
    --start-ps --num-envs 32 --envs-per-server 4 --init-checkpoint "$INIT" --preview-checkpoint "$PV" --require-preview \
    --agent-team $TEAM --control --mosaic pool --checkpoint-dir "$OUT/pool" > "$OUT/pool.txt" 2>&1
  log "pool rc=$?"; clean_ps; disk
fi
need "$OUT/pool/final.pt"
if [ ! -f "$OUT/confirm_pool/results.json" ]; then
  python -m showdown.rl.qd_ppo.confirm_f1 confirm $CONF --preview-checkpoint "$PV" --out "$OUT/confirm_pool" \
    --agent pool="$OUT/pool/final.pt" > "$OUT/confirm_pool.txt" 2>&1
  log "confirm pool rc=$?"; clean_ps
fi
python -m showdown.rl.qd_ppo.confirm_f1 stats --results "$F1/confirm/results.json" --results "$OUT/confirm_pool/results.json" \
  --control control --ni pool --sup "" --init init --out "$OUT/pool_vs_control" \
  --title "F4b: PPO entrenado con el archivo de K como pool de rivales frente al control (protocolo §4.14)" > /dev/null 2>&1
log "stats pool rc=$?"
log "$MARK"
