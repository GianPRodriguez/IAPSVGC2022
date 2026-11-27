#!/bin/bash
# F1 DE LA FAMILIA QD 2.0 (protocolo §4.11, decisión del usuario del 26-11-2026): medir bien antes de
# cualquier mejora.
#   0. Tests.
#   1a. Despliegue sin maldición del ganador: las 10 mejores élites de los archivos de I, J y K juegan los
#       mismos 1000 enfrentamientos val (3 heurísticos por igual); se despliega la de más victorias.
#   1b. Confirmación: I, J, K desplegados, el control y PPO v2 s1 (arranque, descriptivo) juegan los mismos
#       4000 enfrentamientos eval por heurístico (12 000 por agente, bloque de semillas 40001).
#   1c. Estadística pre-registrada: N_I, N_J, N_K (no inferioridad frente al control, margen −1,5 pts) y
#       S_J (J > control contra expectimax), Holm m = 4.
# Preview v2.1b de PPO v2 (PREVIEW_KIND=team_preview_ppo), equipo fijo, observación P5 del checkpoint.
# Resumen: $OUT/confirm/confirm_stats.txt. Marca: QD_F1_DONE.
#
#   bash scripts/server/qd_f1.sh ~/iapsvgc_runs/qd_f1
set -u
OUT=${1:-$HOME/iapsvgc_runs/qd_f1}
QI=${QI:-$HOME/iapsvgc_runs/qd_i2}
QJK=${QJK:-$HOME/iapsvgc_runs/qd_jk2}
INIT=${INIT:-$HOME/iapsvgc_runs/kickoff_ppo_v2/ppo_s1/final.pt}
PV=${PV:-$HOME/iapsvgc_runs/preview_v21b/preview_s0/best.pt}
mkdir -p "$OUT"
source ~/miniconda3/etc/profile.d/conda.sh; conda activate kuroneko
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
export PYTHONUNBUFFERED=1 PS_RL_STRICT=1
unset RL_MASK_ALLY_DAMAGE RL_SYM_AUG RL_HOLDOUT_VAL RL_TRAIN_SUBSET RL_FOE_SYNTH PREVIEW_FORCE_ORDER PS_JOINT_ARCHETYPE \
  PPO_FOE_CHECKPOINTS PPO_FOE_PREVIEW RL_OBS_ABILITIES RL_OBS_CALC PREVIEW_SAMPLE PREVIEW_TEMPERATURE PS_TEAM_SPLIT \
  PREVIEW_PILOT_CHECKPOINT PREVIEW_FOE_PILOT PS_PREVIEW_OWN_TEAM PREVIEW_CHECKPOINT PREVIEW_REQUIRED RL_BD2 PREVIEW_KIND
export PREVIEW_KIND=team_preview_ppo
log() { echo "$(date '+%F %T') $*" >> "$OUT/queue.log"; }
clean_ps() { pkill -f "[p]okemon-showdown.*start" 2>/dev/null; sleep 5; }
TEAM=showdown/rl/data/ddqn_team.txt
CTRL="$QI/control/final.pt"
for f in "$QI/qd/archive.pt" "$QJK/j2/archive.pt" "$QJK/k2/archive.pt" "$CTRL" "$INIT" "$PV"; do
  [ -f "$f" ] || { log "ERROR: falta $f"; log "QD_F1_DONE"; exit 1; }
done
log "inicio commit $(git rev-parse --short HEAD)"
clean_ps

OMP_NUM_THREADS=4 python -m pytest -q -p no:cacheprovider tests/test_qd_confirm_f1.py tests/test_qd_ppo2.py \
  tests/test_frozen_preview.py > "$OUT/tests.txt" 2>&1
rc=$?; log "tests rc=$rc ($(tail -1 "$OUT/tests.txt"))"
[ $rc = 0 ] || { log "ERROR: tests fallan; se detiene"; log "QD_F1_DONE"; exit 1; }

COMMON="--cfg-checkpoint $CTRL --num-envs 32 --envs-per-server 4 --agent-team $TEAM --preview-checkpoint $PV --require-preview"
if [ ! -f "$OUT/deploy/deploy.json" ]; then
  python -m showdown.rl.qd_ppo.confirm_f1 deploy $COMMON --out "$OUT/deploy" --top 10 --games 1000 \
    --archive I="$QI/qd/archive.pt" --archive J="$QJK/j2/archive.pt" --archive K="$QJK/k2/archive.pt" \
    > "$OUT/deploy.txt" 2>&1
  log "deploy rc=$?"; clean_ps
fi
for a in I J K; do [ -f "$OUT/deploy/deploy_$a.pt" ] || { log "ERROR: falta deploy_$a.pt"; log "QD_F1_DONE"; exit 1; }; done
log "desplegados: $(grep '^\[f1\] desplegado' "$OUT/deploy.txt" | sed 's/\[f1\] desplegado //' | tr '\n' ';')"
log "  disco: $(df -h ~ | tail -1 | awk '{print $5}')"

if [ ! -f "$OUT/confirm/results.json" ]; then
  python -m showdown.rl.qd_ppo.confirm_f1 confirm $COMMON --out "$OUT/confirm" --per-foe 4000 \
    --agent I="$OUT/deploy/deploy_I.pt" --agent J="$OUT/deploy/deploy_J.pt" --agent K="$OUT/deploy/deploy_K.pt" \
    --agent control="$CTRL" --agent init="$INIT" > "$OUT/confirm.txt" 2>&1
  log "confirm rc=$?"; clean_ps
fi
python -m showdown.rl.qd_ppo.confirm_f1 stats --results "$OUT/confirm/results.json" > "$OUT/stats.txt" 2>&1
log "stats rc=$?"
log "resumen en $OUT/confirm/confirm_stats.txt"
log "QD_F1_DONE"
