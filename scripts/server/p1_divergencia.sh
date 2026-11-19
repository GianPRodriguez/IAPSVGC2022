#!/bin/bash
# ¿Cuánto se aparta el sorteo de P1 del modo determinista? (19-11-2026, pedido del usuario)
# 5 semillas de v1 contra val: T = 0,5 (200 partidas por rival, mismas semillas que P1: las victorias
# deben repetirse) y T = 1,0 como contraste (100 por rival). Resumen: $OUT/divergencia_p1.md.
#   bash scripts/server/p1_divergencia.sh ~/iapsvgc_runs/p1_div ~/iapsvgc_runs/ppo_v1
set -u
OUT=${1:-$HOME/iapsvgc_runs/p1_div}
V1=${2:-$HOME/iapsvgc_runs/ppo_v1}
mkdir -p "$OUT"
source ~/miniconda3/etc/profile.d/conda.sh; conda activate kuroneko
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
export PYTHONUNBUFFERED=1 PS_RL_STRICT=1
unset RL_OBS_ABILITIES PS_JOINT_ARCHETYPE PREVIEW_FORCE_ORDER PPO_FOE_CHECKPOINTS
log() { echo "$(date '+%F %T') $*" >> "$OUT/queue.log"; }
clean_ps() { pkill -f "pokemon-showdown.*start" 2>/dev/null; sleep 5; }
clean_ps
log "inicio commit $(git rev-parse --short HEAD)"
for T in 0.5 1.0; do
  G=200; [ $T = 1.0 ] && G=100
  for s in 0 1 2 3 4; do
    PS_TEAM_SPLIT=val python -m showdown.rl.ddqn.policy_diag --agent ppo --checkpoint "$V1/ppo_s$s/final.pt" \
      --games $G --port 8429 --agent-team showdown/rl/data/ddqn_team.txt --preview-checkpoint "$V1/preview_v2.pt" \
      --sample --temperature=$T --out "$OUT/T${T}_s${s}_val.json" > "$OUT/T${T}_s${s}_val.txt" 2>&1
    log "T${T}_s$s rc=$?"; clean_ps
  done
done
python scripts/analysis/divergencia_p1.py "$OUT" > "$OUT/resumen.err" 2>&1
log "resumen rc=$?"
log "P1_DIV_DONE"
