#!/bin/bash
# Piloto de velocidad de P5 parte 2a (19-11-2026): ¿cuánto frena la calculadora real en la observación?
# Espera a P1_DIV_DONE (servidor libre). Reinicia el servidor TCP de la calculadora con el código nuevo
# (auras de campo), y corre PPO 30k pasos, 32 entornos, observación nueva: con la calculadora
# (RL_OBS_CALC=1) y con el respaldo en Python (RL_OBS_CALC=0). Resumen: $OUT/resumen.txt.
#   bash scripts/server/p5_speed_pilot.sh ~/iapsvgc_runs/p5_speed ~/iapsvgc_runs/ppo_v1
set -u
OUT=${1:-$HOME/iapsvgc_runs/p5_speed}
V1=${2:-$HOME/iapsvgc_runs/ppo_v1}
mkdir -p "$OUT"
source ~/miniconda3/etc/profile.d/conda.sh; conda activate kuroneko
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
export PYTHONUNBUFFERED=1 PS_RL_STRICT=1
unset RL_OBS_ABILITIES PS_JOINT_ARCHETYPE PREVIEW_FORCE_ORDER PPO_FOE_CHECKPOINTS
log() { echo "$(date '+%F %T') $*" >> "$OUT/queue.log"; }
clean_ps() { pkill -f "pokemon-showdown.*start" 2>/dev/null; sleep 5; }
until grep -q P1_DIV_DONE "$HOME/iapsvgc_runs/p1_div/queue.log" 2>/dev/null; do sleep 30; done
clean_ps
pkill -f "calc_serve[r].js" 2>/dev/null; sleep 2  # el próximo uso levanta el servidor con el código nuevo
log "inicio commit $(git rev-parse --short HEAD)"
for mode in 1 0; do
  RL_OBS_CALC=$mode timeout 1800 python -m showdown.rl.ppo.train --seed 0 --device cuda --start-ps --expand-omega \
    --num-envs 32 --envs-per-server 4 --log-every 5000 --log-every-sec 60 \
    --phases random:6000,greedy_solo:8000,greedy_joint:8000,expectimax_bayesian:8000 \
    --eval-every 10000000 --eval-games 4 --suite-games 0 --agent-team showdown/rl/data/ddqn_team.txt \
    --preview-checkpoint "$V1/preview_v2.pt" --obs-abilities 1 --checkpoint-dir "$OUT/calc$mode" > "$OUT/calc$mode.txt" 2>&1
  log "calc$mode rc=$?"; clean_ps
done
{
  for mode in 1 0; do
    echo "== RL_OBS_CALC=$mode"
    grep -o "DONE[^|]*| steps [^|]*| [0-9.]* step/s | ETA [^|]*| elapsed [^ ]*" "$OUT/calc$mode.txt" | tail -1
    grep "calculadora real en la observación\|observación: corrección" "$OUT/calc$mode.txt" | tail -2
  done
} > "$OUT/resumen.txt"
log "P5_SPEED_DONE"
