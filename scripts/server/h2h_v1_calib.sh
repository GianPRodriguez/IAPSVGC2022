#!/bin/bash
# Calibración del enfrentamiento en espejo (19-11-2026, decisión del usuario): PPO v1 contra el PPO v1
# congelado (semillas 0 y 1, 200 partidas), para saber cuánto da v1 contra sí mismo en esa medición.
# Motivo: P2 y P4 ganaron solo 22–28 % contra v1; la regla de P3 suponía 50 % sin medirlo.
# Espera a KICKOFF_QUEUE24_DONE, deja v1_s{0,1}_h2h en $V2/diag, rehace kickoff_summary24.md y escribe
# H2H_V1_DONE en $V2/h2h_calib.log (la vigesimoquinta cola espera esa marca).
#
#   bash scripts/server/h2h_v1_calib.sh ~/iapsvgc_runs/ppo_v2 ~/iapsvgc_runs/ppo_v1
set -u
V2=${1:-$HOME/iapsvgc_runs/ppo_v2}
V1=${2:-$HOME/iapsvgc_runs/ppo_v1}
source ~/miniconda3/etc/profile.d/conda.sh; conda activate kuroneko
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
export PYTHONUNBUFFERED=1 PS_RL_STRICT=1
unset RL_OBS_ABILITIES PS_JOINT_ARCHETYPE PREVIEW_FORCE_ORDER
LOG="$V2/h2h_calib.log"
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
clean_ps() { pkill -f "pokemon-showdown.*start" 2>/dev/null; sleep 5; }
log "esperando a la vigesimocuarta cola"
until grep -q KICKOFF_QUEUE24_DONE "$V2/queue.log" 2>/dev/null; do sleep 60; done
clean_ps
log "inicio commit $(git rev-parse --short HEAD)"
V1CKS=$(ls "$V1"/ppo_s[0-4]/final.pt | paste -sd, -)
for s in 0 1; do
  PS_TEAM_SPLIT=eval PPO_FOE_CHECKPOINTS="$V1CKS" python -m showdown.rl.ddqn.policy_diag --agent ppo \
    --checkpoint "$V1/ppo_s$s/final.pt" --games 200 --port 8428 --agent-team showdown/rl/data/ddqn_team.txt \
    --preview-checkpoint "$V1/preview_v2.pt" --foes=ppo_v1 --out "$V2/diag/v1_s${s}_h2h.json" \
    > "$V2/diag/v1_s${s}_h2h.txt" 2>&1
  log "diag v1_s${s}_h2h rc=$?"; clean_ps
done
python -m showdown.rl.kickoff_summary24 "$V2" "$V1" > "$V2/summary24_calib.err" 2>&1
log "resumen24 rehecho rc=$?"
log "H2H_V1_DONE"
