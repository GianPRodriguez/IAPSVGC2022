#!/bin/bash
# Decimonovena cola (17-11-2026), aprobada por el usuario: ¿DDQN y PPO juegan contra el plan del
# rival (Trick Room, Tailwind, Fake Out, redirección)? ¿Los rivales heurísticos ejecutan esos planes?
# (memorizacion_equipos.md §11–§12). policy_diag --context con el **equipo del agente al azar en
# cada partida** (--agent-team ''), por decisión del usuario solo para este análisis; rivales y
# equipos propios del grupo eval, 200 partidas por rival.
#
#   DDQN: REG elegida (weight decay 0,03 + dropout 0,1; decimoquinta cola), final.pt, semillas 0 y 1.
#   PPO:  con simetría (duodécima cola), final.pt, semillas 0 y 1.
#
# Espera a KICKOFF_QUEUE18_DONE. Resumen: $OUT/kickoff_summary19.md.
#
#   bash scripts/server/kickoff_queue19.sh ~/iapsvgc_runs/kickoff19 ~/iapsvgc_runs ~/iapsvgc_runs/kickoff18
set -u
OUT=${1:-$HOME/iapsvgc_runs/kickoff19}
R=${2:-$HOME/iapsvgc_runs}
Q18=${3:-$HOME/iapsvgc_runs/kickoff18}
mkdir -p "$OUT"/diag
source ~/miniconda3/etc/profile.d/conda.sh; conda activate kuroneko
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
export PYTHONUNBUFFERED=1 PS_RL_STRICT=1
unset RL_MASK_ALLY_DAMAGE RL_SYM_AUG RL_HOLDOUT_VAL RL_TRAIN_SUBSET RL_FOE_SYNTH
log() { echo "$(date '+%F %T') $*" >> "$OUT/queue.log"; }
clean_ps() { pkill -f "pokemon-showdown.*start" 2>/dev/null; sleep 5; }
log "esperando a la decimoctava cola ($Q18)"
until grep -q KICKOFF_QUEUE18_DONE "$Q18/queue.log" 2>/dev/null; do sleep 60; done
clean_ps
log "inicio commit $(git rev-parse --short HEAD)"
PV=$(sed -n 's/.*preview=\([^ ]*\).*/\1/p' "$R/kickoff3/mini_choice.txt" 2>/dev/null)
[ -f "${PV:-}" ] || PV=""
log "preview: ${PV:-ninguno}"

diag() {  # agente nombre checkpoint
  local agent=$1 name=$2 ck=$3
  [ -f "$ck" ] || { log "diag $name: falta $ck"; return; }
  PS_TEAM_SPLIT=eval python -m showdown.rl.ddqn.policy_diag --agent $agent --checkpoint "$ck" --context \
    --games 200 --port 8414 --agent-team '' ${PV:+--preview-checkpoint "$PV"} \
    --out "$OUT/diag/$name.json" > "$OUT/diag/$name.txt" 2>&1
  log "diag $name rc=$?"; clean_ps
}

diag ddqn ddqn_s0 "$R/kickoff15/R_wd03_s0/final.pt"
diag ppo ppo_s0 "$R/kickoff12/PPO_sym_s0/final.pt"
diag ddqn ddqn_s1 "$R/kickoff15/R_wd03_s1/final.pt"
diag ppo ppo_s1 "$R/kickoff12/PPO_sym_s1/final.pt"

python -m showdown.rl.kickoff_summary19 "$OUT" > "$OUT/summary19.err" 2>&1
log "resumen rc=$?"
log "KICKOFF_QUEUE19_DONE"
