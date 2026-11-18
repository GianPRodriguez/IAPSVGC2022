#!/bin/bash
# KICK-OFF OFICIAL DE PPO (vigesimotercera cola, 18-11-2026; decisión del usuario: primero solo PPO).
# Configuración final, toda por defecto en el código (ddqn_kickoff_checklist.md, "Configuración
# final de los kick-offs"): máscara de daño al aliado, simetría activa (ppo_spec 1.14), entropía
# 0,01, desde cero, validación reservada (fit / val), preview v2, equipo fijo, sin sintéticos.
# 1M pasos (random 150k → greedy_solo 250k → greedy_joint 300k → expectimax 300k), semillas 0–4.
#
# Evaluación de cada semilla (protocolo §3c.2), 200 partidas por rival:
#   final.pt contra eval (resultado principal), best.pt contra eval (secundario), final.pt contra
#   sus propios equipos (fit; brecha de memorización) y final.pt contra greedy_joint en modo
#   arquetipo (Tailwind / Trick Room, memorizacion_equipos.md §14) con --context.
# Resumen: $OUT/kickoff_summary16.md (showdown/rl/kickoff_report.py; DDQN queda sin datos).
#
#   bash scripts/server/kickoff_ppo_oficial.sh ~/iapsvgc_runs/kickoff_ppo1 ~/iapsvgc_runs
set -u
OUT=${1:-$HOME/iapsvgc_runs/kickoff_ppo1}
R=${2:-$HOME/iapsvgc_runs}
mkdir -p "$OUT"/diag
source ~/miniconda3/etc/profile.d/conda.sh; conda activate kuroneko
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
export PYTHONUNBUFFERED=1 PS_RL_STRICT=1
unset RL_MASK_ALLY_DAMAGE RL_SYM_AUG RL_HOLDOUT_VAL RL_TRAIN_SUBSET RL_FOE_SYNTH PREVIEW_FORCE_ORDER PS_JOINT_ARCHETYPE
log() { echo "$(date '+%F %T') $*" >> "$OUT/queue.log"; }
clean_ps() { pkill -f "pokemon-showdown.*start" 2>/dev/null; sleep 5; }
clean_ps
log "inicio commit $(git rev-parse --short HEAD)"
TEAM=showdown/rl/data/ddqn_team.txt
PV=$(sed -n 's/.*preview=\([^ ]*\).*/\1/p' "$R/kickoff3/mini_choice.txt" 2>/dev/null)
[ -f "${PV:-}" ] || { log "ERROR: falta el preview v2 ($PV)"; exit 1; }
log "preview: $PV"

diag() {  # nombre checkpoint grupo [variables...] [--opciones...]
  local name=$1 ck=$2 split=$3; shift 3
  [ -f "$ck" ] || { log "diag $name: falta $ck"; return; }
  local envs=() opts=()
  for x in "$@"; do case "$x" in --*) opts+=("$x");; *) envs+=("$x");; esac; done
  env PS_TEAM_SPLIT=$split ${envs[@]+"${envs[@]}"} python -m showdown.rl.ddqn.policy_diag --agent ppo \
    --checkpoint "$ck" --games 200 --port 8422 --agent-team $TEAM --preview-checkpoint "$PV" \
    ${opts[@]+"${opts[@]}"} --out "$OUT/diag/$name.json" > "$OUT/diag/$name.txt" 2>&1
  log "diag $name rc=$?"; clean_ps
}

for s in 0 1 2 3 4; do
  python -m showdown.rl.ppo.train --seed $s --device cuda --start-ps --expand-omega --num-envs 32 \
    --envs-per-server 4 --log-every 10000 --log-every-sec 120 \
    --phases random:150000,greedy_solo:250000,greedy_joint:300000,expectimax_bayesian:300000 \
    --eval-every 250000 --eval-games 50 --suite-games 0 --agent-team $TEAM --preview-checkpoint "$PV" \
    --checkpoint-dir "$OUT/ppo_s$s" > "$OUT/ppo_s$s.txt" 2>&1
  log "ppo_s$s rc=$?"; clean_ps
  diag ppo_s${s}_final_eval "$OUT/ppo_s$s/final.pt" eval
  diag ppo_s${s}_best_eval "$OUT/ppo_s$s/best.pt" eval
  diag ppo_s${s}_final_own "$OUT/ppo_s$s/final.pt" train RL_HOLDOUT_VAL=1
  diag ppo_s${s}_final_arquetipo "$OUT/ppo_s$s/final.pt" eval PS_JOINT_ARCHETYPE=1 --foes=greedy_joint --context
done
log "KICKOFF_PPO_DONE"

python -m showdown.rl.kickoff_report "$OUT" > "$OUT/kickoff_report.err" 2>&1
log "resumen rc=$?"
log "KICKOFF_PPO_OFICIAL_DONE"
