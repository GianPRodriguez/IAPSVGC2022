#!/bin/bash
# Decimosexta cola: KICK-OFF OFICIAL de DDQN y después de PPO, encadenados (decisión del usuario,
# 17-11-2026). Configuración final (ddqn_kickoff_checklist.md, "Configuración final de los
# kick-offs"), toda por defecto en el código:
#
#   DDQN: máscara del aliado, sin simetría, REG (weight decay + dropout; valores elegidos por la
#         decimoquinta cola con `val`, archivo chosen_reg.txt), validación reservada.
#   PPO:  máscara del aliado, simetría activa, desde cero, validación reservada.
#   Ambos: 1M pasos (currículo random 150k → greedy_solo 250k → greedy_joint 300k →
#          expectimax 300k), 5 semillas (0–4), equipo fijo del agente, preview v2, sin sintéticos.
#
# Evaluación de cada semilla (protocolo §3c.2): final.pt = resultado principal, best.pt =
# secundario; 200 partidas por rival contra los equipos `eval` (decisión del usuario), y final.pt
# contra sus propios equipos (`fit`, 200 por rival) para medir la brecha de memorización.
# Espera a KICKOFF_QUEUE15_DONE. Resumen: $OUT/kickoff_summary16.md.
#
#   bash scripts/server/kickoff_queue16.sh ~/iapsvgc_runs/kickoff16 ~/iapsvgc_runs ~/iapsvgc_runs/kickoff15
set -u
OUT=${1:-$HOME/iapsvgc_runs/kickoff16}
R=${2:-$HOME/iapsvgc_runs}
Q15=${3:-$HOME/iapsvgc_runs/kickoff15}
mkdir -p "$OUT"/diag
source ~/miniconda3/etc/profile.d/conda.sh; conda activate kuroneko
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
export PYTHONUNBUFFERED=1 PS_RL_STRICT=1
unset RL_MASK_ALLY_DAMAGE RL_SYM_AUG RL_HOLDOUT_VAL RL_TRAIN_SUBSET RL_FOE_SYNTH  # valores por defecto
log() { echo "$(date '+%F %T') $*" >> "$OUT/queue.log"; }
clean_ps() { pkill -f "pokemon-showdown.*start" 2>/dev/null; sleep 5; }
log "esperando a la decimoquinta cola ($Q15)"
until grep -q KICKOFF_QUEUE15_DONE "$Q15/queue.log" 2>/dev/null; do sleep 60; done
clean_ps
log "inicio commit $(git rev-parse --short HEAD)"
TEAM=showdown/rl/data/ddqn_team.txt
PV=$(sed -n 's/.*preview=\([^ ]*\).*/\1/p' "$R/kickoff3/mini_choice.txt" 2>/dev/null)
[ -f "${PV:-}" ] || PV=""
REGF=$(cat "$Q15/chosen_reg.txt" 2>/dev/null)
[ -n "$REGF" ] || REGF="--weight-decay 0.01 --dropout 0.1"
log "preview: ${PV:-ninguno}; REG de DDQN: $REGF"
PHASES=random:150000,greedy_solo:250000,greedy_joint:300000,expectimax_bayesian:300000

diag() {  # agente nombre checkpoint grupo [variables...]
  local agent=$1 name=$2 ck=$3 split=$4; shift 4
  [ -f "$ck" ] || { log "diag $name: falta $ck"; return; }
  env PS_TEAM_SPLIT=$split "$@" python -m showdown.rl.ddqn.policy_diag --agent $agent --checkpoint "$ck" \
    --games 200 --port 8408 --agent-team $TEAM ${PV:+--preview-checkpoint "$PV"} \
    --out "$OUT/diag/$name.json" > "$OUT/diag/$name.txt" 2>&1
  log "diag $name rc=$?"; clean_ps
}

evaluate() {  # agente nombre
  local agent=$1 name=$2
  diag $agent ${name}_final_eval "$OUT/$name/final.pt" eval
  diag $agent ${name}_best_eval "$OUT/$name/best.pt" eval
  diag $agent ${name}_final_own "$OUT/$name/final.pt" train RL_HOLDOUT_VAL=1  # propios = fit
}

COMMON="--device cuda --start-ps --expand-omega --num-envs 32 --envs-per-server 4 --log-every 10000 \
  --log-every-sec 120 --phases $PHASES --eval-every 250000 --eval-games 50 --suite-games 0 --agent-team $TEAM"

for s in 0 1 2 3 4; do
  python -m showdown.rl.ddqn.train --seed $s $COMMON --lr 1e-4 --batch-size 256 $REGF \
    ${PV:+--preview-checkpoint "$PV"} --checkpoint-dir "$OUT/ddqn_s$s" > "$OUT/ddqn_s$s.txt" 2>&1
  log "ddqn_s$s rc=$?"; clean_ps
  evaluate ddqn ddqn_s$s
done
log "KICKOFF_DDQN_DONE"

for s in 0 1 2 3 4; do
  python -m showdown.rl.ppo.train --seed $s $COMMON \
    ${PV:+--preview-checkpoint "$PV"} --checkpoint-dir "$OUT/ppo_s$s" > "$OUT/ppo_s$s.txt" 2>&1
  log "ppo_s$s rc=$?"; clean_ps
  evaluate ppo ppo_s$s
done
log "KICKOFF_PPO_DONE"

python -m showdown.rl.kickoff_report "$OUT" > "$OUT/kickoff_report.err" 2>&1
log "resumen rc=$?"
log "KICKOFF_QUEUE16_DONE"
