#!/bin/bash
# Decimoséptima cola (17-11-2026), aprobada por el usuario. Dos preguntas antes del kick-off:
#
#   A. ¿Cuántos pasos? DDQN (con la REG elegida en la decimoquinta cola) y PPO (con simetría) a
#      2M pasos (fases ×2), semillas 0 y 1. La prueba de 2M anterior (undécima cola) fue sin REG.
#   D. ¿Simetría en DDQN sobre REG? DDQN 1M con la REG elegida + RL_SYM_AUG=1, semillas 0 y 1.
#
# Elección sin tocar `eval` (protocolo §3c.2): cada final.pt se mide contra los 30 equipos `val`
# con 200 partidas por rival; `eval` (100 por rival) y los propios (100) quedan como registro.
# Reglas fijadas de antemano (3 heurísticos, 2 semillas agrupadas = 1 200 partidas):
#   A: 2M se recomienda para un método si supera a su referencia de 1M en `val` por más de 4
#      puntos. Referencias de 1M: DDQN = la REG elegida (decimoquinta cola, ya medida en `val`);
#      PPO = PPO con simetría de la duodécima cola (se mide aquí en `val`). Nota: PPO está cerca
#      del techo en `val` (~93–95 %), así que esta medida es poco sensible para él.
#   D: la simetría se recomienda en DDQN si supera a la REG elegida de 1M en `val` por más de 4.
# Espera a KICKOFF_QUEUE15_DONE. Resumen: $OUT/kickoff_summary17.md.
#
#   bash scripts/server/kickoff_queue17.sh ~/iapsvgc_runs/kickoff17 ~/iapsvgc_runs ~/iapsvgc_runs/kickoff15
set -u
OUT=${1:-$HOME/iapsvgc_runs/kickoff17}
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
P1M=random:150000,greedy_solo:250000,greedy_joint:300000,expectimax_bayesian:300000
P2M=random:300000,greedy_solo:500000,greedy_joint:600000,expectimax_bayesian:600000

diag() {  # agente nombre checkpoint grupo partidas [variables...]
  local agent=$1 name=$2 ck=$3 split=$4 games=$5; shift 5
  [ -f "$ck" ] || { log "diag $name: falta $ck"; return; }
  env PS_TEAM_SPLIT=$split "$@" python -m showdown.rl.ddqn.policy_diag --agent $agent --checkpoint "$ck" \
    --games $games --port 8410 --agent-team $TEAM ${PV:+--preview-checkpoint "$PV"} \
    --out "$OUT/diag/$name.json" > "$OUT/diag/$name.txt" 2>&1
  log "diag $name rc=$?"; clean_ps
}

evaluate() {  # agente nombre
  local agent=$1 name=$2
  diag $agent ${name}_final_val "$OUT/$name/final.pt" val 200
  diag $agent ${name}_final_eval "$OUT/$name/final.pt" eval 100  # registro
  diag $agent ${name}_final_own "$OUT/$name/final.pt" train 100 RL_HOLDOUT_VAL=1  # registro (fit)
}

COMMON="--device cuda --start-ps --expand-omega --num-envs 32 --envs-per-server 4 --log-every 10000 \
  --log-every-sec 120 --eval-every 250000 --eval-games 50 --suite-games 0 --agent-team $TEAM"

ddqn() {  # nombre semilla fases [variables...]
  local name=$1 seed=$2 phases=$3; shift 3
  env "$@" python -m showdown.rl.ddqn.train --seed $seed $COMMON --phases $phases --lr 1e-4 \
    --batch-size 256 $REGF ${PV:+--preview-checkpoint "$PV"} --checkpoint-dir "$OUT/$name" > "$OUT/$name.txt" 2>&1
  log "$name rc=$?"; clean_ps
  evaluate ddqn $name
}

ppo() {  # nombre semilla fases
  local name=$1 seed=$2 phases=$3
  python -m showdown.rl.ppo.train --seed $seed $COMMON --phases $phases \
    ${PV:+--preview-checkpoint "$PV"} --checkpoint-dir "$OUT/$name" > "$OUT/$name.txt" 2>&1
  log "$name rc=$?"; clean_ps
  evaluate ppo $name
}

# referencia de PPO a 1M (duodécima cola, con simetría) medida en val
diag ppo PPO1M_s0_final_val "$R/kickoff12/PPO_sym_s0/final.pt" val 200
diag ppo PPO1M_s1_final_val "$R/kickoff12/PPO_sym_s1/final.pt" val 200

ddqn DDQN2M_s0 0 $P2M
ppo PPO2M_s0 0 $P2M
ddqn DSYM_s0 0 $P1M RL_SYM_AUG=1
ddqn DDQN2M_s1 1 $P2M
ppo PPO2M_s1 1 $P2M
ddqn DSYM_s1 1 $P1M RL_SYM_AUG=1

python -m showdown.rl.kickoff_summary17 "$OUT" "$Q15" > "$OUT/summary17.err" 2>&1
log "resumen rc=$?"
log "KICKOFF_QUEUE17_DONE"
