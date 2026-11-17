#!/bin/bash
# Decimoctava cola (17-11-2026): valores Q por hueco en DDQN (opción C, memorizacion_equipos.md
# §10; ddqn_spec 1.21), aprobada por el usuario para antes del kick-off. La red Q pasa de 1 600
# salidas conjuntas a V(s) + A1(s, a1) + A2(s, a2) (+ interacción de rango r), la factorización
# por hueco que distingue a PPO, que generaliza:
#
#   QF0: --q-head factored --q-rank 0 (aditiva)
#   QF8: --q-head factored --q-rank 8 (con interacción entre huecos)
#
# DDQN 1M, configuración por defecto + REG elegida en la decimoquinta cola, semillas 0 y 1.
# Elección sin tocar `eval`: final.pt contra `val` con 200 partidas por rival; se adopta la mejor
# variante si supera a la cabeza conjunta (REG elegida, decimoquinta cola) por más de 4 puntos.
# `eval` (100) y propios (100) quedan como registro.
# Espera a KICKOFF_QUEUE17_DONE. Resumen: $OUT/kickoff_summary18.md.
#
#   bash scripts/server/kickoff_queue18.sh ~/iapsvgc_runs/kickoff18 ~/iapsvgc_runs ~/iapsvgc_runs/kickoff15 ~/iapsvgc_runs/kickoff17
set -u
OUT=${1:-$HOME/iapsvgc_runs/kickoff18}
R=${2:-$HOME/iapsvgc_runs}
Q15=${3:-$HOME/iapsvgc_runs/kickoff15}
Q17=${4:-$HOME/iapsvgc_runs/kickoff17}
mkdir -p "$OUT"/diag
source ~/miniconda3/etc/profile.d/conda.sh; conda activate kuroneko
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
export PYTHONUNBUFFERED=1 PS_RL_STRICT=1
unset RL_MASK_ALLY_DAMAGE RL_SYM_AUG RL_HOLDOUT_VAL RL_TRAIN_SUBSET RL_FOE_SYNTH  # valores por defecto
log() { echo "$(date '+%F %T') $*" >> "$OUT/queue.log"; }
clean_ps() { pkill -f "pokemon-showdown.*start" 2>/dev/null; sleep 5; }
log "esperando a la decimoséptima cola ($Q17)"
until grep -q KICKOFF_QUEUE17_DONE "$Q17/queue.log" 2>/dev/null; do sleep 60; done
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
    --games $games --port 8412 --agent-team $TEAM ${PV:+--preview-checkpoint "$PV"} \
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

ddqn() {  # nombre semilla rango
  local name=$1 seed=$2 rank=$3
  python -m showdown.rl.ddqn.train --seed $seed $COMMON --phases $P1M --lr 1e-4 \
    --batch-size 256 $REGF --q-head factored --q-rank $rank \
    ${PV:+--preview-checkpoint "$PV"} --checkpoint-dir "$OUT/$name" > "$OUT/$name.txt" 2>&1
  log "$name rc=$?"; clean_ps
  evaluate ddqn $name
}

ddqn QF0_s0 0 0
ddqn QF8_s0 0 8
ddqn QF0_s1 1 0
ddqn QF8_s1 1 8

python -m showdown.rl.kickoff_summary18 "$OUT" "$Q15" > "$OUT/summary18.err" 2>&1
log "resumen rc=$?"
log "KICKOFF_QUEUE18_DONE"
