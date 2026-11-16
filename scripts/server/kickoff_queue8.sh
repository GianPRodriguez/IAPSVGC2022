#!/bin/bash
# Octava cola (16-11-2026): memorización de los equipos rivales (cuarta cola: 84 % con equipos
# train frente a 50 % con eval) y cómo reducirla. Espera a KICKOFF_QUEUE7_DONE.
# Resumen: $OUT/kickoff_summary8.md. Ver docs/memorizacion_equipos.md.
#
# Todas las corridas: configuración del mini kick-off (1M pasos, currículo completo, lr 1e-4,
# lote 256, preview v2, equipo fijo) con validación reservada (RL_HOLDOUT_VAL=1: entrena con
# fit = 170 equipos, valida con val = 30). Cada corrida se diagnostica con 100 partidas por rival
# contra sus propios equipos de entrenamiento y contra eval, guardando el equipo de cada partida.
#
#   bash scripts/server/kickoff_queue8.sh ~/iapsvgc_runs/kickoff8 ~/iapsvgc_runs ~/iapsvgc_runs/kickoff7
set -u
OUT=${1:-$HOME/iapsvgc_runs/kickoff8}
R=${2:-$HOME/iapsvgc_runs}
Q7=${3:-$HOME/iapsvgc_runs/kickoff7}
mkdir -p "$OUT"/diag
source ~/miniconda3/etc/profile.d/conda.sh; conda activate kuroneko
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
export PYTHONUNBUFFERED=1 PS_RL_STRICT=1
log() { echo "$(date '+%F %T') $*" >> "$OUT/queue.log"; }
clean_ps() { pkill -f "pokemon-showdown.*start" 2>/dev/null; sleep 5; }
log "esperando a la séptima cola ($Q7)"
until grep -q KICKOFF_QUEUE7_DONE "$Q7/queue.log" 2>/dev/null; do sleep 60; done
clean_ps
log "inicio commit $(git rev-parse --short HEAD)"
TEAM=showdown/rl/data/ddqn_team.txt
PV=$(sed -n 's/.*preview=\([^ ]*\).*/\1/p' "$R/kickoff3/mini_choice.txt" 2>/dev/null)
[ -f "${PV:-}" ] || PV=""
log "preview: ${PV:-ninguno}"
pick() { for f in "$@"; do [ -f "$f" ] && { echo "$f"; return; }; done; echo ""; }

diag() {  # nombre checkpoint grupo [variables...]: 100 partidas por rival, sin sintéticos
  local name=$1 ck=$2 split=$3; shift 3
  [ -f "$ck" ] || { log "diag $name: falta $ck"; return; }
  env PS_TEAM_SPLIT=$split "$@" RL_FOE_SYNTH=0 python -m showdown.rl.ddqn.policy_diag --checkpoint "$ck" \
    --games 100 --port 8392 --agent-team $TEAM ${PV:+--preview-checkpoint "$PV"} \
    --out "$OUT/diag/$name.json" > "$OUT/diag/$name.txt" 2>&1
  log "diag $name rc=$?"; clean_ps
}

run() {  # nombre semilla [variables...]
  local name=$1 seed=$2; shift 2
  env RL_HOLDOUT_VAL=1 "$@" python -m showdown.rl.ddqn.train --seed $seed --device cuda --start-ps --expand-omega \
    --num-envs 32 --envs-per-server 4 --log-every 10000 --log-every-sec 120 \
    --phases random:150000,greedy_solo:250000,greedy_joint:300000,expectimax_bayesian:300000 \
    --eval-every 250000 --eval-games 50 --suite-games 0 --lr 1e-4 --batch-size 256 --agent-team $TEAM \
    ${PV:+--preview-checkpoint "$PV"} --checkpoint-dir "$OUT/$name" > "$OUT/$name.txt" 2>&1
  log "$name rc=$?"; clean_ps
  local ck; ck=$(pick "$OUT/$name/best.pt" "$OUT/$name/final.pt")
  diag ${name}_own "$ck" train RL_HOLDOUT_VAL=1 "$@"
  diag ${name}_eval "$ck" eval
}

# D0. mini kick-off anterior (entrenó con los 200 de train): mismo diagnóstico con equipo por partida
CK0=$(pick "$R/kickoff3/mini_kickoff/best.pt" "$R/kickoff3/mini_kickoff/final.pt")
diag D0_mini_own "$CK0" train
diag D0_mini_eval "$CK0" eval

# Comparación principal primero (con y sin sintéticos), después la curva de escala y la 2.ª semilla
run F170_s0 0
run SY50_s0 0 RL_FOE_SYNTH=0.5
run S50_s0 0 RL_TRAIN_SUBSET=50
run S100_s0 0 RL_TRAIN_SUBSET=100
run F170_s1 1
run SY50_s1 1 RL_FOE_SYNTH=0.5

python -m showdown.rl.ddqn.kickoff_summary8 "$OUT" "$R" > /dev/null 2>"$OUT/summary8.err"
log "resumen rc=$?"
log "KICKOFF_QUEUE8_DONE"
