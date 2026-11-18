#!/bin/bash
# Vigésima cola (18-11-2026), aprobada por el usuario. El agente es una tupla (preview, combate):
# las dos etapas deciden por separado y ambas tienen que jugar bien. Esta cola:
#
#   1. Preview por separado: mismo combate (final.pt de la configuración final: DDQN con REG,
#      PPO con simetría; 2 semillas), tres previews: v2, la formación que más elige v2 jugada
#      siempre, y /team 1234. Tu equipo fijo, rivales eval, 200 partidas por rival. La corrida con
#      v2 lleva --context: combate según el plan del rival, ahora en su distribución.
#   2. Piloto de entrenamiento con equipo propio al azar (--agent-team pool, grupo fit): DDQN y
#      PPO, 1M, semillas 0 y 1, configuración final por defecto. Evaluación contra rivales eval con
#      equipo propio al azar y con tu equipo fijo (200 por rival). El preview v2 se entrenó para tu
#      equipo: con equipos al azar trabaja fuera de su distribución (límite del piloto).
#
# No espera a otra cola (el servidor está libre). Resumen: $OUT/kickoff_summary20.md.
#
#   bash scripts/server/kickoff_queue20.sh ~/iapsvgc_runs/kickoff20 ~/iapsvgc_runs
set -u
OUT=${1:-$HOME/iapsvgc_runs/kickoff20}
R=${2:-$HOME/iapsvgc_runs}
mkdir -p "$OUT"/diag
source ~/miniconda3/etc/profile.d/conda.sh; conda activate kuroneko
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
export PYTHONUNBUFFERED=1 PS_RL_STRICT=1
unset RL_MASK_ALLY_DAMAGE RL_SYM_AUG RL_HOLDOUT_VAL RL_TRAIN_SUBSET RL_FOE_SYNTH PREVIEW_FORCE_ORDER
log() { echo "$(date '+%F %T') $*" >> "$OUT/queue.log"; }
clean_ps() { pkill -f "pokemon-showdown.*start" 2>/dev/null; sleep 5; }
clean_ps
log "inicio commit $(git rev-parse --short HEAD)"
TEAM=showdown/rl/data/ddqn_team.txt
PV=$(sed -n 's/.*preview=\([^ ]*\).*/\1/p' "$R/kickoff3/mini_choice.txt" 2>/dev/null)
[ -f "${PV:-}" ] || PV=""
log "preview: ${PV:-ninguno}"

diag() {  # agente nombre checkpoint equipo(''=al azar) [variables y opciones extra...]
  local agent=$1 name=$2 ck=$3 team=$4; shift 4
  [ -f "$ck" ] || { log "diag $name: falta $ck"; return; }
  local envs=() opts=()
  for x in "$@"; do case "$x" in --*) opts+=("$x");; *) envs+=("$x");; esac; done
  env PS_TEAM_SPLIT=eval ${envs[@]+"${envs[@]}"} python -m showdown.rl.ddqn.policy_diag --agent $agent --checkpoint "$ck" \
    --games 200 --port 8416 --agent-team "$team" ${PV:+--preview-checkpoint "$PV"} ${opts[@]+"${opts[@]}"} \
    --out "$OUT/diag/$name.json" > "$OUT/diag/$name.txt" 2>&1
  log "diag $name rc=$?"; clean_ps
}

moda() {  # formación más frecuente del preview v2 en un diagnóstico
  python - "$1" <<'PY'
import json, sys
from collections import Counter
g = json.load(open(sys.argv[1]))['partidas']
c = Counter(x['formacion'] for x in g if x.get('formacion') and x['rival'] != 'random')
print(c.most_common(1)[0][0] if c else '1234')
PY
}

# ---------------------------------------------------------------- 1. preview por separado
for s in 0 1; do
  for a in ddqn ppo; do
    if [ $a = ddqn ]; then CK="$R/kickoff15/R_wd03_s$s/final.pt"; else CK="$R/kickoff12/PPO_sym_s$s/final.pt"; fi
    diag $a ${a}_s${s}_v2 "$CK" $TEAM --context
    M=$(moda "$OUT/diag/${a}_s${s}_v2.json" 2>/dev/null || echo 1234)
    log "formación más frecuente ${a}_s$s: $M"
    diag $a ${a}_s${s}_moda "$CK" $TEAM PREVIEW_FORCE_ORDER=$M
    diag $a ${a}_s${s}_1234 "$CK" $TEAM PREVIEW_FORCE_ORDER=1234
  done
done

# ---------------------------------------------------------------- 2. piloto con equipo propio al azar
COMMON="--device cuda --start-ps --expand-omega --num-envs 32 --envs-per-server 4 --log-every 10000 \
  --log-every-sec 120 --eval-every 250000 --eval-games 50 --suite-games 0 --agent-team pool \
  --phases random:150000,greedy_solo:250000,greedy_joint:300000,expectimax_bayesian:300000"
for s in 0 1; do
  python -m showdown.rl.ddqn.train --seed $s $COMMON --lr 1e-4 --batch-size 256 \
    ${PV:+--preview-checkpoint "$PV"} --checkpoint-dir "$OUT/pool_ddqn_s$s" > "$OUT/pool_ddqn_s$s.txt" 2>&1
  log "pool_ddqn_s$s rc=$?"; clean_ps
  diag ddqn pool_ddqn_s${s}_azar "$OUT/pool_ddqn_s$s/final.pt" ''
  diag ddqn pool_ddqn_s${s}_fijo "$OUT/pool_ddqn_s$s/final.pt" $TEAM
  python -m showdown.rl.ppo.train --seed $s $COMMON \
    ${PV:+--preview-checkpoint "$PV"} --checkpoint-dir "$OUT/pool_ppo_s$s" > "$OUT/pool_ppo_s$s.txt" 2>&1
  log "pool_ppo_s$s rc=$?"; clean_ps
  diag ppo pool_ppo_s${s}_azar "$OUT/pool_ppo_s$s/final.pt" ''
  diag ppo pool_ppo_s${s}_fijo "$OUT/pool_ppo_s$s/final.pt" $TEAM
done

python -m showdown.rl.kickoff_summary20 "$OUT" "$R" > "$OUT/summary20.err" 2>&1
log "resumen rc=$?"
log "KICKOFF_QUEUE20_DONE"
