#!/bin/bash
# VIGESIMOSEXTA COLA (19-11-2026): preview v2.1 + medición de P5 completo con preview v2.1.
# Decisiones del usuario: preview v2.1 con la misma receta que el v2; medir PPO y DDQN con P5 completo
# y preview v2.1 antes de las corridas oficiales (si algo empeora, se detecta antes de gastar ~11 h);
# P3 y P4 no se adoptan.
#   1. Reinicia el servidor de la calculadora con el código nuevo (auras de campo).
#   2. Preview v2.1: team_preview_ppo, greedy_joint como combate, tu equipo, 200 actualizaciones,
#      entropía 0,005, formaciones canónicas, observación nueva (--obs-abilities 1); se usa best.pt.
#   3. PPO con P5 + preview v2.1 (semillas 0 y 1, 1M, configuración final): val, eval, arquetipo n1 y
#      contra PPO v1 en espejo (v1 con su preview v2; calibración v1-v1 = 59,8 %).
#   4. DDQN con P5 + preview v2.1 (semillas 0 y 1, 1M, REG 0,03): val y eval; referencia DDQN REG
#      contra eval con 200 partidas por rival.
# Espera a P5_SPEED_DONE. Resumen: $OUT/kickoff_summary26.md.
#
#   bash scripts/server/kickoff_queue26.sh ~/iapsvgc_runs/p5full ~/iapsvgc_runs/ppo_v2 ~/iapsvgc_runs/ppo_v1 ~/iapsvgc_runs/kickoff15
set -u
OUT=${1:-$HOME/iapsvgc_runs/p5full}
V2=${2:-$HOME/iapsvgc_runs/ppo_v2}
V1=${3:-$HOME/iapsvgc_runs/ppo_v1}
Q15=${4:-$HOME/iapsvgc_runs/kickoff15}
mkdir -p "$OUT"/diag
source ~/miniconda3/etc/profile.d/conda.sh; conda activate kuroneko
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
export PYTHONUNBUFFERED=1 PS_RL_STRICT=1
unset RL_MASK_ALLY_DAMAGE RL_SYM_AUG RL_HOLDOUT_VAL RL_TRAIN_SUBSET RL_FOE_SYNTH PREVIEW_FORCE_ORDER PS_JOINT_ARCHETYPE \
  PPO_FOE_CHECKPOINTS PPO_FOE_PREVIEW RL_OBS_ABILITIES RL_OBS_CALC
log() { echo "$(date '+%F %T') $*" >> "$OUT/queue.log"; }
clean_ps() { pkill -f "pokemon-showdown.*start" 2>/dev/null; sleep 5; }
log "esperando al piloto de velocidad"
until grep -q P5_SPEED_DONE "$HOME/iapsvgc_runs/p5_speed/queue.log" 2>/dev/null; do sleep 60; done
# Resguardo: si la calculadora deja el entrenamiento por debajo del 40 % de la velocidad sin ella, se detiene.
SP1=$(grep -o "[0-9.]* step/s | ETA 0s" "$HOME/iapsvgc_runs/p5_speed/calc1.txt" 2>/dev/null | tail -1 | cut -d' ' -f1)
SP0=$(grep -o "[0-9.]* step/s | ETA 0s" "$HOME/iapsvgc_runs/p5_speed/calc0.txt" 2>/dev/null | tail -1 | cut -d' ' -f1)
log "piloto: con calculadora ${SP1:-?} pasos/s, sin ella ${SP0:-?} pasos/s"
if [ -z "${SP1:-}" ] || [ -z "${SP0:-}" ] || ! python -c "import sys; sys.exit(0 if float('$SP1') >= 0.4 * float('$SP0') else 1)"; then
  log "ERROR: piloto sin datos o calculadora demasiado lenta; se detiene la cola para revisar"; log "KICKOFF_QUEUE26_DONE"; exit 1
fi
clean_ps
pkill -f "calc_serve[r].js" 2>/dev/null; sleep 2
log "inicio commit $(git rev-parse --short HEAD)"
TEAM=showdown/rl/data/ddqn_team.txt
PV2="$V1/preview_v2.pt"
V1CKS=$(ls "$V1"/ppo_s[0-4]/final.pt | paste -sd, -)
OMP_NUM_THREADS=4 python -m pytest -q -p no:cacheprovider tests/test_obs_abilities.py tests/test_obs_calc.py \
  tests/test_ppo_v2.py tests/test_preview_rl_pilot.py tests/test_rl_obs_v2.py > "$OUT/tests0.txt" 2>&1
rc=$?; log "0 tests rc=$rc ($(tail -1 "$OUT/tests0.txt"))"
[ $rc = 0 ] || { log "ERROR: tests fallan; se detiene la cola"; log "KICKOFF_QUEUE26_DONE"; exit 1; }

# ---------------------------------------------------------------- 2. preview v2.1 (receta del v2)
python -m showdown.rl.team_preview_ppo.train --seed 0 --device cuda --backend live --num-workers 28 \
  --envs-per-server 4 --start-ps --pilot greedy_joint --updates 200 --eval-every 20 \
  --eval-periodic-episodes 256 --eval-baseline pilot --canonical --own-team $TEAM --ent-coef 0.005 \
  --obs-abilities 1 --checkpoint-dir "$OUT/preview_v21" > "$OUT/preview_v21.txt" 2>&1
log "preview_v21 rc=$?"; clean_ps
PV="$OUT/preview_v21/best.pt"
[ -f "$PV" ] || PV="$OUT/preview_v21/final.pt"
[ -f "$PV" ] || { log "ERROR: no hay preview v2.1"; log "KICKOFF_QUEUE26_DONE"; exit 1; }
log "preview v2.1: $PV"

diag() {  # agente nombre checkpoint grupo [VARIABLE=valor...] [--opciones...]
  local agent=$1 name=$2 ck=$3 split=$4; shift 4
  [ -f "$ck" ] || { log "diag $name: falta $ck"; return; }
  local envs=() opts=()
  for x in "$@"; do case "$x" in --*) opts+=("$x");; *) envs+=("$x");; esac; done
  local pv="$PV"
  case "$name" in ddqn_ref_*) pv="$PV2";; esac  # la referencia anterior juega con su preview v2
  env PS_TEAM_SPLIT=$split ${envs[@]+"${envs[@]}"} python -m showdown.rl.ddqn.policy_diag --agent $agent \
    --checkpoint "$ck" --games 200 --port 8430 --agent-team $TEAM --preview-checkpoint "$pv" \
    ${opts[@]+"${opts[@]}"} --out "$OUT/diag/$name.json" > "$OUT/diag/$name.txt" 2>&1
  log "diag $name rc=$?"; clean_ps
}
PHASES="random:150000,greedy_solo:250000,greedy_joint:300000,expectimax_bayesian:300000"
COMMON="--device cuda --start-ps --expand-omega --num-envs 32 --envs-per-server 4 --log-every 10000 \
  --log-every-sec 120 --phases $PHASES --eval-every 250000 --eval-games 50 --suite-games 0 --agent-team $TEAM \
  --preview-checkpoint $PV --obs-abilities 1"

for s in 0 1; do diag ddqn ddqn_ref_s${s}_eval "$Q15/R_wd03_s$s/final.pt" eval; done

# ---------------------------------------------------------------- 3. PPO con P5 + preview v2.1
for s in 0 1; do
  python -m showdown.rl.ppo.train --seed $s $COMMON --checkpoint-dir "$OUT/ppo_p5_s$s" > "$OUT/ppo_p5_s$s.txt" 2>&1
  log "ppo_p5_s$s rc=$?"; clean_ps
  C="$OUT/ppo_p5_s$s/final.pt"
  diag ppo ppo_p5_s${s}_val "$C" val
  diag ppo ppo_p5_s${s}_eval "$C" eval
  diag ppo ppo_p5_s${s}_arq1_eval "$C" eval PS_JOINT_ARCHETYPE=1 --foes=greedy_joint --context
  diag ppo ppo_p5_s${s}_h2h "$C" eval PPO_FOE_CHECKPOINTS="$V1CKS" PPO_FOE_PREVIEW="$PV2" --foes=ppo_v1
done

# ---------------------------------------------------------------- 4. DDQN con P5 + preview v2.1
for s in 0 1; do
  python -m showdown.rl.ddqn.train --seed $s $COMMON --lr 1e-4 --batch-size 256 \
    --checkpoint-dir "$OUT/ddqn_p5_s$s" > "$OUT/ddqn_p5_s$s.txt" 2>&1
  log "ddqn_p5_s$s rc=$?"; clean_ps
  C="$OUT/ddqn_p5_s$s/final.pt"
  diag ddqn ddqn_p5_s${s}_val "$C" val
  diag ddqn ddqn_p5_s${s}_eval "$C" eval
done

python -m showdown.rl.kickoff_summary26 "$OUT" "$V2" "$V1" "$Q15" > "$OUT/summary26.err" 2>&1
log "resumen rc=$?"
log "KICKOFF_QUEUE26_DONE"
