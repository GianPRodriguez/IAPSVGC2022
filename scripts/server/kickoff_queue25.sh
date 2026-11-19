#!/bin/bash
# VIGESIMOQUINTA COLA (18-11-2026): P5, corrección de habilidades en la observación (rl_env_spec v1.32).
# Decisión del usuario: P5 se activa en PPO y DDQN (por defecto en los entrenamientos nuevos). Esta cola
# mide su efecto con la configuración final de cada agente: 1M, semillas 0 y 1, contra val y eval.
#   PPO P5 frente a PPO v1 congelado (referencia de la vigesimocuarta cola); también arquetipo nivel 1
#   y contra PPO v1 en espejo.
#   DDQN P5 frente a DDQN REG 0,03 (decimoquinta cola; eval repetido con 200 partidas por rival).
# Espera a KICKOFF_QUEUE24_DONE. Resumen: $OUT/kickoff_summary25.md.
#
#   bash scripts/server/kickoff_queue25.sh ~/iapsvgc_runs/p5 ~/iapsvgc_runs/ppo_v2 ~/iapsvgc_runs/ppo_v1 ~/iapsvgc_runs/kickoff15
set -u
OUT=${1:-$HOME/iapsvgc_runs/p5}
V2=${2:-$HOME/iapsvgc_runs/ppo_v2}
V1=${3:-$HOME/iapsvgc_runs/ppo_v1}
Q15=${4:-$HOME/iapsvgc_runs/kickoff15}
mkdir -p "$OUT"/diag
source ~/miniconda3/etc/profile.d/conda.sh; conda activate kuroneko
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
export PYTHONUNBUFFERED=1 PS_RL_STRICT=1
unset RL_MASK_ALLY_DAMAGE RL_SYM_AUG RL_HOLDOUT_VAL RL_TRAIN_SUBSET RL_FOE_SYNTH PREVIEW_FORCE_ORDER PS_JOINT_ARCHETYPE PPO_FOE_CHECKPOINTS RL_OBS_ABILITIES
log() { echo "$(date '+%F %T') $*" >> "$OUT/queue.log"; }
clean_ps() { pkill -f "pokemon-showdown.*start" 2>/dev/null; sleep 5; }
# Espera configurable (19-11-2026: tras la calibración de v1 contra v1, h2h_v1_calib.sh).
WAIT_FILE=${WAIT_FILE:-$V2/queue.log}
WAIT_MARK=${WAIT_MARK:-KICKOFF_QUEUE24_DONE}
log "esperando $WAIT_MARK en $WAIT_FILE"
until grep -q "$WAIT_MARK" "$WAIT_FILE" 2>/dev/null; do sleep 60; done
clean_ps
log "inicio commit $(git rev-parse --short HEAD)"
TEAM=showdown/rl/data/ddqn_team.txt
PV="$V1/preview_v2.pt"
V1CKS=$(ls "$V1"/ppo_s[0-4]/final.pt | paste -sd, -)
OMP_NUM_THREADS=4 python -m pytest -q -p no:cacheprovider tests/test_obs_abilities.py tests/test_ppo_v2.py \
  tests/test_ddqn_regularization.py tests/test_kickoff_report.py > "$OUT/tests0.txt" 2>&1
rc=$?; log "0 tests rc=$rc ($(tail -1 "$OUT/tests0.txt"))"
[ $rc = 0 ] || { log "ERROR: tests fallan; se detiene la cola"; log "KICKOFF_QUEUE25_DONE"; exit 1; }

diag() {  # agente nombre checkpoint grupo [VARIABLE=valor...] [--opciones...]
  local agent=$1 name=$2 ck=$3 split=$4; shift 4
  [ -f "$ck" ] || { log "diag $name: falta $ck"; return; }
  local envs=() opts=()
  for x in "$@"; do case "$x" in --*) opts+=("$x");; *) envs+=("$x");; esac; done
  env PS_TEAM_SPLIT=$split ${envs[@]+"${envs[@]}"} python -m showdown.rl.ddqn.policy_diag --agent $agent \
    --checkpoint "$ck" --games 200 --port 8427 --agent-team $TEAM --preview-checkpoint "$PV" \
    ${opts[@]+"${opts[@]}"} --out "$OUT/diag/$name.json" > "$OUT/diag/$name.txt" 2>&1
  log "diag $name rc=$?"; clean_ps
}
PHASES="random:150000,greedy_solo:250000,greedy_joint:300000,expectimax_bayesian:300000"
COMMON="--device cuda --start-ps --expand-omega --num-envs 32 --envs-per-server 4 --log-every 10000 \
  --log-every-sec 120 --phases $PHASES --eval-every 250000 --eval-games 50 --suite-games 0 --agent-team $TEAM \
  --preview-checkpoint $PV --obs-abilities 1"

# ---------------------------------------------------------------- referencia DDQN (eval a 200)
for s in 0 1; do diag ddqn ddqn_ref_s${s}_eval "$Q15/R_wd03_s$s/final.pt" eval; done

# ---------------------------------------------------------------- PPO con P5
for s in 0 1; do
  python -m showdown.rl.ppo.train --seed $s $COMMON --checkpoint-dir "$OUT/ppo_p5_s$s" > "$OUT/ppo_p5_s$s.txt" 2>&1
  log "ppo_p5_s$s rc=$?"; clean_ps
  C="$OUT/ppo_p5_s$s/final.pt"
  diag ppo ppo_p5_s${s}_val "$C" val
  diag ppo ppo_p5_s${s}_eval "$C" eval
  diag ppo ppo_p5_s${s}_arq1_eval "$C" eval PS_JOINT_ARCHETYPE=1 --foes=greedy_joint --context
  diag ppo ppo_p5_s${s}_h2h "$C" eval PPO_FOE_CHECKPOINTS="$V1CKS" --foes=ppo_v1
done

# ---------------------------------------------------------------- DDQN con P5 (configuración final: REG 0,03)
for s in 0 1; do
  python -m showdown.rl.ddqn.train --seed $s $COMMON --lr 1e-4 --batch-size 256 \
    --checkpoint-dir "$OUT/ddqn_p5_s$s" > "$OUT/ddqn_p5_s$s.txt" 2>&1
  log "ddqn_p5_s$s rc=$?"; clean_ps
  C="$OUT/ddqn_p5_s$s/final.pt"
  diag ddqn ddqn_p5_s${s}_val "$C" val
  diag ddqn ddqn_p5_s${s}_eval "$C" eval
done

python -m showdown.rl.kickoff_summary25 "$OUT" "$V2" "$V1" "$Q15" > "$OUT/summary25.err" 2>&1
log "resumen rc=$?"
log "KICKOFF_QUEUE25_DONE"
