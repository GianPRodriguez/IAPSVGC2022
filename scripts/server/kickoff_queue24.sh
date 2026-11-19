#!/bin/bash
# VIGESIMOCUARTA COLA (18-11-2026): candidatos para PPO v2 (memorizacion_equipos.md §17).
# Decisión del usuario: encolar P1, P2, P3 y P4, solo para PPO, reportado como PPO v2; la v1
# (kick-off oficial) queda congelada en ~/iapsvgc_runs/ppo_v1 (solo lectura, MANIFEST.md5).
#
#   Referencia v1: final.pt de las 5 semillas contra val; semillas 0 y 1 contra el rival de
#                  arquetipos nivel 2 (val y eval). eval y arquetipo nivel 1 ya están en el kick-off.
#   P1: sin reentrenar, sorteo de la política (T = 1,0 y 0,5) con las 5 semillas de v1: val, eval y
#       arquetipo nivel 1.
#   P2: 1M con la mitad de la fase greedy_joint en modo arquetipo nivel 2 (greedy_joint_arq).
#   P4: P2 + entropía 0,03.
#   P3: 1M con la mitad de la fase expectimax contra el PPO v1 congelado en espejo (ppo_v1).
#   P2/P4/P3: semillas 0 y 1; cada una contra val, eval, arquetipo n1 (eval), n2 (val y eval) y
#             contra PPO v1 (espejo, 200 partidas).
# Reglas de decisión: encabezado de showdown/rl/kickoff_summary24.py. Resumen: $OUT/kickoff_summary24.md.
#
#   bash scripts/server/kickoff_queue24.sh ~/iapsvgc_runs/ppo_v2 ~/iapsvgc_runs/ppo_v1
set -u
OUT=${1:-$HOME/iapsvgc_runs/ppo_v2}
V1=${2:-$HOME/iapsvgc_runs/ppo_v1}
mkdir -p "$OUT"/diag
source ~/miniconda3/etc/profile.d/conda.sh; conda activate kuroneko
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
export PYTHONUNBUFFERED=1 PS_RL_STRICT=1
unset RL_MASK_ALLY_DAMAGE RL_SYM_AUG RL_HOLDOUT_VAL RL_TRAIN_SUBSET RL_FOE_SYNTH PREVIEW_FORCE_ORDER PS_JOINT_ARCHETYPE PPO_FOE_CHECKPOINTS
log() { echo "$(date '+%F %T') $*" >> "$OUT/queue.log"; }
clean_ps() { pkill -f "pokemon-showdown.*start" 2>/dev/null; sleep 5; }
clean_ps
log "inicio commit $(git rev-parse --short HEAD)"
TEAM=showdown/rl/data/ddqn_team.txt
PV="$V1/preview_v2.pt"
[ -f "$PV" ] || { log "ERROR: falta el preview v2 congelado ($PV)"; exit 1; }
( cd "$V1" && md5sum -c --quiet MANIFEST.md5 ) >> "$OUT/queue.log" 2>&1 \
  && log "v1 congelado verificado (MANIFEST.md5)" || { log "ERROR: v1 congelado no coincide con MANIFEST.md5"; exit 1; }
V1CKS=$(ls "$V1"/ppo_s[0-4]/final.pt | paste -sd, -)
log "PPO_FOE_CHECKPOINTS=$V1CKS"

OMP_NUM_THREADS=4 python -m pytest -q -p no:cacheprovider tests/test_ppo_v2.py tests/test_joint_archetype.py \
  tests/test_kickoff_report.py tests/test_rl_symmetry.py > "$OUT/tests0.txt" 2>&1
rc=$?; log "0 tests rc=$rc ($(tail -1 "$OUT/tests0.txt"))"
[ $rc = 0 ] || { log "ERROR: tests fallan; se detiene la cola"; log "KICKOFF_QUEUE24_DONE"; exit 1; }

diag() {  # nombre checkpoint grupo [VARIABLE=valor...] [--opciones...]
  local name=$1 ck=$2 split=$3; shift 3
  [ -f "$ck" ] || { log "diag $name: falta $ck"; return; }
  local envs=() opts=()
  for x in "$@"; do case "$x" in --*) opts+=("$x");; *) envs+=("$x");; esac; done
  env PS_TEAM_SPLIT=$split ${envs[@]+"${envs[@]}"} python -m showdown.rl.ddqn.policy_diag --agent ppo \
    --checkpoint "$ck" --games 200 --port 8424 --agent-team $TEAM --preview-checkpoint "$PV" \
    ${opts[@]+"${opts[@]}"} --out "$OUT/diag/$name.json" > "$OUT/diag/$name.txt" 2>&1
  log "diag $name rc=$?"; clean_ps
}

# ---------------------------------------------------------------- referencia v1
for s in 0 1 2 3 4; do diag v1_s${s}_val "$V1/ppo_s$s/final.pt" val; done
for s in 0 1; do
  diag v1_s${s}_arq2_val "$V1/ppo_s$s/final.pt" val --foes=greedy_joint_arq --context
  diag v1_s${s}_arq2_eval "$V1/ppo_s$s/final.pt" eval --foes=greedy_joint_arq --context
done

# ---------------------------------------------------------------- P1: sorteo
for T in 1.0 0.5; do
  for s in 0 1 2 3 4; do
    C="$V1/ppo_s$s/final.pt"
    diag p1_T${T}_s${s}_val "$C" val --sample --temperature=$T
    diag p1_T${T}_s${s}_eval "$C" eval --sample --temperature=$T
    diag p1_T${T}_s${s}_arq1 "$C" eval PS_JOINT_ARCHETYPE=1 --foes=greedy_joint --context --sample --temperature=$T
  done
done
log "P1 terminado"

# ---------------------------------------------------------------- P2, P4, P3
BASE_A="random:150000,greedy_solo:250000"
train() {  # nombre semilla fases [VARIABLE=valor...] [--opciones...]
  local name=$1 s=$2 phases=$3; shift 3
  local envs=() opts=()
  for x in "$@"; do case "$x" in --*) opts+=("$x");; *) envs+=("$x");; esac; done
  env ${envs[@]+"${envs[@]}"} python -m showdown.rl.ppo.train --seed $s --device cuda --start-ps --expand-omega \
    --num-envs 32 --envs-per-server 4 --log-every 10000 --log-every-sec 120 --phases "$phases" \
    --eval-every 250000 --eval-games 50 --suite-games 0 --agent-team $TEAM --preview-checkpoint "$PV" \
    ${opts[@]+"${opts[@]}"} --checkpoint-dir "$OUT/${name}_s$s" > "$OUT/${name}_s$s.txt" 2>&1
  log "${name}_s$s rc=$?"; clean_ps
  local C="$OUT/${name}_s$s/final.pt"
  diag ${name}_s${s}_val "$C" val
  diag ${name}_s${s}_eval "$C" eval
  diag ${name}_s${s}_arq1_eval "$C" eval PS_JOINT_ARCHETYPE=1 --foes=greedy_joint --context
  diag ${name}_s${s}_arq2_val "$C" val --foes=greedy_joint_arq --context
  diag ${name}_s${s}_arq2_eval "$C" eval --foes=greedy_joint_arq --context
  diag ${name}_s${s}_h2h "$C" eval PPO_FOE_CHECKPOINTS="$V1CKS" --foes=ppo_v1
}
P2_PHASES="$BASE_A,greedy_joint:150000,greedy_joint_arq:150000,expectimax_bayesian:300000"
P3_PHASES="$BASE_A,greedy_joint:300000,expectimax_bayesian:150000,ppo_v1:150000"
for s in 0 1; do train p2 $s "$P2_PHASES"; done
for s in 0 1; do train p4 $s "$P2_PHASES" --ent-coef=0.03; done
for s in 0 1; do train p3 $s "$P3_PHASES" PPO_FOE_CHECKPOINTS="$V1CKS"; done

python -m showdown.rl.kickoff_summary24 "$OUT" "$V1" > "$OUT/summary24.err" 2>&1
log "resumen rc=$?"
log "KICKOFF_QUEUE24_DONE"
