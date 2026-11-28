#!/bin/bash
# PPO Y DDQN ENTRENADOS CONTRA GREEDY_JOINT NIVEL 3 (protocolo §4.20, aprobado por el usuario el 28-11-2026;
# reconsideración de P4 con la evidencia de D3, §5.17). De aquí salen las versiones oficiales PPO v3 / DDQN v2.
#
#   Brazo L3: igual que el brazo C de la cola v3 (versión anterior + 1M pasos en series BO3, máscara del Protect
#   consecutivo, sin memoria, arranque en caliente) salvo la mezcla de rivales, que incluye el nivel 3:
#     random:150k, greedy_solo:200k, greedy_joint:200k, greedy_joint_arq3:200k, expectimax_bayesian:250k
#     (C usó random:150k, greedy_solo:250k, greedy_joint:300k, expectimax_bayesian:300k), --expand-omega.
#   El brazo C se reutiliza de la cola v3 (~/iapsvgc_runs/v3/<fam>_C_s<k>/final.pt): misma receta salvo los rivales.
#   Semillas 0–2 por familia; entrenamientos de uno en uno.
#   Evaluación: L3, C y la versión anterior (prev), policy_diag sobre eval, 200 partidas por rival (random, solo,
#   joint, expectimax y nivel 3), --context, máscara activada, de a $JOBS en paralelo (un puerto cada una).
#   Informe: scripts/analysis/l3_report.py -> $OUT/l3_report.md. Marca: L3_DONE.
#
#   JOBS=6 bash scripts/server/l3_queue.sh ~/iapsvgc_runs/l3
set -u
OUT=${1:-$HOME/iapsvgc_runs/l3}
JOBS=${JOBS:-6}
V3=${V3:-$HOME/iapsvgc_runs/v3}
GHLOG=${GHLOG:-$HOME/iapsvgc_runs/gh_arq/queue.log}
PV=${PV:-$HOME/iapsvgc_runs/preview_v21b/preview_s0/best.pt}
DDQN1=${DDQN1:-$HOME/iapsvgc_runs/kickoff_ddqn_v2}
mkdir -p "$OUT"/diag
source ~/miniconda3/etc/profile.d/conda.sh; conda activate kuroneko
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
PPO2=${PPO2:-$REPO/checkpoints/ppo_v2}
export PYTHONUNBUFFERED=1 PS_RL_STRICT=1
unset RL_MASK_ALLY_DAMAGE RL_SYM_AUG RL_HOLDOUT_VAL RL_TRAIN_SUBSET RL_FOE_SYNTH PREVIEW_FORCE_ORDER PS_JOINT_ARCHETYPE \
  PPO_FOE_CHECKPOINTS PPO_FOE_PREVIEW RL_OBS_ABILITIES RL_OBS_CALC PREVIEW_SAMPLE PREVIEW_TEMPERATURE PS_TEAM_SPLIT \
  RL_OBS_SERIES_MEMORY RL_MASK_PROTECT_STREAK RL_SERIES_BO3 PREVIEW_SERIES_CONDITION PREVIEW_KIND
log() { echo "$(date '+%F %T') $*" >> "$OUT/queue.log"; }
clean_ps() { pkill -f "[p]okemon-showdown.*start" 2>/dev/null; sleep 5; }
kill_port() {
  pkill -f "[p]okemon-showdown start.* $1\$" 2>/dev/null
  pkill -f "[p]s_seeded_start.js.* $1\$" 2>/dev/null
}
TEAM=showdown/rl/data/ddqn_team.txt
PHASES=random:150000,greedy_solo:200000,greedy_joint:200000,greedy_joint_arq3:200000,expectimax_bayesian:250000

log "esperando el fin de la cola de G y H ($GHLOG)"
until grep -v "vigía caché" "$GHLOG" 2>/dev/null | tail -n 1 | grep -q " GH_ARQ_DONE\$"; do sleep 120; done
[ -f "$PV" ] || { log "ERROR: falta el preview $PV"; log "L3_DONE"; exit 1; }
log "inicio commit $(git rev-parse --short HEAD); JOBS=$JOBS; fases L3 $PHASES"
clean_ps
pkill -f "calc_serve[r].js" 2>/dev/null; sleep 2

OMP_NUM_THREADS=4 python -m pytest -q -p no:cacheprovider tests/test_l3_report.py tests/test_joint_archetype_v3.py \
  tests/test_series_memory_v3.py tests/test_rl_ppo.py tests/test_rl_ddqn.py tests/test_ppo_v2.py > "$OUT/tests.txt" 2>&1
rc=$?; log "tests rc=$rc ($(tail -1 "$OUT/tests.txt"))"
[ $rc = 0 ] || { log "ERROR: tests fallan; se detiene"; log "L3_DONE"; exit 1; }

prev_ck() { if [ "$1" = ppo ]; then echo "$PPO2/ppo_s$2/final.pt"; else echo "$DDQN1/ddqn_s$2/final.pt"; fi; }

train() {  # familia semilla
  local fam=$1 s=$2 dir="$OUT/${1}_L3_s${2}"
  if [ -f "$dir/final.pt" ]; then log "${fam}_L3_s$s ya entrenado (reanudación)"; return; fi
  local ck; ck=$(prev_ck "$fam" "$s")
  [ -f "$ck" ] || { log "ERROR ${fam}_L3_s$s: falta $ck"; return; }
  local extra=(); [ "$fam" = ddqn ] && extra=(--lr 1e-4 --batch-size 256)
  local t0=$SECONDS
  python -m showdown.rl.$fam.train --seed "$s" --device cuda --start-ps --expand-omega --num-envs 32 \
    --envs-per-server 4 --log-every 10000 --log-every-sec 120 --phases "$PHASES" \
    --eval-every 250000 --eval-games 50 --suite-games 0 --agent-team $TEAM --preview-checkpoint "$PV" \
    --obs-abilities 1 --series-bo3 1 --mask-protect-streak 1 --series-memory 0 --init-from "$ck" \
    ${extra[@]+"${extra[@]}"} --checkpoint-dir "$dir" > "$dir.txt" 2>&1
  log "${fam}_L3_s$s rc=$? $(( (SECONDS - t0) / 60 )) min"; clean_ps
  grep -h "arranque en caliente" "$dir.txt" 2>/dev/null | tail -1 | cut -c1-200 | sed "s/^/  ${fam}_L3_s$s: /" >> "$OUT/queue.log"
}
for fam in ppo ddqn; do for s in 0 1 2; do train $fam $s; done; done
log "entrenamientos terminados"

diag_port() {  # puerto nombre familia checkpoint
  local port=$1 name=$2 fam=$3 ck=$4
  [ -s "$OUT/diag/$name.json" ] && { log "diag $name ya hecho (reanudación)"; return; }
  [ -f "$ck" ] || { log "diag $name: falta $ck"; return; }
  local t0=$SECONDS
  env PS_TEAM_SPLIT=eval python -m showdown.rl.ddqn.policy_diag --agent "$fam" --checkpoint "$ck" --port "$port" \
    --agent-team $TEAM --preview-checkpoint "$PV" --games 200 --context --mask-protect 1 \
    --foes random,greedy_solo,greedy_joint,expectimax_bayesian,greedy_joint_arq3 \
    --out "$OUT/diag/$name.json" > "$OUT/diag/$name.txt" 2>&1
  log "diag $name rc=$? puerto=$port $(( (SECONDS - t0) / 60 )) min"
  kill_port "$port"
}
declare -A PORT_PID
free_port() {
  local i p pid
  for ((i = 0; i < JOBS; i++)); do
    p=$((8450 + i)); pid=${PORT_PID[$p]:-}
    if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then echo "$p"; return; fi
  done
}
launch() {
  local p
  while :; do p=$(free_port); [ -n "$p" ] && break; wait -n 2>/dev/null || sleep 5; done
  diag_port "$p" "$@" &
  PORT_PID[$p]=$!
}
for fam in ppo ddqn; do
  for s in 0 1 2; do
    launch ev_${fam}_L3_s$s $fam "$OUT/${fam}_L3_s$s/final.pt"
    launch ev_${fam}_C_s$s $fam "$V3/${fam}_C_s$s/final.pt"
    launch ev_${fam}_prev_s$s $fam "$(prev_ck $fam $s)"
  done
done
wait
log "evaluaciones terminadas"
python scripts/analysis/l3_report.py "$OUT" > "$OUT/l3_report.err" 2>&1
log "informe rc=$?"
log "L3_DONE"
