#!/bin/bash
# Cola nocturna previa al kick-off de DDQN (etapas 1–6 del checklist).
# Cada etapa escribe sus resultados en $OUT; una etapa que falla NO detiene las siguientes.
# Al final se genera $OUT/kickoff_summary.md (showdown.rl.ddqn.kickoff_summary).
#
#   bash scripts/server/kickoff_queue.sh ~/iapsvgc_runs/kickoff
set -u
OUT=${1:-$HOME/iapsvgc_runs/kickoff}
mkdir -p "$OUT"/tune "$OUT"/ablation
source ~/miniconda3/etc/profile.d/conda.sh; conda activate kuroneko
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
export PYTHONUNBUFFERED=1 PS_RL_STRICT=1
log() { echo "$(date '+%F %T') $*" >> "$OUT/queue.log"; }
clean_ps() { pkill -f "pokemon-showdown.*start" 2>/dev/null; sleep 5; }
log "inicio commit $(git rev-parse --short HEAD)"

TEAM=showdown/rl/data/ddqn_team.txt
COMMON="--seed 0 --device cuda --start-ps --expand-omega --num-envs 32 --envs-per-server 4 --log-every 10000 --log-every-sec 120"

# ---------------------------------------------------------------- 1. reanudación
mkdir -p "$OUT/resume"
python -m showdown.rl.ddqn.train $COMMON --phases random:6000,greedy_solo:6000 --eval-every 6000 \
  --eval-games 10 --suite-games 0 --checkpoint-dir "$OUT/resume/ck" > "$OUT/resume/train1.txt" 2>&1
python -m showdown.rl.ddqn.train $COMMON --phases random:6000,greedy_solo:12000 --eval-every 6000 \
  --eval-games 10 --suite-games 0 --checkpoint-dir "$OUT/resume/ck" --resume "$OUT/resume/ck/final.pt" \
  > "$OUT/resume/train2.txt" 2>&1
log "1 reanudación rc=$?"; clean_ps

# ---------------------------------------------------------------- 2. auditoría de la observación
python -m showdown.rl.obs_audit --episodes 25 --port 8360 --agent-team $TEAM --out "$OUT/obs_audit.json" \
  > "$OUT/obs_audit.txt" 2>&1
log "2 auditoría rc=$?"; clean_ps

# ---------------------------------------------------------------- 3. probe del preview
python -m showdown.inference_sim.run_preview_value_probe --n 3000 --num-workers 28 --envs-per-server 4 \
  --start-ps --out "$OUT/preview_value_probe.json" > "$OUT/preview_probe.txt" 2>&1
log "3 probe rc=$?"; clean_ps

# ---------------------------------------------------------------- 4. preview PPO
python -m showdown.rl.team_preview_ppo.train --seed 0 --device cuda --backend live --num-workers 28 \
  --envs-per-server 4 --start-ps --pilot greedy_joint --updates 20 --eval-every 5 \
  --eval-periodic-episodes 256 --checkpoint-dir "$OUT/preview_ppo" > "$OUT/preview_ppo.txt" 2>&1
log "4 preview PPO rc=$?"; clean_ps
PREVIEW=""
for f in "$OUT/preview_ppo/best.pt" "$OUT/preview_ppo/final.pt"; do
  if [ -f "$f" ]; then PREVIEW="--preview-checkpoint $f"; echo "$f" > "$OUT/preview_used.txt"; break; fi
done
[ -z "$PREVIEW" ] && echo "(ninguno: /team 1234)" > "$OUT/preview_used.txt"

# ---------------------------------------------------------------- 5. tuning de DDQN
TUNE="--phases random:150000,greedy_solo:250000 --eval-every 200000 --eval-games 50 --suite-games 0"
tune() {  # nombre flags...
  local name=$1; shift
  python -m showdown.rl.ddqn.train $COMMON $TUNE $PREVIEW --checkpoint-dir "$OUT/tune/$name" "$@" \
    > "$OUT/tune/$name.txt" 2>&1
  log "5 tuning $name rc=$?"; clean_ps
}
tune T0_base
tune T1_lr1e-4 --lr 1e-4
tune T2_lr6e-4 --lr 6e-4
tune T3_gamma0.99 --gamma 0.99
tune T4_hard2000 --target-update-every 2000
tune T5_layernorm --layer-norm
tune T6_trainfreq2 --train-freq 2
tune T7_batch512 --batch-size 512

# ---------------------------------------------------------------- 6. ablación codificador × simetría
abl() {  # celda semilla env...
  local cell=$1 seed=$2; shift 2
  env "$@" python -m showdown.rl.ddqn.train $COMMON $TUNE $PREVIEW --seed $seed \
    --checkpoint-dir "$OUT/ablation/${cell}_s${seed}" > "$OUT/ablation/${cell}_s${seed}.txt" 2>&1
  log "6 ablación ${cell} s${seed} rc=$?"; clean_ps
}
# (estructurado, simetría, s0) = T0_base: se reutiliza
mkdir -p "$OUT/ablation"; ln -sfn "$OUT/tune/T0_base" "$OUT/ablation/struct_sym_s0"
for s in 1 2; do abl struct_sym $s RL_SYM_AUG=1; done
for s in 0 1 2; do abl struct_nosym $s RL_SYM_AUG=0; done
for s in 0 1 2; do abl flat_sym $s RL_OBS_ENCODER=flat; done
for s in 0 1 2; do abl flat_nosym $s RL_OBS_ENCODER=flat RL_SYM_AUG=0; done

# ---------------------------------------------------------------- resumen
python -m showdown.rl.ddqn.kickoff_summary "$OUT" > /dev/null 2>&1
log "resumen rc=$?"
log "KICKOFF_QUEUE_DONE"
