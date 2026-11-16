#!/bin/bash
# Quinta cola previa al kick-off (16-11-2026): verificación del preview v2 y efecto de la
# aumentación por simetría a 1M pasos (DDQN) y en PPO. Espera a KICKOFF_QUEUE4_DONE.
# Resumen: $OUT/kickoff_summary5.md.
#
#   bash scripts/server/kickoff_queue5.sh ~/iapsvgc_runs/kickoff5 ~/iapsvgc_runs/kickoff2 ~/iapsvgc_runs/kickoff3 ~/iapsvgc_runs/kickoff4
set -u
OUT=${1:-$HOME/iapsvgc_runs/kickoff5}
Q2=${2:-$HOME/iapsvgc_runs/kickoff2}
Q3=${3:-$HOME/iapsvgc_runs/kickoff3}
Q4=${4:-$HOME/iapsvgc_runs/kickoff4}
mkdir -p "$OUT"
source ~/miniconda3/etc/profile.d/conda.sh; conda activate kuroneko
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
export PYTHONUNBUFFERED=1 PS_RL_STRICT=1
log() { echo "$(date '+%F %T') $*" >> "$OUT/queue.log"; }
clean_ps() { pkill -f "pokemon-showdown.*start" 2>/dev/null; sleep 5; }
log "esperando a la cuarta cola ($Q4)"
until grep -q KICKOFF_QUEUE4_DONE "$Q4/queue.log" 2>/dev/null; do sleep 60; done
clean_ps
log "inicio commit $(git rev-parse --short HEAD)"
TEAM=showdown/rl/data/ddqn_team.txt
PV="$(cat "$Q2/preview_best.txt" 2>/dev/null)"
[ -f "$PV" ] || PV=""

# ---------------------------------------------------------------- 1. verificación del preview v2
if [ -n "$PV" ]; then
  python -m showdown.rl.preview_check --checkpoint "$PV" --own-team $TEAM --n 512 --replays-n 12 \
    --num-workers 28 --start-ps --out-dir "$OUT/preview_check" > "$OUT/preview_check.txt" 2>&1
  log "1 verificación del preview rc=$?"
else
  log "1 verificación del preview: sin checkpoint v2"
fi
clean_ps

# ---------------------------------------------------------------- 2. DDQN 1M pasos sin simetría
# Misma configuración que el mini kick-off de la cola 3 (lr, lote, preview), solo cambia la
# aumentación por simetría: aísla su efecto con el currículo completo y 1M pasos.
CHOICE="$(cat "$Q3/mini_choice.txt" 2>/dev/null)"
LR=$(echo "$CHOICE" | sed -n 's/.*lr=\([^ ]*\).*/\1/p')
BS=$(echo "$CHOICE" | sed -n 's/.*batch=\([^ ]*\).*/\1/p')
MPV=$(echo "$CHOICE" | sed -n 's/.*preview=\([^ ]*\).*/\1/p')
echo "lr=${LR:-1e-4} batch=${BS:-256} preview=${MPV} RL_SYM_AUG=0" > "$OUT/mini_nosym_config.txt"
RL_SYM_AUG=0 python -m showdown.rl.ddqn.train --seed 0 --device cuda --start-ps --expand-omega \
  --num-envs 32 --envs-per-server 4 --log-every 10000 --log-every-sec 120 \
  --phases random:150000,greedy_solo:250000,greedy_joint:300000,expectimax_bayesian:300000 \
  --eval-every 250000 --eval-games 50 --suite-games 0 --lr ${LR:-1e-4} --batch-size ${BS:-256} \
  ${MPV:+--preview-checkpoint "$MPV"} --checkpoint-dir "$OUT/mini_kickoff_nosym" > "$OUT/mini_kickoff_nosym.txt" 2>&1
log "2 DDQN 1M sin simetría rc=$?"; clean_ps

# ---------------------------------------------------------------- 3. PPO sin simetría
# Pareado con el piloto de PPO de la cola 2 (misma configuración y preview, simetría activada).
RL_SYM_AUG=0 python -m showdown.rl.ppo.train --seed 0 --device cuda --start-ps --expand-omega --num-envs 32 \
  --envs-per-server 4 --phases random:150000,greedy_solo:250000 --suite-games 100 \
  --log-every 10000 --log-every-sec 120 ${PV:+--preview-checkpoint "$PV"} --checkpoint-dir "$OUT/ppo_pilot_nosym" \
  > "$OUT/ppo_pilot_nosym.txt" 2>&1
log "3 PPO sin simetría rc=$?"; clean_ps

python -m showdown.rl.ddqn.kickoff_summary5 "$OUT" "$Q2" "$Q3" > /dev/null 2>&1
log "resumen rc=$?"
log "KICKOFF_QUEUE5_DONE"
