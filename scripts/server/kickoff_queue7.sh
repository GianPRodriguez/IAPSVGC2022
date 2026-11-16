#!/bin/bash
# Séptima cola (16-11-2026): teorías sobre los ataques de daño al propio aliado (~7 %) y los
# cambios voluntarios (~7 %) de DDQN, y validez del preview v2 para otros combates.
# Espera a KICKOFF_QUEUE6_DONE. Resumen: $OUT/kickoff_summary7.md.
#
#   bash scripts/server/kickoff_queue7.sh ~/iapsvgc_runs/kickoff7 ~/iapsvgc_runs ~/iapsvgc_runs/kickoff6
set -u
OUT=${1:-$HOME/iapsvgc_runs/kickoff7}
R=${2:-$HOME/iapsvgc_runs}
Q6=${3:-$HOME/iapsvgc_runs/kickoff6}
mkdir -p "$OUT"/diag
source ~/miniconda3/etc/profile.d/conda.sh; conda activate kuroneko
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
export PYTHONUNBUFFERED=1 PS_RL_STRICT=1
log() { echo "$(date '+%F %T') $*" >> "$OUT/queue.log"; }
clean_ps() { pkill -f "pokemon-showdown.*start" 2>/dev/null; sleep 5; }
log "esperando a la sexta cola ($Q6)"
until grep -q KICKOFF_QUEUE6_DONE "$Q6/queue.log" 2>/dev/null; do sleep 60; done
clean_ps
log "inicio commit $(git rev-parse --short HEAD)"
TEAM=showdown/rl/data/ddqn_team.txt
PV1="$(cat "$R/kickoff/preview_used.txt" 2>/dev/null)"; [ -f "$PV1" ] || PV1=""
PV2="$(cat "$R/kickoff2/preview_best.txt" 2>/dev/null)"; [ -f "$PV2" ] || PV2=""

diag() {  # nombre checkpoint preview partidas
  local name=$1 ck=$2 pv=$3 games=${4:-25}
  [ -f "$ck" ] || { log "diag $name: falta $ck"; return; }
  python -m showdown.rl.ddqn.policy_diag --checkpoint "$ck" --games $games --port 8390 --agent-team $TEAM \
    ${pv:+--preview-checkpoint "$pv"} --out "$OUT/diag/$name.json" > "$OUT/diag/$name.txt" 2>&1
  log "diag $name rc=$?"; clean_ps
}
pick() { for f in "$@"; do [ -f "$f" ] && { echo "$f"; return; }; done; echo ""; }

# ---------------------------------------------------------------- D1. ¿bajan con el entrenamiento?
diag D1_T1_200k "$R/kickoff/tune/T1_lr1e-4/step_200000.pt" "$PV1"
diag D1_T1_400k "$(pick "$R/kickoff/tune/T1_lr1e-4/step_400000.pt" "$R/kickoff/tune/T1_lr1e-4/final.pt")" "$PV1"
MPV=$(sed -n 's/.*preview=\([^ ]*\).*/\1/p' "$R/kickoff3/mini_choice.txt" 2>/dev/null)
for s in 250000 500000 750000 1000000; do
  diag D1_mini_${s} "$(pick "$R/kickoff3/mini_kickoff/step_${s}.pt" "$R/kickoff3/mini_kickoff/final.pt")" "$MPV"
done

# ---------------------------------------------------------------- D2. ¿tiene que ver la simetría?
for s in 0 1 2; do
  diag D2_sym_s$s "$(pick "$R/kickoff/ablation/struct_sym_s$s/step_400000.pt" "$R/kickoff/ablation/struct_sym_s$s/final.pt")" "$PV1"
  diag D2_nosym_s$s "$(pick "$R/kickoff/ablation/struct_nosym_s$s/step_400000.pt" "$R/kickoff/ablation/struct_nosym_s$s/final.pt")" "$PV1"
done

# ---------------------------------------------------------------- D3. preview v1 frente a v2 con combate DDQN
CK3="$(pick "$R/kickoff/tune/T1_lr1e-4/best.pt" "$R/kickoff/tune/T1_lr1e-4/final.pt")"
diag D3_preview_v1 "$CK3" "$PV1" 50
diag D3_preview_v2 "$CK3" "$PV2" 50

# ---------------------------------------------------------------- D4. preview v2 con expectimax en el combate
if [ -n "$PV2" ]; then
  python -m showdown.rl.preview_check --checkpoint "$PV2" --own-team $TEAM --n 256 --num-workers 28 \
    --start-ps --pilot expectimax_bayesian --skip-replays --out-dir "$OUT/preview_check_expectimax" \
    > "$OUT/preview_check_expectimax.txt" 2>&1
  log "D4 preview con expectimax rc=$?"; clean_ps
fi

# ---------------------------------------------------------------- P5 / P6. pilotos A/B (config de T0_base)
TUNE="--seed 0 --device cuda --start-ps --expand-omega --num-envs 32 --envs-per-server 4 --log-every 10000 --log-every-sec 120 --phases random:150000,greedy_solo:250000 --eval-every 200000 --eval-games 50 --suite-games 0"
RL_REWARD_HP_SELF=1 python -m showdown.rl.ddqn.train $TUNE ${PV1:+--preview-checkpoint "$PV1"} \
  --checkpoint-dir "$OUT/P5_reward_hp_self" > "$OUT/P5_reward_hp_self.txt" 2>&1
log "P5 recompensa simétrica rc=$?"; clean_ps
diag P5_final "$(pick "$OUT/P5_reward_hp_self/final.pt")" "$PV1"
RL_MASK_ALLY_DAMAGE=1 python -m showdown.rl.ddqn.train $TUNE ${PV1:+--preview-checkpoint "$PV1"} \
  --checkpoint-dir "$OUT/P6_mask_ally_damage" > "$OUT/P6_mask_ally_damage.txt" 2>&1
log "P6 máscara de daño al aliado rc=$?"; clean_ps
RL_MASK_ALLY_DAMAGE=1 diag P6_final "$(pick "$OUT/P6_mask_ally_damage/final.pt")" "$PV1"
diag T0_base_final "$(pick "$R/kickoff/tune/T0_base/step_400000.pt" "$R/kickoff/tune/T0_base/final.pt")" "$PV1"

python -m showdown.rl.ddqn.kickoff_summary7 "$OUT" "$R" > /dev/null 2>&1
log "resumen rc=$?"
log "KICKOFF_QUEUE7_DONE"
