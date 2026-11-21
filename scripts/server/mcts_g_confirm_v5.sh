#!/bin/bash
# CONFIRMACIÓN DE MCTS+PUCT v5 (agente G, fase 3c): hipótesis W1–W4 + control W5, pre-registradas el
# 21-11-2026 (docs/protocolo_evaluacion.md §4.3). Solo baselines. v6 (terreno) no mejoró a v5.
#
#   W1 v5 > greedy_joint · W2 v5 > expectimax · W3 v5 > greedy_solo   (binomial, C7 + C8)
#   W4 v5 > v4, ambos contra greedy_joint, mismas series              (McNemar, C7 + C8)
#   Holm m = 4.  W5 v5 > random (C7, control).  C7 = 9801–9817, C8 = 9901–9917. Preview greedy.
# Caché de la calculadora por corrida y borrada al terminar (regla del usuario: nunca > 250 GB).
# Resumen: $OUT/confirm_summary.txt. Marca: MCTS_G_CONFIRM_V5_DONE.
set -u
OUT=${1:-$HOME/iapsvgc_runs/mcts_g_confirm_v5}
FROZEN=${FROZEN:-cce1356}
WAIT_FILE=${WAIT_FILE:-$HOME/iapsvgc_runs/mcts_g_v6/queue.log}
WAIT_PAT=${WAIT_PAT:-MCTS_G_V6_DONE}
PROCS=${PROCS:-28}
mkdir -p "$OUT"/reports "$OUT"/raw
source ~/miniconda3/etc/profile.d/conda.sh; conda activate kuroneko
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
export PYTHONUNBUFFERED=1
unset MCTS_PUCT_VERSION MCTS_SIMS MCTS_SEED MCTS_S3_CALC MCTS_PREVIEW MCTS_PREVIEW_CKPT PS_TEAM_SPLIT PS_PREVIEW_A \
  PS_PREVIEW_B PS_B_VALIDATED PS_SP_INFO_A PS_SP_INFO_B MCTS_STATS_FILE RL_OBS_ABILITIES RL_OBS_CALC PREVIEW_FORCE_ORDER
log() { echo "$(date '+%F %T') $*" >> "$OUT/queue.log"; }
clean_ps() { pkill -f "pokemon-showdown.*start" 2>/dev/null; sleep 5; }

# run <config> <seed> <version> <player_b> [args...]
run() {
  local cfg=$1 seed=$2 ver=$3 pb=$4; shift 4
  local rpt="$OUT/reports/${cfg}_s${seed}.txt"
  if [ -s "$rpt" ]; then log "$cfg s$seed ya hecho (reanudación)"; return 0; fi
  rm -f logs/robust_e2e_report.txt
  export DAMAGE_CALC_CACHE="$OUT/calc_cache_run.sqlite"
  rm -f "$DAMAGE_CALC_CACHE" "$DAMAGE_CALC_CACHE"-wal "$DAMAGE_CALC_CACHE"-shm
  MCTS_PUCT_VERSION=$ver python -m showdown.inference_sim.run_robust_e2e --teams 300 --procs "$PROCS" \
    --seed "$seed" --restart-ps --player-a mcts_puct --player-b "$pb" --preview-a greedy "$@" \
    > "$OUT/raw/${cfg}_s${seed}.txt" 2>&1
  local rc=$?
  rm -f "$DAMAGE_CALC_CACHE" "$DAMAGE_CALC_CACHE"-wal "$DAMAGE_CALC_CACHE"-shm
  if [ ! -s logs/robust_e2e_report.txt ]; then
    log "ERROR: $cfg s$seed rc=$rc sin reporte (ver raw/${cfg}_s${seed}.txt)"; clean_ps; return 1
  fi
  cp logs/robust_e2e_report.txt "$rpt.tmp" && mv "$rpt.tmp" "$rpt"
  log "$cfg s$seed rc=$rc $(grep -E '^win_rate_player_a|^player_a_wins|^series_errors' "$rpt" | tr '\n' ' ')"
  clean_ps
  return $rc
}

log "esperando /$WAIT_PAT/ en $WAIT_FILE"
until grep -qE "$WAIT_PAT" "$WAIT_FILE" 2>/dev/null; do sleep 60; done
clean_ps
HEAD_C=$(git rev-parse --short HEAD)
log "inicio commit $HEAD_C (congelado $FROZEN)"
if ! git diff --quiet "$FROZEN" HEAD -- showdown/rl/mcts_puct showdown/battle_bots/greedy_joint \
     showdown/battle_bots/greedy_solo showdown/battle_bots/expectimax_bayesian; then
  log "ERROR: el código congelado cambió desde $FROZEN; no se confirma (pre-registro §4.3)"
  log "MCTS_G_CONFIRM_V5_DONE"; exit 1
fi
OMP_NUM_THREADS=4 python -m pytest -q -p no:cacheprovider tests/test_mcts_puct_s3c.py tests/test_mcts_puct_s3b.py \
  tests/test_mcts_puct_s3.py > "$OUT/tests.txt" 2>&1
rc=$?; log "tests rc=$rc ($(tail -1 "$OUT/tests.txt"))"
[ $rc = 0 ] || { log "ERROR: tests fallan; se detiene"; log "MCTS_G_CONFIRM_V5_DONE"; exit 1; }

PROCS_SAVE=$PROCS; PROCS=4
rm -f "$OUT/anomalies_smoke.tsv"
MCTS_STATS_FILE="$OUT/anomalies_smoke.tsv" run smoke 9001 5 greedy_joint --teams 20 --timeout 1200
PROCS=$PROCS_SAVE
SERR=$(grep -oE '^series_errors: *[0-9]+' "$OUT/reports/smoke_s9001.txt" 2>/dev/null | grep -oE '[0-9]+$')
NANOM=$(cat "$OUT/anomalies_smoke.tsv" 2>/dev/null | wc -l)
log "smoke: series_errors=${SERR:-?} anomalías=${NANOM}"
if [ "${SERR:-1}" != 0 ] || [ "$NANOM" != 0 ]; then
  log "ERROR: smoke con errores o anomalías; se detiene"; log "MCTS_G_CONFIRM_V5_DONE"; exit 1
fi
export MCTS_STATS_FILE="$OUT/anomalies.tsv"

for blk in 98 99; do
  for s in $(seq ${blk}01 ${blk}17); do run W1 "$s" 5 greedy_joint --timeout 1200; done
  for s in $(seq ${blk}01 ${blk}17); do run W4ref "$s" 4 greedy_joint --timeout 1200; done
  for s in $(seq ${blk}01 ${blk}17); do run W2 "$s" 5 expectimax_bayesian --timeout 2400; done
  for s in $(seq ${blk}01 ${blk}17); do run W3 "$s" 5 greedy_solo --timeout 1200; done
  if [ $blk = 98 ]; then
    for s in $(seq 9801 9817); do run W5 "$s" 5 random --timeout 1200; done
  fi
done

S="$OUT/confirm_summary.txt"
{
  echo "=== Confirmación MCTS+PUCT v5 (G) — commit $HEAD_C (código congelado $FROZEN) ==="
  echo "Pre-registro: docs/protocolo_evaluacion.md §4.3. Preview greedy."
  echo "Anomalías del player: $(cut -f1 "$OUT/anomalies.tsv" 2>/dev/null | sort | uniq -c | tr '\n' ' ')"
  PS_=""
  for h in W1 W2 W3; do
    echo; echo "--- $h C7+C8"
    line=$(python -m showdown.inference_sim.confirm_stats binom "$OUT"/raw/${h}_s98*.txt "$OUT"/raw/${h}_s99*.txt)
    echo "$line"; PS_="$PS_ $(echo "$line" | grep -oE ' p=[^ ]+' | head -1 | cut -d= -f2)"
    echo "--- $h C7"; python -m showdown.inference_sim.confirm_stats binom "$OUT"/raw/${h}_s98*.txt
    echo "--- $h C8"; python -m showdown.inference_sim.confirm_stats binom "$OUT"/raw/${h}_s99*.txt
  done
  echo; echo "--- W4 C7+C8 (McNemar v5 frente a v4, ambos contra joint)"
  line=$(python -m showdown.inference_sim.confirm_stats mcnemar --a "$OUT"/raw/W1_s9[89]*.txt --b "$OUT"/raw/W4ref_s9[89]*.txt)
  echo "$line"; PS_="$PS_ $(echo "$line" | grep -oE ' p=[^ ]+' | head -1 | cut -d= -f2)"
  echo "--- W4 C7"; python -m showdown.inference_sim.confirm_stats mcnemar --a "$OUT"/raw/W1_s98*.txt --b "$OUT"/raw/W4ref_s98*.txt
  echo "--- W4 C8"; python -m showdown.inference_sim.confirm_stats mcnemar --a "$OUT"/raw/W1_s99*.txt --b "$OUT"/raw/W4ref_s99*.txt
  echo "--- v4 contra joint C7+C8 (referencia)"; python -m showdown.inference_sim.confirm_stats binom "$OUT"/raw/W4ref_s9[89]*.txt
  echo; echo "--- Holm (m=4; p de W1, W2, W3, W4 en ese orden)"; python -m showdown.inference_sim.confirm_stats holm $PS_
  echo; echo "--- W5 C7 (control contra random)"; python -m showdown.inference_sim.confirm_stats binom "$OUT"/raw/W5_s98*.txt
} > "$S" 2>&1
log "estadística escrita en $S"
log "MCTS_G_CONFIRM_V5_DONE"
