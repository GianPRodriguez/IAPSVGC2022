#!/bin/bash
# CONFIRMACIÓN DE MCTS+PUCT v4 (agente G, fase 3b): hipótesis V1–V4 + control V5, pre-registradas el
# 20-11-2026 (docs/protocolo_evaluacion.md §4.2). Solo baselines. Decisión del usuario: encolar tras la
# cola del preview Nash (MCTS_G_NASH_DONE).
#
#   V1 v4 > greedy_joint · V2 v4 > expectimax · V3 v4 > greedy_solo   (binomial, C5 + C6)
#   V4 v4 > v3.1, ambos contra greedy_joint, mismas series            (McNemar, C5 + C6)
#   Holm m = 4.  V5 v4 > random (C5, control).
#   C5 = 9601–9617, C6 = 9701–9717.
# Preview (igual para v4 y v3.1): regla del diagnóstico §7 sobre $NASH/nash_summary.txt
# (nash_decision): Nash de la semilla adoptada o greedy.
# Caché de la calculadora por corrida y borrada al terminar (regla del usuario: nunca > 250 GB).
# Resumen: $OUT/confirm_summary.txt. Marca: MCTS_G_CONFIRM_V4_DONE.
set -u
OUT=${1:-$HOME/iapsvgc_runs/mcts_g_confirm_v4}
NASH=${NASH:-$HOME/iapsvgc_runs/mcts_g_nash}
FROZEN=${FROZEN:-0df75cb}
WAIT_FILE=${WAIT_FILE:-$NASH/queue.log}
WAIT_PAT=${WAIT_PAT:-MCTS_G_NASH_DONE}
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
    --seed "$seed" --restart-ps --player-a mcts_puct --player-b "$pb" "${PREV_ARGS[@]}" "$@" \
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
if ! git diff --quiet "$FROZEN" HEAD -- showdown/rl/mcts_puct showdown/rl/team_preview_nash showdown/rl/frozen_preview.py \
     showdown/battle_bots/greedy_joint showdown/battle_bots/greedy_solo showdown/battle_bots/expectimax_bayesian; then
  log "ERROR: el código congelado cambió desde $FROZEN; no se confirma (pre-registro §4.2)"
  log "MCTS_G_CONFIRM_V4_DONE"; exit 1
fi

# Preview según la regla del diagnóstico §7.
DEC=$(python -m showdown.inference_sim.mcts_g_summary --nash-decision "$NASH/nash_summary.txt" 2>/dev/null || echo greedy)
case "$DEC" in
  P1) CK="$NASH/preview_nash_s0/final.pt" ;;
  P2) CK="$NASH/preview_nash_s1/final.pt" ;;
  *)  CK="" ;;
esac
if [ -n "$CK" ] && [ -s "$CK" ]; then
  export MCTS_PREVIEW=nash MCTS_PREVIEW_CKPT="$CK"; PREV_ARGS=()
  log "preview: Nash ($DEC, $CK) según diagnóstico §7"
else
  PREV_ARGS=(--preview-a greedy)
  log "preview: greedy (decisión §7: $DEC)"
fi

OMP_NUM_THREADS=4 python -m pytest -q -p no:cacheprovider tests/test_mcts_puct_s3b.py tests/test_mcts_puct_s3.py \
  tests/test_mcts_g_summary.py > "$OUT/tests.txt" 2>&1
rc=$?; log "tests rc=$rc ($(tail -1 "$OUT/tests.txt"))"
[ $rc = 0 ] || { log "ERROR: tests fallan; se detiene"; log "MCTS_G_CONFIRM_V4_DONE"; exit 1; }

PROCS_SAVE=$PROCS; PROCS=4
rm -f "$OUT/anomalies_smoke.tsv"
MCTS_STATS_FILE="$OUT/anomalies_smoke.tsv" run smoke 9001 4 greedy_joint --teams 20 --timeout 1200
PROCS=$PROCS_SAVE
SERR=$(grep -oE '^series_errors: *[0-9]+' "$OUT/reports/smoke_s9001.txt" 2>/dev/null | grep -oE '[0-9]+$')
NANOM=$(cat "$OUT/anomalies_smoke.tsv" 2>/dev/null | wc -l)
log "smoke: series_errors=${SERR:-?} anomalías=${NANOM} ($(cut -f1 "$OUT/anomalies_smoke.tsv" 2>/dev/null | sort | uniq -c | tr '\n' ' '))"
if [ "${SERR:-1}" != 0 ] || [ "$NANOM" != 0 ]; then
  log "ERROR: smoke con errores o anomalías; se detiene"; log "MCTS_G_CONFIRM_V4_DONE"; exit 1
fi
export MCTS_STATS_FILE="$OUT/anomalies.tsv"

for blk in 96 97; do
  for s in $(seq ${blk}01 ${blk}17); do run V1 "$s" 4 greedy_joint --timeout 1200; done
  for s in $(seq ${blk}01 ${blk}17); do run V4ref "$s" 3 greedy_joint --timeout 1200; done
  for s in $(seq ${blk}01 ${blk}17); do run V2 "$s" 4 expectimax_bayesian --timeout 2400; done
  for s in $(seq ${blk}01 ${blk}17); do run V3 "$s" 4 greedy_solo --timeout 1200; done
  if [ $blk = 96 ]; then
    for s in $(seq 9601 9617); do run V5 "$s" 4 random --timeout 1200; done
  fi
done

S="$OUT/confirm_summary.txt"
{
  echo "=== Confirmación MCTS+PUCT v4 (G) — commit $HEAD_C (código congelado $FROZEN) ==="
  echo "Pre-registro: docs/protocolo_evaluacion.md §4.2. Preview: $DEC ${CK:-greedy}."
  echo "Anomalías del player: $(cut -f1 "$OUT/anomalies.tsv" 2>/dev/null | sort | uniq -c | tr '\n' ' ')"
  PS_=""
  for h in V1 V2 V3; do
    echo; echo "--- $h C5+C6"
    line=$(python -m showdown.inference_sim.confirm_stats binom "$OUT"/raw/${h}_s96*.txt "$OUT"/raw/${h}_s97*.txt)
    echo "$line"; PS_="$PS_ $(echo "$line" | grep -oE ' p=[^ ]+' | head -1 | cut -d= -f2)"
    echo "--- $h C5"; python -m showdown.inference_sim.confirm_stats binom "$OUT"/raw/${h}_s96*.txt
    echo "--- $h C6"; python -m showdown.inference_sim.confirm_stats binom "$OUT"/raw/${h}_s97*.txt
  done
  echo; echo "--- V4 C5+C6 (McNemar v4 frente a v3.1, ambos contra joint)"
  line=$(python -m showdown.inference_sim.confirm_stats mcnemar --a "$OUT"/raw/V1_s9[67]*.txt --b "$OUT"/raw/V4ref_s9[67]*.txt)
  echo "$line"; PS_="$PS_ $(echo "$line" | grep -oE ' p=[^ ]+' | head -1 | cut -d= -f2)"
  echo "--- V4 C5"; python -m showdown.inference_sim.confirm_stats mcnemar --a "$OUT"/raw/V1_s96*.txt --b "$OUT"/raw/V4ref_s96*.txt
  echo "--- V4 C6"; python -m showdown.inference_sim.confirm_stats mcnemar --a "$OUT"/raw/V1_s97*.txt --b "$OUT"/raw/V4ref_s97*.txt
  echo "--- v3.1 contra joint C5+C6 (referencia)"; python -m showdown.inference_sim.confirm_stats binom "$OUT"/raw/V4ref_s9[67]*.txt
  echo; echo "--- Holm (m=4; p de V1, V2, V3, V4 en ese orden)"; python -m showdown.inference_sim.confirm_stats holm $PS_
  echo; echo "--- V5 C5 (control contra random)"; python -m showdown.inference_sim.confirm_stats binom "$OUT"/raw/V5_s96*.txt
} > "$S" 2>&1
log "estadística escrita en $S"
log "MCTS_G_CONFIRM_V4_DONE"
