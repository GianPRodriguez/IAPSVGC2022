#!/bin/bash
# EXPLORACIÓN DE MCTS+PUCT v4 = S3 fase 3b (agente G, 20-11-2026; decisión del usuario: seguir con 3b).
# Spec v4.0 §4.3. Mismas semillas que v3 para comparar en pareado con v3_n200 (reportes y logs de
# V3OUT enlazados): Δ por semilla (mcts_g_summary) y McNemar por serie (confirm_stats).
# Solo baselines como rival (greedy_joint validado). Exploración: no respalda afirmaciones.
#
# 1. Tests de MCTS. 2. Smoke v4 (se detiene ante cualquier anomalía). 3. v4_n200, 9101–9117.
# Caché de la calculadora por corrida y borrada al terminar (regla del usuario: nunca > 250 GB).
# Resumen: $OUT/summary.md + $OUT/mcnemar.txt. Marca: MCTS_G_V4_DONE.
#
#   cd ~/iapsvgc_scratch/spinfo/IAPSVGC2022 && git pull
#   nohup setsid bash scripts/server/mcts_g_v4.sh ~/iapsvgc_runs/mcts_g_v4 > ~/iapsvgc_runs/mcts_g_v4.out 2>&1 < /dev/null &
set -u
OUT=${1:-$HOME/iapsvgc_runs/mcts_g_v4}
V3OUT=${V3OUT:-$HOME/iapsvgc_runs/mcts_g_v3}
PROCS=${PROCS:-28}
mkdir -p "$OUT"/reports "$OUT"/raw
source ~/miniconda3/etc/profile.d/conda.sh; conda activate kuroneko
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
export PYTHONUNBUFFERED=1
unset MCTS_SIMS MCTS_SEED MCTS_S3_CALC MCTS_PREVIEW PS_TEAM_SPLIT PS_PREVIEW_A PS_PREVIEW_B PS_B_VALIDATED MCTS_STATS_FILE
export MCTS_PUCT_VERSION=4
log() { echo "$(date '+%F %T') $*" >> "$OUT/queue.log"; }
clean_ps() { pkill -f "pokemon-showdown.*start" 2>/dev/null; sleep 5; }

run() {
  local cfg=$1 seed=$2 teams=$3; shift 3
  local rpt="$OUT/reports/${cfg}_s${seed}.txt"
  if [ -s "$rpt" ]; then log "$cfg s$seed ya hecho (reanudación)"; return 0; fi
  rm -f logs/robust_e2e_report.txt
  export DAMAGE_CALC_CACHE="$OUT/calc_cache_run.sqlite"
  rm -f "$DAMAGE_CALC_CACHE" "$DAMAGE_CALC_CACHE"-wal "$DAMAGE_CALC_CACHE"-shm
  python -m showdown.inference_sim.run_robust_e2e --teams "$teams" --procs "$PROCS" --seed "$seed" \
    --restart-ps --player-a mcts_puct --player-b greedy_joint "$@" > "$OUT/raw/${cfg}_s${seed}.txt" 2>&1
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

clean_ps
log "inicio commit $(git rev-parse --short HEAD) (MCTS_PUCT_VERSION=$MCTS_PUCT_VERSION)"
for f in "$V3OUT"/reports/v3_n200_s91*.txt; do
  [ -f "$f" ] && ln -sf "$f" "$OUT/reports/$(basename "$f")"
done
log "referencia v3: $(ls "$OUT"/reports/v3_n200_s91*.txt 2>/dev/null | wc -l) reportes enlazados"

OMP_NUM_THREADS=4 python -m pytest -q -p no:cacheprovider tests/test_mcts_puct_s3b.py tests/test_mcts_puct_s3.py \
  tests/test_mcts_puct_v2.py tests/test_rl_mcts_puct.py tests/test_rl_mcts_puct_nash.py > "$OUT/tests.txt" 2>&1
rc=$?; log "1 tests rc=$rc ($(tail -1 "$OUT/tests.txt"))"
[ $rc = 0 ] || { log "ERROR: tests fallan; se detiene"; log "MCTS_G_V4_DONE"; exit 1; }

PROCS_SAVE=$PROCS; PROCS=4
rm -f "$OUT/anomalies_smoke.tsv"
MCTS_STATS_FILE="$OUT/anomalies_smoke.tsv" run smoke 9001 20 --preview-a greedy --timeout 1200
PROCS=$PROCS_SAVE
SERR=$(grep -oE '^series_errors: *[0-9]+' "$OUT/reports/smoke_s9001.txt" 2>/dev/null | grep -oE '[0-9]+$')
NANOM=$(cat "$OUT/anomalies_smoke.tsv" 2>/dev/null | wc -l)
log "2 smoke: series_errors=${SERR:-?} anomalías=${NANOM} ($(cut -f1 "$OUT/anomalies_smoke.tsv" 2>/dev/null | sort | uniq -c | tr '\n' ' '))"
if [ "${SERR:-1}" != 0 ] || [ "$NANOM" != 0 ]; then
  log "ERROR: smoke con errores o anomalías; se detiene"; log "MCTS_G_V4_DONE"; exit 1
fi
export MCTS_STATS_FILE="$OUT/anomalies.tsv"

for s in $(seq 9101 9117); do run v4_n200 "$s" 300 --preview-a greedy --timeout 1200; done

log "anomalías: $(cut -f1 "$OUT/anomalies.tsv" 2>/dev/null | sort | uniq -c | tr '\n' ' ')"
python -m showdown.inference_sim.mcts_g_summary "$OUT" v3_n200 > "$OUT/summary.err" 2>&1
log "resumen rc=$?"
python -m showdown.inference_sim.confirm_stats mcnemar --a "$OUT"/raw/v4_n200_s91*.txt \
  --b "$V3OUT"/raw/v3_n200_s91*.txt > "$OUT/mcnemar.txt" 2>&1
log "McNemar v4 vs v3: $(cat "$OUT/mcnemar.txt" | tr '\n' ' ')"
log "MCTS_G_V4_DONE"
