#!/bin/bash
# PASO 0 DE H 2.1 (23-11-2026): diagnóstico de H 2.0, sin hipótesis ni decisión de default.
# H 2.0 contra greedy_joint en 9101–9102 × 300 con MCTS_NASH_DIAG_FILE: por cada decisión, dispersión de
# la matriz de la raíz, concentración de σ*/τ*, qué elegirían otras reglas (más visitada, mejor respuesta
# al prior q, mezcla) y qué elige G 5.2 en el mismo estado. No cambia la decisión de H (el win rate debe
# coincidir con N2 de la cola mcts_h_v2 en esas semillas). Resumen: $OUT/diag_summary.txt. Marca: MCTS_H_DIAG_DONE.
OUT=${1:-$HOME/iapsvgc_runs/mcts_h_diag}
SEEDS=${SEEDS:-9101 9102}
PROCS=${PROCS:-28}
mkdir -p "$OUT"/reports "$OUT"/raw "$OUT"/diag
source ~/miniconda3/etc/profile.d/conda.sh; conda activate kuroneko
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
export PYTHONUNBUFFERED=1
unset MCTS_PUCT_VERSION MCTS_NASH_VERSION MCTS_NASH_SIMS MCTS_NASH_SEED MCTS_SIMS MCTS_SEED MCTS_S3_CALC MCTS_PREVIEW MCTS_PREVIEW_CKPT MCTS_PREVIEW_DECISION \
  MCTS_PREVIEW_PHASE MCTS_PREVIEW_DEADLINE PS_TEAM_SPLIT PS_PREVIEW_A PS_PREVIEW_B PS_B_VALIDATED PS_SP_INFO_A \
  PS_SP_INFO_B PS_SP_INFO_DEBUG MCTS_STATS_FILE RL_OBS_ABILITIES RL_OBS_CALC PREVIEW_FORCE_ORDER MCTS_NASH_DIAG_G
log() { echo "$(date '+%F %T') $*" >> "$OUT/queue.log"; }
clean_ps() { pkill -f "[p]okemon-showdown.*start" 2>/dev/null; sleep 5; }
log "inicio commit $(git rev-parse --short HEAD)"
OMP_NUM_THREADS=4 python -m pytest -q -p no:cacheprovider tests/test_mcts_puct_nash_v2.py > "$OUT/tests.txt" 2>&1
rc=$?; log "tests rc=$rc ($(tail -1 "$OUT/tests.txt"))"
[ $rc = 0 ] || { log "ERROR: tests fallan"; log "MCTS_H_DIAG_DONE"; exit 1; }
clean_ps
for s in $SEEDS; do
  rpt="$OUT/reports/D_s$s.txt"; [ -s "$rpt" ] && continue
  rm -f logs/robust_e2e_report.txt "$OUT"/diag/s$s.jsonl.*
  export DAMAGE_CALC_CACHE="$OUT/calc_cache_run.sqlite"
  rm -f "$DAMAGE_CALC_CACHE" "$DAMAGE_CALC_CACHE"-wal "$DAMAGE_CALC_CACHE"-shm
  t0=$(date +%s)
  env MCTS_NASH_VERSION=2 MCTS_NASH_DIAG_FILE="$OUT/diag/s$s.jsonl" MCTS_STATS_FILE="$OUT/anomalies_D.tsv" \
    python -m showdown.inference_sim.run_robust_e2e --teams 300 --procs "$PROCS" --seed "$s" --restart-ps \
    --player-a mcts_puct_nash --player-b greedy_joint --timeout 3600 > "$OUT/raw/D_s$s.txt" 2>&1
  rc=$?
  rm -f "$DAMAGE_CALC_CACHE" "$DAMAGE_CALC_CACHE"-wal "$DAMAGE_CALC_CACHE"-shm
  [ -s logs/robust_e2e_report.txt ] && cp logs/robust_e2e_report.txt "$rpt"
  log "D s$s rc=$rc $(( $(date +%s) - t0 ))s $(grep -E '^player_a_wins|^series_errors' "$rpt" 2>/dev/null | tr '\n' ' ') filas=$(cat "$OUT"/diag/s$s.jsonl.[0-9]* 2>/dev/null | wc -l) errores_diag=$(cat "$OUT"/diag/s$s.jsonl.*.err 2>/dev/null | wc -l)"
  clean_ps
done
python -m showdown.rl.mcts_puct_nash.diag $(ls "$OUT"/diag/*.jsonl.* | grep -v '\.err$') > "$OUT/diag_summary.txt" 2>&1
log "resumen en $OUT/diag_summary.txt"
log "MCTS_H_DIAG_DONE"
