#!/bin/bash
# PREVIEW NASH PARA G: entrenamiento live + medición (20-11-2026, decisión del usuario).
# Regla fijada de antemano: docs/mcts_puct_diagnostico.md §7. Solo baselines (piloto y rival
# greedy_joint). Encolada tras la exploración de v4 (MCTS_G_V4_DONE).
#
# 1. Tests (preview Nash, pool live, frozen preview, MCTS).
# 2. Entrena team_preview_nash live (greedy_joint, 200 updates, PS_TEAM_SPLIT=fit), semillas 0 y 1.
# 3. Smoke: G con preview Nash s0 en equipos eval; se detiene si el preview cae al orden fijo o hay
#    cualquier otra anomalía.
# 4. Medición (G v3.1 vs greedy_joint, --team-split eval, 9101–9134, 100 BO3/semilla):
#    P0 preview greedy, P1 Nash s0, P2 Nash s1. McNemar P1/P2 vs P0 + Holm m=2.
# 5. Secundaria (memorización): P1 y P2 con los 300 equipos (all), 9101–9117, pareado con v3_n200.
# Caché de la calculadora por corrida y borrada al terminar (regla del usuario: nunca > 250 GB).
# Resumen: $OUT/nash_summary.txt. Marca: MCTS_G_NASH_DONE.
#
#   cd ~/iapsvgc_scratch/spinfo/IAPSVGC2022 && git pull
#   nohup setsid bash scripts/server/mcts_g_nash_preview.sh ~/iapsvgc_runs/mcts_g_nash > ~/iapsvgc_runs/mcts_g_nash.out 2>&1 < /dev/null &
set -u
OUT=${1:-$HOME/iapsvgc_runs/mcts_g_nash}
V3OUT=${V3OUT:-$HOME/iapsvgc_runs/mcts_g_v3}
WAIT_FILE=${WAIT_FILE:-$HOME/iapsvgc_runs/mcts_g_v4/queue.log}
WAIT_PAT=${WAIT_PAT:-MCTS_G_V4_DONE}
PROCS=${PROCS:-28}
mkdir -p "$OUT"/reports "$OUT"/raw
source ~/miniconda3/etc/profile.d/conda.sh; conda activate kuroneko
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
export PYTHONUNBUFFERED=1
unset MCTS_SIMS MCTS_SEED MCTS_S3_CALC MCTS_PREVIEW MCTS_PREVIEW_CKPT PS_TEAM_SPLIT PS_PREVIEW_A PS_PREVIEW_B \
  PS_B_VALIDATED MCTS_STATS_FILE RL_OBS_ABILITIES RL_OBS_CALC PREVIEW_FORCE_ORDER PREVIEW_SAMPLE PREVIEW_TEMPERATURE
export MCTS_PUCT_VERSION=3
log() { echo "$(date '+%F %T') $*" >> "$OUT/queue.log"; }
clean_ps() { pkill -f "pokemon-showdown.*start" 2>/dev/null; sleep 5; }

# run <config> <seed> <teams> [args...]  (el preview se elige con MCTS_PREVIEW / MCTS_PREVIEW_CKPT)
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

if [ -n "$WAIT_FILE" ]; then
  log "esperando /$WAIT_PAT/ en $WAIT_FILE"
  until grep -qE "$WAIT_PAT" "$WAIT_FILE" 2>/dev/null; do sleep 60; done
fi
clean_ps
log "inicio commit $(git rev-parse --short HEAD)"

# ---------------------------------------------------------------- 1. tests
OMP_NUM_THREADS=4 python -m pytest -q -p no:cacheprovider tests/test_team_preview_nash.py tests/test_preview_live.py \
  tests/test_frozen_preview.py tests/test_mcts_puct_s3.py > "$OUT/tests.txt" 2>&1
rc=$?; log "1 tests rc=$rc ($(tail -1 "$OUT/tests.txt"))"
[ $rc = 0 ] || { log "ERROR: tests fallan; se detiene"; log "MCTS_G_NASH_DONE"; exit 1; }

# ---------------------------------------------------------------- 2. entrenamiento
for ps in 0 1; do
  CK="$OUT/preview_nash_s$ps"
  if [ -s "$CK/final.pt" ] && grep -q 'provenance: live' "$CK/reporte.txt" 2>/dev/null; then
    log "preview_nash_s$ps ya entrenado (reanudación)"; continue
  fi
  PS_TEAM_SPLIT=fit python -m showdown.rl.team_preview_nash.train --seed $ps --device cuda --backend live \
    --num-workers 28 --envs-per-server 4 --start-ps --pilot greedy_joint --updates 200 \
    --checkpoint-dir "$CK" > "$OUT/preview_nash_s$ps.txt" 2>&1
  rc=$?
  log "preview_nash_s$ps rc=$rc $(grep -E '^(loss|v_star|provenance)' "$CK/reporte.txt" 2>/dev/null | tr '\n' ' ')"
  clean_ps
  if [ ! -s "$CK/final.pt" ] || ! grep -q 'provenance: live' "$CK/reporte.txt" 2>/dev/null; then
    log "ERROR: preview_nash_s$ps sin checkpoint live; se detiene"; log "MCTS_G_NASH_DONE"; exit 1
  fi
done

# ---------------------------------------------------------------- 3. smoke (preview Nash s0, eval)
PROCS_SAVE=$PROCS; PROCS=4
rm -f "$OUT/anomalies_smoke.tsv"
MCTS_PREVIEW=nash MCTS_PREVIEW_CKPT="$OUT/preview_nash_s0/final.pt" MCTS_STATS_FILE="$OUT/anomalies_smoke.tsv" \
  run smoke_nash 9001 20 --team-split eval --timeout 1200
PROCS=$PROCS_SAVE
SERR=$(grep -oE '^series_errors: *[0-9]+' "$OUT/reports/smoke_nash_s9001.txt" 2>/dev/null | grep -oE '[0-9]+$')
NANOM=$(cat "$OUT/anomalies_smoke.tsv" 2>/dev/null | wc -l)
log "3 smoke: series_errors=${SERR:-?} anomalías=${NANOM} ($(cut -f1 "$OUT/anomalies_smoke.tsv" 2>/dev/null | sort | uniq -c | tr '\n' ' '))"
if [ "${SERR:-1}" != 0 ] || [ "$NANOM" != 0 ]; then
  log "ERROR: smoke con errores o el preview Nash no carga (preview_fallback); se detiene"
  log "MCTS_G_NASH_DONE"; exit 1
fi
export MCTS_STATS_FILE="$OUT/anomalies.tsv"

# ---------------------------------------------------------------- 4. medición en equipos eval
for s in $(seq 9101 9134); do run P0 "$s" 100 --team-split eval --timeout 1200; done
for s in $(seq 9101 9134); do
  MCTS_PREVIEW=nash MCTS_PREVIEW_CKPT="$OUT/preview_nash_s0/final.pt" run P1 "$s" 100 --team-split eval --timeout 1200
done
for s in $(seq 9101 9134); do
  MCTS_PREVIEW=nash MCTS_PREVIEW_CKPT="$OUT/preview_nash_s1/final.pt" run P2 "$s" 100 --team-split eval --timeout 1200
done

S="$OUT/nash_summary.txt"
{
  echo "=== Preview Nash para G — commit $(git rev-parse --short HEAD) — regla: diagnóstico §7 ==="
  echo "Anomalías: $(cut -f1 "$OUT/anomalies.tsv" 2>/dev/null | sort | uniq -c | tr '\n' ' ')"
  for c in P0 P1 P2; do
    echo "--- $c (eval, 9101–9134)"; python -m showdown.inference_sim.confirm_stats binom "$OUT"/raw/${c}_s91*.txt
  done
  PS_=""
  for c in P1 P2; do
    echo "--- McNemar $c vs P0"
    line=$(python -m showdown.inference_sim.confirm_stats mcnemar --a "$OUT"/raw/${c}_s91*.txt --b "$OUT"/raw/P0_s91*.txt)
    echo "$line"
    PS_="$PS_ $(echo "$line" | grep -oE ' p=[^ ]+' | head -1 | cut -d= -f2)"
  done
  echo "--- Holm (m=2; P1, P2)"; python -m showdown.inference_sim.confirm_stats holm $PS_
} > "$S" 2>&1
log "4 decisión: ver $S"

# ---------------------------------------------------------------- 5. secundaria: todos los equipos
for s in $(seq 9101 9117); do
  MCTS_PREVIEW=nash MCTS_PREVIEW_CKPT="$OUT/preview_nash_s0/final.pt" run P1all "$s" 300 --timeout 1200
done
for s in $(seq 9101 9117); do
  MCTS_PREVIEW=nash MCTS_PREVIEW_CKPT="$OUT/preview_nash_s1/final.pt" run P2all "$s" 300 --timeout 1200
done
{
  echo; echo "=== Secundaria (memorización): 300 equipos (all), pareado con v3_n200 (preview greedy) ==="
  for c in P1all P2all; do
    echo "--- $c"; python -m showdown.inference_sim.confirm_stats binom "$OUT"/raw/${c}_s91*.txt
    echo "--- McNemar $c vs v3_n200"
    python -m showdown.inference_sim.confirm_stats mcnemar --a "$OUT"/raw/${c}_s91*.txt --b "$V3OUT"/raw/v3_n200_s91*.txt
  done
  echo "Anomalías finales: $(cut -f1 "$OUT/anomalies.tsv" 2>/dev/null | sort | uniq -c | tr '\n' ' ')"
} >> "$S" 2>&1
log "5 secundaria escrita en $S"
log "MCTS_G_NASH_DONE"
