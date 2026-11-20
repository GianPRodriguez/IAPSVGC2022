#!/bin/bash
# MEDICIÓN DE MCTS+PUCT v3 = modelo S3, fase 3a (agente G, 20-11-2026; plan acordado con el usuario:
# medir 3a antes de decidir 3b/3c). Spec v3.0 y diagnóstico: docs/mcts_puct_spec.md,
# docs/mcts_puct_diagnostico.md. Misma metodología que mcts_g_v2.sh y mismas semillas, para comparar
# en pareado con v2 (los reportes v2_n200 de V2OUT se enlazan en $OUT/reports).
#
# Se corre desde el clon AISLADO (~/iapsvgc_scratch/spinfo/IAPSVGC2022).
#
# 0. Espera opcional (WAIT_FILE / WAIT_PAT; vacío = arranca ya). WAIT_SKIP_LINES como en v2.
# 1. Tests de MCTS (S3, v2, v1, MCTS-Nash, resumen). Si fallan, se detiene.
# 2. Smoke con la calculadora real: v3 vs greedy_joint, 20 equipos. Se detiene si hay errores de serie,
#    excepciones o fallbacks del player, errores de la calculadora o pedidos sin respuesta.
# 3. Principal: v3, 200 sims, preview greedy, semillas 9101-9117 × 300 (pareado con v2_n200).
# 4. Pipeline G real: v3 con su preview Nash, semillas 9101-9105.
# 5. Barrido: v3 con 100 y 400 sims, semillas 9101-9105 (¿con un modelo mejor, más sims ayudan?).
# Resumen: $OUT/summary.md con Δ frente a v2_n200. Marca: MCTS_G_V3_DONE.
#
#   cd ~/iapsvgc_scratch/spinfo/IAPSVGC2022 && git pull
#   nohup setsid bash scripts/server/mcts_g_v3.sh ~/iapsvgc_runs/mcts_g_v3 > ~/iapsvgc_runs/mcts_g_v3.out 2>&1 < /dev/null &
set -u
OUT=${1:-$HOME/iapsvgc_runs/mcts_g_v3}
V2OUT=${V2OUT:-$HOME/iapsvgc_runs/mcts_g_v2}
WAIT_FILE=${WAIT_FILE:-}
WAIT_PAT=${WAIT_PAT:-}
WAIT_SKIP_LINES=${WAIT_SKIP_LINES:-$( [ -n "$WAIT_FILE" ] && wc -l < "$WAIT_FILE" 2>/dev/null || echo 0)}
PROCS=${PROCS:-28}
mkdir -p "$OUT"/reports "$OUT"/raw
source ~/miniconda3/etc/profile.d/conda.sh; conda activate kuroneko
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
export PYTHONUNBUFFERED=1
unset MCTS_PUCT_VERSION MCTS_SIMS MCTS_SEED MCTS_S3_CALC PS_TEAM_SPLIT PS_PREVIEW_A PS_PREVIEW_B PS_B_VALIDATED
log() { echo "$(date '+%F %T') $*" >> "$OUT/queue.log"; }
clean_ps() { pkill -f "pokemon-showdown.*start" 2>/dev/null; sleep 5; }

run() {
  local cfg=$1 seed=$2 teams=$3; shift 3
  local rpt="$OUT/reports/${cfg}_s${seed}.txt"
  if [ -s "$rpt" ]; then log "$cfg s$seed ya hecho (reanudación)"; return 0; fi
  rm -f logs/robust_e2e_report.txt
  python -m showdown.inference_sim.run_robust_e2e --teams "$teams" --procs "$PROCS" --seed "$seed" \
    --restart-ps --player-a mcts_puct --player-b greedy_joint "$@" > "$OUT/raw/${cfg}_s${seed}.txt" 2>&1
  local rc=$?
  if [ ! -s logs/robust_e2e_report.txt ]; then
    log "ERROR: $cfg s$seed rc=$rc sin reporte (ver raw/${cfg}_s${seed}.txt)"; clean_ps; return 1
  fi
  cp logs/robust_e2e_report.txt "$rpt.tmp" && mv "$rpt.tmp" "$rpt"
  log "$cfg s$seed rc=$rc $(grep -E '^win_rate_player_a|^player_a_wins|^series_errors' "$rpt" | tr '\n' ' ')"
  clean_ps
  return $rc
}

if [ -n "$WAIT_FILE" ]; then
  log "esperando /$WAIT_PAT/ en $WAIT_FILE (después de la línea $WAIT_SKIP_LINES)"
  until tail -n +"$((WAIT_SKIP_LINES + 1))" "$WAIT_FILE" 2>/dev/null | grep -qE "$WAIT_PAT"; do sleep 60; done
fi
clean_ps
log "inicio commit $(git rev-parse --short HEAD)"

# Referencia v2 en pareado: enlazar sus reportes v2_n200.
for f in "$V2OUT"/reports/v2_n200_s*.txt; do
  [ -f "$f" ] && ln -sf "$f" "$OUT/reports/$(basename "$f")"
done
log "referencia v2: $(ls "$OUT"/reports/v2_n200_s*.txt 2>/dev/null | wc -l) reportes enlazados de $V2OUT"

# ---------------------------------------------------------------- 1. tests
OMP_NUM_THREADS=4 python -m pytest -q -p no:cacheprovider tests/test_mcts_puct_s3.py tests/test_mcts_puct_v2.py \
  tests/test_rl_mcts_puct.py tests/test_rl_mcts_puct_nash.py tests/test_mcts_g_summary.py > "$OUT/tests.txt" 2>&1
rc=$?; log "1 tests rc=$rc ($(tail -1 "$OUT/tests.txt"))"
[ $rc = 0 ] || { log "ERROR: tests fallan; se detiene"; log "MCTS_G_V3_DONE"; exit 1; }

# ---------------------------------------------------------------- 2. smoke (calculadora real)
PROCS_SAVE=$PROCS; PROCS=4
rm -f "$OUT/anomalies_smoke.tsv"
MCTS_STATS_FILE="$OUT/anomalies_smoke.tsv" run smoke 9001 20 --preview-a greedy --timeout 1200
PROCS=$PROCS_SAVE
SERR=$(grep -oE '^series_errors: *[0-9]+' "$OUT/reports/smoke_s9001.txt" 2>/dev/null | grep -oE '[0-9]+$')
NANOM=$(cat "$OUT/anomalies_smoke.tsv" 2>/dev/null | wc -l)
KINDS=$(cut -f1 "$OUT/anomalies_smoke.tsv" 2>/dev/null | sort | uniq -c | tr '\n' ' ')
log "2 smoke: series_errors=${SERR:-?} anomalías=${NANOM} (${KINDS:-ninguna})"
if [ "${SERR:-1}" != 0 ] || [ "$NANOM" != 0 ]; then
  log "ERROR: smoke con errores, fallbacks o fallos de la calculadora (ver anomalies_smoke.tsv); se detiene"
  log "MCTS_G_V3_DONE"; exit 1
fi
export MCTS_STATS_FILE="$OUT/anomalies.tsv"

# ---------------------------------------------------------------- 3. principal
for s in $(seq 9101 9117); do run v3_n200 "$s" 300 --preview-a greedy --timeout 1200; done

# ---------------------------------------------------------------- 4. pipeline G real (preview Nash)
for s in $(seq 9101 9105); do run v3_n200_ownpreview "$s" 300 --timeout 1200; done

# ---------------------------------------------------------------- 5. barrido
for n in 100 400; do
  for s in $(seq 9101 9105); do MCTS_SIMS=$n run "v3_n$n" "$s" 300 --preview-a greedy --timeout 2400; done
done

log "anomalías en las corridas largas: $(cut -f1 "$OUT/anomalies.tsv" 2>/dev/null | sort | uniq -c | tr '\n' ' ')"
python -m showdown.inference_sim.mcts_g_summary "$OUT" v2_n200 > "$OUT/summary.err" 2>&1
log "resumen rc=$?"
log "MCTS_G_V3_DONE"
