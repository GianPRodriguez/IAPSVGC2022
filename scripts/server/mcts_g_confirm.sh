#!/bin/bash
# CONFIRMACIÓN DE MCTS+PUCT v3 (agente G), hipótesis G1–G2 pre-registradas el 20-11-2026
# (docs/protocolo_evaluacion.md §4.1). Decisión del usuario: confirmar v3 contra joint y contra
# expectimax en bloques nuevos antes de la fase 3b.
#
#   G1: G v3.1 > greedy_joint        C3 = 9401–9417 + C4 = 9501–9517
#   G2: G v3.1 > expectimax v1.9     C3 + C4
#   G3: G v3.1 > greedy_solo         C3 + C4          (Holm m = 3 sobre G1–G3)
#   G4: G v3.1 > random              C3               (control de cordura, fuera de Holm)
# Solo baselines (usuario: ningún agente RL). Preview greedy (por defecto en v3.1; se pasa
# --preview-a greedy igualmente).
#   (GS1 eliminada el 20-11-2026 antes de medir: no existe checkpoint entrenado del preview Nash;
#    se habría medido el orden fijo /team 1234. Ver protocolo §4.1.)
# Estadística: confirm_stats binom (binomial exacta + IC por cluster bootstrap) por hipótesis y por
# bloque, y Holm con m = 3. Resumen: $OUT/confirm_summary.txt. Marca: MCTS_G_CONFIRM_DONE.
#
# Se corre desde el clon AISLADO. Espera MCTS_G_V3_DONE (WAIT_FILE/WAIT_PAT) y aborta si el código
# congelado cambió desde FROZEN (bb9ef00 = v3.1).
#
#   cd ~/iapsvgc_scratch/spinfo/IAPSVGC2022 && git pull
#   nohup setsid bash scripts/server/mcts_g_confirm.sh ~/iapsvgc_runs/mcts_g_confirm > ~/iapsvgc_runs/mcts_g_confirm.out 2>&1 < /dev/null &
set -u
OUT=${1:-$HOME/iapsvgc_runs/mcts_g_confirm}
FROZEN=${FROZEN:-bb9ef00}
WAIT_FILE=${WAIT_FILE:-$HOME/iapsvgc_runs/mcts_g_v3/queue.log}
WAIT_PAT=${WAIT_PAT:-MCTS_G_V3_DONE}
PROCS=${PROCS:-28}
mkdir -p "$OUT"/reports "$OUT"/raw
source ~/miniconda3/etc/profile.d/conda.sh; conda activate kuroneko
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
export PYTHONUNBUFFERED=1
unset MCTS_PUCT_VERSION MCTS_SIMS MCTS_SEED MCTS_S3_CALC PS_TEAM_SPLIT PS_PREVIEW_A PS_PREVIEW_B PS_B_VALIDATED \
  PS_SP_INFO_A PS_SP_INFO_B MCTS_STATS_FILE
log() { echo "$(date '+%F %T') $*" >> "$OUT/queue.log"; }
clean_ps() { pkill -f "pokemon-showdown.*start" 2>/dev/null; sleep 5; }

# run <config> <seed> <player_b> [args...]
run() {
  local cfg=$1 seed=$2 pb=$3; shift 3
  local rpt="$OUT/reports/${cfg}_s${seed}.txt"
  if [ -s "$rpt" ]; then log "$cfg s$seed ya hecho (reanudación)"; return 0; fi
  rm -f logs/robust_e2e_report.txt
  # Caché en disco de la calculadora POR CORRIDA, borrada al terminar: la compartida
  # (logs/calc_cache.sqlite) creció sin límite con las claves de S3 (HP actual) hasta 360 GB y
  # llenó el disco el 20-11-2026 a las 11:36. No cambia ningún valor de daño (solo la caché).
  export DAMAGE_CALC_CACHE="$OUT/calc_cache_run.sqlite"
  rm -f "$DAMAGE_CALC_CACHE" "$DAMAGE_CALC_CACHE"-wal "$DAMAGE_CALC_CACHE"-shm
  python -m showdown.inference_sim.run_robust_e2e --teams 300 --procs "$PROCS" --seed "$seed" \
    --restart-ps --player-a mcts_puct --player-b "$pb" "$@" > "$OUT/raw/${cfg}_s${seed}.txt" 2>&1
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
  log "ERROR: el código congelado cambió desde $FROZEN; no se confirma (pre-registro §4.1)"
  git diff --stat "$FROZEN" HEAD -- showdown/rl/mcts_puct showdown/battle_bots >> "$OUT/queue.log"
  log "MCTS_G_CONFIRM_DONE"; exit 1
fi

OMP_NUM_THREADS=4 python -m pytest -q -p no:cacheprovider tests/test_mcts_puct_s3.py tests/test_mcts_puct_v2.py \
  > "$OUT/tests.txt" 2>&1
rc=$?; log "tests rc=$rc ($(tail -1 "$OUT/tests.txt"))"
[ $rc = 0 ] || { log "ERROR: tests fallan; se detiene"; log "MCTS_G_CONFIRM_DONE"; exit 1; }
export MCTS_STATS_FILE="$OUT/anomalies.tsv"

# Orden por bloque: así C3 completo (G1–G4) queda antes que C4.
for s in $(seq 9401 9417); do run G1 "$s" greedy_joint --preview-a greedy --timeout 1200; done
for s in $(seq 9401 9417); do run G2 "$s" expectimax_bayesian --preview-a greedy --timeout 2400; done
for s in $(seq 9401 9417); do run G3 "$s" greedy_solo --preview-a greedy --timeout 1200; done
for s in $(seq 9401 9417); do run G4 "$s" random --preview-a greedy --timeout 1200; done
for s in $(seq 9501 9517); do run G1 "$s" greedy_joint --preview-a greedy --timeout 1200; done
for s in $(seq 9501 9517); do run G2 "$s" expectimax_bayesian --preview-a greedy --timeout 2400; done
for s in $(seq 9501 9517); do run G3 "$s" greedy_solo --preview-a greedy --timeout 1200; done

# ---------------------------------------------------------------- estadística
S="$OUT/confirm_summary.txt"
{
  echo "=== Confirmación MCTS+PUCT v3 (G) — commit $HEAD_C (código congelado $FROZEN) ==="
  echo "Pre-registro: docs/protocolo_evaluacion.md §4.1. Anomalías del player: $(cut -f1 "$OUT/anomalies.tsv" 2>/dev/null | sort | uniq -c | tr '\n' ' ')"
  PS_=""
  for h in G1 G2 G3; do
    echo; echo "--- $h C3+C4"
    line=$(python -m showdown.inference_sim.confirm_stats binom "$OUT"/raw/${h}_s94*.txt "$OUT"/raw/${h}_s95*.txt)
    echo "$line"
    PS_="$PS_ $(echo "$line" | grep -oE ' p=[^ ]+' | head -1 | cut -d= -f2)"
    echo "--- $h C3";    python -m showdown.inference_sim.confirm_stats binom "$OUT"/raw/${h}_s94*.txt
    echo "--- $h C4";    python -m showdown.inference_sim.confirm_stats binom "$OUT"/raw/${h}_s95*.txt
  done
  echo; echo "--- Holm (m=3; p de G1, G2, G3 en ese orden)"; python -m showdown.inference_sim.confirm_stats holm $PS_
  echo; echo "--- G4 C3 (control de cordura contra random)"; python -m showdown.inference_sim.confirm_stats binom "$OUT"/raw/G4_s94*.txt
} > "$S" 2>&1
log "estadística escrita en $S"
log "MCTS_G_CONFIRM_DONE"
