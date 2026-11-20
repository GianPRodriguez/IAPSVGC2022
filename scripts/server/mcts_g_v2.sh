#!/bin/bash
# MEDICIÓN DE MCTS+PUCT v2 (agente G, 20-11-2026; plan acordado con el usuario, encolada tras las colas
# actuales). Diagnóstico y spec: docs/mcts_puct_diagnostico.md, docs/mcts_puct_spec.md v2.0.
#
# Se corre desde el clon AISLADO (~/iapsvgc_scratch/spinfo/IAPSVGC2022) para no tocar el clon principal:
# run_robust_e2e reescribe logs/robust_e2e_report.txt dentro del repo.
#
# 0. Espera la marca de fin de la cola actual (WAIT_FILE / WAIT_PAT; por defecto el kick-off de DDQN v2:
#    KICKOFF_DDQN_V2_DONE; no PILOT_DONE: el usuario decidió completar las semillas 2-4, 20-11-2026).
# 1. Tests de MCTS (v2, v1, MCTS-Nash, barrido). Si fallan, se detiene.
# 2. Smoke: v2 vs greedy_joint, 20 equipos. Se detiene si hay errores de serie, excepciones o fallbacks
#    del adaptador (avisos "mcts_puct:" del player).
# 3. Principal: v2, 200 sims, preview greedy (aísla el combate, como el factorial de expectimax),
#    semillas 9101-9117 × 300 equipos.
# 4. Pipeline G real: v2 con su preview propio (team_preview_nash congelado), semillas 9101-9105.
# 5. Referencia v1 (MCTS_PUCT_VERSION=1, ~26 s/decisión): semilla 9101, 100 equipos.
# 6. Barrido real de N_sim (50, 100, 400; 200 ya está en 3), semillas 9101-9105, preview greedy.
# Resumen: $OUT/summary.md (python -m showdown.inference_sim.mcts_g_summary). Marca: MCTS_G_V2_DONE.
#
#   cd ~/iapsvgc_scratch/spinfo/IAPSVGC2022 && git pull
#   nohup bash scripts/server/mcts_g_v2.sh ~/iapsvgc_runs/mcts_g_v2 > /dev/null 2>&1 &
set -u
OUT=${1:-$HOME/iapsvgc_runs/mcts_g_v2}
WAIT_FILE=${WAIT_FILE:-$HOME/iapsvgc_runs/kickoff_ddqn_v2/queue.log}
WAIT_PAT=${WAIT_PAT:-KICKOFF_DDQN_V2_DONE}
# Líneas de WAIT_FILE que se ignoran (marcas de corridas anteriores en el mismo log). Por defecto,
# las que ya existen al lanzar la cola: la marca del piloto de DDQN ya está en ese log.
WAIT_SKIP_LINES=${WAIT_SKIP_LINES:-$(wc -l < "$WAIT_FILE" 2>/dev/null || echo 0)}
PROCS=${PROCS:-28}
mkdir -p "$OUT"/reports "$OUT"/raw
source ~/miniconda3/etc/profile.d/conda.sh; conda activate kuroneko
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
export PYTHONUNBUFFERED=1
unset MCTS_PUCT_VERSION MCTS_SIMS MCTS_SEED PS_TEAM_SPLIT PS_PREVIEW_A PS_PREVIEW_B PS_B_VALIDATED
log() { echo "$(date '+%F %T') $*" >> "$OUT/queue.log"; }
clean_ps() { pkill -f "pokemon-showdown.*start" 2>/dev/null; sleep 5; }

# run <config> <seed> <teams> [args...]: una corrida de run_robust_e2e contra greedy_joint validado.
run() {
  local cfg=$1 seed=$2 teams=$3; shift 3
  local rpt="$OUT/reports/${cfg}_s${seed}.txt"
  if [ -s "$rpt" ]; then log "$cfg s$seed ya hecho (reanudación)"; return 0; fi
  rm -f logs/robust_e2e_report.txt  # que una corrida fallida no copie el reporte anterior
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

# ---------------------------------------------------------------- 1. tests
OMP_NUM_THREADS=4 python -m pytest -q -p no:cacheprovider tests/test_mcts_puct_v2.py tests/test_rl_mcts_puct.py \
  tests/test_rl_mcts_puct_nash.py tests/test_mcts_sims_sweep.py tests/test_mcts_g_summary.py > "$OUT/tests.txt" 2>&1
rc=$?; log "1 tests rc=$rc ($(tail -1 "$OUT/tests.txt"))"
[ $rc = 0 ] || { log "ERROR: tests fallan; se detiene"; log "MCTS_G_V2_DONE"; exit 1; }

# ---------------------------------------------------------------- 2. smoke
PROCS_SAVE=$PROCS; PROCS=4
rm -f "$OUT/anomalies_smoke.tsv"
MCTS_STATS_FILE="$OUT/anomalies_smoke.tsv" run smoke 9001 20 --preview-a greedy --timeout 1200
PROCS=$PROCS_SAVE
SERR=$(grep -oE '^series_errors: *[0-9]+' "$OUT/reports/smoke_s9001.txt" | grep -oE '[0-9]+$')
NANOM=$(cat "$OUT/anomalies_smoke.tsv" 2>/dev/null | wc -l)
log "2 smoke: series_errors=${SERR:-?} anomalías_mcts=${NANOM} (excepciones/fallbacks del player)"
if [ "${SERR:-1}" != 0 ] || [ "$NANOM" != 0 ]; then
  log "ERROR: smoke con errores o fallbacks (ver anomalies_smoke.tsv y raw/smoke_s9001.txt); se detiene"
  log "MCTS_G_V2_DONE"; exit 1
fi
# En las corridas largas se siguen contando (no detienen la cola; van al resumen).
export MCTS_STATS_FILE="$OUT/anomalies.tsv"

# ---------------------------------------------------------------- 3. principal (combate aislado)
for s in $(seq 9101 9117); do run v2_n200 "$s" 300 --preview-a greedy --timeout 1200; done

# ---------------------------------------------------------------- 4. pipeline G real (preview Nash)
for s in $(seq 9101 9105); do run v2_n200_ownpreview "$s" 300 --timeout 1200; done

# ---------------------------------------------------------------- 5. referencia v1
MCTS_PUCT_VERSION=1 run v1_n200 9101 100 --preview-a greedy --timeout 2400

# ---------------------------------------------------------------- 6. barrido de N_sim
for n in 50 100 400; do
  for s in $(seq 9101 9105); do MCTS_SIMS=$n run "v2_n$n" "$s" 300 --preview-a greedy --timeout 2400; done
done

python -m showdown.inference_sim.mcts_g_summary "$OUT" > "$OUT/summary.err" 2>&1
log "resumen rc=$?"
log "MCTS_G_V2_DONE"
