#!/bin/bash
# PASO A tras H 2.1 (24-11-2026): medición descriptiva y diagnóstico, sin hipótesis ni cambio de default.
#   1. G 5.2 contra expectimax_bayesian (Ge) y contra greedy_solo (Gs) en C15 = 10601–10617 y C16 =
#      10701–10717: completa la comparación H3 frente a G 5.2 en las mismas series que Xe/Xs de la cola
#      mcts_h_v21 (McNemar descriptivo; contra joint ya está Xj frente a Gj).
#   2. Paso 0 repetido con H3 (matriz 8/6/4 + decisión por visitas) en 9101–9102 contra greedy_joint con
#      MCTS_NASH_DIAG_FILE (dispersión de la raíz y acuerdo con G 5.2 en el mismo estado). Las victorias
#      deben coincidir con H3 de la exploración (200 y 193).
# Espera a MCTS_H_V21_DONE. Caché por corrida y borrada al terminar (regla: nunca > 250 GB).
# Resumen: $OUT/stepA_summary.txt. Marca: MCTS_H_STEPA_DONE.
set -u
OUT=${1:-$HOME/iapsvgc_runs/mcts_h_stepA}
FROZEN=${FROZEN:-4db8267}   # mismo código de agentes que la cola de H 2.1
V21=${V21:-$HOME/iapsvgc_runs/mcts_h_v21}   # Xj/Xe/Xs/Gj de H 2.1
PROCS=${PROCS:-28}
mkdir -p "$OUT"/reports "$OUT"/raw
source ~/miniconda3/etc/profile.d/conda.sh; conda activate kuroneko
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
export PYTHONUNBUFFERED=1
unset MCTS_PUCT_VERSION MCTS_NASH_VERSION MCTS_NASH_SIMS MCTS_NASH_SEED MCTS_SIMS MCTS_SEED MCTS_S3_CALC MCTS_PREVIEW MCTS_PREVIEW_CKPT MCTS_PREVIEW_DECISION \
  MCTS_PREVIEW_PHASE MCTS_PREVIEW_DEADLINE PS_TEAM_SPLIT PS_PREVIEW_A PS_PREVIEW_B PS_B_VALIDATED PS_SP_INFO_A \
  PS_SP_INFO_B PS_SP_INFO_DEBUG MCTS_STATS_FILE RL_OBS_ABILITIES RL_OBS_CALC PREVIEW_FORCE_ORDER \
  MCTS_NASH_MATRIX MCTS_NASH_DECISION MCTS_NASH_SOLVER MCTS_NASH_DIAG_FILE MCTS_NASH_DIAG_G
log() { echo "$(date '+%F %T') $*" >> "$OUT/queue.log"; }
clean_ps() { pkill -f "[p]okemon-showdown.*start" 2>/dev/null; sleep 5; }

# variante -> entorno del agente (H 2.0 = sin variables; g = G 5.2)
arm_env() {
  case $1 in
    h20) echo "MCTS_NASH_VERSION=2" ;;
    h1)  echo "MCTS_NASH_VERSION=2 MCTS_NASH_MATRIX=8,6,4" ;;
    h2)  echo "MCTS_NASH_VERSION=2 MCTS_NASH_DECISION=visits" ;;
    h3)  echo "MCTS_NASH_VERSION=2 MCTS_NASH_MATRIX=8,6,4 MCTS_NASH_DECISION=visits" ;;
    h4)  echo "MCTS_NASH_VERSION=2 MCTS_NASH_SOLVER=rm+ MCTS_NASH_SIMS=350" ;;
    g)   echo "MCTS_PUCT_VERSION=5" ;;
    *)   echo "ERROR_ARM" ;;
  esac
}

# run <config> <seed> <variante> <player_b> [args...]
run() {
  local cfg=$1 seed=$2 arm=$3 pb=$4; shift 4
  local pa=mcts_puct_nash venv; venv=$(arm_env "$arm")
  [ "$arm" = g ] && pa=mcts_puct
  local rpt="$OUT/reports/${cfg}_s${seed}.txt"
  if [ -s "$rpt" ]; then log "$cfg s$seed ya hecho (reanudación)"; return 0; fi
  rm -f logs/robust_e2e_report.txt
  export DAMAGE_CALC_CACHE="$OUT/calc_cache_run.sqlite"
  rm -f "$DAMAGE_CALC_CACHE" "$DAMAGE_CALC_CACHE"-wal "$DAMAGE_CALC_CACHE"-shm
  local t0=$(date +%s)
  env $venv MCTS_STATS_FILE="$OUT/anomalies_${cfg}.tsv" python -m showdown.inference_sim.run_robust_e2e \
    --teams 300 --procs "$PROCS" --seed "$seed" --restart-ps --player-a $pa --player-b "$pb" "$@" \
    > "$OUT/raw/${cfg}_s${seed}.txt" 2>&1
  local rc=$?
  rm -f "$DAMAGE_CALC_CACHE" "$DAMAGE_CALC_CACHE"-wal "$DAMAGE_CALC_CACHE"-shm
  if [ ! -s logs/robust_e2e_report.txt ]; then
    log "ERROR: $cfg s$seed rc=$rc sin reporte (ver raw/${cfg}_s${seed}.txt)"; clean_ps; return 1
  fi
  cp logs/robust_e2e_report.txt "$rpt.tmp" && mv "$rpt.tmp" "$rpt"
  echo "$(( $(date +%s) - t0 ))" > "$OUT/reports/${cfg}_s${seed}.sec"
  log "$cfg s$seed rc=$rc $(cat "$OUT/reports/${cfg}_s${seed}.sec")s $(grep -E '^player_a_wins|^series_errors' "$rpt" | tr '\n' ' ')"
  clean_ps
  return $rc
}
nfb() { local n; n=$(grep -c '^preview_search_fallback' "$OUT/anomalies_$1.tsv" 2>/dev/null); echo "${n:-0}"; }
nerr() { cat "$OUT"/reports/$1_s*.txt 2>/dev/null | grep -oE '^series_errors: *[0-9]+' | awk '{s+=$2} END{print s+0}'; }
secs() { cat "$OUT"/reports/$1_s*.sec 2>/dev/null | awk '{s+=$1; n++} END{if(n) printf "%.0f s/semilla", s/n}'; }
secn() { cat "$OUT"/reports/$1_s*.sec 2>/dev/null | awk '{s+=$1; n++} END{if(n) printf "%.0f", s/n; else print 99999}'; }
mc() { python -m showdown.inference_sim.confirm_stats mcnemar --a $1 --b $2; }

log "esperando /MCTS_H_V21_DONE/ en $V21/queue.log"
until grep -q MCTS_H_V21_DONE "$V21/queue.log" 2>/dev/null; do sleep 60; done
clean_ps
HEAD_C=$(git rev-parse --short HEAD)
log "inicio commit $HEAD_C (código de agentes de $FROZEN)"
if ! git diff --quiet "$FROZEN" HEAD -- showdown/rl/mcts_puct showdown/rl/mcts_puct_nash showdown/battle_bots/greedy_joint \
     showdown/battle_bots/greedy_solo showdown/battle_bots/expectimax_bayesian \
     showdown/inference_sim/validated_greedy_player.py showdown/stat_inference.py showdown/meta_sp_prior.py; then
  log "ERROR: el código de los agentes cambió desde $FROZEN; no es comparable con la cola de H 2.1"
  log "MCTS_H_STEPA_DONE"; exit 1
fi
OMP_NUM_THREADS=4 python -m pytest -q -p no:cacheprovider tests/test_mcts_puct_nash_v21.py tests/test_mcts_puct_nash_v2.py > "$OUT/tests.txt" 2>&1
rc=$?; log "tests rc=$rc ($(tail -1 "$OUT/tests.txt"))"
[ $rc = 0 ] || { log "ERROR: tests fallan"; log "MCTS_H_STEPA_DONE"; exit 1; }

# ------------------------------------------------ 1. G 5.2 contra expectimax y solo (C15, C16)
for blk in 106 107; do
  for s in $(seq ${blk}01 ${blk}17); do run Ge "$s" g expectimax_bayesian --timeout 3000; done
  for s in $(seq ${blk}01 ${blk}17); do run Gs "$s" g greedy_solo --timeout 1800; done
done

# ------------------------------------------------ 2. paso 0 con H3
mkdir -p "$OUT/diag"
for s in 9101 9102; do
  rpt="$OUT/reports/D3_s$s.txt"; [ -s "$rpt" ] && continue
  rm -f logs/robust_e2e_report.txt "$OUT"/diag/s$s.jsonl.*
  export DAMAGE_CALC_CACHE="$OUT/calc_cache_run.sqlite"
  rm -f "$DAMAGE_CALC_CACHE" "$DAMAGE_CALC_CACHE"-wal "$DAMAGE_CALC_CACHE"-shm
  t0=$(date +%s)
  env $(arm_env h3) MCTS_NASH_DIAG_FILE="$OUT/diag/s$s.jsonl" MCTS_STATS_FILE="$OUT/anomalies_D3.tsv" \
    python -m showdown.inference_sim.run_robust_e2e --teams 300 --procs "$PROCS" --seed "$s" --restart-ps \
    --player-a mcts_puct_nash --player-b greedy_joint --timeout 3600 > "$OUT/raw/D3_s$s.txt" 2>&1
  rc=$?
  rm -f "$DAMAGE_CALC_CACHE" "$DAMAGE_CALC_CACHE"-wal "$DAMAGE_CALC_CACHE"-shm
  [ -s logs/robust_e2e_report.txt ] && cp logs/robust_e2e_report.txt "$rpt"
  log "D3 s$s rc=$rc $(( $(date +%s) - t0 ))s $(grep -E '^player_a_wins|^series_errors' "$rpt" 2>/dev/null | tr '\n' ' ') filas=$(cat "$OUT"/diag/s$s.jsonl.[0-9]* 2>/dev/null | wc -l) errores_diag=$(cat "$OUT"/diag/s$s.jsonl.*.err 2>/dev/null | wc -l)"
  clean_ps
done

S="$OUT/stepA_summary.txt"
{
  echo "=== Paso A tras H 2.1 — commit $HEAD_C (agentes de $FROZEN); descriptivo ==="
  for p in "e:expectimax" "s:solo"; do
    k=${p%%:*}; r=${p#*:}
    echo; echo "--- G 5.2 contra $r (G$k, $(secs G$k); fallbacks preview: $(nfb G$k), series_errors: $(nerr G$k))"
    python -m showdown.inference_sim.confirm_stats binom "$OUT"/raw/G${k}_s10[67]*.txt
    echo "    C15: $(python -m showdown.inference_sim.confirm_stats binom "$OUT"/raw/G${k}_s106*.txt)"
    echo "    C16: $(python -m showdown.inference_sim.confirm_stats binom "$OUT"/raw/G${k}_s107*.txt)"
    echo "--- H3 (X$k de mcts_h_v21) frente a G 5.2 contra $r (McNemar, mismas series)"
    mc "$V21/raw/X${k}_s10[67]*.txt" "$OUT/raw/G${k}_s10[67]*.txt"
  done
  echo; echo "--- referencia contra joint: H3 (Xj) frente a G 5.2 (Gj) de mcts_h_v21"
  mc "$V21/raw/Xj_s10[67]*.txt" "$V21/raw/Gj_s10[67]*.txt"
  echo; echo "=== Paso 0 con H3 (9101–9102 contra joint; exploración H3: 200 y 193) ==="
  python -m showdown.rl.mcts_puct_nash.diag $(ls "$OUT"/diag/*.jsonl.* | grep -v '\.err$')
} > "$S" 2>&1
log "resumen en $S"
log "MCTS_H_STEPA_DONE"
