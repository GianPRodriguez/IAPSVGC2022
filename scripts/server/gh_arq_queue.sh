#!/bin/bash
# G Y H CONTRA ARQUETIPOS (protocolo §4.19, decisión del usuario del 27-11-2026). Exploración, bloque X1 = 11001–11017,
# × 300 BO3 por semilla. Solo baselines (greedy_joint en modo arquetipo); H nunca juega contra G.
#
#   0. Espera a que termine D3 (última línea de ~/iapsvgc_runs/arq3/queue.log = ARQ3_DONE, sin las del vigía).
#   1. Tests y smoke bajo carga (G con las tres extensiones contra el nivel 3, 56 equipos).
#   2. Punto 1 (descriptivo): G 5.2 y H 2.2 contra greedy_joint nivel 0 (G0, H0) y nivel 3 (G3, H3).
#   3. Puntos 2–4 (exploración contra el nivel 3): + arquetipos (MCTS_ARCH, GA/HA), + serie BO3 (MCTS_SERIES,
#      GS/HS), + hoja con campo (MCTS_LEAF_FIELD, GF/HF).
#   Estadística: scripts/server/gh_arq_stats.py. Resumen: $OUT/gh_arq_summary.txt. Marca: GH_ARQ_DONE.
# Caché de la calculadora por corrida y borrada al terminar (regla: nunca > 250 GB).
#
#   FROZEN=<sha> bash scripts/server/gh_arq_queue.sh ~/iapsvgc_runs/gh_arq
set -u
OUT=${1:-$HOME/iapsvgc_runs/gh_arq}
FROZEN=${FROZEN:?commit congelado del pre-registro §4.19 (FROZEN=<sha>)}
PROCS=${PROCS:-28}
ARQLOG=${ARQLOG:-$HOME/iapsvgc_runs/arq3/queue.log}
mkdir -p "$OUT"/reports "$OUT"/raw
source ~/miniconda3/etc/profile.d/conda.sh; conda activate kuroneko
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
export PYTHONUNBUFFERED=1
unset MCTS_PUCT_VERSION MCTS_NASH_VERSION MCTS_NASH_SIMS MCTS_NASH_SEED MCTS_SIMS MCTS_SEED MCTS_S3_CALC MCTS_PREVIEW \
  MCTS_PREVIEW_CKPT MCTS_PREVIEW_DECISION MCTS_PREVIEW_PHASE MCTS_PREVIEW_DEADLINE PS_TEAM_SPLIT PS_PREVIEW_A PS_PREVIEW_B \
  PS_B_VALIDATED PS_SP_INFO_A PS_SP_INFO_B PS_SP_INFO_DEBUG MCTS_STATS_FILE RL_OBS_ABILITIES RL_OBS_CALC \
  PREVIEW_FORCE_ORDER MCTS_NASH_MATRIX MCTS_NASH_DECISION MCTS_NASH_SOLVER MCTS_NASH_DIAG_FILE MCTS_NASH_DIAG_G \
  MCTS_NASH_BACKUP_MIX MCTS_ARCH MCTS_SERIES MCTS_LEAF_FIELD PS_JOINT_ARCHETYPE PREVIEW_KIND
log() { echo "$(date '+%F %T') $*" >> "$OUT/queue.log"; }
clean_ps() { pkill -f "[p]okemon-showdown.*start" 2>/dev/null; sleep 5; }

log "esperando el fin de D3 ($ARQLOG)"
until grep -v "vigía caché" "$ARQLOG" 2>/dev/null | tail -n 1 | grep -q " ARQ3_DONE\$"; do sleep 120; done
while ps -eo cmd | grep -q "[r]un_robust_e2e --teams"; do log "hay una corrida activa; espero"; sleep 300; done
HEAD_C=$(git rev-parse --short HEAD)
log "inicio commit $HEAD_C (congelado $FROZEN)"
if ! git merge-base --is-ancestor "$FROZEN" HEAD || ! git diff --quiet "$FROZEN" HEAD -- showdown/rl/mcts_puct \
     showdown/rl/mcts_puct_nash showdown/battle_bots/greedy_joint showdown/battle_bots/greedy_solo \
     showdown/inference_sim/validated_greedy_player.py showdown/inference_sim/confirm_stats.py scripts/server/gh_arq_stats.py; then
  log "ERROR: el código congelado cambió desde $FROZEN o no está; no se mide (pre-registro §4.19)"
  log "GH_ARQ_DONE"; exit 1
fi
clean_ps
OMP_NUM_THREADS=4 python -m pytest -q -p no:cacheprovider tests/test_mcts_gh_ext.py tests/test_joint_archetype_v3.py \
  tests/test_mcts_puct_s3c.py tests/test_mcts_search_preview.py tests/test_mcts_puct_nash_v22.py tests/test_rl_mcts_puct.py \
  tests/test_rl_mcts_puct_nash.py > "$OUT/tests.txt" 2>&1
rc=$?; log "tests rc=$rc ($(tail -1 "$OUT/tests.txt"))"
[ $rc = 0 ] || { log "ERROR: tests fallan; se detiene"; log "GH_ARQ_DONE"; exit 1; }

# brazo -> entorno explícito (no depende de los defaults del player)
G_ENV="MCTS_PUCT_VERSION=5"
H_ENV="MCTS_NASH_VERSION=2 MCTS_NASH_MATRIX=8,6,4 MCTS_NASH_DECISION=visits MCTS_NASH_SOLVER=rm+"
arm_env() {
  case $1 in
    G0) echo "$G_ENV" ;;                     G3) echo "$G_ENV PS_JOINT_ARCHETYPE=3" ;;
    H0) echo "$H_ENV" ;;                     H3) echo "$H_ENV PS_JOINT_ARCHETYPE=3" ;;
    GA) echo "$G_ENV PS_JOINT_ARCHETYPE=3 MCTS_ARCH=1" ;;
    GS) echo "$G_ENV PS_JOINT_ARCHETYPE=3 MCTS_SERIES=1" ;;
    GF) echo "$G_ENV PS_JOINT_ARCHETYPE=3 MCTS_LEAF_FIELD=1" ;;
    HA) echo "$H_ENV PS_JOINT_ARCHETYPE=3 MCTS_ARCH=1" ;;
    HS) echo "$H_ENV PS_JOINT_ARCHETYPE=3 MCTS_SERIES=1" ;;
    HF) echo "$H_ENV PS_JOINT_ARCHETYPE=3 MCTS_LEAF_FIELD=1" ;;
    SMOKE) echo "$G_ENV PS_JOINT_ARCHETYPE=3 MCTS_ARCH=1 MCTS_SERIES=1 MCTS_LEAF_FIELD=1" ;;
    *) echo "ERROR_ARM" ;;
  esac
}
# run <brazo> <semilla> [args...]
run() {
  local arm=$1 seed=$2; shift 2
  local pa=mcts_puct venv; venv=$(arm_env "$arm")
  case $arm in H*) pa=mcts_puct_nash ;; esac
  local rpt="$OUT/reports/${arm}_s${seed}.txt"
  if [ -s "$rpt" ]; then log "$arm s$seed ya hecho (reanudación)"; return 0; fi
  rm -f logs/robust_e2e_report.txt
  export DAMAGE_CALC_CACHE="$OUT/calc_cache_run.sqlite"
  rm -f "$DAMAGE_CALC_CACHE" "$DAMAGE_CALC_CACHE"-wal "$DAMAGE_CALC_CACHE"-shm
  local t0=$(date +%s)
  env $venv MCTS_STATS_FILE="$OUT/anomalies_${arm}.tsv" python -m showdown.inference_sim.run_robust_e2e \
    --teams 300 --procs "$PROCS" --seed "$seed" --restart-ps --player-a $pa --player-b greedy_joint "$@" \
    > "$OUT/raw/${arm}_s${seed}.txt" 2>&1
  local rc=$?
  rm -f "$DAMAGE_CALC_CACHE" "$DAMAGE_CALC_CACHE"-wal "$DAMAGE_CALC_CACHE"-shm
  if [ ! -s logs/robust_e2e_report.txt ]; then
    log "ERROR: $arm s$seed rc=$rc sin reporte (ver raw/${arm}_s${seed}.txt)"; clean_ps; return 1
  fi
  cp logs/robust_e2e_report.txt "$rpt.tmp" && mv "$rpt.tmp" "$rpt"
  echo "$(( $(date +%s) - t0 ))" > "$OUT/reports/${arm}_s${seed}.sec"
  log "$arm s$seed rc=$rc $(cat "$OUT/reports/${arm}_s${seed}.sec")s $(grep -E '^player_a_wins|^series_errors' "$rpt" | tr '\n' ' ')"
  clean_ps
  return $rc
}

# ---------------------------------------------------------------- smoke bajo carga
rm -f "$OUT/anomalies_SMOKE.tsv"
run SMOKE 9002 --teams 56 --timeout 2400
SERR=$(grep -oE '^series_errors: *[0-9]+' "$OUT/reports/SMOKE_s9002.txt" 2>/dev/null | grep -oE '[0-9]+$')
NOM=$(grep -vc '^preview_search_fallback' "$OUT/anomalies_SMOKE.tsv" 2>/dev/null); NOM=${NOM:-0}
NFB=$(grep -c '^preview_search_fallback' "$OUT/anomalies_SMOKE.tsv" 2>/dev/null); NFB=${NFB:-0}
log "smoke: series_errors=${SERR:-?} fallbacks_preview=$NFB otras_anomalías=$NOM"
if [ "${SERR:-1}" != 0 ] || [ "$NOM" != 0 ] || [ "$NFB" -gt 3 ]; then
  log "ERROR: smoke con errores, anomalías o > 5 % de fallbacks; se detiene"; log "GH_ARQ_DONE"; exit 1
fi

# ---------------------------------------------------------------- X1 = 11001–11017
for arm in G0 G3 H0 H3 GA GS GF HA HS HF; do
  case $arm in H*) TO=3000 ;; *) TO=1800 ;; esac
  for s in $(seq 11001 11017); do run "$arm" "$s" --timeout $TO; done
  log "brazo $arm terminado"
done
S="$OUT/gh_arq_summary.txt"
python scripts/server/gh_arq_stats.py "$OUT/raw" "$OUT/reports" > "$S" 2>&1
log "estadística escrita en $S"
log "GH_ARQ_DONE"
