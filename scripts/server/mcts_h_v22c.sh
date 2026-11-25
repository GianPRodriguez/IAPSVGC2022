#!/bin/bash
# CONFIRMACIÓN DE H 2.2 = B2 (H 2.1 + RM+ con 200 sims), pre-registrada el 25-11-2026 antes de medir
# (docs/mcts_puct_nash_spec.md §4e y docs/protocolo_evaluacion.md §4.9). Solo baselines; G 5.2 como
# referencia descriptiva contra el mismo rival (H nunca juega contra G).
#
#   Bloques nuevos C17 = 10801–10817 y C18 = 10901–10917, × 300 BO3:
#     Bj/Be/Bs = B2 contra greedy_joint / expectimax_bayesian / greedy_solo
#     Hj       = H 2.1 contra greedy_joint (pareja de U4)
#     Gj       = G 5.2 contra greedy_joint (descriptivo, mismo coste que B2)
#   U1–U3 B2 > baseline (binomial); U4 B2 no inferior a H 2.1 (McNemar, margen −1,5 pts, p unilateral);
#   Holm m = 4; además s/semilla de Bj ≤ 85 % del de Hj. Si todo pasa, B2 = H 2.2 (default del player);
#   si no, H se queda en 2.1. Estadística: scripts/server/h22c_stats.py.
# Actualiza el clon de medición (ff-only) si no hay corridas activas y exige el commit congelado.
# Caché de la calculadora por corrida y borrada al terminar (regla: nunca > 250 GB).
# Resumen: $OUT/h22c_summary.txt. Marca: MCTS_H_V22C_DONE.
set -u
OUT=${1:-$HOME/iapsvgc_runs/mcts_h_v22c}
FROZEN=${FROZEN:?congelar el commit del pre-registro de la confirmación de H 2.2 (FROZEN=<sha>)}
REPO=${REPO:?clon de medición (REPO=<ruta>)}
PROCS=${PROCS:-28}
mkdir -p "$OUT"/reports "$OUT"/raw
source ~/miniconda3/etc/profile.d/conda.sh; conda activate kuroneko
export PYTHONUNBUFFERED=1
unset MCTS_PUCT_VERSION MCTS_NASH_VERSION MCTS_NASH_SIMS MCTS_NASH_SEED MCTS_SIMS MCTS_SEED MCTS_S3_CALC MCTS_PREVIEW MCTS_PREVIEW_CKPT MCTS_PREVIEW_DECISION \
  MCTS_PREVIEW_PHASE MCTS_PREVIEW_DEADLINE PS_TEAM_SPLIT PS_PREVIEW_A PS_PREVIEW_B PS_B_VALIDATED PS_SP_INFO_A \
  PS_SP_INFO_B PS_SP_INFO_DEBUG MCTS_STATS_FILE RL_OBS_ABILITIES RL_OBS_CALC PREVIEW_FORCE_ORDER \
  MCTS_NASH_MATRIX MCTS_NASH_DECISION MCTS_NASH_SOLVER MCTS_NASH_DIAG_FILE MCTS_NASH_DIAG_G MCTS_NASH_BACKUP_MIX
log() { echo "$(date '+%F %T') $*" >> "$OUT/queue.log"; }
clean_ps() { pkill -f "[p]okemon-showdown.*start" 2>/dev/null; sleep 5; }

# ------------------------------------------------ actualiza el clon (sin corridas activas)
while ps -eo cmd | grep -q "[r]un_robust_e2e --teams"; do log "hay una corrida activa; espero"; sleep 300; done
cd "$REPO"
if [ -n "$(git -c core.fileMode=false status --porcelain --untracked-files=no -- . ':!logs')" ]; then
  log "ERROR: el clon $REPO tiene cambios locales fuera de logs/; no se actualiza"; log "MCTS_H_V22C_DONE"; exit 1
fi
if ! { git fetch -q origin && git merge -q --ff-only origin/feature/Champions; }; then
  log "ERROR: no se pudo actualizar el clon (ff-only)"; log "MCTS_H_V22C_DONE"; exit 1
fi
if ! git merge-base --is-ancestor "$FROZEN" HEAD; then
  log "ERROR: $FROZEN no está en el clon tras actualizar"; log "MCTS_H_V22C_DONE"; exit 1
fi

# variante -> entorno (explícito: no depende de los defaults del player)
arm_env() {
  case $1 in
    h21) echo "MCTS_NASH_VERSION=2 MCTS_NASH_MATRIX=8,6,4 MCTS_NASH_DECISION=visits MCTS_NASH_SOLVER=fp" ;;
    b2)  echo "MCTS_NASH_VERSION=2 MCTS_NASH_MATRIX=8,6,4 MCTS_NASH_DECISION=visits MCTS_NASH_SOLVER=rm+" ;;
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
nerr() { cat "$OUT"/reports/$1_s10[89]*.txt 2>/dev/null | grep -oE '^series_errors: *[0-9]+' | awk '{s+=$2} END{print s+0}'; }

clean_ps
HEAD_C=$(git rev-parse --short HEAD)
log "inicio commit $HEAD_C (congelado $FROZEN)"
if ! git diff --quiet "$FROZEN" HEAD -- showdown/rl/mcts_puct showdown/rl/mcts_puct_nash showdown/battle_bots/greedy_joint \
     showdown/battle_bots/greedy_solo showdown/battle_bots/expectimax_bayesian \
     showdown/inference_sim/validated_greedy_player.py showdown/stat_inference.py showdown/meta_sp_prior.py \
     showdown/inference_sim/confirm_stats.py scripts/server/h22c_stats.py; then
  log "ERROR: el código congelado cambió desde $FROZEN; no se mide (pre-registro §4.9)"
  log "MCTS_H_V22C_DONE"; exit 1
fi
OMP_NUM_THREADS=4 python -m pytest -q -p no:cacheprovider tests/test_mcts_puct_nash_v22.py tests/test_mcts_puct_nash_v21.py \
  tests/test_mcts_puct_nash_v2.py tests/test_rl_mcts_puct_nash.py tests/test_mcts_puct_s3c.py tests/test_mcts_puct_s3.py \
  tests/test_mcts_search_preview.py > "$OUT/tests.txt" 2>&1
rc=$?; log "tests rc=$rc ($(tail -1 "$OUT/tests.txt"))"
[ $rc = 0 ] || { log "ERROR: tests fallan; se detiene"; log "MCTS_H_V22C_DONE"; exit 1; }

# ------------------------------------------------ smoke de B2 bajo carga (56 equipos)
rm -f "$OUT/anomalies_smoke_b2.tsv"
run smoke_b2 9001 b2 greedy_joint --teams 56 --timeout 2400
SERR=$(grep -oE '^series_errors: *[0-9]+' "$OUT/reports/smoke_b2_s9001.txt" 2>/dev/null | grep -oE '[0-9]+$')
NFB=$(nfb smoke_b2); NOM=$(grep -vc '^preview_search_fallback' "$OUT/anomalies_smoke_b2.tsv" 2>/dev/null); NOM=${NOM:-0}
log "smoke b2: series_errors=${SERR:-?} fallbacks_preview=$NFB otras_anomalías=$NOM"
if [ "${SERR:-1}" != 0 ] || [ "$NOM" != 0 ] || [ "$NFB" -gt 3 ]; then
  log "ERROR: smoke b2 con errores, anomalías o > 5 % de fallbacks; se detiene"; log "MCTS_H_V22C_DONE"; exit 1
fi

# ------------------------------------------------ confirmación (C17, C18)
for blk in 108 109; do
  for s in $(seq ${blk}01 ${blk}17); do run Bj "$s" b2 greedy_joint --timeout 3000; done
  for s in $(seq ${blk}01 ${blk}17); do run Hj "$s" h21 greedy_joint --timeout 3000; done
  for s in $(seq ${blk}01 ${blk}17); do run Be "$s" b2 expectimax_bayesian --timeout 3600; done
  for s in $(seq ${blk}01 ${blk}17); do run Bs "$s" b2 greedy_solo --timeout 3000; done
  for s in $(seq ${blk}01 ${blk}17); do run Gj "$s" g greedy_joint --timeout 1800; done
done

S="$OUT/h22c_summary.txt"
{
  echo "commit $HEAD_C (congelado $FROZEN). Pre-registro: protocolo §4.9, spec de H §4e."
  for c in Bj Be Bs Hj Gj; do echo "$c: fallbacks preview $(nfb $c), series_errors $(nerr $c)"; done
  echo
  python scripts/server/h22c_stats.py "$OUT/raw" "$OUT/reports"
} > "$S" 2>&1
log "estadística escrita en $S"
log "decisión: $(grep '^Decisión pre-registrada' "$S")"
log "MCTS_H_V22C_DONE"
