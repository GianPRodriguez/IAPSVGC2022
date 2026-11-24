#!/bin/bash
# AGENTE H 2.2, EXPLORACIÓN (variantes de H3 = candidato de H 2.1), pre-registrada el 24-11-2026 antes de
# medir (docs/mcts_puct_nash_spec.md §4d y docs/protocolo_evaluacion.md §4.8). Solo baselines; H nunca
# juega contra G. Sin confirmación en esta cola (se encola aparte, en 10801+, con aprobación).
#
#   Exploración (9101–9117 × 300, contra greedy_joint, mismas series):
#     R  = H3 (H3 de la cola mcts_h_v21, matriz 8/6/4 + decisión por visitas; mismo código, se reutiliza)
#     B1 = matriz más chica  MCTS_NASH_MATRIX=6,4,3 + visitas
#     B2 = H3 + RM+ en caliente con 200 sims (MCTS_NASH_SOLVER=rm+): prueba de NO INFERIORIDAD y velocidad
#     B3 = H3 + backup mixto  MCTS_NASH_BACKUP_MIX=0.5 (sube ½·v_Nash + ½·media del nodo)
#   Regla de superioridad (B1, B3): 0 errores de serie, fallbacks del preview ≤ 5 %, McNemar frente a R con
#   diferencia > 0 y p < 0,05; gana la mayor diferencia (empate: menor tiempo por semilla).
#   Regla de no inferioridad (B2): 0 errores, fallbacks ≤ 5 %, cota inferior unilateral 95 % de la
#   diferencia pareada frente a R > −1,5 pts y tiempo por semilla ≤ 85 % del de R.
#   Si nada pasa, H 2.2 = H 2.1 (H3) y no hay confirmación.
# Espera a MCTS_H_STEPA_DONE; luego actualiza el clon de medición (ff-only) y exige el commit congelado.
# Caché de la calculadora por corrida y borrada al terminar (regla: nunca > 250 GB).
# Resumen: $OUT/h22_summary.txt. Marca: MCTS_H_V22_DONE.
set -u
OUT=${1:-$HOME/iapsvgc_runs/mcts_h_v22}
FROZEN=${FROZEN:?congelar el commit del pre-registro de H 2.2 (FROZEN=<sha>)}
REPO=${REPO:?clon de medición (REPO=<ruta>)}
STEPA=${STEPA:-$HOME/iapsvgc_runs/mcts_h_stepA}
V21=${V21:-$HOME/iapsvgc_runs/mcts_h_v21}       # H3 contra joint en 9101–9117 (referencia R)
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

# ------------------------------------------------ espera al paso A y actualiza el clon
log "esperando MCTS_H_STEPA_DONE en $STEPA/queue.log"
until grep -q MCTS_H_STEPA_DONE "$STEPA/queue.log" 2>/dev/null; do sleep 120; done
while ps -eo cmd | grep -q "[r]un_robust_e2e --teams"; do sleep 60; done
cd "$REPO"
# los logs/ que escriben las corridas pueden estar modificados; el resto debe estar limpio
if [ -n "$(git status --porcelain --untracked-files=no -- . ':!logs')" ]; then
  log "ERROR: el clon $REPO tiene cambios locales fuera de logs/; no se actualiza"; log "MCTS_H_V22_DONE"; exit 1
fi
if ! { git fetch -q origin && git merge -q --ff-only origin/feature/Champions; }; then
  log "ERROR: no se pudo actualizar el clon (ff-only)"; log "MCTS_H_V22_DONE"; exit 1
fi
if ! git merge-base --is-ancestor "$FROZEN" HEAD; then
  log "ERROR: $FROZEN no está en el clon tras actualizar"; log "MCTS_H_V22_DONE"; exit 1
fi

arm_env() {
  case $1 in
    b1) echo "MCTS_NASH_VERSION=2 MCTS_NASH_MATRIX=6,4,3 MCTS_NASH_DECISION=visits" ;;
    b2) echo "MCTS_NASH_VERSION=2 MCTS_NASH_MATRIX=8,6,4 MCTS_NASH_DECISION=visits MCTS_NASH_SOLVER=rm+" ;;
    b3) echo "MCTS_NASH_VERSION=2 MCTS_NASH_MATRIX=8,6,4 MCTS_NASH_DECISION=visits MCTS_NASH_BACKUP_MIX=0.5" ;;
    *)  echo "ERROR_ARM" ;;
  esac
}

# run <config> <seed> <variante> <player_b> [args...]
run() {
  local cfg=$1 seed=$2 arm=$3 pb=$4; shift 4
  local venv; venv=$(arm_env "$arm")
  local rpt="$OUT/reports/${cfg}_s${seed}.txt"
  if [ -s "$rpt" ]; then log "$cfg s$seed ya hecho (reanudación)"; return 0; fi
  rm -f logs/robust_e2e_report.txt
  export DAMAGE_CALC_CACHE="$OUT/calc_cache_run.sqlite"
  rm -f "$DAMAGE_CALC_CACHE" "$DAMAGE_CALC_CACHE"-wal "$DAMAGE_CALC_CACHE"-shm
  local t0=$(date +%s)
  env $venv MCTS_STATS_FILE="$OUT/anomalies_${cfg}.tsv" python -m showdown.inference_sim.run_robust_e2e \
    --teams 300 --procs "$PROCS" --seed "$seed" --restart-ps --player-a mcts_puct_nash --player-b "$pb" "$@" \
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
secs() { cat "$1"_s91*.sec 2>/dev/null | awk '{s+=$1; n++} END{if(n) printf "%.0f", s/n; else print 99999}'; }
mc() { python -m showdown.inference_sim.confirm_stats mcnemar --a $1 --b $2; }

clean_ps
HEAD_C=$(git rev-parse --short HEAD)
log "inicio commit $HEAD_C (congelado $FROZEN)"
if ! git diff --quiet "$FROZEN" HEAD -- showdown/rl/mcts_puct showdown/rl/mcts_puct_nash showdown/battle_bots/greedy_joint \
     showdown/battle_bots/greedy_solo showdown/battle_bots/expectimax_bayesian \
     showdown/inference_sim/validated_greedy_player.py showdown/stat_inference.py showdown/meta_sp_prior.py; then
  log "ERROR: el código congelado cambió desde $FROZEN; no se mide (pre-registro §4.8)"
  log "MCTS_H_V22_DONE"; exit 1
fi
NR=$(ls "$V21"/raw/H3_s91*.txt 2>/dev/null | wc -l)
[ "$NR" = 17 ] || { log "ERROR: faltan las 17 semillas de R (H3) en $V21/raw ($NR)"; log "MCTS_H_V22_DONE"; exit 1; }
OMP_NUM_THREADS=4 python -m pytest -q -p no:cacheprovider tests/test_mcts_puct_nash_v22.py tests/test_mcts_puct_nash_v21.py \
  tests/test_mcts_puct_nash_v2.py tests/test_rl_mcts_puct_nash.py tests/test_mcts_puct_s3c.py tests/test_mcts_puct_s3.py \
  tests/test_mcts_search_preview.py > "$OUT/tests.txt" 2>&1
rc=$?; log "tests rc=$rc ($(tail -1 "$OUT/tests.txt"))"
[ $rc = 0 ] || { log "ERROR: tests fallan; se detiene"; log "MCTS_H_V22_DONE"; exit 1; }

# ------------------------------------------------ smoke: las tres variantes bajo carga (56 equipos)
for a in b1 b2 b3; do
  rm -f "$OUT/anomalies_smoke_$a.tsv"
  run "smoke_$a" 9001 "$a" greedy_joint --teams 56 --timeout 2400
  SERR=$(grep -oE '^series_errors: *[0-9]+' "$OUT/reports/smoke_${a}_s9001.txt" 2>/dev/null | grep -oE '[0-9]+$')
  NFB=$(nfb "smoke_$a"); NOM=$(grep -vc '^preview_search_fallback' "$OUT/anomalies_smoke_$a.tsv" 2>/dev/null); NOM=${NOM:-0}
  log "smoke $a: series_errors=${SERR:-?} fallbacks_preview=$NFB otras_anomalías=$NOM"
  if [ "${SERR:-1}" != 0 ] || [ "$NOM" != 0 ] || [ "$NFB" -gt 3 ]; then
    log "ERROR: smoke $a con errores, anomalías o > 5 % de fallbacks; se detiene"; log "MCTS_H_V22_DONE"; exit 1
  fi
done

# ------------------------------------------------ exploración
for a in B1:b1 B2:b2 B3:b3; do
  for s in $(seq 9101 9117); do run "${a%%:*}" "$s" "${a#*:}" greedy_joint --timeout 3000; done
done

S="$OUT/h22_summary.txt"
SR=$(secs "$V21/reports/H3")
{
  echo "=== Agente H 2.2: exploración — commit $HEAD_C (congelado $FROZEN) ==="
  echo "Pre-registro: docs/mcts_puct_nash_spec.md §4d, protocolo §4.8. R = H3 de mcts_h_v21 ($SR s/semilla)."
  echo; echo "--- R (H3) contra joint"; python -m showdown.inference_sim.confirm_stats binom "$V21"/raw/H3_s91*.txt
  for c in B1 B2 B3; do
    echo; echo "--- $c contra joint ($(secs "$OUT/reports/$c") s/semilla; fallbacks preview: $(nfb $c), series_errors: $(nerr $c))"
    python -m showdown.inference_sim.confirm_stats binom "$OUT"/raw/${c}_s91*.txt
    echo "--- $c frente a R"; mc "$OUT/raw/${c}_s91*.txt" "$V21/raw/H3_s91*.txt"
  done
} > "$S" 2>&1
log "exploración escrita en $S"

python "$REPO/scripts/server/h22_rules.py" "$S" "$(nfb B1) $(nfb B2) $(nfb B3)" "$(nerr B1) $(nerr B2) $(nerr B3)" \
  "$(secs "$OUT/reports/B1") $(secs "$OUT/reports/B2") $(secs "$OUT/reports/B3")" "$SR" >> "$S" 2>&1
log "reglas: $(grep -E '^Brazo ganador|^B2 no inferior' "$S" | tr '\n' ' ')"
log "MCTS_H_V22_DONE"
