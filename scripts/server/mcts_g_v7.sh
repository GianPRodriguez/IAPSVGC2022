#!/bin/bash
# FASE 3e (OBJETOS) PARA G: MCTS+PUCT v7 frente a v5.2, pre-registrado el 22-11-2026 antes de medir
# (docs/mcts_puct_diagnostico.md §10 y docs/protocolo_evaluacion.md §4.5). Solo baselines.
#
#   Ambos con el preview por defecto de G (búsqueda `mean`, deadline 90 s; la fase de S3 del preview
#   sigue a la versión: 3c en v5, 3e en v7) y el sp_info por defecto del código congelado.
#   Exploración (9101–9117 × 300, contra greedy_joint, mismas series): E0 = v5.2, E1 = v7.
#     Pasa a confirmación si 0 errores, fallbacks del preview ≤ 5 % y McNemar E1 frente a E0 con
#     diferencia > 0 y p < 0,05. Si no, G se queda en v5.2 y la cola termina.
#   Confirmación (C11 = 10201–10217, C12 = 10301–10317): Z1 v7 > greedy_joint, Z2 v7 > expectimax,
#     Z3 v7 > greedy_solo (binomial con IC cluster), Z4 v7 > v5.2 (McNemar, ambos contra joint);
#     Holm m = 4. v7 pasa a ser el default si Z4 se confirma y Z1–Z3 se mantienen.
# Caché de la calculadora por corrida y borrada al terminar (regla: nunca > 250 GB).
# Resumen: $OUT/v7_summary.txt. Marca: MCTS_G_V7_DONE.
set -u
OUT=${1:-$HOME/iapsvgc_runs/mcts_g_v7}
FROZEN=${FROZEN:?congelar el commit de 3e (FROZEN=<sha>)}
WAIT_FILE=${WAIT_FILE:-$HOME/iapsvgc_runs/mcts_g_spinfo/queue.log}
WAIT_PAT=${WAIT_PAT:-MCTS_G_SPINFO_DONE}
PROCS=${PROCS:-28}
mkdir -p "$OUT"/reports "$OUT"/raw
source ~/miniconda3/etc/profile.d/conda.sh; conda activate kuroneko
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
export PYTHONUNBUFFERED=1
unset MCTS_PUCT_VERSION MCTS_SIMS MCTS_SEED MCTS_S3_CALC MCTS_PREVIEW MCTS_PREVIEW_CKPT MCTS_PREVIEW_DECISION \
  MCTS_PREVIEW_PHASE MCTS_PREVIEW_DEADLINE PS_TEAM_SPLIT PS_PREVIEW_A PS_PREVIEW_B PS_B_VALIDATED PS_SP_INFO_A \
  PS_SP_INFO_B PS_SP_INFO_DEBUG MCTS_STATS_FILE RL_OBS_ABILITIES RL_OBS_CALC PREVIEW_FORCE_ORDER
log() { echo "$(date '+%F %T') $*" >> "$OUT/queue.log"; }
clean_ps() { pkill -f "[p]okemon-showdown.*start" 2>/dev/null; sleep 5; }

# run <config> <seed> <version> <player_b> [args...]   (G con su preview y sp_info por defecto)
run() {
  local cfg=$1 seed=$2 ver=$3 pb=$4; shift 4
  local rpt="$OUT/reports/${cfg}_s${seed}.txt"
  if [ -s "$rpt" ]; then log "$cfg s$seed ya hecho (reanudación)"; return 0; fi
  rm -f logs/robust_e2e_report.txt
  export DAMAGE_CALC_CACHE="$OUT/calc_cache_run.sqlite"
  rm -f "$DAMAGE_CALC_CACHE" "$DAMAGE_CALC_CACHE"-wal "$DAMAGE_CALC_CACHE"-shm
  local t0=$(date +%s)
  MCTS_PUCT_VERSION=$ver MCTS_STATS_FILE="$OUT/anomalies_${cfg}.tsv" python -m showdown.inference_sim.run_robust_e2e \
    --teams 300 --procs "$PROCS" --seed "$seed" --restart-ps --player-a mcts_puct --player-b "$pb" "$@" \
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

log "esperando /$WAIT_PAT/ en $WAIT_FILE"
until grep -qE "$WAIT_PAT" "$WAIT_FILE" 2>/dev/null; do sleep 60; done
clean_ps
HEAD_C=$(git rev-parse --short HEAD)
log "inicio commit $HEAD_C (congelado $FROZEN)"
if ! git diff --quiet "$FROZEN" HEAD -- showdown/rl/mcts_puct showdown/battle_bots/greedy_joint \
     showdown/battle_bots/greedy_solo showdown/battle_bots/expectimax_bayesian \
     showdown/inference_sim/validated_greedy_player.py showdown/stat_inference.py showdown/meta_sp_prior.py; then
  log "ERROR: el código congelado cambió desde $FROZEN; no se mide (pre-registro §4.5)"
  log "MCTS_G_V7_DONE"; exit 1
fi
OMP_NUM_THREADS=4 python -m pytest -q -p no:cacheprovider tests/test_mcts_puct_s3e.py tests/test_mcts_puct_s3c.py \
  tests/test_mcts_puct_s3b.py tests/test_mcts_puct_s3.py tests/test_mcts_search_preview.py > "$OUT/tests.txt" 2>&1
rc=$?; log "tests rc=$rc ($(tail -1 "$OUT/tests.txt"))"
[ $rc = 0 ] || { log "ERROR: tests fallan; se detiene"; log "MCTS_G_V7_DONE"; exit 1; }

# ------------------------------------------------ smoke: v7 bajo carga (28 procs)
rm -f "$OUT/anomalies_smoke.tsv"
run smoke 9001 7 greedy_joint --teams 56 --timeout 1800
SERR=$(grep -oE '^series_errors: *[0-9]+' "$OUT/reports/smoke_s9001.txt" 2>/dev/null | grep -oE '[0-9]+$')
NFB=$(nfb smoke); NOM=$(grep -vc '^preview_search_fallback' "$OUT/anomalies_smoke.tsv" 2>/dev/null); NOM=${NOM:-0}
log "smoke: series_errors=${SERR:-?} fallbacks_preview=$NFB otras_anomalías=$NOM ($(cut -f1 "$OUT/anomalies_smoke.tsv" 2>/dev/null | sort | uniq -c | tr '\n' ' '))"
if [ "${SERR:-1}" != 0 ] || [ "$NOM" != 0 ] || [ "$NFB" -gt 3 ]; then
  log "ERROR: smoke con errores, anomalías o > 5 % de fallbacks; se detiene"; log "MCTS_G_V7_DONE"; exit 1
fi

# ------------------------------------------------ exploración
for s in $(seq 9101 9117); do run E0 "$s" 5 greedy_joint --timeout 1800; done
for s in $(seq 9101 9117); do run E1 "$s" 7 greedy_joint --timeout 1800; done

S="$OUT/v7_summary.txt"
mc() { python -m showdown.inference_sim.confirm_stats mcnemar --a $1 --b $2; }
{
  echo "=== Fase 3e (objetos): G v7 frente a v5.2 — commit $HEAD_C (congelado $FROZEN) ==="
  echo "Pre-registro: docs/mcts_puct_diagnostico.md §10, protocolo §4.5. Preview por defecto (búsqueda mean)."
  for c in E0 E1; do
    echo; echo "--- $c contra joint ($(secs $c); fallbacks preview: $(nfb $c), series_errors: $(nerr $c))"
    python -m showdown.inference_sim.confirm_stats binom "$OUT"/raw/${c}_s91*.txt
  done
  echo; echo "--- E1 (v7) frente a E0 (v5.2)"; mc "$OUT/raw/E1_s91*.txt" "$OUT/raw/E0_s91*.txt"
} > "$S" 2>&1
log "exploración escrita en $S"

GO=$(python - "$S" "$(nfb E1)" "$(nerr E1)" <<'PY'
import re, sys
txt = open(sys.argv[1], encoding='utf-8').read()
fb, er = int(sys.argv[2]), int(sys.argv[3])
m = re.search(r'--- E1 \(v7\) frente a E0 \(v5.2\)\n.*?diferencia=([+-][\d.]+) pts p=([\d.e+-]+)', txt, re.S)
ok = bool(m) and er == 0 and fb <= 0.05 * 5100 and float(m.group(1)) > 0 and float(m.group(2)) < 0.05
print('yes' if ok else 'no')
PY
)
log "regla pre-registrada (pasa a confirmación): $GO"
echo; echo "Pasa a confirmación: $GO" >> "$S"
if [ "$GO" != yes ]; then log "v7 no mejora a v5.2 en exploración; G se queda en v5.2"; log "MCTS_G_V7_DONE"; exit 0; fi

# ------------------------------------------------ confirmación (C11, C12)
for blk in 102 103; do
  for s in $(seq ${blk}01 ${blk}17); do run Zj7 "$s" 7 greedy_joint --timeout 1800; done
  for s in $(seq ${blk}01 ${blk}17); do run Zj5 "$s" 5 greedy_joint --timeout 1800; done
  for s in $(seq ${blk}01 ${blk}17); do run Ze7 "$s" 7 expectimax_bayesian --timeout 2400; done
  for s in $(seq ${blk}01 ${blk}17); do run Zs7 "$s" 7 greedy_solo --timeout 1800; done
done
{
  echo; echo "=== Confirmación (C11 = 10201–10217, C12 = 10301–10317) ==="
  PS_=""
  for h in Zj7:Z1_contra_joint Ze7:Z2_contra_expectimax Zs7:Z3_contra_solo; do
    c=${h%%:*}; echo; echo "--- ${h#*:} ($c, $(secs $c); fallbacks preview: $(nfb $c), series_errors: $(nerr $c))"
    line=$(python -m showdown.inference_sim.confirm_stats binom "$OUT"/raw/${c}_s10[23]*.txt); echo "$line"
    PS_="$PS_ $(echo "$line" | grep -oE ' p=[^ ]+' | head -1 | cut -d= -f2)"
    echo "    C11: $(python -m showdown.inference_sim.confirm_stats binom "$OUT"/raw/${c}_s102*.txt)"
    echo "    C12: $(python -m showdown.inference_sim.confirm_stats binom "$OUT"/raw/${c}_s103*.txt)"
  done
  echo; echo "--- Z4 v7 frente a v5.2 (McNemar, contra joint; C11+C12)"
  line=$(mc "$OUT/raw/Zj7_s10[23]*.txt" "$OUT/raw/Zj5_s10[23]*.txt"); echo "$line"
  PS_="$PS_ $(echo "$line" | grep -oE ' p=[^ ]+' | head -1 | cut -d= -f2)"
  echo "    C11: $(mc "$OUT/raw/Zj7_s102*.txt" "$OUT/raw/Zj5_s102*.txt")"
  echo "    C12: $(mc "$OUT/raw/Zj7_s103*.txt" "$OUT/raw/Zj5_s103*.txt")"
  echo "--- Zj5 (v5.2 contra joint, referencia)"; python -m showdown.inference_sim.confirm_stats binom "$OUT"/raw/Zj5_s10[23]*.txt
  echo; echo "--- Holm (m=4; p de Z1, Z2, Z3, Z4)"; python -m showdown.inference_sim.confirm_stats holm $PS_
} >> "$S" 2>&1
log "estadística escrita en $S"
log "MCTS_G_V7_DONE"
