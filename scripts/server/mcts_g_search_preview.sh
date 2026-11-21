#!/bin/bash
# PREVIEW POR BÚSQUEDA PARA G (MCTS+PUCT v5), pre-registrado el 21-11-2026 antes de medir
# (docs/mcts_puct_diagnostico.md §8 y docs/protocolo_evaluacion.md §4.4). Solo baselines.
#
#   Exploración (9101–9117 × 300, contra greedy_joint): S0 preview greedy, S1 search-mean,
#     S2 search-maximin. Elegible = 0 errores y fallbacks del preview ≤ 5 % de las series. Se
#     elige la elegible con mayor diferencia McNemar frente a S0; pasa a confirmación si
#     diferencia > 0 y p < 0,05. Si no, G se queda con el preview greedy y la cola termina.
#   Confirmación (C9 = 10001–10017, C10 = 10101–10117): X1 search > greedy contra greedy_joint,
#     X2 search > greedy contra expectimax (McNemar, mismas series; Holm m = 2).
# Caché de la calculadora por corrida y borrada al terminar (regla: nunca > 250 GB).
# Resumen: $OUT/search_summary.txt. Marca: MCTS_G_SEARCH_DONE.
set -u
OUT=${1:-$HOME/iapsvgc_runs/mcts_g_search}
FROZEN=${FROZEN:-118e998}
WAIT_FILE=${WAIT_FILE:-$HOME/iapsvgc_runs/mcts_g_confirm_v5/queue.log}
WAIT_PAT=${WAIT_PAT:-MCTS_G_CONFIRM_V5_DONE}
PROCS=${PROCS:-28}
export MCTS_PREVIEW_DEADLINE=${MCTS_PREVIEW_DEADLINE:-90}
mkdir -p "$OUT"/reports "$OUT"/raw
source ~/miniconda3/etc/profile.d/conda.sh; conda activate kuroneko
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
export PYTHONUNBUFFERED=1
unset MCTS_PUCT_VERSION MCTS_SIMS MCTS_SEED MCTS_S3_CALC MCTS_PREVIEW MCTS_PREVIEW_CKPT MCTS_PREVIEW_DECISION \
  MCTS_PREVIEW_PHASE PS_TEAM_SPLIT PS_PREVIEW_A PS_PREVIEW_B PS_B_VALIDATED PS_SP_INFO_A PS_SP_INFO_B MCTS_STATS_FILE \
  RL_OBS_ABILITIES RL_OBS_CALC PREVIEW_FORCE_ORDER
log() { echo "$(date '+%F %T') $*" >> "$OUT/queue.log"; }
clean_ps() { pkill -f "pokemon-showdown.*start" 2>/dev/null; sleep 5; }

# run <config> <seed> <preview: greedy|mean|maximin> <player_b> [args...]   (G = v5)
run() {
  local cfg=$1 seed=$2 pv=$3 pb=$4; shift 4
  local rpt="$OUT/reports/${cfg}_s${seed}.txt"
  if [ -s "$rpt" ]; then log "$cfg s$seed ya hecho (reanudación)"; return 0; fi
  rm -f logs/robust_e2e_report.txt
  export DAMAGE_CALC_CACHE="$OUT/calc_cache_run.sqlite"
  rm -f "$DAMAGE_CALC_CACHE" "$DAMAGE_CALC_CACHE"-wal "$DAMAGE_CALC_CACHE"-shm
  local t0=$(date +%s) rc
  if [ "$pv" = greedy ]; then
    MCTS_PUCT_VERSION=5 MCTS_STATS_FILE="$OUT/anomalies_${cfg}.tsv" python -m showdown.inference_sim.run_robust_e2e \
      --teams 300 --procs "$PROCS" --seed "$seed" --restart-ps --player-a mcts_puct --player-b "$pb" \
      --preview-a greedy "$@" > "$OUT/raw/${cfg}_s${seed}.txt" 2>&1
    rc=$?
  else
    MCTS_PUCT_VERSION=5 MCTS_PREVIEW=search MCTS_PREVIEW_DECISION=$pv MCTS_STATS_FILE="$OUT/anomalies_${cfg}.tsv" \
      python -m showdown.inference_sim.run_robust_e2e --teams 300 --procs "$PROCS" --seed "$seed" --restart-ps \
      --player-a mcts_puct --player-b "$pb" "$@" > "$OUT/raw/${cfg}_s${seed}.txt" 2>&1
    rc=$?
  fi
  rm -f "$DAMAGE_CALC_CACHE" "$DAMAGE_CALC_CACHE"-wal "$DAMAGE_CALC_CACHE"-shm
  if [ ! -s logs/robust_e2e_report.txt ]; then
    log "ERROR: $cfg s$seed rc=$rc sin reporte (ver raw/${cfg}_s${seed}.txt)"; clean_ps; return 1
  fi
  cp logs/robust_e2e_report.txt "$rpt.tmp" && mv "$rpt.tmp" "$rpt"
  log "$cfg s$seed rc=$rc $(( $(date +%s) - t0 ))s $(grep -E '^player_a_wins|^series_errors' "$rpt" | tr '\n' ' ')"
  clean_ps
  return $rc
}
nfb() { local n; n=$(grep -c '^preview_search_fallback' "$OUT/anomalies_$1.tsv" 2>/dev/null); echo "${n:-0}"; }
nerr() { cat "$OUT"/reports/$1_s*.txt 2>/dev/null | grep -oE '^series_errors: *[0-9]+' | awk '{s+=$2} END{print s+0}'; }

log "esperando /$WAIT_PAT/ en $WAIT_FILE"
until grep -qE "$WAIT_PAT" "$WAIT_FILE" 2>/dev/null; do sleep 60; done
clean_ps
HEAD_C=$(git rev-parse --short HEAD)
log "inicio commit $HEAD_C (congelado $FROZEN) deadline=${MCTS_PREVIEW_DEADLINE}s"
if ! git diff --quiet "$FROZEN" HEAD -- showdown/rl/mcts_puct showdown/battle_bots/greedy_joint \
     showdown/battle_bots/greedy_solo showdown/battle_bots/expectimax_bayesian; then
  log "ERROR: el código congelado cambió desde $FROZEN; no se mide (pre-registro §4.4)"
  log "MCTS_G_SEARCH_DONE"; exit 1
fi
OMP_NUM_THREADS=4 python -m pytest -q -p no:cacheprovider tests/test_mcts_search_preview.py tests/test_mcts_puct_s3c.py \
  tests/test_mcts_puct_s3b.py tests/test_mcts_puct_s3.py > "$OUT/tests.txt" 2>&1
rc=$?; log "tests rc=$rc ($(tail -1 "$OUT/tests.txt"))"
[ $rc = 0 ] || { log "ERROR: tests fallan; se detiene"; log "MCTS_G_SEARCH_DONE"; exit 1; }

# ------------------------------------------------ smoke: preview por búsqueda bajo carga (28 procs)
rm -f "$OUT/anomalies_smoke.tsv"
run smoke 9001 mean greedy_joint --teams 56 --timeout 1800
SERR=$(grep -oE '^series_errors: *[0-9]+' "$OUT/reports/smoke_s9001.txt" 2>/dev/null | grep -oE '[0-9]+$')
NFB=$(nfb smoke); NOM=$(grep -vc '^preview_search_fallback' "$OUT/anomalies_smoke.tsv" 2>/dev/null); NOM=${NOM:-0}
log "smoke: series_errors=${SERR:-?} fallbacks_preview=$NFB otras_anomalías=$NOM ($(cut -f1 "$OUT/anomalies_smoke.tsv" 2>/dev/null | sort | uniq -c | tr '\n' ' '))"
if [ "${SERR:-1}" != 0 ] || [ "$NOM" != 0 ] || [ "$NFB" -gt 3 ]; then
  log "ERROR: smoke con errores, anomalías o > 5 % de fallbacks; se detiene"; log "MCTS_G_SEARCH_DONE"; exit 1
fi

# ------------------------------------------------ exploración
for s in $(seq 9101 9117); do run S0 "$s" greedy greedy_joint --timeout 1200; done
for s in $(seq 9101 9117); do run S1 "$s" mean greedy_joint --timeout 1800; done
for s in $(seq 9101 9117); do run S2 "$s" maximin greedy_joint --timeout 1800; done

S="$OUT/search_summary.txt"
mc() { python -m showdown.inference_sim.confirm_stats mcnemar --a $1 --b $2; }
{
  echo "=== Preview por búsqueda para G v5 — commit $HEAD_C (congelado $FROZEN), deadline ${MCTS_PREVIEW_DEADLINE}s ==="
  echo "Pre-registro: docs/mcts_puct_diagnostico.md §8, protocolo §4.4."
  for c in S0 S1 S2; do
    echo; echo "--- $c contra joint (fallbacks preview: $(nfb $c), series_errors: $(nerr $c))"
    python -m showdown.inference_sim.confirm_stats binom "$OUT"/raw/${c}_s91*.txt
  done
  echo; echo "--- S1 (mean) frente a S0 (greedy)"; mc "$OUT/raw/S1_s91*.txt" "$OUT/raw/S0_s91*.txt"
  echo "--- S2 (maximin) frente a S0 (greedy)"; mc "$OUT/raw/S2_s91*.txt" "$OUT/raw/S0_s91*.txt"
} > "$S" 2>&1
log "exploración escrita en $S"

# Regla de selección (pre-registrada)
SEL=$(python - "$S" "$(nfb S1)" "$(nfb S2)" "$(nerr S1)" "$(nerr S2)" <<'PY'
import re, sys
txt = open(sys.argv[1], encoding='utf-8').read()
fb = {'S1': int(sys.argv[2]), 'S2': int(sys.argv[3])}
er = {'S1': int(sys.argv[4]), 'S2': int(sys.argv[5])}
best = None
for c, name in (('S1', 'mean'), ('S2', 'maximin')):
    m = re.search(rf'--- {c} \(\w+\) frente a S0 \(greedy\)\n.*?emparejadas=(\d+).*?diferencia=([+-][\d.]+) pts p=([\d.e+-]+)', txt, re.S)
    if not m:
        continue
    n, d, p = int(m.group(1)), float(m.group(2)), float(m.group(3))
    if er[c] != 0 or fb[c] > 0.05 * 5100:
        print(f'# {c} no elegible (errores={er[c]}, fallbacks={fb[c]})', file=sys.stderr); continue
    if best is None or d > best[1]:
        best = (name, d, p)
if best and best[1] > 0 and best[2] < 0.05:
    print(best[0])
else:
    print('none')
PY
)
log "selección pre-registrada: $SEL"
echo; echo "Selección pre-registrada: $SEL" >> "$S"
if [ "$SEL" = none ]; then
  log "ninguna variante mejora al preview greedy; G se queda con el greedy"; log "MCTS_G_SEARCH_DONE"; exit 0
fi

# ------------------------------------------------ confirmación (C9, C10)
for blk in 100 101; do
  for s in $(seq ${blk}01 ${blk}17); do run X1s "$s" "$SEL" greedy_joint --timeout 1800; done
  for s in $(seq ${blk}01 ${blk}17); do run X1g "$s" greedy greedy_joint --timeout 1200; done
  for s in $(seq ${blk}01 ${blk}17); do run X2s "$s" "$SEL" expectimax_bayesian --timeout 2400; done
  for s in $(seq ${blk}01 ${blk}17); do run X2g "$s" greedy expectimax_bayesian --timeout 2400; done
done
{
  echo; echo "=== Confirmación (variante $SEL; C9 = 10001–10017, C10 = 10101–10117) ==="
  PS_=""
  for h in X1 X2; do
    echo; echo "--- $h C9+C10 (search frente a greedy)"
    line=$(mc "$OUT/raw/${h}s_s10[01]*.txt" "$OUT/raw/${h}g_s10[01]*.txt"); echo "$line"
    PS_="$PS_ $(echo "$line" | grep -oE ' p=[^ ]+' | head -1 | cut -d= -f2)"
    echo "--- $h C9"; mc "$OUT/raw/${h}s_s100*.txt" "$OUT/raw/${h}g_s100*.txt"
    echo "--- $h C10"; mc "$OUT/raw/${h}s_s101*.txt" "$OUT/raw/${h}g_s101*.txt"
    for v in s g; do
      echo "--- ${h}$v C9+C10 (fallbacks preview: $(nfb ${h}$v))"
      python -m showdown.inference_sim.confirm_stats binom "$OUT"/raw/${h}${v}_s10[01]*.txt
    done
  done
  echo; echo "--- Holm (m=2; p de X1, X2)"; python -m showdown.inference_sim.confirm_stats holm $PS_
} >> "$S" 2>&1
log "estadística escrita en $S"
log "MCTS_G_SEARCH_DONE"
