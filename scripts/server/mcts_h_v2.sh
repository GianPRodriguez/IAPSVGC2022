#!/bin/bash
# AGENTE H 2.0 (MCTS–PUCT/Nash con lo aprendido en G), pre-registrado el 22-11-2026 antes de medir
# (docs/mcts_puct_nash_spec.md v2.0 y docs/protocolo_evaluacion.md §4.6). Solo baselines (y G 5.2 como
# referencia contra el mismo rival; H nunca juega contra G).
#
#   H 2.0 = SM-MCTS open-loop con S3 fase 3c (combate de G v5), Top-K por S3, selección desacoplada contra
#   la mezcla de Nash, valor de Nash al padre, 200 sims; preview por búsqueda de G (`mean`, 90 s); SP meta.
#   H 1.7 = SM-MCTS B1 original (S2, árbol cerrado, 50 sims) con el mismo preview y SP.
#   Exploración (9101–9117 × 300, contra greedy_joint): N1 = H 1.7, N2 = H 2.0. Pasa a confirmación si N2
#     tiene 0 errores, fallbacks del preview ≤ 5 % y McNemar N2 frente a N1 con diferencia > 0 y p < 0,05.
#     Descriptivo: N2 frente a G 5.2 (E0 de la cola 3e, mismas series).
#   Confirmación (C13 = 10401–10417, C14 = 10501–10517): W1 H2 > greedy_joint, W2 H2 > expectimax,
#     W3 H2 > greedy_solo (binomial con IC cluster), W4 H2 frente a G 5.2 (McNemar bilateral, ambos contra
#     joint, mismas series); Holm m = 4. H 2.0 pasa a ser el H de la Conclusión si W1–W3 se confirman.
# Caché de la calculadora por corrida y borrada al terminar (regla: nunca > 250 GB).
# Resumen: $OUT/h_summary.txt. Marca: MCTS_H_V2_DONE.
set -u
OUT=${1:-$HOME/iapsvgc_runs/mcts_h_v2}
FROZEN=${FROZEN:?congelar el commit de H 2.0 (FROZEN=<sha>)}
GREF=${GREF:-$HOME/iapsvgc_runs/mcts_g_v7/raw}   # G 5.2 contra joint en 9101–9117 (E0 de la cola 3e)
WAIT_FILE=${WAIT_FILE:-$HOME/iapsvgc_runs/mcts_g_v7/queue.log}
WAIT_PAT=${WAIT_PAT:-MCTS_G_V7_DONE}
PROCS=${PROCS:-28}
mkdir -p "$OUT"/reports "$OUT"/raw
source ~/miniconda3/etc/profile.d/conda.sh; conda activate kuroneko
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
export PYTHONUNBUFFERED=1
unset MCTS_PUCT_VERSION MCTS_NASH_VERSION MCTS_NASH_SIMS MCTS_NASH_SEED MCTS_SIMS MCTS_SEED MCTS_S3_CALC MCTS_PREVIEW MCTS_PREVIEW_CKPT MCTS_PREVIEW_DECISION \
  MCTS_PREVIEW_PHASE MCTS_PREVIEW_DEADLINE PS_TEAM_SPLIT PS_PREVIEW_A PS_PREVIEW_B PS_B_VALIDATED PS_SP_INFO_A \
  PS_SP_INFO_B PS_SP_INFO_DEBUG MCTS_STATS_FILE RL_OBS_ABILITIES RL_OBS_CALC PREVIEW_FORCE_ORDER
log() { echo "$(date '+%F %T') $*" >> "$OUT/queue.log"; }
clean_ps() { pkill -f "[p]okemon-showdown.*start" 2>/dev/null; sleep 5; }

# run <config> <seed> <agente:versión> <player_b> [args...]   (h:2, h:1 o g:5; preview y sp_info por defecto)
run() {
  local cfg=$1 seed=$2 av=$3 pb=$4; shift 4
  local pa=mcts_puct_nash venv="MCTS_NASH_VERSION=${av#*:}"
  [ "${av%%:*}" = g ] && { pa=mcts_puct; venv="MCTS_PUCT_VERSION=${av#*:}"; }
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

log "esperando /$WAIT_PAT/ en $WAIT_FILE"
until grep -qE "$WAIT_PAT" "$WAIT_FILE" 2>/dev/null; do sleep 60; done
clean_ps
HEAD_C=$(git rev-parse --short HEAD)
log "inicio commit $HEAD_C (congelado $FROZEN)"
if ! git diff --quiet "$FROZEN" HEAD -- showdown/rl/mcts_puct showdown/rl/mcts_puct_nash showdown/battle_bots/greedy_joint \
     showdown/battle_bots/greedy_solo showdown/battle_bots/expectimax_bayesian \
     showdown/inference_sim/validated_greedy_player.py showdown/stat_inference.py showdown/meta_sp_prior.py; then
  log "ERROR: el código congelado cambió desde $FROZEN; no se mide (pre-registro §4.6)"
  log "MCTS_H_V2_DONE"; exit 1
fi
OMP_NUM_THREADS=4 python -m pytest -q -p no:cacheprovider tests/test_mcts_puct_nash_v2.py tests/test_rl_mcts_puct_nash.py \
  tests/test_mcts_puct_s3c.py tests/test_mcts_puct_s3.py tests/test_mcts_search_preview.py > "$OUT/tests.txt" 2>&1
rc=$?; log "tests rc=$rc ($(tail -1 "$OUT/tests.txt"))"
[ $rc = 0 ] || { log "ERROR: tests fallan; se detiene"; log "MCTS_H_V2_DONE"; exit 1; }

# ------------------------------------------------ smoke: H 2.0 bajo carga (28 procs)
rm -f "$OUT/anomalies_smoke.tsv"
run smoke 9001 h:2 greedy_joint --teams 56 --timeout 2400
SERR=$(grep -oE '^series_errors: *[0-9]+' "$OUT/reports/smoke_s9001.txt" 2>/dev/null | grep -oE '[0-9]+$')
NFB=$(nfb smoke); NOM=$(grep -vc '^preview_search_fallback' "$OUT/anomalies_smoke.tsv" 2>/dev/null); NOM=${NOM:-0}
log "smoke: series_errors=${SERR:-?} fallbacks_preview=$NFB otras_anomalías=$NOM ($(cut -f1 "$OUT/anomalies_smoke.tsv" 2>/dev/null | sort | uniq -c | tr '\n' ' '))"
if [ "${SERR:-1}" != 0 ] || [ "$NOM" != 0 ] || [ "$NFB" -gt 3 ]; then
  log "ERROR: smoke con errores, anomalías o > 5 % de fallbacks; se detiene"; log "MCTS_H_V2_DONE"; exit 1
fi

# ------------------------------------------------ exploración
for s in $(seq 9101 9117); do run N2 "$s" h:2 greedy_joint --timeout 2400; done
# N1_SEEDS: desviación del 22-11-2026 (protocolo §4.6): N1 cortado en 4 semillas (10,8 %, ~900–1400 s/semilla)
for s in ${N1_SEEDS:-$(seq 9101 9117)}; do run N1 "$s" h:1 greedy_joint --timeout 2400; done

S="$OUT/h_summary.txt"
mc() { python -m showdown.inference_sim.confirm_stats mcnemar --a $1 --b $2; }
{
  echo "=== Agente H 2.0 frente a H 1.7 — commit $HEAD_C (congelado $FROZEN) ==="
  echo "Pre-registro: docs/mcts_puct_nash_spec.md v2.0, protocolo §4.6. Preview por búsqueda de G, SP meta."
  for c in N1 N2; do
    echo; echo "--- $c contra joint ($(secs $c); fallbacks preview: $(nfb $c), series_errors: $(nerr $c))"
    python -m showdown.inference_sim.confirm_stats binom "$OUT"/raw/${c}_s91*.txt
  done
  echo; echo "--- N2 (H 2.0) frente a N1 (H 1.7)"; mc "$OUT/raw/N2_s91*.txt" "$OUT/raw/N1_s91*.txt"
  echo; echo "--- descriptivo: N2 (H 2.0) frente a G 5.2 (E0 de la cola 3e)"; mc "$OUT/raw/N2_s91*.txt" "$GREF/E0_s91*.txt"
} > "$S" 2>&1
log "exploración escrita en $S"

GO=$(python - "$S" "$(nfb N2)" "$(nerr N2)" <<'PY'
import re, sys
txt = open(sys.argv[1], encoding='utf-8').read()
fb, er = int(sys.argv[2]), int(sys.argv[3])
m = re.search(r'--- N2 \(H 2.0\) frente a N1 \(H 1.7\)\n.*?diferencia=([+-][\d.]+) pts p=([\d.e+-]+)', txt, re.S)
ok = bool(m) and er == 0 and fb <= 0.05 * 5100 and float(m.group(1)) > 0 and float(m.group(2)) < 0.05
print('yes' if ok else 'no')
PY
)
log "regla pre-registrada (pasa a confirmación): $GO"
echo; echo "Pasa a confirmación: $GO" >> "$S"
if [ "$GO" != yes ]; then log "H 2.0 no mejora a H 1.7 en exploración; se revisa antes de confirmar"; log "MCTS_H_V2_DONE"; exit 0; fi

# ------------------------------------------------ confirmación (C13, C14)
for blk in 104 105; do
  for s in $(seq ${blk}01 ${blk}17); do run Wj "$s" h:2 greedy_joint --timeout 2400; done
  for s in $(seq ${blk}01 ${blk}17); do run Gj "$s" g:5 greedy_joint --timeout 1800; done
  for s in $(seq ${blk}01 ${blk}17); do run We "$s" h:2 expectimax_bayesian --timeout 3000; done
  for s in $(seq ${blk}01 ${blk}17); do run Ws "$s" h:2 greedy_solo --timeout 2400; done
done
{
  echo; echo "=== Confirmación (C13 = 10401–10417, C14 = 10501–10517) ==="
  PS_=""
  for h in Wj:W1_contra_joint We:W2_contra_expectimax Ws:W3_contra_solo; do
    c=${h%%:*}; echo; echo "--- ${h#*:} ($c, $(secs $c); fallbacks preview: $(nfb $c), series_errors: $(nerr $c))"
    line=$(python -m showdown.inference_sim.confirm_stats binom "$OUT"/raw/${c}_s10[45]*.txt); echo "$line"
    PS_="$PS_ $(echo "$line" | grep -oE ' p=[^ ]+' | head -1 | cut -d= -f2)"
    echo "    C13: $(python -m showdown.inference_sim.confirm_stats binom "$OUT"/raw/${c}_s104*.txt)"
    echo "    C14: $(python -m showdown.inference_sim.confirm_stats binom "$OUT"/raw/${c}_s105*.txt)"
  done
  echo; echo "--- W4 H 2.0 frente a G 5.2 (McNemar bilateral, contra joint; C13+C14)"
  line=$(mc "$OUT/raw/Wj_s10[45]*.txt" "$OUT/raw/Gj_s10[45]*.txt"); echo "$line"
  PS_="$PS_ $(echo "$line" | grep -oE ' p=[^ ]+' | head -1 | cut -d= -f2)"
  echo "    C13: $(mc "$OUT/raw/Wj_s104*.txt" "$OUT/raw/Gj_s104*.txt")"
  echo "    C14: $(mc "$OUT/raw/Wj_s105*.txt" "$OUT/raw/Gj_s105*.txt")"
  echo "--- Gj (G 5.2 contra joint, referencia; $(secs Gj))"; python -m showdown.inference_sim.confirm_stats binom "$OUT"/raw/Gj_s10[45]*.txt
  echo; echo "--- Holm (m=4; p de W1, W2, W3, W4)"; python -m showdown.inference_sim.confirm_stats holm $PS_
} >> "$S" 2>&1
log "estadística escrita en $S"
log "MCTS_H_V2_DONE"
