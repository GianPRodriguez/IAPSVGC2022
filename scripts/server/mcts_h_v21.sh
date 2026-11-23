#!/bin/bash
# AGENTE H 2.1 (variantes de H 2.0 tras el paso 0), pre-registrado el 23-11-2026 antes de medir
# (docs/mcts_puct_nash_spec.md §4c y docs/protocolo_evaluacion.md §4.7). Solo baselines (G 5.2 como
# referencia descriptiva contra el mismo rival; H nunca juega contra G).
#
#   Exploración (9101–9117 × 300, contra greedy_joint, mismas series):
#     H0 = H 2.0 (N2 de la cola mcts_h_v2, mismo código por defecto; se reutiliza)
#     H1 = matriz chica  MCTS_NASH_MATRIX=8,6,4 (raíz 8, interior 6, rival 4)
#     H2 = decisión por visitas  MCTS_NASH_DECISION=visits
#     H3 = H1 + H2
#     H4 = RM+ con arranque en caliente (10 it./backup, 200 en la raíz) y 350 sims (≈ tiempo de H 2.0)
#   Regla: de los brazos con 0 errores de serie, fallbacks del preview ≤ 5 % y McNemar frente a H0 con
#   diferencia > 0 y p < 0,05, gana el de mayor diferencia (empate: menor tiempo por semilla). Si ninguno
#   pasa, H se queda en 2.0 y no hay confirmación.
#   Confirmación (C15 = 10601–10617, C16 = 10701–10717): V1 H2.1 > greedy_joint, V2 H2.1 > expectimax,
#   V3 H2.1 > greedy_solo (binomial), V4 H2.1 > H2.0 (McNemar, contra joint, mismas series; H 2.0 se
#   vuelve a medir); Holm m = 4. Descriptivo: H2.1 frente a G 5.2 (Gj, contra joint).
# Caché de la calculadora por corrida y borrada al terminar (regla: nunca > 250 GB).
# Resumen: $OUT/h21_summary.txt. Marca: MCTS_H_V21_DONE.
set -u
OUT=${1:-$HOME/iapsvgc_runs/mcts_h_v21}
FROZEN=${FROZEN:?congelar el commit del pre-registro de H 2.1 (FROZEN=<sha>)}
H0REF=${H0REF:-$HOME/iapsvgc_runs/mcts_h_v2/raw}   # N2 = H 2.0 contra joint en 9101–9117
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

clean_ps
HEAD_C=$(git rev-parse --short HEAD)
log "inicio commit $HEAD_C (congelado $FROZEN)"
if ! git diff --quiet "$FROZEN" HEAD -- showdown/rl/mcts_puct showdown/rl/mcts_puct_nash showdown/battle_bots/greedy_joint \
     showdown/battle_bots/greedy_solo showdown/battle_bots/expectimax_bayesian \
     showdown/inference_sim/validated_greedy_player.py showdown/stat_inference.py showdown/meta_sp_prior.py; then
  log "ERROR: el código congelado cambió desde $FROZEN; no se mide (pre-registro §4.7)"
  log "MCTS_H_V21_DONE"; exit 1
fi
NH0=$(ls "$H0REF"/N2_s91*.txt 2>/dev/null | wc -l)
[ "$NH0" = 17 ] || { log "ERROR: faltan las 17 semillas de H0 (N2) en $H0REF ($NH0)"; log "MCTS_H_V21_DONE"; exit 1; }
OMP_NUM_THREADS=4 python -m pytest -q -p no:cacheprovider tests/test_mcts_puct_nash_v21.py tests/test_mcts_puct_nash_v2.py \
  tests/test_rl_mcts_puct_nash.py tests/test_mcts_puct_s3c.py tests/test_mcts_puct_s3.py tests/test_mcts_search_preview.py \
  > "$OUT/tests.txt" 2>&1
rc=$?; log "tests rc=$rc ($(tail -1 "$OUT/tests.txt"))"
[ $rc = 0 ] || { log "ERROR: tests fallan; se detiene"; log "MCTS_H_V21_DONE"; exit 1; }

# ------------------------------------------------ smoke: las cuatro variantes bajo carga (56 equipos)
for a in h1 h2 h3 h4; do
  rm -f "$OUT/anomalies_smoke_$a.tsv"
  run "smoke_$a" 9001 "$a" greedy_joint --teams 56 --timeout 2400
  SERR=$(grep -oE '^series_errors: *[0-9]+' "$OUT/reports/smoke_${a}_s9001.txt" 2>/dev/null | grep -oE '[0-9]+$')
  NFB=$(nfb "smoke_$a"); NOM=$(grep -vc '^preview_search_fallback' "$OUT/anomalies_smoke_$a.tsv" 2>/dev/null); NOM=${NOM:-0}
  log "smoke $a: series_errors=${SERR:-?} fallbacks_preview=$NFB otras_anomalías=$NOM"
  if [ "${SERR:-1}" != 0 ] || [ "$NOM" != 0 ] || [ "$NFB" -gt 3 ]; then
    log "ERROR: smoke $a con errores, anomalías o > 5 % de fallbacks; se detiene"; log "MCTS_H_V21_DONE"; exit 1
  fi
done

# ------------------------------------------------ exploración
for a in H1:h1 H2:h2 H3:h3 H4:h4; do
  for s in $(seq 9101 9117); do run "${a%%:*}" "$s" "${a#*:}" greedy_joint --timeout 3000; done
done

S="$OUT/h21_summary.txt"
{
  echo "=== Agente H 2.1: exploración — commit $HEAD_C (congelado $FROZEN) ==="
  echo "Pre-registro: docs/mcts_puct_nash_spec.md §4c, protocolo §4.7. H0 = N2 de mcts_h_v2 (H 2.0)."
  echo; echo "--- H0 (H 2.0) contra joint"; python -m showdown.inference_sim.confirm_stats binom "$H0REF"/N2_s91*.txt
  for c in H1 H2 H3 H4; do
    echo; echo "--- $c contra joint ($(secs $c); fallbacks preview: $(nfb $c), series_errors: $(nerr $c))"
    python -m showdown.inference_sim.confirm_stats binom "$OUT"/raw/${c}_s91*.txt
    echo "--- $c frente a H0"; mc "$OUT/raw/${c}_s91*.txt" "$H0REF/N2_s91*.txt"
  done
} > "$S" 2>&1
log "exploración escrita en $S"

WIN=$(python - "$S" "$(nfb H1) $(nfb H2) $(nfb H3) $(nfb H4)" "$(nerr H1) $(nerr H2) $(nerr H3) $(nerr H4)" \
      "$(secn H1) $(secn H2) $(secn H3) $(secn H4)" <<'PY'
import re, sys
txt = open(sys.argv[1], encoding='utf-8').read()
fb = [int(x) for x in sys.argv[2].split()]; er = [int(x) for x in sys.argv[3].split()]
sec = [int(x) for x in sys.argv[4].split()]
best = None
for k, c in enumerate(['H1', 'H2', 'H3', 'H4']):
    m = re.search(rf'--- {c} frente a H0\n.*?diferencia=([+-][\d.]+) pts p=([\d.e+-]+)', txt, re.S)
    if not m:
        continue
    d, p = float(m.group(1)), float(m.group(2))
    if er[k] == 0 and fb[k] <= 0.05 * 5100 and d > 0 and p < 0.05:
        key = (d, -sec[k])
        if best is None or key > best[0]:
            best = (key, c)
print(best[1] if best else 'ninguno')
PY
)
log "regla pre-registrada (brazo ganador): $WIN"
echo; echo "Brazo ganador: $WIN" >> "$S"
if [ "$WIN" = ninguno ]; then log "ninguna variante mejora a H 2.0; H se queda en 2.0"; log "MCTS_H_V21_DONE"; exit 0; fi
ARM=$(echo "$WIN" | tr 'H' 'h')

# ------------------------------------------------ confirmación (C15, C16)
for blk in 106 107; do
  for s in $(seq ${blk}01 ${blk}17); do run Xj "$s" "$ARM" greedy_joint --timeout 3000; done
  for s in $(seq ${blk}01 ${blk}17); do run Hj "$s" h20 greedy_joint --timeout 3000; done
  for s in $(seq ${blk}01 ${blk}17); do run Xe "$s" "$ARM" expectimax_bayesian --timeout 3600; done
  for s in $(seq ${blk}01 ${blk}17); do run Xs "$s" "$ARM" greedy_solo --timeout 3000; done
  for s in $(seq ${blk}01 ${blk}17); do run Gj "$s" g greedy_joint --timeout 1800; done
done
{
  echo; echo "=== Confirmación de $WIN (C15 = 10601–10617, C16 = 10701–10717) ==="
  PS_=""
  for h in Xj:V1_contra_joint Xe:V2_contra_expectimax Xs:V3_contra_solo; do
    c=${h%%:*}; echo; echo "--- ${h#*:} ($c, $(secs $c); fallbacks preview: $(nfb $c), series_errors: $(nerr $c))"
    line=$(python -m showdown.inference_sim.confirm_stats binom "$OUT"/raw/${c}_s10[67]*.txt); echo "$line"
    PS_="$PS_ $(echo "$line" | grep -oE ' p=[^ ]+' | head -1 | cut -d= -f2)"
    echo "    C15: $(python -m showdown.inference_sim.confirm_stats binom "$OUT"/raw/${c}_s106*.txt)"
    echo "    C16: $(python -m showdown.inference_sim.confirm_stats binom "$OUT"/raw/${c}_s107*.txt)"
  done
  echo; echo "--- V4 $WIN frente a H 2.0 (McNemar, contra joint; C15+C16)"
  line=$(mc "$OUT/raw/Xj_s10[67]*.txt" "$OUT/raw/Hj_s10[67]*.txt"); echo "$line"
  PS_="$PS_ $(echo "$line" | grep -oE ' p=[^ ]+' | head -1 | cut -d= -f2)"
  echo "    C15: $(mc "$OUT/raw/Xj_s106*.txt" "$OUT/raw/Hj_s106*.txt")"
  echo "    C16: $(mc "$OUT/raw/Xj_s107*.txt" "$OUT/raw/Hj_s107*.txt")"
  echo "--- Hj (H 2.0 contra joint, $(secs Hj))"; python -m showdown.inference_sim.confirm_stats binom "$OUT"/raw/Hj_s10[67]*.txt
  echo; echo "--- descriptivo: $WIN frente a G 5.2 (McNemar, contra joint; Gj $(secs Gj))"
  mc "$OUT/raw/Xj_s10[67]*.txt" "$OUT/raw/Gj_s10[67]*.txt"
  python -m showdown.inference_sim.confirm_stats binom "$OUT"/raw/Gj_s10[67]*.txt
  echo; echo "--- Holm (m=4; p de V1, V2, V3, V4; V4 exige además diferencia > 0)"; python -m showdown.inference_sim.confirm_stats holm $PS_
} >> "$S" 2>&1
log "estadística escrita en $S"
log "MCTS_H_V21_DONE"
