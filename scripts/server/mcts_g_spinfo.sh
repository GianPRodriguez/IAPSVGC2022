#!/bin/bash
# ¿APORTA EL MOTOR DE INFERENCIA DE SP A G (MCTS+PUCT v5)? Pre-registrado el 21-11-2026 antes de
# medir (docs/mcts_puct_diagnostico.md §9). Solo baselines. Preview greedy fijo (aislado del
# preview por búsqueda).
#
#   Conteo: smoke con PS_SP_INFO_DEBUG=1 (¿cuántos stats estrecha la inferencia en partidas de G?).
#   Exploración (9101–9117 × 300, contra greedy_joint, mismas series):
#     I0 inference (default actual) · I1 oracle_spe (techo: Velocidad real del rival) · I2 meta
#     (sin inferencia; los 6 stats del rival al reparto más frecuente del prior meta LOO).
#   Regla: si I2 no es significativamente peor que I0 (no: diferencia < 0 y p < 0,05) y además
#   |diferencia| ≤ 1 pt o I2 es significativamente mejor → el default de G pasa a meta (inferencia
#   apagada). I1 es diagnóstico (valor máximo de conocer la Velocidad).
# Caché de la calculadora por corrida y borrada al terminar (regla: nunca > 250 GB).
# Resumen: $OUT/spinfo_summary.txt. Marca: MCTS_G_SPINFO_DONE.
set -u
OUT=${1:-$HOME/iapsvgc_runs/mcts_g_spinfo}
FROZEN=${FROZEN:-d247fab}
WAIT_FILE=${WAIT_FILE:-$HOME/iapsvgc_runs/mcts_g_search/queue.log}
WAIT_PAT=${WAIT_PAT:-MCTS_G_SEARCH_DONE}
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
clean_ps() { pkill -f "pokemon-showdown.*start" 2>/dev/null; sleep 5; }

# run <config> <seed> <sp_info_a> [args...]   (G v5, preview greedy, contra greedy_joint)
run() {
  local cfg=$1 seed=$2 spi=$3; shift 3
  local rpt="$OUT/reports/${cfg}_s${seed}.txt"
  if [ -s "$rpt" ]; then log "$cfg s$seed ya hecho (reanudación)"; return 0; fi
  rm -f logs/robust_e2e_report.txt
  export DAMAGE_CALC_CACHE="$OUT/calc_cache_run.sqlite"
  rm -f "$DAMAGE_CALC_CACHE" "$DAMAGE_CALC_CACHE"-wal "$DAMAGE_CALC_CACHE"-shm
  local t0=$(date +%s)
  MCTS_PUCT_VERSION=5 MCTS_STATS_FILE="$OUT/anomalies.tsv" python -m showdown.inference_sim.run_robust_e2e \
    --teams 300 --procs "$PROCS" --seed "$seed" --restart-ps --player-a mcts_puct --player-b greedy_joint \
    --preview-a greedy --sp-info-a "$spi" "$@" > "$OUT/raw/${cfg}_s${seed}.txt" 2>&1
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

log "esperando /$WAIT_PAT/ en $WAIT_FILE"
until grep -qE "$WAIT_PAT" "$WAIT_FILE" 2>/dev/null; do sleep 60; done
clean_ps
HEAD_C=$(git rev-parse --short HEAD)
log "inicio commit $HEAD_C (congelado $FROZEN)"
if ! git diff --quiet "$FROZEN" HEAD -- showdown/rl/mcts_puct showdown/battle_bots/greedy_joint \
     showdown/inference_sim/validated_greedy_player.py showdown/stat_inference.py showdown/meta_sp_prior.py; then
  log "ERROR: el código congelado cambió desde $FROZEN; no se mide (diagnóstico §9)"
  log "MCTS_G_SPINFO_DONE"; exit 1
fi

# ------------------------------------------------ conteo de estrechamiento (G con inferencia)
PS_SP_INFO_DEBUG=1 run count 9001 inference --teams 56 --timeout 1200
grep -h '^\[SP-INFO\].*mode=inference' "$OUT/raw/count_s9001.txt" | python -c "
import re, sys
n = st = nar = res = obs = 0
for line in sys.stdin:
    g = dict(re.findall(r'(\w+)=([\w./]+)', line))
    n += 1; nar += int(g.get('estrechados', 0)); obs += int(g.get('obs', 0))
    r, t = g.get('stats_resueltos', '0/0').split('/'); res += int(r); st += int(t)
print(f'partidas={n} stats={st} estrechados={nar} resueltos={res} observaciones={obs}')
" > "$OUT/count.txt" 2>&1
log "conteo inferencia en G: $(cat "$OUT/count.txt")"

# ------------------------------------------------ exploración
for s in $(seq 9101 9117); do run I0 "$s" inference --timeout 1200; done
for s in $(seq 9101 9117); do run I1 "$s" oracle_spe --timeout 1200; done
for s in $(seq 9101 9117); do run I2 "$s" meta --timeout 1200; done

S="$OUT/spinfo_summary.txt"
mc() { python -m showdown.inference_sim.confirm_stats mcnemar --a $1 --b $2; }
secs() { cat "$OUT"/reports/$1_s91*.sec 2>/dev/null | awk '{s+=$1; n++} END{if(n) printf "%.0f s/semilla (%d)", s/n, n}'; }
{
  echo "=== Motor de inferencia de SP en G v5 — commit $HEAD_C (congelado $FROZEN) ==="
  echo "Pre-registro: docs/mcts_puct_diagnostico.md §9. Preview greedy, contra greedy_joint, 9101–9117."
  echo "Conteo (56 series, PS_SP_INFO_DEBUG=1): $(cat "$OUT/count.txt")"
  echo "Anomalías del player: $(cut -f1 "$OUT/anomalies.tsv" 2>/dev/null | sort | uniq -c | tr '\n' ' ')"
  for c in I0 I1 I2; do
    echo; echo "--- $c ($(secs $c))"; python -m showdown.inference_sim.confirm_stats binom "$OUT"/raw/${c}_s91*.txt
  done
  echo; echo "--- I1 (oracle_spe) frente a I0 (inference)"; mc "$OUT/raw/I1_s91*.txt" "$OUT/raw/I0_s91*.txt"
  echo "--- I2 (meta) frente a I0 (inference)"; mc "$OUT/raw/I2_s91*.txt" "$OUT/raw/I0_s91*.txt"
  if ls ~/iapsvgc_runs/mcts_g_search/raw/S0_s91*.txt >/dev/null 2>&1; then
    echo "--- Reproducibilidad: I0 frente a S0 de la cola del preview (misma config)"
    mc "$OUT/raw/I0_s91*.txt" "$HOME/iapsvgc_runs/mcts_g_search/raw/S0_s91*.txt"
  fi
} > "$S" 2>&1
log "estadística escrita en $S"
log "MCTS_G_SPINFO_DONE"
