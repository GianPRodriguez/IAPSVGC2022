#!/bin/bash
# Tercera cola previa al kick-off (16-11-2026): diagnóstico de la política, lr 1e-4 + lote 512 y
# mini kick-off de 1M pasos con el currículo completo. Registra la carga del servidor.
# Espera a KICKOFF_QUEUE2_DONE. Resumen: $OUT/kickoff_summary3.md.
#
#   bash scripts/server/kickoff_queue3.sh ~/iapsvgc_runs/kickoff3 ~/iapsvgc_runs/kickoff ~/iapsvgc_runs/kickoff2
set -u
OUT=${1:-$HOME/iapsvgc_runs/kickoff3}
Q1=${2:-$HOME/iapsvgc_runs/kickoff}
Q2=${3:-$HOME/iapsvgc_runs/kickoff2}
mkdir -p "$OUT"
source ~/miniconda3/etc/profile.d/conda.sh; conda activate kuroneko
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
export PYTHONUNBUFFERED=1 PS_RL_STRICT=1
log() { echo "$(date '+%F %T') $*" >> "$OUT/queue.log"; }
stage() { echo "$1" > "$OUT/.stage"; log "inicio $1"; }
clean_ps() { pkill -f "pokemon-showdown.*start" 2>/dev/null; sleep 5; }

# Registro de la carga: CPU total ocupada y CPU de procesos de OTROS usuarios, cada 60 s.
monitor() {
  echo "ts,etapa,cpu_ocupada,cpu_otros_usuarios" > "$OUT/load.csv"
  while true; do
    local idle others
    idle=$(top -bn1 | awk -F',' '/%Cpu\(s\)/ {for(i=1;i<=NF;i++) if($i ~ / id/){gsub(/[^0-9.]/,"",$i); print $i}}')
    others=$(top -bn1 -o %CPU | awk -v me="$(whoami)" 'NR>7 && $2!=me {s+=$9} END{printf "%.0f", s}')
    echo "$(date +%s),$(cat "$OUT/.stage" 2>/dev/null),$(awk -v i="${idle:-0}" 'BEGIN{printf "%.1f", 100-i}'),${others:-0}" >> "$OUT/load.csv"
    sleep 60
  done
}

log "esperando a la segunda cola ($Q2)"
until grep -q KICKOFF_QUEUE2_DONE "$Q2/queue.log" 2>/dev/null; do sleep 60; done
clean_ps
monitor & MON=$!
trap 'kill $MON 2>/dev/null' EXIT
log "inicio commit $(git rev-parse --short HEAD)"
TEAM=showdown/rl/data/ddqn_team.txt
OLD_PV="$(cat "$Q1/preview_used.txt" 2>/dev/null)"
[ -f "$OLD_PV" ] || OLD_PV=""
COMMON="--device cuda --start-ps --expand-omega --num-envs 32 --envs-per-server 4 --log-every 10000 --log-every-sec 120"
TUNE="--phases random:150000,greedy_solo:250000 --eval-every 200000 --eval-games 50 --suite-games 0"

# ---------------------------------------------------------------- 4. diagnóstico de la política
stage diag
CK="$Q1/tune/T1_lr1e-4/best.pt"
[ -f "$CK" ] || CK="$Q1/tune/T1_lr1e-4/final.pt"
python -m showdown.rl.ddqn.policy_diag --checkpoint "$CK" --games 50 --port 8370 --agent-team $TEAM \
  ${OLD_PV:+--preview-checkpoint "$OLD_PV"} --out "$OUT/policy_diag.json" > "$OUT/policy_diag.txt" 2>&1
log "4 diagnóstico rc=$?"; clean_ps

# ---------------------------------------------------------------- 2. lr 1e-4 + lote 512
stage combo
python -m showdown.rl.ddqn.train --seed 0 $COMMON $TUNE ${OLD_PV:+--preview-checkpoint "$OLD_PV"} \
  --lr 1e-4 --batch-size 512 --checkpoint-dir "$OUT/combo_lr1e-4_b512" > "$OUT/combo.txt" 2>&1
log "2 combinado rc=$?"; clean_ps

# ---------------------------------------------------------------- elección para el mini kick-off
CHOICE=$(python -m showdown.rl.ddqn.kickoff_summary3 --choose "$OUT" "$Q1" "$Q2")
echo "$CHOICE" > "$OUT/mini_choice.txt"
log "configuración elegida: $CHOICE"
LR=$(echo "$CHOICE" | sed -n 's/.*lr=\([^ ]*\).*/\1/p')
BS=$(echo "$CHOICE" | sed -n 's/.*batch=\([^ ]*\).*/\1/p')
PV=$(echo "$CHOICE" | sed -n 's/.*preview=\([^ ]*\).*/\1/p')

# ---------------------------------------------------------------- 1. mini kick-off (1M pasos)
stage mini_kickoff
python -m showdown.rl.ddqn.train --seed 0 $COMMON \
  --phases random:150000,greedy_solo:250000,greedy_joint:300000,expectimax_bayesian:300000 \
  --eval-every 250000 --eval-games 50 --suite-games 0 --lr ${LR:-1e-4} --batch-size ${BS:-256} \
  ${PV:+--preview-checkpoint "$PV"} --checkpoint-dir "$OUT/mini_kickoff" > "$OUT/mini_kickoff.txt" 2>&1
log "1 mini kick-off rc=$?"; clean_ps

stage resumen
python -m showdown.rl.ddqn.kickoff_summary3 "$OUT" "$Q1" "$Q2" > /dev/null 2>&1
log "resumen rc=$?"
log "KICKOFF_QUEUE3_DONE"
