#!/bin/bash
# P1b (19-11-2026, pedido del usuario): preview no determinista con el preview v2.1. Mismo algoritmo
# (sortea de su propia política, solo entre las 90 formaciones canónicas con que se entrenó).
# Combate: PPO con P5 + preview v2.1 de la vigesimosexta cola (semillas 0 y 1), contra val, 200 partidas
# por rival. Modos: preview T = 1,0 y T = 0,5 con combate determinista, y preview T = 0,5 + combate
# T = 0,5 (modo contra personas). Referencia determinista: ppo_p5_s{0,1}_val de la cola 26.
# Regla (como P1): se adopta si pierde como mucho 2 puntos en val. Espera a KICKOFF_QUEUE26_DONE.
#   bash scripts/server/p1b_preview.sh ~/iapsvgc_runs/p1b ~/iapsvgc_runs/p5full
set -u
OUT=${1:-$HOME/iapsvgc_runs/p1b}
Q26=${2:-$HOME/iapsvgc_runs/p5full}
mkdir -p "$OUT/diag"
source ~/miniconda3/etc/profile.d/conda.sh; conda activate kuroneko
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
export PYTHONUNBUFFERED=1 PS_RL_STRICT=1
unset RL_OBS_ABILITIES RL_OBS_CALC PS_JOINT_ARCHETYPE PREVIEW_FORCE_ORDER PPO_FOE_CHECKPOINTS PREVIEW_SAMPLE PREVIEW_TEMPERATURE
log() { echo "$(date '+%F %T') $*" >> "$OUT/queue.log"; }
clean_ps() { pkill -f "pokemon-showdown.*start" 2>/dev/null; sleep 5; }
log "esperando a la vigesimosexta cola"
until grep -q KICKOFF_QUEUE26_DONE "$Q26/queue.log" 2>/dev/null; do sleep 60; done
clean_ps
log "inicio commit $(git rev-parse --short HEAD)"
PV="$Q26/preview_v21/best.pt"; [ -f "$PV" ] || PV="$Q26/preview_v21/final.pt"
run() {  # nombre semilla [VARIABLE=valor...] [--opciones...]
  local name=$1 s=$2; shift 2
  local envs=() opts=()
  for x in "$@"; do case "$x" in --*) opts+=("$x");; *) envs+=("$x");; esac; done
  local C="$Q26/ppo_p5_s$s/final.pt"
  [ -f "$C" ] && [ -f "$PV" ] || { log "$name s$s: falta $C o $PV"; return; }
  env PS_TEAM_SPLIT=val PREVIEW_SAMPLE=1 ${envs[@]+"${envs[@]}"} python -m showdown.rl.ddqn.policy_diag --agent ppo \
    --checkpoint "$C" --games 200 --port 8431 --agent-team showdown/rl/data/ddqn_team.txt --preview-checkpoint "$PV" \
    ${opts[@]+"${opts[@]}"} --out "$OUT/diag/${name}_s${s}_val.json" > "$OUT/diag/${name}_s${s}_val.txt" 2>&1
  log "$name s$s rc=$?"; clean_ps
}
for s in 0 1; do
  run pv_T1.0 $s PREVIEW_TEMPERATURE=1.0
  run pv_T0.5 $s PREVIEW_TEMPERATURE=0.5
  run pv_T0.5_cb_T0.5 $s PREVIEW_TEMPERATURE=0.5 --sample --temperature=0.5
done
python scripts/analysis/divergencia_preview.py "$OUT" "$Q26" > "$OUT/resumen.err" 2>&1
log "resumen rc=$?"
log "P1B_DONE"
