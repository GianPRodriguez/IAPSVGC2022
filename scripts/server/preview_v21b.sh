#!/bin/bash
# Reentrenar el preview v2.1 sin los equipos de val (19-11-2026, decisión del usuario; memorizacion §18.4).
# Misma receta que el v2 (greedy_joint como combate, tu equipo, 200 actualizaciones, entropía 0,005,
# formaciones canónicas, best.pt) y la observación nueva, pero con PS_TEAM_SPLIT=fit (170 equipos):
# val queda desconocido también para el preview. Dos semillas (0 y 1) para ver la variación.
# Después, combate fijo (PPO y DDQN con P5, semillas 0 y 1) con cada preview nuevo, contra val y eval.
# Referencia: el mismo combate con el preview v2 (ablación, cruces B y D).
# Regla (fijada antes de ver resultados): se elige el preview nuevo con mejor val (PPO y DDQN agrupados);
# se adopta si en val no pierde más de 2 puntos frente al v2 (el v2 conoce los equipos de val).
#   bash scripts/server/preview_v21b.sh ~/iapsvgc_runs/preview_v21b ~/iapsvgc_runs/p5full ~/iapsvgc_runs/ablacion_p5
set -u
OUT=${1:-$HOME/iapsvgc_runs/preview_v21b}
Q26=${2:-$HOME/iapsvgc_runs/p5full}
ABL=${3:-$HOME/iapsvgc_runs/ablacion_p5}
mkdir -p "$OUT/diag"
source ~/miniconda3/etc/profile.d/conda.sh; conda activate kuroneko
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
export PYTHONUNBUFFERED=1 PS_RL_STRICT=1
unset RL_OBS_ABILITIES RL_OBS_CALC PS_JOINT_ARCHETYPE PREVIEW_FORCE_ORDER PPO_FOE_CHECKPOINTS PREVIEW_SAMPLE PREVIEW_TEMPERATURE
log() { echo "$(date '+%F %T') $*" >> "$OUT/queue.log"; }
clean_ps() { pkill -f "pokemon-showdown.*start" 2>/dev/null; sleep 5; }
clean_ps
log "inicio commit $(git rev-parse --short HEAD)"
TEAM=showdown/rl/data/ddqn_team.txt
for ps in 0 1; do
  PS_TEAM_SPLIT=fit python -m showdown.rl.team_preview_ppo.train --seed $ps --device cuda --backend live --num-workers 28 \
    --envs-per-server 4 --start-ps --pilot greedy_joint --updates 200 --eval-every 20 \
    --eval-periodic-episodes 256 --eval-baseline pilot --canonical --own-team $TEAM --ent-coef 0.005 \
    --obs-abilities 1 --checkpoint-dir "$OUT/preview_s$ps" > "$OUT/preview_s$ps.txt" 2>&1
  log "preview_s$ps rc=$?"; clean_ps
done
diag() {  # agente nombre checkpoint grupo preview
  local agent=$1 name=$2 ck=$3 split=$4 pv=$5
  [ -f "$ck" ] && [ -f "$pv" ] || { log "diag $name: falta $ck o $pv"; return; }
  PS_TEAM_SPLIT=$split python -m showdown.rl.ddqn.policy_diag --agent $agent --checkpoint "$ck" --games 200 \
    --port 8434 --agent-team $TEAM --preview-checkpoint "$pv" --out "$OUT/diag/$name.json" > "$OUT/diag/$name.txt" 2>&1
  log "diag $name rc=$?"; clean_ps
}
for ps in 0 1; do
  PV="$OUT/preview_s$ps/best.pt"; [ -f "$PV" ] || PV="$OUT/preview_s$ps/final.pt"
  for s in 0 1; do
    for sp in val eval; do
      diag ppo pv${ps}_ppo_s${s}_$sp "$Q26/ppo_p5_s$s/final.pt" $sp "$PV"
      diag ddqn pv${ps}_ddqn_s${s}_$sp "$Q26/ddqn_p5_s$s/final.pt" $sp "$PV"
    done
  done
done
python scripts/analysis/preview_v21b.py "$OUT" "$ABL" > "$OUT/resumen.err" 2>&1
log "resumen rc=$?"
log "PREVIEW_V21B_DONE"
