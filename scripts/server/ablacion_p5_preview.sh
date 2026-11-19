#!/bin/bash
# Separar el efecto de P5 (observación nueva) del preview v2.1 (19-11-2026, decisión del usuario).
# Solo evaluaciones sobre checkpoints existentes (200 partidas por rival, val y eval, semillas 0 y 1):
#   A: PPO v1 + preview v2.1        (frente a PPO v1 + v2)
#   B: PPO con P5 + preview v2      (frente a PPO con P5 + v2.1, cola 26)
#   C: DDQN REG 0,03 + preview v2.1 (frente a DDQN REG + v2)
#   D: DDQN con P5 + preview v2     (frente a DDQN con P5 + v2.1, cola 26)
# Resumen: $OUT/ablacion.md (scripts/analysis/ablacion_p5_preview.py).
#   bash scripts/server/ablacion_p5_preview.sh ~/iapsvgc_runs/ablacion_p5 ~/iapsvgc_runs/p5full ~/iapsvgc_runs/ppo_v1 ~/iapsvgc_runs/kickoff15
set -u
OUT=${1:-$HOME/iapsvgc_runs/ablacion_p5}
Q26=${2:-$HOME/iapsvgc_runs/p5full}
V1=${3:-$HOME/iapsvgc_runs/ppo_v1}
Q15=${4:-$HOME/iapsvgc_runs/kickoff15}
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
PV2="$V1/preview_v2.pt"
PV21="$Q26/preview_v21/best.pt"; [ -f "$PV21" ] || PV21="$Q26/preview_v21/final.pt"
diag() {  # agente nombre checkpoint grupo preview
  local agent=$1 name=$2 ck=$3 split=$4 pv=$5
  [ -f "$ck" ] && [ -f "$pv" ] || { log "diag $name: falta $ck o $pv"; return; }
  PS_TEAM_SPLIT=$split python -m showdown.rl.ddqn.policy_diag --agent $agent --checkpoint "$ck" --games 200 \
    --port 8433 --agent-team showdown/rl/data/ddqn_team.txt --preview-checkpoint "$pv" \
    --out "$OUT/diag/$name.json" > "$OUT/diag/$name.txt" 2>&1
  log "diag $name rc=$?"; clean_ps
}
for s in 0 1; do
  for sp in val eval; do
    diag ppo A_ppo_v1_pv21_s${s}_$sp "$V1/ppo_s$s/final.pt" $sp "$PV21"
    diag ppo B_ppo_p5_pv2_s${s}_$sp "$Q26/ppo_p5_s$s/final.pt" $sp "$PV2"
    diag ddqn C_ddqn_ref_pv21_s${s}_$sp "$Q15/R_wd03_s$s/final.pt" $sp "$PV21"
    diag ddqn D_ddqn_p5_pv2_s${s}_$sp "$Q26/ddqn_p5_s$s/final.pt" $sp "$PV2"
  done
done
python scripts/analysis/ablacion_p5_preview.py "$OUT" "$Q26" "$V1" "$Q15" "$HOME/iapsvgc_runs/ppo_v2" > "$OUT/resumen.err" 2>&1
log "resumen rc=$?"
log "ABLACION_DONE"
