#!/bin/bash
# KICK-OFF OFICIAL DE PPO v2 (25-11-2026). Receta fijada en memorizacion_equipos §17–§18:
#   PPO v2 = PPO v1 + P5 completo (calculadora real en la observación, --obs-abilities 1) + preview
#   v2.1 sin val (el mismo de DDQN v1: preview_v21b/preview_s0/best.pt) + sorteo T = 0,5 solo al jugar
#   contra personas (no afecta el entrenamiento). Resto = configuración del kick-off de PPO v1
#   (simetría ON, máscara del aliado, validación reservada, equipo fijo, 1M pasos por semilla).
# Decisión del usuario (25-11-2026): entrenar PPO v2 antes de los agentes QD; I, J y K arrancan desde
# el final.pt de la semilla con mejor `val` (regla fijada antes de entrenar). Comparación limpia de
# algoritmos: PPO v2 frente a DDQN v1 (mismo preview y misma observación).
#
# Por semilla (0–4): entrenamiento; final.pt contra eval (principal), best.pt contra eval, final.pt
# contra sus propios equipos, final.pt contra greedy_joint en modo arquetipo, final.pt contra val.
# Resumen: $OUT/kickoff_summary16.md (incluye DDQN v1 por enlaces a sus diag) y $OUT/ppo_v2_seed.txt
# (semilla elegida por val). Marca: KICKOFF_PPO_V2_DONE.
#
#   bash scripts/server/kickoff_ppo_v2.sh ~/iapsvgc_runs/kickoff_ppo_v2
set -u
OUT=${1:-$HOME/iapsvgc_runs/kickoff_ppo_v2}
PV=${PV:-$HOME/iapsvgc_runs/preview_v21b/preview_s0/best.pt}
DDQN1=${DDQN1:-$HOME/iapsvgc_runs/kickoff_ddqn_v2}
mkdir -p "$OUT"/diag
source ~/miniconda3/etc/profile.d/conda.sh; conda activate kuroneko
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
export PYTHONUNBUFFERED=1 PS_RL_STRICT=1
unset RL_MASK_ALLY_DAMAGE RL_SYM_AUG RL_HOLDOUT_VAL RL_TRAIN_SUBSET RL_FOE_SYNTH PREVIEW_FORCE_ORDER PS_JOINT_ARCHETYPE \
  PPO_FOE_CHECKPOINTS PPO_FOE_PREVIEW RL_OBS_ABILITIES RL_OBS_CALC PREVIEW_SAMPLE PREVIEW_TEMPERATURE PS_TEAM_SPLIT
log() { echo "$(date '+%F %T') $*" >> "$OUT/queue.log"; }
clean_ps() { pkill -f "[p]okemon-showdown.*start" 2>/dev/null; sleep 5; }

[ -f "$PV" ] || { log "ERROR: falta el preview $PV"; log "KICKOFF_PPO_V2_DONE"; exit 1; }
clean_ps
pkill -f "calc_serve[r].js" 2>/dev/null; sleep 2
log "inicio commit $(git rev-parse --short HEAD); preview $PV"
TEAM=showdown/rl/data/ddqn_team.txt

diag() {  # nombre checkpoint grupo [VARIABLE=valor...] [--opciones...]
  local name=$1 ck=$2 split=$3; shift 3
  [ -s "$OUT/diag/$name.json" ] && { log "diag $name ya hecho (reanudación)"; return; }
  [ -f "$ck" ] || { log "diag $name: falta $ck"; return; }
  local envs=() opts=()
  for x in "$@"; do case "$x" in --*) opts+=("$x");; *) envs+=("$x");; esac; done
  env PS_TEAM_SPLIT=$split ${envs[@]+"${envs[@]}"} python -m showdown.rl.ddqn.policy_diag --agent ppo \
    --checkpoint "$ck" --games 200 --port 8422 --agent-team $TEAM --preview-checkpoint "$PV" \
    ${opts[@]+"${opts[@]}"} --out "$OUT/diag/$name.json" > "$OUT/diag/$name.txt" 2>&1
  log "diag $name rc=$?"; clean_ps
}
seed() {  # semilla
  local s=$1
  if [ -f "$OUT/ppo_s$s/final.pt" ]; then
    log "ppo_s$s ya entrenado (reanudación)"
  else
    python -m showdown.rl.ppo.train --seed $s --device cuda --start-ps --expand-omega --num-envs 32 \
      --envs-per-server 4 --log-every 10000 --log-every-sec 120 \
      --phases random:150000,greedy_solo:250000,greedy_joint:300000,expectimax_bayesian:300000 \
      --eval-every 250000 --eval-games 50 --suite-games 0 --agent-team $TEAM --preview-checkpoint "$PV" \
      --obs-abilities 1 --checkpoint-dir "$OUT/ppo_s$s" > "$OUT/ppo_s$s.txt" 2>&1
    log "ppo_s$s rc=$?"; clean_ps
  fi
  diag ppo_s${s}_final_eval "$OUT/ppo_s$s/final.pt" eval
  diag ppo_s${s}_best_eval "$OUT/ppo_s$s/best.pt" eval
  diag ppo_s${s}_final_own "$OUT/ppo_s$s/final.pt" train RL_HOLDOUT_VAL=1
  diag ppo_s${s}_final_arquetipo "$OUT/ppo_s$s/final.pt" eval PS_JOINT_ARCHETYPE=1 --foes=greedy_joint --context
  diag ppo_s${s}_final_val "$OUT/ppo_s$s/final.pt" val
  grep -h "calculadora real en la observación" "$OUT/ppo_s$s.txt" 2>/dev/null | tail -1 | cut -c1-300 | sed "s/^/  ppo_s$s calc: /" >> "$OUT/queue.log"
}

for s in 0 1 2 3 4; do seed $s; done

# DDQN v1 en el mismo resumen (comparación limpia de algoritmos: mismo preview y observación)
for f in "$DDQN1"/diag/ddqn_s[0-4]_{final_eval,best_eval,final_own}.json; do
  [ -f "$f" ] && ln -sf "$f" "$OUT/diag/$(basename "$f")"
done
python -m showdown.rl.kickoff_report "$OUT" > "$OUT/kickoff_report.err" 2>&1
log "resumen rc=$?"

# Semilla de arranque de I/J/K: mejor final.pt contra val (media de los 3 heurísticos), regla previa
python - "$OUT" > "$OUT/ppo_v2_seed.txt" 2>&1 <<'PY'
import sys
from pathlib import Path
from showdown.rl.kickoff_report import _load, rate
out = Path(sys.argv[1])
rows = []
for s in range(5):
    r = rate(_load(out / 'diag' / f'ppo_s{s}_final_val.json'))
    if r:
        rows.append((r[0], s))
        print(f'semilla {s}: val {100 * r[0]:.1f} %')
best = max(rows) if rows else None
print(f'ELEGIDA {best[1]}' if best else 'ELEGIDA ninguna')
PY
log "semilla para I/J/K: $(grep ELEGIDA "$OUT/ppo_v2_seed.txt")"
log "KICKOFF_PPO_V2_DONE"
