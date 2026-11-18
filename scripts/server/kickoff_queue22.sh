#!/bin/bash
# Vigesimosegunda cola (18-11-2026): preview entrenado con su propio combate (memorizacion_equipos.md
# §15), aprobado por el usuario ("aplicar recomendación" tras la parte 1 de la vigésima cola: con el
# combate final, adaptar la formación al rival aporta solo +4,1 puntos en DDQN y +2,8 en PPO).
#
# El preview v2 se entrenó con greedy_joint como combate; aquí se entrena un preview por agente con
# SU combate congelado (final.pt semilla 0: DDQN con REG 0,03 / PPO con simetría) en nuestro lado y
# greedy_joint en el rival, y solo con los 170 equipos de fit (así val queda desconocido también
# para el preview). Misma receta que v2 (cola 2): 200 actualizaciones, entropía 0,005, tu equipo.
# Después: mismo combate (semillas 0 y 1) con v2 y con el preview nuevo, contra val (elige: más de
# 4 puntos) y eval (registro), 200 partidas por rival.
# Espera a KICKOFF_QUEUE21_DONE. Resumen: $OUT/kickoff_summary22.md.
#
#   bash scripts/server/kickoff_queue22.sh ~/iapsvgc_runs/kickoff22 ~/iapsvgc_runs ~/iapsvgc_runs/kickoff21
set -u
OUT=${1:-$HOME/iapsvgc_runs/kickoff22}
R=${2:-$HOME/iapsvgc_runs}
Q21=${3:-$HOME/iapsvgc_runs/kickoff21}
mkdir -p "$OUT"/diag
source ~/miniconda3/etc/profile.d/conda.sh; conda activate kuroneko
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
export PYTHONUNBUFFERED=1 PS_RL_STRICT=1
unset RL_MASK_ALLY_DAMAGE RL_SYM_AUG RL_HOLDOUT_VAL RL_TRAIN_SUBSET RL_FOE_SYNTH PREVIEW_FORCE_ORDER PS_JOINT_ARCHETYPE
log() { echo "$(date '+%F %T') $*" >> "$OUT/queue.log"; }
clean_ps() { pkill -f "pokemon-showdown.*start" 2>/dev/null; sleep 5; }
log "esperando a la vigesimoprimera cola ($Q21)"
until grep -q KICKOFF_QUEUE21_DONE "$Q21/queue.log" 2>/dev/null; do sleep 60; done
clean_ps
log "inicio commit $(git rev-parse --short HEAD)"
TEAM=showdown/rl/data/ddqn_team.txt
PV2=$(sed -n 's/.*preview=\([^ ]*\).*/\1/p' "$R/kickoff3/mini_choice.txt" 2>/dev/null)
log "preview v2: $PV2"
OMP_NUM_THREADS=4 python -m pytest -q -p no:cacheprovider tests/test_preview_rl_pilot.py > "$OUT/tests0.txt" 2>&1
log "0 tests rc=$? ($(tail -1 "$OUT/tests0.txt"))"

declare -A CK=( [ddqn]="$R/kickoff15/R_wd03_s0/final.pt" [ppo]="$R/kickoff12/PPO_sym_s0/final.pt" )
declare -A CK1=( [ddqn]="$R/kickoff15/R_wd03_s1/final.pt" [ppo]="$R/kickoff12/PPO_sym_s1/final.pt" )

# ---------------------------------------------------------------- 1. entrenar los previews
for a in ddqn ppo; do
  PS_TEAM_SPLIT=fit PREVIEW_PILOT_CHECKPOINT="${CK[$a]}" PREVIEW_FOE_PILOT=greedy_joint \
  DDQN_DEVICE=cpu PPO_DEVICE=cpu python -m showdown.rl.team_preview_ppo.train --seed 0 --device cuda \
    --backend live --num-workers 28 --envs-per-server 4 --start-ps --pilot $a --updates 200 --eval-every 20 \
    --eval-periodic-episodes 256 --eval-baseline pilot --canonical --own-team $TEAM --ent-coef 0.005 \
    --checkpoint-dir "$OUT/preview_$a" > "$OUT/preview_$a.txt" 2>&1
  log "preview_$a rc=$?"; clean_ps
done

# ---------------------------------------------------------------- 2. mismo combate, v2 frente al nuevo
diag() {  # agente nombre checkpoint grupo preview
  local agent=$1 name=$2 ck=$3 split=$4 pv=$5
  [ -f "$ck" ] && [ -f "$pv" ] || { log "diag $name: falta $ck o $pv"; return; }
  PS_TEAM_SPLIT=$split python -m showdown.rl.ddqn.policy_diag --agent $agent --checkpoint "$ck" \
    --games 200 --port 8420 --agent-team $TEAM --preview-checkpoint "$pv" \
    --out "$OUT/diag/$name.json" > "$OUT/diag/$name.txt" 2>&1
  log "diag $name rc=$?"; clean_ps
}
for a in ddqn ppo; do
  NEW=$(ls "$OUT/preview_$a/best.pt" "$OUT/preview_$a/final.pt" 2>/dev/null | head -1)
  log "preview nuevo $a: ${NEW:-ninguno}"
  for s in 0 1; do
    if [ $s = 0 ]; then C="${CK[$a]}"; else C="${CK1[$a]}"; fi
    diag $a ${a}_s${s}_v2_val "$C" val "$PV2"
    diag $a ${a}_s${s}_nuevo_val "$C" val "$NEW"
    diag $a ${a}_s${s}_nuevo_eval "$C" eval "$NEW"
  done
done

python -m showdown.rl.kickoff_summary22 "$OUT" > "$OUT/summary22.err" 2>&1
log "resumen rc=$?"
log "KICKOFF_QUEUE22_DONE"
