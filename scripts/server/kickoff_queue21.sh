#!/bin/bash
# Vigesimoprimera cola (18-11-2026), aprobada por el usuario (memorizacion_equipos.md §14):
#
#   1. Rival de arquetipos: greedy_joint con PS_JOINT_ARCHETYPE=1 (activa Tailwind siempre que no
#      esté puesto y Trick Room cuando su lado es más lento). Mide si DDQN y PPO resisten planes de
#      velocidad que los heurísticos normales casi nunca ejecutan (cola 19: 0–1 % Trick Room,
#      0–4 % Tailwind). final.pt de la configuración final, 2 semillas, tu equipo, rivales eval,
#      200 partidas, solo contra joint, con --context.
#   2. Ajuste de entropía de PPO (igualar el esfuerzo de ajuste con DDQN, que tuvo el de REG):
#      ent_coef 0,003 y 0,03 frente a 0,01 (por defecto), 1M, semillas 0 y 1, configuración final.
#      Elección con val (200 por rival, final.pt): se recomienda si supera a 0,01 por más de 4.
#
# Espera a KICKOFF_QUEUE20_DONE. Resumen: $OUT/kickoff_summary21.md.
#
#   bash scripts/server/kickoff_queue21.sh ~/iapsvgc_runs/kickoff21 ~/iapsvgc_runs ~/iapsvgc_runs/kickoff20
set -u
OUT=${1:-$HOME/iapsvgc_runs/kickoff21}
R=${2:-$HOME/iapsvgc_runs}
Q20=${3:-$HOME/iapsvgc_runs/kickoff20}
mkdir -p "$OUT"/diag
source ~/miniconda3/etc/profile.d/conda.sh; conda activate kuroneko
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
export PYTHONUNBUFFERED=1 PS_RL_STRICT=1
unset RL_MASK_ALLY_DAMAGE RL_SYM_AUG RL_HOLDOUT_VAL RL_TRAIN_SUBSET RL_FOE_SYNTH PREVIEW_FORCE_ORDER PS_JOINT_ARCHETYPE
log() { echo "$(date '+%F %T') $*" >> "$OUT/queue.log"; }
clean_ps() { pkill -f "pokemon-showdown.*start" 2>/dev/null; sleep 5; }
log "esperando a la vigésima cola ($Q20)"
until grep -q KICKOFF_QUEUE20_DONE "$Q20/queue.log" 2>/dev/null; do sleep 60; done
clean_ps
log "inicio commit $(git rev-parse --short HEAD)"
TEAM=showdown/rl/data/ddqn_team.txt
PV=$(sed -n 's/.*preview=\([^ ]*\).*/\1/p' "$R/kickoff3/mini_choice.txt" 2>/dev/null)
[ -f "${PV:-}" ] || PV=""
log "preview: ${PV:-ninguno}"

OMP_NUM_THREADS=4 python -m pytest -q -p no:cacheprovider tests/test_joint_archetype.py > "$OUT/tests0.txt" 2>&1
log "0 tests rc=$? ($(tail -1 "$OUT/tests0.txt"))"

diag() {  # agente nombre checkpoint grupo partidas [variables...] -- [opciones...]
  local agent=$1 name=$2 ck=$3 split=$4 games=$5; shift 5
  [ -f "$ck" ] || { log "diag $name: falta $ck"; return; }
  local envs=() opts=()
  for x in "$@"; do case "$x" in --*) opts+=("$x");; *) envs+=("$x");; esac; done
  env PS_TEAM_SPLIT=$split ${envs[@]+"${envs[@]}"} python -m showdown.rl.ddqn.policy_diag --agent $agent \
    --checkpoint "$ck" --games $games --port 8418 --agent-team $TEAM ${PV:+--preview-checkpoint "$PV"} \
    ${opts[@]+"${opts[@]}"} --out "$OUT/diag/$name.json" > "$OUT/diag/$name.txt" 2>&1
  log "diag $name rc=$?"; clean_ps
}

# ---------------------------------------------------------------- 1. rival de arquetipos
for s in 0 1; do
  diag ddqn ddqn_s${s}_arquetipo "$R/kickoff15/R_wd03_s$s/final.pt" eval 200 PS_JOINT_ARCHETYPE=1 --foes=greedy_joint --context
  diag ppo ppo_s${s}_arquetipo "$R/kickoff12/PPO_sym_s$s/final.pt" eval 200 PS_JOINT_ARCHETYPE=1 --foes=greedy_joint --context
done

# ---------------------------------------------------------------- 2. entropía de PPO
COMMON="--device cuda --start-ps --expand-omega --num-envs 32 --envs-per-server 4 --log-every 10000 \
  --log-every-sec 120 --eval-every 250000 --eval-games 50 --suite-games 0 --agent-team $TEAM \
  --phases random:150000,greedy_solo:250000,greedy_joint:300000,expectimax_bayesian:300000"
for s in 0 1; do
  for e in 003 03; do
    name=ENT${e}_s$s
    python -m showdown.rl.ppo.train --seed $s $COMMON --ent-coef 0.$e ${PV:+--preview-checkpoint "$PV"} \
      --checkpoint-dir "$OUT/$name" > "$OUT/$name.txt" 2>&1
    log "$name rc=$?"; clean_ps
    diag ppo ${name}_final_val "$OUT/$name/final.pt" val 200
    diag ppo ${name}_final_eval "$OUT/$name/final.pt" eval 100
    diag ppo ${name}_final_own "$OUT/$name/final.pt" train 100 RL_HOLDOUT_VAL=1
  done
done

python -m showdown.rl.kickoff_summary21 "$OUT" "$R" > "$OUT/summary21.err" 2>&1
log "resumen rc=$?"
log "KICKOFF_QUEUE21_DONE"
