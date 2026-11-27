#!/bin/bash
# PPO v3 / DDQN v2 (protocolo §4.16, fijado el 27-11-2026 antes de medir). Decisiones del usuario: +1M pasos
# por semilla, 3 semillas por brazo, brazos C (control: +1M en series BO3 con máscara, SIN memoria) y V (igual,
# CON memoria), orden P1 → S1 → PPO → DDQN. PPO y DDQN reciben los mismos cambios (juego limpio).
#
#   P1  máscara del Protect consecutivo sin reentrenar: PPO v2 s0–s4 y DDQN v1 s0–s4, 200 partidas por rival
#       sobre eval, con y sin máscara (mismas semillas, pareado).
#   S1  preview condicionado a la serie sin reentrenar: los mismos checkpoints en series BO3 (100 por rival),
#       con y sin --preview-series.
#   V   por familia y semilla 0–2: C y V desde la versión anterior (+1M pasos, mezcla de rivales del kick-off,
#       sorteo de casi-empates por defecto), evaluación de final.pt en series BO3 sobre eval con --seed 40000;
#       descriptivas: la versión anterior en las mismas series y C/V con --foe-tie.
#   Informe: $OUT/v3_report.md (scripts/analysis/v3_report.py). Marca: V3_DONE.
#
#   bash scripts/server/v3_queue.sh ~/iapsvgc_runs/v3
set -u
OUT=${1:-$HOME/iapsvgc_runs/v3}
PV=${PV:-$HOME/iapsvgc_runs/preview_v21b/preview_s0/best.pt}
DDQN1=${DDQN1:-$HOME/iapsvgc_runs/kickoff_ddqn_v2}
mkdir -p "$OUT"/diag
source ~/miniconda3/etc/profile.d/conda.sh; conda activate kuroneko
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
PPO2=${PPO2:-$REPO/checkpoints/ppo_v2}
export PYTHONUNBUFFERED=1 PS_RL_STRICT=1
unset RL_MASK_ALLY_DAMAGE RL_SYM_AUG RL_HOLDOUT_VAL RL_TRAIN_SUBSET RL_FOE_SYNTH PREVIEW_FORCE_ORDER PS_JOINT_ARCHETYPE \
  PPO_FOE_CHECKPOINTS PPO_FOE_PREVIEW RL_OBS_ABILITIES RL_OBS_CALC PREVIEW_SAMPLE PREVIEW_TEMPERATURE PS_TEAM_SPLIT \
  RL_OBS_SERIES_MEMORY RL_MASK_PROTECT_STREAK RL_SERIES_BO3 PREVIEW_SERIES_CONDITION
log() { echo "$(date '+%F %T') $*" >> "$OUT/queue.log"; }
clean_ps() { pkill -f "[p]okemon-showdown.*start" 2>/dev/null; sleep 5; }
TEAM=showdown/rl/data/ddqn_team.txt

[ -f "$PV" ] || { log "ERROR: falta el preview $PV"; log "V3_DONE"; exit 1; }
clean_ps
pkill -f "calc_serve[r].js" 2>/dev/null; sleep 2
log "inicio commit $(git rev-parse --short HEAD); preview $PV"

prev_ck() {  # familia semilla -> checkpoint de la versión anterior
  if [ "$1" = ppo ]; then echo "$PPO2/ppo_s$2/final.pt"; else echo "$DDQN1/ddqn_s$2/final.pt"; fi
}

diag() {  # nombre familia checkpoint [--opciones de policy_diag...]
  local name=$1 fam=$2 ck=$3; shift 3
  [ -s "$OUT/diag/$name.json" ] && { log "diag $name ya hecho (reanudación)"; return; }
  [ -f "$ck" ] || { log "diag $name: falta $ck"; return; }
  env PS_TEAM_SPLIT=eval python -m showdown.rl.ddqn.policy_diag --agent "$fam" --checkpoint "$ck" --port 8424 \
    --agent-team $TEAM --preview-checkpoint "$PV" "$@" --out "$OUT/diag/$name.json" > "$OUT/diag/$name.txt" 2>&1
  log "diag $name rc=$?"; clean_ps
}

# ---------------- P1: máscara sin reentrenar ----------------
for fam in ppo ddqn; do
  for s in 0 1 2 3 4; do
    ck=$(prev_ck $fam $s)
    diag p1_${fam}_s${s}_off $fam "$ck" --games 200 --mask-protect 0
    diag p1_${fam}_s${s}_on $fam "$ck" --games 200 --mask-protect 1
  done
done
log "P1 terminado"

# ---------------- S1: preview condicionado sin reentrenar ----------------
for fam in ppo ddqn; do
  for s in 0 1 2 3 4; do
    ck=$(prev_ck $fam $s)
    diag s1_${fam}_s${s}_off $fam "$ck" --games 100 --series --mask-protect 0
    diag s1_${fam}_s${s}_on $fam "$ck" --games 100 --series --mask-protect 0 --preview-series
  done
done
log "S1 terminado"

# ---------------- V: brazos C y V ----------------
train() {  # familia brazo semilla
  local fam=$1 arm=$2 s=$3 dir="$OUT/${1}_${2}_s${3}"
  local mem=0; [ "$arm" = V ] && mem=1
  if [ -f "$dir/final.pt" ]; then log "${fam}_${arm}_s$s ya entrenado (reanudación)"; return; fi
  local ck; ck=$(prev_ck $fam $s)
  [ -f "$ck" ] || { log "ERROR ${fam}_${arm}_s$s: falta $ck"; return; }
  local extra=(); [ "$fam" = ddqn ] && extra=(--lr 1e-4 --batch-size 256)
  python -m showdown.rl.$fam.train --seed $s --device cuda --start-ps --expand-omega --num-envs 32 \
    --envs-per-server 4 --log-every 10000 --log-every-sec 120 \
    --phases random:150000,greedy_solo:250000,greedy_joint:300000,expectimax_bayesian:300000 \
    --eval-every 250000 --eval-games 50 --suite-games 0 --agent-team $TEAM --preview-checkpoint "$PV" \
    --obs-abilities 1 --series-bo3 1 --mask-protect-streak 1 --series-memory $mem --init-from "$ck" \
    ${extra[@]+"${extra[@]}"} --checkpoint-dir "$dir" > "$dir.txt" 2>&1
  log "${fam}_${arm}_s$s rc=$?"; clean_ps
  grep -h "arranque en caliente" "$dir.txt" 2>/dev/null | tail -1 | cut -c1-240 | sed "s/^/  ${fam}_${arm}_s$s: /" >> "$OUT/queue.log"
}

for fam in ppo ddqn; do
  for s in 0 1 2; do
    for arm in C V; do
      train $fam $arm $s
      diag v_${fam}_${arm}_s${s} $fam "$OUT/${fam}_${arm}_s$s/final.pt" --games 100 --series --seed 40000
    done
    diag prev_${fam}_s${s} $fam "$(prev_ck $fam $s)" --games 100 --series --seed 40000
  done
  for s in 0 1 2; do
    for arm in C V; do
      diag vtie_${fam}_${arm}_s${s} $fam "$OUT/${fam}_${arm}_s$s/final.pt" --games 100 --series --seed 40000 --foe-tie
    done
  done
  log "familia $fam terminada"
done

python scripts/analysis/v3_report.py "$OUT" > "$OUT/v3_report.err" 2>&1
log "informe rc=$?"
log "V3_DONE"
