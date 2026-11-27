#!/bin/bash
# PPO v3 / DDQN v2 (protocolo §4.16, fijado el 27-11-2026 antes de medir). Decisiones del usuario: +1M pasos
# por semilla, 3 semillas por brazo, brazos C (control: +1M en series BO3 con máscara, SIN memoria) y V (igual,
# CON memoria), orden P1 → S1 → PPO → DDQN; evaluaciones en paralelo y S1 con 3 semillas. PPO y DDQN reciben los
# mismos cambios (juego limpio).
#
#   P1  máscara del Protect consecutivo sin reentrenar: PPO v2 s0–s4 y DDQN v1 s0–s4, 200 partidas por rival
#       sobre eval, con y sin máscara (mismas semillas, pareado), con --context (eventos por partida).
#   D1/D2 diagnósticos descriptivos desde P1 sin máscara: $OUT/d1_report.md (scripts/analysis/archetype_diag.py).
#   S1  preview condicionado a la serie sin reentrenar: PPO v2 s0–s2 y DDQN v1 s0–s2 en series BO3 (100 por
#       rival), con y sin --preview-series.
#   V   por familia: los 6 entrenamientos (C y V, semillas 0–2, +1M pasos desde la versión anterior, mezcla de
#       rivales del kick-off) DE UNO EN UNO; después sus evaluaciones de final.pt en series BO3 sobre eval con
#       --seed 40000, más las descriptivas (versión anterior en las mismas series; C/V con --foe-tie).
#   Informe: $OUT/v3_report.md (scripts/analysis/v3_report.py). Marca: V3_DONE.
#
# Paralelismo: las evaluaciones (policy_diag, un proceso de CPU cada una) corren de a $JOBS (6 por defecto), cada
# una en su puerto (8430…) y limpiando solo su servidor de Showdown. Nunca se solapan con un entrenamiento.
# El servidor es compartido: revisar la carga antes de lanzar y bajar JOBS si hace falta.
#
#   JOBS=6 bash scripts/server/v3_queue.sh ~/iapsvgc_runs/v3
set -u
OUT=${1:-$HOME/iapsvgc_runs/v3}
JOBS=${JOBS:-6}
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
clean_ps() { pkill -f "[p]okemon-showdown.*start" 2>/dev/null; sleep 5; }      # solo sin evaluaciones en curso
kill_port() {  # servidor de Showdown de un puerto (lanzador normal o con semilla)
  pkill -f "[p]okemon-showdown start.* $1\$" 2>/dev/null
  pkill -f "[p]s_seeded_start.js.* $1\$" 2>/dev/null
}
TEAM=showdown/rl/data/ddqn_team.txt

[ -f "$PV" ] || { log "ERROR: falta el preview $PV"; log "V3_DONE"; exit 1; }
clean_ps
pkill -f "calc_serve[r].js" 2>/dev/null; sleep 2
log "inicio commit $(git rev-parse --short HEAD); preview $PV; JOBS=$JOBS"

prev_ck() {  # familia semilla -> checkpoint de la versión anterior
  if [ "$1" = ppo ]; then echo "$PPO2/ppo_s$2/final.pt"; else echo "$DDQN1/ddqn_s$2/final.pt"; fi
}

diag_port() {  # puerto nombre familia checkpoint [--opciones de policy_diag...]
  local port=$1 name=$2 fam=$3 ck=$4; shift 4
  [ -s "$OUT/diag/$name.json" ] && { log "diag $name ya hecho (reanudación)"; return; }
  [ -f "$ck" ] || { log "diag $name: falta $ck"; return; }
  local t0=$SECONDS
  env PS_TEAM_SPLIT=eval python -m showdown.rl.ddqn.policy_diag --agent "$fam" --checkpoint "$ck" --port "$port" \
    --agent-team $TEAM --preview-checkpoint "$PV" "$@" --out "$OUT/diag/$name.json" > "$OUT/diag/$name.txt" 2>&1
  local rc=$?
  log "diag $name rc=$rc puerto=$port $(( (SECONDS - t0) / 60 )) min"
  kill_port "$port"
}

# Reserva de puertos: un puerto por evaluación en curso (8430 … 8430+JOBS-1).
declare -A PORT_PID
free_port() {
  local i p pid
  for ((i = 0; i < JOBS; i++)); do
    p=$((8430 + i)); pid=${PORT_PID[$p]:-}
    if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then echo "$p"; return; fi
  done
}
launch() {  # nombre familia checkpoint [--opciones...]: en segundo plano en cuanto haya un puerto libre
  local p
  while :; do
    p=$(free_port)
    [ -n "$p" ] && break
    wait -n 2>/dev/null || sleep 5
  done
  diag_port "$p" "$@" &
  PORT_PID[$p]=$!
}
wait_all() { wait; }

# ---------------- P1: máscara sin reentrenar (en paralelo) ----------------
for fam in ppo ddqn; do
  for s in 0 1 2 3 4; do
    ck=$(prev_ck $fam $s)
    launch p1_${fam}_s${s}_off $fam "$ck" --games 200 --mask-protect 0 --context
    launch p1_${fam}_s${s}_on $fam "$ck" --games 200 --mask-protect 1 --context
  done
done
wait_all
log "P1 terminado"
# D1/D2 (descriptivos): victorias por arquetipo, ¿los heurísticos ejecutan Trick Room / Tailwind?, uso propio de
# Tailwind y preview real por arquetipo, desde las evaluaciones sin máscara de P1 (llevan --context).
python scripts/analysis/archetype_diag.py --out "$OUT/d1_report.md" "$OUT/diag/p1_ppo_s*_off.json" \
  "$OUT/diag/p1_ddqn_s*_off.json" > "$OUT/d1_report.err" 2>&1
log "D1/D2 rc=$?"

# ---------------- S1: preview condicionado sin reentrenar (en paralelo, 3 semillas) ----------------
for fam in ppo ddqn; do
  for s in 0 1 2; do
    ck=$(prev_ck $fam $s)
    launch s1_${fam}_s${s}_off $fam "$ck" --games 100 --series --mask-protect 0
    launch s1_${fam}_s${s}_on $fam "$ck" --games 100 --series --mask-protect 0 --preview-series
  done
done
wait_all
log "S1 terminado"

# ---------------- V: brazos C y V ----------------
train() {  # familia brazo semilla
  local fam=$1 arm=$2 s=$3 dir="$OUT/${1}_${2}_s${3}"
  local mem=0; [ "$arm" = V ] && mem=1
  if [ -f "$dir/final.pt" ]; then log "${fam}_${arm}_s$s ya entrenado (reanudación)"; return; fi
  local ck; ck=$(prev_ck $fam $s)
  [ -f "$ck" ] || { log "ERROR ${fam}_${arm}_s$s: falta $ck"; return; }
  local extra=(); [ "$fam" = ddqn ] && extra=(--lr 1e-4 --batch-size 256)
  local t0=$SECONDS
  python -m showdown.rl.$fam.train --seed $s --device cuda --start-ps --expand-omega --num-envs 32 \
    --envs-per-server 4 --log-every 10000 --log-every-sec 120 \
    --phases random:150000,greedy_solo:250000,greedy_joint:300000,expectimax_bayesian:300000 \
    --eval-every 250000 --eval-games 50 --suite-games 0 --agent-team $TEAM --preview-checkpoint "$PV" \
    --obs-abilities 1 --series-bo3 1 --mask-protect-streak 1 --series-memory $mem --init-from "$ck" \
    ${extra[@]+"${extra[@]}"} --checkpoint-dir "$dir" > "$dir.txt" 2>&1
  log "${fam}_${arm}_s$s rc=$? $(( (SECONDS - t0) / 60 )) min"; clean_ps
  grep -h "arranque en caliente" "$dir.txt" 2>/dev/null | tail -1 | cut -c1-240 | sed "s/^/  ${fam}_${arm}_s$s: /" >> "$OUT/queue.log"
}

for fam in ppo ddqn; do
  # entrenamientos de uno en uno (nada en paralelo: la limpieza global de Showdown es segura)
  for s in 0 1 2; do
    for arm in C V; do
      train $fam $arm $s
    done
  done
  # evaluaciones de la familia en paralelo
  for s in 0 1 2; do
    for arm in C V; do
      launch v_${fam}_${arm}_s${s} $fam "$OUT/${fam}_${arm}_s$s/final.pt" --games 100 --series --seed 40000
    done
    launch prev_${fam}_s${s} $fam "$(prev_ck $fam $s)" --games 100 --series --seed 40000
    for arm in C V; do
      launch vtie_${fam}_${arm}_s${s} $fam "$OUT/${fam}_${arm}_s$s/final.pt" --games 100 --series --seed 40000 --foe-tie
    done
  done
  wait_all
  log "familia $fam terminada"
done

python scripts/analysis/v3_report.py "$OUT" > "$OUT/v3_report.err" 2>&1
log "informe rc=$?"
log "V3_DONE"
