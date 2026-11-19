#!/bin/bash
# Corridas oficiales con P5 completo + preview v2.1 (PPO v2 y kick-off de DDQN), encadenadas con condición
# (decisión del usuario 19-11-2026: "encadenar con la condición").
#
# Espera a P1B_DONE (después de la vigesimosexta cola y de P1b). CONDICIÓN para seguir (si falla, se
# detiene y escribe GATE_FAIL):
#   - PPO y DDQN con P5 + preview v2.1 (cola 26) no pierden más de 2 puntos en val frente a su referencia
#     (PPO v1 semillas 0 y 1; DDQN REG 0,03), partidas agrupadas;
#   - la calculadora real no falló en esos entrenamientos (respaldo < 1 % y pedidos sin respuesta < 1 %).
# Luego, en el orden ORDER (parámetro pendiente de decisión del usuario), 1M × 5 semillas por agente con la
# configuración final, observación nueva y preview v2.1, y por semilla (200 partidas por rival): final.pt
# contra eval (principal), best.pt contra eval, final.pt contra sus propios equipos y contra greedy_joint
# en modo arquetipo; con H2H=1, PPO también contra PPO v1 en espejo (v1 con su preview v2).
# Resumen: $OUT/kickoff_summary16.md (showdown/rl/kickoff_report.py).
#
#   ORDER="ppo ddqn" H2H=1 bash scripts/server/kickoff_oficial_v2.sh ~/iapsvgc_runs/oficial_v2 \
#       ~/iapsvgc_runs/p5full ~/iapsvgc_runs/ppo_v2 ~/iapsvgc_runs/ppo_v1 ~/iapsvgc_runs/kickoff15 ~/iapsvgc_runs/p1b
set -u
OUT=${1:-$HOME/iapsvgc_runs/oficial_v2}
Q26=${2:-$HOME/iapsvgc_runs/p5full}
V2=${3:-$HOME/iapsvgc_runs/ppo_v2}
V1=${4:-$HOME/iapsvgc_runs/ppo_v1}
Q15=${5:-$HOME/iapsvgc_runs/kickoff15}
P1B=${6:-$HOME/iapsvgc_runs/p1b}
: "${ORDER:?falta ORDER (orden de las corridas oficiales, decisión del usuario)}"
H2H=${H2H:-0}
mkdir -p "$OUT"/diag
source ~/miniconda3/etc/profile.d/conda.sh; conda activate kuroneko
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
export PYTHONUNBUFFERED=1 PS_RL_STRICT=1
unset RL_MASK_ALLY_DAMAGE RL_SYM_AUG RL_HOLDOUT_VAL RL_TRAIN_SUBSET RL_FOE_SYNTH PREVIEW_FORCE_ORDER PS_JOINT_ARCHETYPE \
  PPO_FOE_CHECKPOINTS PPO_FOE_PREVIEW RL_OBS_ABILITIES RL_OBS_CALC PREVIEW_SAMPLE PREVIEW_TEMPERATURE
log() { echo "$(date '+%F %T') $*" >> "$OUT/queue.log"; }
clean_ps() { pkill -f "pokemon-showdown.*start" 2>/dev/null; sleep 5; }
log "esperando a P1b ($P1B) | ORDER=$ORDER H2H=$H2H"
until grep -q P1B_DONE "$P1B/queue.log" 2>/dev/null; do sleep 60; done

# ---------------------------------------------------------------- condición
python - "$Q26" "$V2" "$Q15" > "$OUT/gate.txt" 2>&1 <<'PY'
import re, sys
from pathlib import Path
from showdown.rl.kickoff_summary24 import _load, diff, pooled
q26, v2, q15 = (Path(a) for a in sys.argv[1:4])
g = lambda d, n: _load(d / 'diag' / f'{n}.json')
ok = True
for name, ref, cand in (
        ('PPO', [g(v2, f'v1_s{s}_val') for s in (0, 1)], [g(q26, f'ppo_p5_s{s}_val') for s in (0, 1)]),
        ('DDQN', [g(q15, f'R_wd03_s{s}_final_val') for s in (0, 1)], [g(q26, f'ddqn_p5_s{s}_val') for s in (0, 1)])):
    d = diff(pooled(cand), pooled(ref))
    print(name, 'Δval', d)
    if d is None or d[0] < -2.0:
        ok = False
for txt in sorted(q26.glob('*_p5_s?.txt')):
    lines = [l for l in txt.read_text(errors='replace').splitlines() if 'calculadora real en la observación' in l]
    fb = re.search(r"'fallback_rate': ([0-9.e-]+)", lines[-1]) if lines else None
    ms = re.search(r"'missing_rate': ([0-9.e-]+)", lines[-1]) if lines else None
    print(txt.name, 'respaldo', fb and fb.group(1), 'sin respuesta', ms and ms.group(1))
    if not (fb and ms) or float(fb.group(1)) >= 0.01 or float(ms.group(1)) >= 0.01:
        ok = False
print('GATE', 'OK' if ok else 'FAIL')
sys.exit(0 if ok else 1)
PY
if [ $? != 0 ]; then log "GATE_FAIL ($(tail -1 "$OUT/gate.txt")); se detiene para revisión"; log "KICKOFF_OFICIAL_V2_DONE"; exit 1; fi
log "condición cumplida: $(tr '\n' ' ' < "$OUT/gate.txt")"

clean_ps
log "inicio commit $(git rev-parse --short HEAD)"
TEAM=showdown/rl/data/ddqn_team.txt
PV="$Q26/preview_v21/best.pt"; [ -f "$PV" ] || PV="$Q26/preview_v21/final.pt"
[ -f "$PV" ] || { log "ERROR: falta el preview v2.1"; log "KICKOFF_OFICIAL_V2_DONE"; exit 1; }
PV2="$V1/preview_v2.pt"
V1CKS=$(ls "$V1"/ppo_s[0-4]/final.pt | paste -sd, -)
log "preview v2.1: $PV"

diag() {  # agente nombre checkpoint grupo [VARIABLE=valor...] [--opciones...]
  local agent=$1 name=$2 ck=$3 split=$4; shift 4
  [ -f "$ck" ] || { log "diag $name: falta $ck"; return; }
  local envs=() opts=()
  for x in "$@"; do case "$x" in --*) opts+=("$x");; *) envs+=("$x");; esac; done
  env PS_TEAM_SPLIT=$split ${envs[@]+"${envs[@]}"} python -m showdown.rl.ddqn.policy_diag --agent $agent \
    --checkpoint "$ck" --games 200 --port 8432 --agent-team $TEAM --preview-checkpoint "$PV" \
    ${opts[@]+"${opts[@]}"} --out "$OUT/diag/$name.json" > "$OUT/diag/$name.txt" 2>&1
  log "diag $name rc=$?"; clean_ps
}
PHASES="random:150000,greedy_solo:250000,greedy_joint:300000,expectimax_bayesian:300000"
COMMON="--device cuda --start-ps --expand-omega --num-envs 32 --envs-per-server 4 --log-every 10000 \
  --log-every-sec 120 --phases $PHASES --eval-every 250000 --eval-games 50 --suite-games 0 --agent-team $TEAM \
  --preview-checkpoint $PV --obs-abilities 1"
for a in $ORDER; do
  for s in 0 1 2 3 4; do
    if [ $a = ddqn ]; then EXTRA="--lr 1e-4 --batch-size 256"; else EXTRA=""; fi
    python -m showdown.rl.$a.train --seed $s $COMMON $EXTRA --checkpoint-dir "$OUT/${a}_s$s" > "$OUT/${a}_s$s.txt" 2>&1
    log "${a}_s$s rc=$?"; clean_ps
    diag $a ${a}_s${s}_final_eval "$OUT/${a}_s$s/final.pt" eval
    diag $a ${a}_s${s}_best_eval "$OUT/${a}_s$s/best.pt" eval
    diag $a ${a}_s${s}_final_own "$OUT/${a}_s$s/final.pt" train RL_HOLDOUT_VAL=1
    diag $a ${a}_s${s}_final_arquetipo "$OUT/${a}_s$s/final.pt" eval PS_JOINT_ARCHETYPE=1 --foes=greedy_joint --context
    if [ $a = ppo ] && [ "$H2H" = 1 ]; then
      diag ppo ppo_s${s}_final_h2h "$OUT/ppo_s$s/final.pt" eval PPO_FOE_CHECKPOINTS="$V1CKS" PPO_FOE_PREVIEW="$PV2" --foes=ppo_v1
    fi
  done
  log "OFICIAL_${a}_DONE"
done
python -m showdown.rl.kickoff_report "$OUT" > "$OUT/kickoff_report.err" 2>&1
log "resumen rc=$?"
log "KICKOFF_OFICIAL_V2_DONE"
