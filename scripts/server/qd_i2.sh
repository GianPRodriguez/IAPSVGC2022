#!/bin/bash
# AGENTE I 2.0 (QD-PPO desde PPO v2), decisiones del usuario del 25-11-2026 (docs/qd_ppo_spec.md §I 2.0):
# arrancar desde PPO v2 (semilla con mejor val, regla previa), control "PPO v2 + el mismo presupuesto" y
# misma receta luego para J y K.
#
#   0. Espera a KICKOFF_PPO_V2_DONE y toma la semilla elegida (ppo_v2_seed.txt) → INIT = final.pt.
#   1. Tests.
#   2. Preview QD-PGA v1.6 en vivo con piloto = PPO v2 (INIT) y rival greedy_joint, contextos con el
#      equipo propio fijo y observación P5. Desplegado = la élite con mejor reevaluación (best_elite.pt).
#   3. Control: PPO v2 sigue entrenando 1M pasos sin archivo (mismo mosaico, preview, equipo).
#   4. I 2.0: MAP-Elites + PPO desde INIT, 1M pasos (calibración, comparación pareada, reevaluación en val).
#   5. Exploración (descriptiva): final.pt del control, best_elite.pt de I e INIT, los tres con el preview
#      QD-PGA, contra eval (200 partidas por rival, policy_diag). La confirmación se pre-registra aparte.
# Caché de la calculadora: la de cada proceso (regla: nunca > 250 GB; se vigila el disco).
# Resumen: $OUT/qd_i2_summary.txt. Marca: QD_I2_DONE.
#
#   bash scripts/server/qd_i2.sh ~/iapsvgc_runs/qd_i2 ~/iapsvgc_runs/kickoff_ppo_v2
set -u
OUT=${1:-$HOME/iapsvgc_runs/qd_i2}
KO=${2:-$HOME/iapsvgc_runs/kickoff_ppo_v2}
mkdir -p "$OUT"/diag
source ~/miniconda3/etc/profile.d/conda.sh; conda activate kuroneko
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
export PYTHONUNBUFFERED=1 PS_RL_STRICT=1
unset RL_MASK_ALLY_DAMAGE RL_SYM_AUG RL_HOLDOUT_VAL RL_TRAIN_SUBSET RL_FOE_SYNTH PREVIEW_FORCE_ORDER PS_JOINT_ARCHETYPE \
  PPO_FOE_CHECKPOINTS PPO_FOE_PREVIEW RL_OBS_ABILITIES RL_OBS_CALC PREVIEW_SAMPLE PREVIEW_TEMPERATURE PS_TEAM_SPLIT \
  PREVIEW_PILOT_CHECKPOINT PREVIEW_FOE_PILOT PS_PREVIEW_OWN_TEAM PREVIEW_CHECKPOINT PREVIEW_REQUIRED
log() { echo "$(date '+%F %T') $*" >> "$OUT/queue.log"; }
clean_ps() { pkill -f "[p]okemon-showdown.*start" 2>/dev/null; sleep 5; }
TEAM=showdown/rl/data/ddqn_team.txt

log "esperando KICKOFF_PPO_V2_DONE en $KO"
until grep -q KICKOFF_PPO_V2_DONE "$KO/queue.log" 2>/dev/null; do sleep 120; done
S=$(grep -o "ELEGIDA [0-9]*" "$KO/ppo_v2_seed.txt" 2>/dev/null | grep -o "[0-9]*")
INIT="$KO/ppo_s${S:-x}/final.pt"
[ -f "$INIT" ] || { log "ERROR: sin semilla elegida o falta $INIT"; log "QD_I2_DONE"; exit 1; }
log "inicio commit $(git rev-parse --short HEAD); PPO v2 semilla $S: $INIT"
clean_ps

OMP_NUM_THREADS=4 python -m pytest -q -p no:cacheprovider tests/test_qd_ppo2.py tests/test_rl_qd_ppo.py \
  tests/test_team_preview_qd_pga.py tests/test_frozen_preview.py tests/test_parallel_training.py \
  tests/test_preview_rl_pilot.py > "$OUT/tests.txt" 2>&1
rc=$?; log "tests rc=$rc ($(tail -1 "$OUT/tests.txt"))"
[ $rc = 0 ] || { log "ERROR: tests fallan; se detiene"; log "QD_I2_DONE"; exit 1; }

# ---------------------------------------------------------------- 2. preview QD-PGA (piloto PPO v2)
PV="$OUT/preview/best_elite.pt"
if [ ! -f "$PV" ]; then
  PREVIEW_PILOT_CHECKPOINT="$INIT" PREVIEW_FOE_PILOT=greedy_joint python -m showdown.rl.team_preview_qd_pga.train \
    --seed 0 --device cuda --backend live --num-workers 28 --envs-per-server 4 --start-ps --pilot ppo \
    --own-team $TEAM --obs-abilities 1 --checkpoint-dir "$OUT/preview" > "$OUT/preview.txt" 2>&1
  log "preview rc=$?"; clean_ps
fi
[ -f "$PV" ] || { log "ERROR: falta $PV"; log "QD_I2_DONE"; exit 1; }
log "preview: $(grep -E '^(coverage|best_f|MEAN_SCORE)' "$OUT/preview/reporte.txt" | tr '\n' ' ')"

# ---------------------------------------------------------------- 3–4. control e I 2.0
train2() {  # nombre [opciones...]
  local name=$1; shift
  if [ -f "$OUT/$name/summary.json" ]; then log "$name ya hecho (reanudación)"; return; fi
  python -m showdown.rl.qd_ppo.train2 --seed 0 --device cuda --start-ps --num-envs 32 --envs-per-server 4 \
    --init-checkpoint "$INIT" --preview-checkpoint "$PV" --require-preview --agent-team $TEAM \
    --checkpoint-dir "$OUT/$name" "$@" > "$OUT/$name.txt" 2>&1
  log "$name rc=$?"; clean_ps
  log "  disco: $(df -h ~ | tail -1 | awk '{print $5}')"
}
train2 control --control
train2 qd

# ---------------------------------------------------------------- 5. exploración contra eval
diag() {  # nombre checkpoint
  local name=$1 ck=$2
  [ -s "$OUT/diag/$name.json" ] && { log "diag $name ya hecho"; return; }
  [ -f "$ck" ] || { log "diag $name: falta $ck"; return; }
  PS_TEAM_SPLIT=eval python -m showdown.rl.ddqn.policy_diag --agent ppo --family qd_ppo --checkpoint "$ck" \
    --games 200 --port 8455 --agent-team $TEAM --preview-checkpoint "$PV" \
    --out "$OUT/diag/$name.json" > "$OUT/diag/$name.txt" 2>&1
  log "diag $name rc=$?"; clean_ps
}
diag i2_eval "$OUT/qd/best_elite.pt"
diag control_eval "$OUT/control/final.pt"
diag init_qdpreview_eval "$INIT"

python - "$OUT" "$KO" "$S" > "$OUT/qd_i2_summary.txt" 2>&1 <<'PY'
import json, sys
from pathlib import Path
from showdown.rl.kickoff_report import FOES, HEUR, _load, rate
out, ko, s = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
rows = [('I 2.0 (best_elite, preview QD)', out / 'diag' / 'i2_eval.json'),
        ('Control: PPO v2 + 1M (preview QD)', out / 'diag' / 'control_eval.json'),
        (f'PPO v2 s{s} (arranque, preview QD)', out / 'diag' / 'init_qdpreview_eval.json'),
        (f'PPO v2 s{s} (su preview v2.1)', ko / 'diag' / f'ppo_s{s}_final_eval.json')]
print('=== I 2.0: exploración contra eval (200 partidas por rival; descriptivo) ===')
print(f"{'agente':40s} " + ' '.join(f'{f:>12s}' for f in FOES) + '   3 heur.')
for name, p in rows:
    res = _load(p)
    if not res:
        print(f'{name:40s} sin datos'); continue
    cols = []
    for f in FOES:
        r = rate(res, (f,))
        cols.append(f'{100 * r[0]:11.1f}%' if r else '         --')
    h = rate(res)
    print(f'{name:40s} ' + ' '.join(cols) + (f'   {100 * h[0]:.1f}%' if h else ''))
for name in ('qd', 'control'):
    p = out / name / 'summary.json'
    if p.is_file():
        d = json.loads(p.read_text())
        keep = {k: d.get(k) for k in ('global_steps', 'generations', 'bounds', 'coverage', 'qd_score', 'cells',
                                      'children', 'accepts', 'new_cells', 'replaced', 'rejected_gmin',
                                      'best_reeval', 'reeval')}
        print(f'\n--- {name}: ' + json.dumps({k: v for k, v in keep.items() if v is not None}, default=str))
PY
log "resumen escrito en $OUT/qd_i2_summary.txt"
log "QD_I2_DONE"
