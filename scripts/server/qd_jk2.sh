#!/bin/bash
# AGENTES J 2.0 (PGA-ME fiel) y K 2.0 (PPGA fiel), decisión del usuario del 25-11-2026
# (docs/qd_pga_map_elites_spec.md §J 2.0, docs/qd_ppga_spec.md §K 2.0). Misma base que I 2.0:
# arranque desde la semilla de PPO v2 elegida por val, preview v2.1b de PPO v2 (team_preview_ppo vía
# PREVIEW_KIND; el QD-PGA era degenerado con el equipo fijo, decisión del usuario del 26-11-2026),
# mosaico de PPO v2, comparación pareada, reevaluación en val; el control es el de qd_i2 (PPO v2 + 1M
# pasos con el mismo preview, mosaico y equipo), que sirve para los tres.
#
#   0. Espera a QD_I2_DONE (un método a la vez en el servidor).
#   1. Tests.  2. J 2.0 (1M pasos de entorno).  3. K 2.0 (1M pasos de entorno).
#   4. Exploración (descriptiva): best_elite.pt de J y K contra eval (200 partidas por rival, mismo preview),
#      junto a I 2.0, el control y PPO v2 de qd_i2 (mismo preview). La confirmación se pre-registra aparte.
# Resumen: $OUT/qd_jk2_summary.txt. Marca: QD_JK2_DONE.
#
#   bash scripts/server/qd_jk2.sh ~/iapsvgc_runs/qd_jk2 ~/iapsvgc_runs/qd_i2 ~/iapsvgc_runs/kickoff_ppo_v2
set -u
OUT=${1:-$HOME/iapsvgc_runs/qd_jk2}
QI=${2:-$HOME/iapsvgc_runs/qd_i2}
KO=${3:-$HOME/iapsvgc_runs/kickoff_ppo_v2}
mkdir -p "$OUT"/diag
source ~/miniconda3/etc/profile.d/conda.sh; conda activate kuroneko
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
export PYTHONUNBUFFERED=1 PS_RL_STRICT=1
unset RL_MASK_ALLY_DAMAGE RL_SYM_AUG RL_HOLDOUT_VAL RL_TRAIN_SUBSET RL_FOE_SYNTH PREVIEW_FORCE_ORDER PS_JOINT_ARCHETYPE \
  PPO_FOE_CHECKPOINTS PPO_FOE_PREVIEW RL_OBS_ABILITIES RL_OBS_CALC PREVIEW_SAMPLE PREVIEW_TEMPERATURE PS_TEAM_SPLIT \
  PREVIEW_PILOT_CHECKPOINT PREVIEW_FOE_PILOT PS_PREVIEW_OWN_TEAM PREVIEW_CHECKPOINT PREVIEW_REQUIRED RL_BD2 PREVIEW_KIND
log() { echo "$(date '+%F %T') $*" >> "$OUT/queue.log"; }
clean_ps() { pkill -f "[p]okemon-showdown.*start" 2>/dev/null; sleep 5; }
TEAM=showdown/rl/data/ddqn_team.txt

log "esperando QD_I2_DONE en $QI"
until grep -q QD_I2_DONE "$QI/queue.log" 2>/dev/null; do sleep 120; done
S=$(grep -o "ELEGIDA [0-9]*" "$KO/ppo_v2_seed.txt" 2>/dev/null | grep -o "[0-9]*")
INIT="$KO/ppo_s${S:-x}/final.pt"
PV=${PV:-$HOME/iapsvgc_runs/preview_v21b/preview_s0/best.pt}
export PREVIEW_KIND=team_preview_ppo
for f in "$INIT" "$PV"; do [ -f "$f" ] || { log "ERROR: falta $f"; log "QD_JK2_DONE"; exit 1; }; done
log "inicio commit $(git rev-parse --short HEAD); PPO v2 semilla $S; preview $PV"
clean_ps

OMP_NUM_THREADS=4 python -m pytest -q -p no:cacheprovider tests/test_qd_pga2.py tests/test_qd_ppga2.py \
  tests/test_qd_ppo2.py tests/test_parallel_training.py > "$OUT/tests.txt" 2>&1
rc=$?; log "tests rc=$rc ($(tail -1 "$OUT/tests.txt"))"
[ $rc = 0 ] || { log "ERROR: tests fallan; se detiene"; log "QD_JK2_DONE"; exit 1; }

run() {  # nombre módulo
  local name=$1 mod=$2
  if [ -f "$OUT/$name/summary.json" ]; then log "$name ya hecho (reanudación)"; return; fi
  python -m "$mod" --seed 0 --device cuda --start-ps --num-envs 32 --envs-per-server 4 \
    --init-checkpoint "$INIT" --preview-checkpoint "$PV" --require-preview --agent-team $TEAM \
    --checkpoint-dir "$OUT/$name" > "$OUT/$name.txt" 2>&1
  log "$name rc=$?"; clean_ps
  log "  disco: $(df -h ~ | tail -1 | awk '{print $5}')"
}
run j2 showdown.rl.qd_pga.train_j2
run k2 showdown.rl.qd_ppga.train_k2

diag() {  # nombre checkpoint familia
  local name=$1 ck=$2 fam=$3
  [ -s "$OUT/diag/$name.json" ] && { log "diag $name ya hecho"; return; }
  [ -f "$ck" ] || { log "diag $name: falta $ck"; return; }
  PS_TEAM_SPLIT=eval python -m showdown.rl.ddqn.policy_diag --agent ppo --family $fam --checkpoint "$ck" \
    --games 200 --port 8455 --agent-team $TEAM --preview-checkpoint "$PV" \
    --out "$OUT/diag/$name.json" > "$OUT/diag/$name.txt" 2>&1
  log "diag $name rc=$?"; clean_ps
}
diag j2_eval "$OUT/j2/best_elite.pt" qd_pga
diag k2_eval "$OUT/k2/best_elite.pt" qd_ppga

python - "$OUT" "$QI" "$KO" "$S" > "$OUT/qd_jk2_summary.txt" 2>&1 <<'PY'
import json, sys
from pathlib import Path
from showdown.rl.kickoff_report import FOES, _load, rate
out, qi, ko, s = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
rows = [('J 2.0 PGA-ME (best_elite)', out / 'diag' / 'j2_eval.json'),
        ('K 2.0 PPGA (best_elite)', out / 'diag' / 'k2_eval.json'),
        ('I 2.0 QD-PPO (best_elite)', qi / 'diag' / 'i2_eval.json'),
        ('Control: PPO v2 + 1M', qi / 'diag' / 'control_eval.json'),
        (f'PPO v2 s{s} (arranque)', qi / 'diag' / 'init_eval.json')]
print('=== Familia QD 2.0: exploración contra eval (200 partidas por rival, preview v2.1b de PPO v2; descriptivo) ===')
print(f"{'agente':32s} " + ' '.join(f'{f:>12s}' for f in FOES) + '   3 heur.')
for name, p in rows:
    res = _load(p)
    if not res:
        print(f'{name:32s} sin datos'); continue
    cols = [(lambda r: f'{100 * r[0]:11.1f}%' if r else '         --')(rate(res, (f,))) for f in FOES]
    h = rate(res)
    print(f'{name:32s} ' + ' '.join(cols) + (f'   {100 * h[0]:.1f}%' if h else ''))
for name in ('j2', 'k2'):
    p = out / name / 'summary.json'
    if p.is_file():
        d = json.loads(p.read_text())
        drop = ('bounds',)
        print(f'\n--- {name}: ' + json.dumps({k: v for k, v in d.items() if k not in drop}, default=str))
PY
log "resumen escrito en $OUT/qd_jk2_summary.txt"
log "QD_JK2_DONE"
