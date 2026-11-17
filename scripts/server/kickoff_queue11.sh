#!/bin/bash
# Undécima cola (16-11-2026): ensayo general del kick-off de DDQN (B) y corrida de 2M pasos para
# fijar su duración (C). Configuración final decidida por el usuario: valores por defecto de
# rl_env_spec v1.28 (máscara de daño al aliado activa, simetría apagada) + validación reservada
# (activa por defecto en los entrenadores desde el 16-11-2026: entrena con fit, valida con val)
# y sin equipos sintéticos. Espera a KICKOFF_QUEUE10_DONE. Resumen: $OUT/kickoff_summary11.md.
#
# B: 1M pasos, semillas 0 y 1 (las mismas que F170 de la octava cola, que difiere solo en la
#    máscara y la simetría). C: 2M pasos, fases ×2, validación cada 250k.
# Cada corrida se diagnostica con best.pt (propios y eval) y con final.pt (eval): así se ve si la
# validación reservada elige mejor que quedarse con el último checkpoint.
#
#   bash scripts/server/kickoff_queue11.sh ~/iapsvgc_runs/kickoff11 ~/iapsvgc_runs ~/iapsvgc_runs/kickoff10
set -u
OUT=${1:-$HOME/iapsvgc_runs/kickoff11}
R=${2:-$HOME/iapsvgc_runs}
Q10=${3:-$HOME/iapsvgc_runs/kickoff10}
mkdir -p "$OUT"/diag
source ~/miniconda3/etc/profile.d/conda.sh; conda activate kuroneko
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
export PYTHONUNBUFFERED=1 PS_RL_STRICT=1
unset RL_MASK_ALLY_DAMAGE RL_SYM_AUG RL_HOLDOUT_VAL RL_TRAIN_SUBSET RL_FOE_SYNTH  # valores por defecto
log() { echo "$(date '+%F %T') $*" >> "$OUT/queue.log"; }
clean_ps() { pkill -f "pokemon-showdown.*start" 2>/dev/null; sleep 5; }
log "esperando a la décima cola ($Q10)"
until grep -q KICKOFF_QUEUE10_DONE "$Q10/queue.log" 2>/dev/null; do sleep 60; done
clean_ps
log "inicio commit $(git rev-parse --short HEAD)"
TEAM=showdown/rl/data/ddqn_team.txt
PV=$(sed -n 's/.*preview=\([^ ]*\).*/\1/p' "$R/kickoff3/mini_choice.txt" 2>/dev/null)
[ -f "${PV:-}" ] || PV=""
log "preview: ${PV:-ninguno}"

# 0. tests de lo que cambió desde la suite de la novena cola
OMP_NUM_THREADS=4 python -m pytest -q -p no:cacheprovider tests/test_foe_pool_and_synth.py \
  tests/test_resume_keeps_best.py tests/test_ally_mask_and_reward.py > "$OUT/tests0.txt" 2>&1
log "0 tests rc=$? ($(tail -1 "$OUT/tests0.txt"))"

diag() {  # nombre checkpoint grupo [variables...]
  local name=$1 ck=$2 split=$3; shift 3
  [ -f "$ck" ] || { log "diag $name: falta $ck"; return; }
  env PS_TEAM_SPLIT=$split "$@" python -m showdown.rl.ddqn.policy_diag --checkpoint "$ck" \
    --games 100 --port 8398 --agent-team $TEAM ${PV:+--preview-checkpoint "$PV"} \
    --out "$OUT/diag/$name.json" > "$OUT/diag/$name.txt" 2>&1
  log "diag $name rc=$?"; clean_ps
}

run() {  # nombre semilla fases
  local name=$1 seed=$2 phases=$3
  python -m showdown.rl.ddqn.train --seed $seed --device cuda --start-ps --expand-omega \
    --num-envs 32 --envs-per-server 4 --log-every 10000 --log-every-sec 120 \
    --phases "$phases" --eval-every 250000 --eval-games 50 --suite-games 0 --lr 1e-4 \
    --batch-size 256 --agent-team $TEAM ${PV:+--preview-checkpoint "$PV"} \
    --checkpoint-dir "$OUT/$name" > "$OUT/$name.txt" 2>&1
  log "$name rc=$?"; clean_ps
  diag ${name}_best_own "$OUT/$name/best.pt" train RL_HOLDOUT_VAL=1  # propios = fit
  diag ${name}_best_eval "$OUT/$name/best.pt" eval
  diag ${name}_final_eval "$OUT/$name/final.pt" eval
}

P1M=random:150000,greedy_solo:250000,greedy_joint:300000,expectimax_bayesian:300000
P2M=random:300000,greedy_solo:500000,greedy_joint:600000,expectimax_bayesian:600000
run B_s0 0 $P1M
run B_s1 1 $P1M
run C2M_s0 0 $P2M

# suite completa al final
OMP_NUM_THREADS=4 python -m pytest -q -p no:cacheprovider tests > "$OUT/suite.txt" 2>&1
log "suite rc=$? ($(tail -1 "$OUT/suite.txt"))"

python - "$OUT" "$R/kickoff8" <<'PY'
import json, math, sys
from pathlib import Path
d, q8 = Path(sys.argv[1]), Path(sys.argv[2])
HEUR = ('greedy_solo', 'greedy_joint', 'expectimax_bayesian')

def heur(p):
    try:
        r = json.loads(p.read_text())['resultados']
    except Exception:
        return None
    w = sum(r.get(f, {}).get('win', 0) for f in HEUR)
    n = sum(sum(r.get(f, {}).values()) for f in HEUR)
    cells = ' / '.join(f"{r.get(f, {}).get('win', 0)}" for f in HEUR)
    return (w / n if n else float('nan'), n, cells)

def fmt(x):
    return '—' if x is None else f'{100 * x[0]:.1f} %'

L = ['# Resumen de la undécima cola: ensayo general (B) y 2M pasos (C)', '',
     'Configuración final: máscara de daño al aliado activa, simetría apagada, validación '
     'reservada (entrena con `fit`, elige `best.pt` con `val`), sin sintéticos. Diagnósticos: '
     '100 partidas por rival, sin exploración; cifras = 3 heurísticos (300 partidas).', '',
     '## 1. Equipos propios frente a `eval`', '',
     '| Corrida | best.pt propios | best.pt `eval` | final.pt `eval` | Brecha (best) | `eval` solo / joint / expectimax (best) |',
     '|---|---|---|---|---|---|']
for name, desc in (('B_s0', 'B, semilla 0 (1M)'), ('B_s1', 'B, semilla 1 (1M)'), ('C2M_s0', 'C, semilla 0 (2M)')):
    own, ev, fe = (heur(d / 'diag' / f'{name}_{k}.json') for k in ('best_own', 'best_eval', 'final_eval'))
    gap = '—'
    if own and ev:
        se = 100 * math.sqrt(own[0] * (1 - own[0]) / own[1] + ev[0] * (1 - ev[0]) / ev[1])
        gap = f'{100 * (ev[0] - own[0]):+.1f} (± {se:.1f})'
    L.append(f"| {desc} | {fmt(own)} | {fmt(ev)} | {fmt(fe)} | {gap} | {ev[2] if ev else '—'} |")
L += ['', 'Referencia (octava cola, máscara apagada y simetría activa, best.pt):', '',
      '| Corrida | Propios | `eval` |', '|---|---|---|']
for name in ('F170_s0', 'F170_s1'):
    L.append(f"| {name} | {fmt(heur(q8 / 'diag' / f'{name}_own.json'))} | {fmt(heur(q8 / 'diag' / f'{name}_eval.json'))} |")
L += ['', '## 2. Validación reservada (`val`) durante el entrenamiento', '',
      '| Corrida | Paso | Media `val` (con random) | random | solo | joint | expectimax |', '|---|---|---|---|---|---|---|']
for name in ('B_s0', 'B_s1', 'C2M_s0'):
    mf = d / name / 'train_metrics.jsonl'
    if not mf.is_file():
        continue
    for line in mf.read_text(errors='replace').splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get('kind') == 'eval':
            rows = {x.get('opponent', x.get('foe', '?')): x['win_rate'] for x in r.get('rows', [])}
            vals = [x['win_rate'] for x in r.get('rows', [])]
            cells = ' | '.join(f'{100 * v:.0f} %' for v in vals)
            L.append(f"| {name} | {r['step'] // 1000}k | {100 * r['mean_wr']:.1f} % | {cells} |")
    b = d / name / 'best.json'
    if b.is_file():
        bj = json.loads(b.read_text())
        L.append(f"| {name} | **best.pt = {bj.get('step', 0) // 1000}k** | {100 * float(bj.get('mean_wr', 0)):.1f} % | | | | |")
    for line in mf.read_text(errors='replace').splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get('kind') == 'foe_state' and 'total' in r:  # solo el registro final (no las ventanas)
            L.append(f"| {name} | espejo de rivales | `{r['total']}` | | | | |")
suite = (d / 'suite.txt').read_text(errors='replace').strip().splitlines() if (d / 'suite.txt').is_file() else ['sin datos']
fails = [l for l in suite if l.startswith('FAILED') or l.startswith('ERROR')]
L += ['', '## 3. Suite completa de tests', '', '```', suite[-1]] + fails[:30] + ['```', '']
(d / 'kickoff_summary11.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
PY
log "resumen rc=$?"
log "KICKOFF_QUEUE11_DONE"
