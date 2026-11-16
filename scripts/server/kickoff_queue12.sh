#!/bin/bash
# Duodécima cola (16-11-2026): PPO de 1M pasos con la configuración final (E) y nueva evaluación
# de la simetría, ambas aprobadas por el usuario. Pregunta 1: ¿PPO sigue generalizando a equipos
# no vistos con el currículo completo (joint y expectimax), o en los pilotos de 400k generalizaba
# por haber entrenado menos? Pregunta 2: ¿la simetría mejora la generalización (piloto: +6 puntos
# en eval, una semilla)? La decisión de apagarla se tomó con validaciones sobre equipos de
# entrenamiento. PPO empieza desde cero (decisión del usuario: sin --bc-checkpoint).
#
# Configuración final: máscara de daño al aliado activa, validación reservada (fit / val), sin
# sintéticos, equipo fijo, preview v2. Simetría: apagada (por defecto) frente a RL_SYM_AUG=1.
# Semillas 0 y 1. Cada corrida: best.pt con equipos propios y eval, final.pt con eval.
# Espera a KICKOFF_QUEUE11_DONE. Resumen: $OUT/kickoff_summary12.md.
#
#   bash scripts/server/kickoff_queue12.sh ~/iapsvgc_runs/kickoff12 ~/iapsvgc_runs ~/iapsvgc_runs/kickoff11
set -u
OUT=${1:-$HOME/iapsvgc_runs/kickoff12}
R=${2:-$HOME/iapsvgc_runs}
Q11=${3:-$HOME/iapsvgc_runs/kickoff11}
mkdir -p "$OUT"/diag
source ~/miniconda3/etc/profile.d/conda.sh; conda activate kuroneko
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
export PYTHONUNBUFFERED=1 PS_RL_STRICT=1
unset RL_MASK_ALLY_DAMAGE RL_SYM_AUG RL_HOLDOUT_VAL RL_TRAIN_SUBSET RL_FOE_SYNTH  # valores por defecto
log() { echo "$(date '+%F %T') $*" >> "$OUT/queue.log"; }
clean_ps() { pkill -f "pokemon-showdown.*start" 2>/dev/null; sleep 5; }
log "esperando a la undécima cola ($Q11)"
until grep -q KICKOFF_QUEUE11_DONE "$Q11/queue.log" 2>/dev/null; do sleep 60; done
clean_ps
log "inicio commit $(git rev-parse --short HEAD)"
TEAM=showdown/rl/data/ddqn_team.txt
PV=$(sed -n 's/.*preview=\([^ ]*\).*/\1/p' "$R/kickoff3/mini_choice.txt" 2>/dev/null)
[ -f "${PV:-}" ] || PV=""
log "preview: ${PV:-ninguno}"

diag() {  # nombre checkpoint grupo [variables...]
  local name=$1 ck=$2 split=$3; shift 3
  [ -f "$ck" ] || { log "diag $name: falta $ck"; return; }
  env PS_TEAM_SPLIT=$split "$@" python -m showdown.rl.ddqn.policy_diag --agent ppo --checkpoint "$ck" \
    --games 100 --port 8400 --agent-team $TEAM ${PV:+--preview-checkpoint "$PV"} \
    --out "$OUT/diag/$name.json" > "$OUT/diag/$name.txt" 2>&1
  log "diag $name rc=$?"; clean_ps
}

run() {  # nombre semilla [variables...]
  local name=$1 seed=$2; shift 2
  env "$@" python -m showdown.rl.ppo.train --seed $seed --device cuda --start-ps --expand-omega \
    --num-envs 32 --envs-per-server 4 --log-every 10000 --log-every-sec 120 \
    --phases random:150000,greedy_solo:250000,greedy_joint:300000,expectimax_bayesian:300000 \
    --eval-every 250000 --eval-games 50 --suite-games 0 --agent-team $TEAM \
    ${PV:+--preview-checkpoint "$PV"} --checkpoint-dir "$OUT/$name" > "$OUT/$name.txt" 2>&1
  log "$name rc=$?"; clean_ps
  diag ${name}_best_own "$OUT/$name/best.pt" train RL_HOLDOUT_VAL=1  # propios = fit
  diag ${name}_best_eval "$OUT/$name/best.pt" eval
  diag ${name}_final_eval "$OUT/$name/final.pt" eval
}

# Alternando simetría para que, si la cola se corta, haya al menos un par comparable
run PPO_nosym_s0 0
run PPO_sym_s0 0 RL_SYM_AUG=1
run PPO_nosym_s1 1
run PPO_sym_s1 1 RL_SYM_AUG=1

python - "$OUT" "$R" <<'PY'
import json, math, sys
from pathlib import Path
d, R = Path(sys.argv[1]), Path(sys.argv[2])
HEUR = ('greedy_solo', 'greedy_joint', 'expectimax_bayesian')

def heur(p):
    try:
        r = json.loads(p.read_text())['resultados']
    except Exception:
        return None
    w = sum(r.get(f, {}).get('win', 0) for f in HEUR)
    n = sum(sum(r.get(f, {}).values()) for f in HEUR)
    return (w / n if n else float('nan'), n, ' / '.join(str(r.get(f, {}).get('win', 0)) for f in HEUR))

def fmt(x):
    return '—' if x is None else f'{100 * x[0]:.1f} %'

RUNS = (('PPO_nosym_s0', 'sin simetría, semilla 0'), ('PPO_sym_s0', 'con simetría, semilla 0'),
        ('PPO_nosym_s1', 'sin simetría, semilla 1'), ('PPO_sym_s1', 'con simetría, semilla 1'))
L = ['# Resumen de la duodécima cola: PPO 1M con la configuración final y simetría', '',
     'Máscara del aliado activa, validación reservada (`fit` / `val`), sin sintéticos, desde cero. '
     'Diagnósticos: 100 partidas por rival, sin exploración; cifras = 3 heurísticos (300 partidas).', '',
     '## 1. Equipos propios frente a `eval`', '',
     '| Corrida | best.pt propios | best.pt `eval` | final.pt `eval` | Brecha (best) | `eval` solo / joint / expectimax (best) |',
     '|---|---|---|---|---|---|']
ev_by = {}
for name, desc in RUNS:
    own, ev, fe = (heur(d / 'diag' / f'{name}_{k}.json') for k in ('best_own', 'best_eval', 'final_eval'))
    ev_by[name] = ev
    gap = '—'
    if own and ev:
        se = 100 * math.sqrt(own[0] * (1 - own[0]) / own[1] + ev[0] * (1 - ev[0]) / ev[1])
        gap = f'{100 * (ev[0] - own[0]):+.1f} (± {se:.1f})'
    L.append(f"| {desc} | {fmt(own)} | {fmt(ev)} | {fmt(fe)} | {gap} | {ev[2] if ev else '—'} |")
L += ['', '## 2. Simetría (best.pt contra `eval`)', '', '| Semilla | Sin | Con | Diferencia |', '|---|---|---|---|']
diffs = []
for s in (0, 1):
    a, b = ev_by.get(f'PPO_nosym_s{s}'), ev_by.get(f'PPO_sym_s{s}')
    if a and b:
        se = 100 * math.sqrt(a[0] * (1 - a[0]) / a[1] + b[0] * (1 - b[0]) / b[1])
        diffs.append(b[0] - a[0])
        L.append(f'| {s} | {fmt(a)} | {fmt(b)} | {100 * (b[0] - a[0]):+.1f} (± {se:.1f}) |')
if diffs:
    L.append(f'| **media** | | | **{100 * sum(diffs) / len(diffs):+.1f}** |')
L += ['', 'Referencias (equipos `eval`, 3 heurísticos): pilotos PPO 400k (cola 9) con simetría 80,0 %, '
      'sin simetría 74,0 %; DDQN 1M (cola 8) 34–39 %; ensayo general DDQN (cola 11) en '
      '`kickoff11/kickoff_summary11.md`.', '',
      '## 3. Validación reservada (`val`) durante el entrenamiento', '',
      '| Corrida | Paso | Media `val` (con random) | Por rival |', '|---|---|---|---|']
for name, _ in RUNS:
    mf = d / name / 'train_metrics.jsonl'
    if not mf.is_file():
        continue
    for line in mf.read_text(errors='replace').splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get('kind') == 'eval':
            cells = ' / '.join(f"{100 * x['win_rate']:.0f}" for x in r.get('rows', []))
            L.append(f"| {name} | {r['step'] // 1000}k | {100 * r['mean_wr']:.1f} % | {cells} |")
        elif r.get('kind') == 'foe_state' and 'total' in r:
            L.append(f"| {name} | espejo de rivales | `{r['total']}` | |")
    b = d / name / 'best.json'
    if b.is_file():
        bj = json.loads(b.read_text())
        L.append(f"| {name} | **best.pt = {bj.get('step', 0) // 1000}k** | {100 * float(bj.get('mean_wr', 0)):.1f} % | |")
(d / 'kickoff_summary12.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
PY
log "resumen rc=$?"
log "KICKOFF_QUEUE12_DONE"
