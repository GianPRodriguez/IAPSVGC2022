#!/bin/bash
# Decimotercera cola (16-11-2026): ¿qué causó la mejora de +11,8 puntos en eval del ensayo general
# de DDQN (undécima cola) frente a F170 (octava cola)? Entre ambos cambiaron dos cosas: la máscara
# de daño al aliado (apagada → activa) y la simetría (activa → apagada). Esta cola corre la
# combinación que falta, máscara activa + simetría activa, con las mismas semillas:
#
#   | máscara \ simetría | activa              | apagada          |
#   | apagada            | F170 (cola 8)       | —                |
#   | activa             | MSYM (esta cola)    | B (cola 11)      |
#
# MSYM frente a F170 = efecto de la máscara; MSYM frente a B = efecto de la simetría (en DDQN; la
# duodécima cola lo mide en PPO). Configuración idéntica a B salvo RL_SYM_AUG=1.
# Espera a KICKOFF_QUEUE12_DONE. Resumen: $OUT/kickoff_summary13.md.
#
#   bash scripts/server/kickoff_queue13.sh ~/iapsvgc_runs/kickoff13 ~/iapsvgc_runs ~/iapsvgc_runs/kickoff12
set -u
OUT=${1:-$HOME/iapsvgc_runs/kickoff13}
R=${2:-$HOME/iapsvgc_runs}
Q12=${3:-$HOME/iapsvgc_runs/kickoff12}
mkdir -p "$OUT"/diag
source ~/miniconda3/etc/profile.d/conda.sh; conda activate kuroneko
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
export PYTHONUNBUFFERED=1 PS_RL_STRICT=1
unset RL_MASK_ALLY_DAMAGE RL_SYM_AUG RL_HOLDOUT_VAL RL_TRAIN_SUBSET RL_FOE_SYNTH  # valores por defecto
log() { echo "$(date '+%F %T') $*" >> "$OUT/queue.log"; }
clean_ps() { pkill -f "pokemon-showdown.*start" 2>/dev/null; sleep 5; }
log "esperando a la duodécima cola ($Q12)"
until grep -q KICKOFF_QUEUE12_DONE "$Q12/queue.log" 2>/dev/null; do sleep 60; done
clean_ps
log "inicio commit $(git rev-parse --short HEAD)"
TEAM=showdown/rl/data/ddqn_team.txt
PV=$(sed -n 's/.*preview=\([^ ]*\).*/\1/p' "$R/kickoff3/mini_choice.txt" 2>/dev/null)
[ -f "${PV:-}" ] || PV=""
log "preview: ${PV:-ninguno}"

diag() {  # nombre checkpoint grupo [variables...]
  local name=$1 ck=$2 split=$3; shift 3
  [ -f "$ck" ] || { log "diag $name: falta $ck"; return; }
  env PS_TEAM_SPLIT=$split "$@" python -m showdown.rl.ddqn.policy_diag --checkpoint "$ck" \
    --games 100 --port 8402 --agent-team $TEAM ${PV:+--preview-checkpoint "$PV"} \
    --out "$OUT/diag/$name.json" > "$OUT/diag/$name.txt" 2>&1
  log "diag $name rc=$?"; clean_ps
}

run() {  # nombre semilla
  local name=$1 seed=$2
  RL_SYM_AUG=1 python -m showdown.rl.ddqn.train --seed $seed --device cuda --start-ps --expand-omega \
    --num-envs 32 --envs-per-server 4 --log-every 10000 --log-every-sec 120 \
    --phases random:150000,greedy_solo:250000,greedy_joint:300000,expectimax_bayesian:300000 \
    --eval-every 250000 --eval-games 50 --suite-games 0 --lr 1e-4 --batch-size 256 \
    --agent-team $TEAM ${PV:+--preview-checkpoint "$PV"} \
    --checkpoint-dir "$OUT/$name" > "$OUT/$name.txt" 2>&1
  log "$name rc=$?"; clean_ps
  diag ${name}_best_own "$OUT/$name/best.pt" train RL_HOLDOUT_VAL=1  # propios = fit
  diag ${name}_best_eval "$OUT/$name/best.pt" eval
  diag ${name}_final_eval "$OUT/$name/final.pt" eval
}

run MSYM_s0 0
run MSYM_s1 1

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
    return (w / n if n else float('nan'), n)

def fmt(x):
    return '—' if x is None else f'{100 * x[0]:.1f} %'

def diff(a, b):
    if not a or not b:
        return '—', None
    se = 100 * math.sqrt(a[0] * (1 - a[0]) / a[1] + b[0] * (1 - b[0]) / b[1])
    return f'{100 * (b[0] - a[0]):+.1f} (± {se:.1f})', b[0] - a[0]

L = ['# Resumen de la decimotercera cola: máscara del aliado frente a simetría en DDQN', '',
     'DDQN 1M, currículo completo, validación reservada, sin sintéticos. Cifras: best.pt contra los '
     '3 heurísticos (300 partidas), 100 partidas por rival, sin exploración.', '',
     '## 1. Las tres configuraciones contra `eval`', '',
     '| Semilla | F170: sin máscara, con simetría (cola 8) | MSYM: con máscara, con simetría | B: con máscara, sin simetría (cola 11) |',
     '|---|---|---|---|']
cells = {}
for s in (0, 1):
    f = heur(R / 'kickoff8' / 'diag' / f'F170_s{s}_eval.json')
    m = heur(d / 'diag' / f'MSYM_s{s}_best_eval.json')
    b = heur(R / 'kickoff11' / 'diag' / f'B_s{s}_best_eval.json')
    cells[s] = (f, m, b)
    L.append(f'| {s} | {fmt(f)} | {fmt(m)} | {fmt(b)} |')
L += ['', '## 2. Efectos', '', '| Efecto | Semilla 0 | Semilla 1 | Media |', '|---|---|---|---|']
for label, i, j in (('Máscara del aliado (MSYM − F170)', 0, 1), ('Apagar la simetría (B − MSYM)', 1, 2),
                    ('Total (B − F170)', 0, 2)):
    row, vals = [], []
    for s in (0, 1):
        txt, v = diff(cells[s][i], cells[s][j])
        row.append(txt)
        if v is not None:
            vals.append(v)
    mean = f'{100 * sum(vals) / len(vals):+.1f}' if vals else '—'
    L.append(f'| {label} | {row[0]} | {row[1]} | {mean} |')
L += ['', '## 3. Detalle de MSYM', '', '| Corrida | best.pt propios | best.pt `eval` | final.pt `eval` | best.pt (paso, `val`) |',
      '|---|---|---|---|---|']
for s in (0, 1):
    n = f'MSYM_s{s}'
    own, ev, fe = (heur(d / 'diag' / f'{n}_{k}.json') for k in ('best_own', 'best_eval', 'final_eval'))
    bj = {}
    try:
        bj = json.loads((d / n / 'best.json').read_text())
    except Exception:
        pass
    L.append(f"| {n} | {fmt(own)} | {fmt(ev)} | {fmt(fe)} | {bj.get('step', 0) // 1000}k, "
             f"{100 * float(bj.get('mean_wr', 0)):.1f} % |")
for s in (0, 1):
    mf = d / f'MSYM_s{s}' / 'train_metrics.jsonl'
    if mf.is_file():
        for line in mf.read_text(errors='replace').splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get('kind') == 'foe_state' and 'total' in r:
                L.append(f"\nEspejo de rivales MSYM_s{s}: `{r['total']}`")
(d / 'kickoff_summary13.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
PY
log "resumen rc=$?"
log "KICKOFF_QUEUE13_DONE"
