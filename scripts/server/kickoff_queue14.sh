#!/bin/bash
# Decimocuarta cola (16-11-2026): remedios contra la memorización de equipos rivales en DDQN
# (memorizacion_equipos.md §7–§8), aprobados por el usuario. §7 mostró que DDQN responde a la
# combinación completa del equipo rival: gana 60 % contra equipos eval cuyos 6 sets conoce, frente
# a 88 % con sus propios equipos. Dos remedios, 1M pasos, semillas 0 y 1, misma configuración que
# el ensayo general B (undécima cola: máscara del aliado activa, simetría apagada, validación
# reservada), que es la referencia:
#
#   REG: (a) regularización: AdamW con weight decay 0,01 + dropout 0,1 en la red Q.
#   FBD: (b) borrado de cada Pokémon rival no activo de la obs con p = 0,5 al aprender.
#
# Criterio fijado de antemano: un remedio se adopta si mejora a B contra eval en ambas semillas y
# por más de ~5 puntos de media (B: 49,3 / 47,0 %).
# Espera a KICKOFF_QUEUE13_DONE. Resumen: $OUT/kickoff_summary14.md.
#
#   bash scripts/server/kickoff_queue14.sh ~/iapsvgc_runs/kickoff14 ~/iapsvgc_runs ~/iapsvgc_runs/kickoff13
set -u
OUT=${1:-$HOME/iapsvgc_runs/kickoff14}
R=${2:-$HOME/iapsvgc_runs}
Q13=${3:-$HOME/iapsvgc_runs/kickoff13}
mkdir -p "$OUT"/diag
source ~/miniconda3/etc/profile.d/conda.sh; conda activate kuroneko
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
export PYTHONUNBUFFERED=1 PS_RL_STRICT=1
unset RL_MASK_ALLY_DAMAGE RL_SYM_AUG RL_HOLDOUT_VAL RL_TRAIN_SUBSET RL_FOE_SYNTH  # valores por defecto
log() { echo "$(date '+%F %T') $*" >> "$OUT/queue.log"; }
clean_ps() { pkill -f "pokemon-showdown.*start" 2>/dev/null; sleep 5; }
log "esperando a la decimotercera cola ($Q13)"
until grep -q KICKOFF_QUEUE13_DONE "$Q13/queue.log" 2>/dev/null; do sleep 60; done
clean_ps
log "inicio commit $(git rev-parse --short HEAD)"
TEAM=showdown/rl/data/ddqn_team.txt
PV=$(sed -n 's/.*preview=\([^ ]*\).*/\1/p' "$R/kickoff3/mini_choice.txt" 2>/dev/null)
[ -f "${PV:-}" ] || PV=""
log "preview: ${PV:-ninguno}"

# 0. tests de lo nuevo
OMP_NUM_THREADS=4 python -m pytest -q -p no:cacheprovider tests/test_ddqn_regularization.py > "$OUT/tests0.txt" 2>&1
log "0 tests rc=$? ($(tail -1 "$OUT/tests0.txt"))"

diag() {  # nombre checkpoint grupo [variables...]
  local name=$1 ck=$2 split=$3; shift 3
  [ -f "$ck" ] || { log "diag $name: falta $ck"; return; }
  env PS_TEAM_SPLIT=$split "$@" python -m showdown.rl.ddqn.policy_diag --checkpoint "$ck" \
    --games 100 --port 8404 --agent-team $TEAM ${PV:+--preview-checkpoint "$PV"} \
    --out "$OUT/diag/$name.json" > "$OUT/diag/$name.txt" 2>&1
  log "diag $name rc=$?"; clean_ps
}

run() {  # nombre semilla [opciones del entrenador...]
  local name=$1 seed=$2; shift 2
  python -m showdown.rl.ddqn.train "$@" --seed $seed --device cuda --start-ps --expand-omega \
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

# Alternando remedios para que, si la cola se corta, haya al menos una semilla de cada uno
run REG_s0 0 --weight-decay 0.01 --dropout 0.1
run FBD_s0 0 --foe-bench-dropout 0.5
run REG_s1 1 --weight-decay 0.01 --dropout 0.1
run FBD_s1 1 --foe-bench-dropout 0.5

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

L = ['# Resumen de la decimocuarta cola: remedios contra la memorización en DDQN', '',
     'DDQN 1M, configuración del ensayo general B. REG = AdamW (weight decay 0,01) + dropout 0,1; '
     'FBD = borrado de Pokémon rivales no activos (p = 0,5). Cifras: 3 heurísticos (300 partidas), '
     '100 partidas por rival, sin exploración. Criterio: se adopta si mejora a B en `eval` en ambas '
     'semillas y por más de ~5 puntos de media.', '',
     '## 1. Contra `eval` y propios', '',
     '| Corrida | best.pt propios | best.pt `eval` | final.pt `eval` | Brecha | `eval` solo / joint / expectimax | best.pt (paso, `val`) |',
     '|---|---|---|---|---|---|---|']
ev = {}
for name in ('B_s0', 'B_s1', 'REG_s0', 'REG_s1', 'FBD_s0', 'FBD_s1'):
    base = (R / 'kickoff11') if name.startswith('B_') else d
    own, e, fe = (heur(base / 'diag' / f'{name}_{k}.json') for k in ('best_own', 'best_eval', 'final_eval'))
    ev[name] = e
    gap = f'{100 * (e[0] - own[0]):+.1f}' if own and e else '—'
    bj = {}
    try:
        bj = json.loads((base / name / 'best.json').read_text())
    except Exception:
        pass
    L.append(f"| {name} | {fmt(own)} | {fmt(e)} | {fmt(fe)} | {gap} | {e[2] if e else '—'} | "
             f"{bj.get('step', 0) // 1000}k, {100 * float(bj.get('mean_wr', 0)):.1f} % |")
L += ['', '## 2. Frente a B (best.pt contra `eval`)', '',
      '| Remedio | Semilla 0 | Semilla 1 | Media | ¿Cumple el criterio? |', '|---|---|---|---|---|']
for rem in ('REG', 'FBD'):
    diffs, cells = [], []
    for s in (0, 1):
        a, b = ev.get(f'B_s{s}'), ev.get(f'{rem}_s{s}')
        if a and b:
            se = 100 * math.sqrt(a[0] * (1 - a[0]) / a[1] + b[0] * (1 - b[0]) / b[1])
            diffs.append(b[0] - a[0])
            cells.append(f'{100 * (b[0] - a[0]):+.1f} (± {se:.1f})')
        else:
            cells.append('—')
    ok = len(diffs) == 2 and all(x > 0 for x in diffs) and sum(diffs) / 2 > 0.05
    mean = f'{100 * sum(diffs) / len(diffs):+.1f}' if diffs else '—'
    L.append(f"| {rem} | {cells[0]} | {cells[1]} | {mean} | {'sí' if ok else 'no'} |")
for name in ('REG_s0', 'REG_s1', 'FBD_s0', 'FBD_s1'):
    mf = d / name / 'train_metrics.jsonl'
    if mf.is_file():
        for line in mf.read_text(errors='replace').splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get('kind') == 'foe_state' and 'total' in r:
                L.append(f"\nEspejo de rivales {name}: `{r['total']}`")
(d / 'kickoff_summary14.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
PY
log "resumen rc=$?"
log "KICKOFF_QUEUE14_DONE"
