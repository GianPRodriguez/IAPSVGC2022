#!/bin/bash
# Decimoquinta cola (17-11-2026): ajuste de los hiperparámetros de la regularización (REG) de DDQN
# antes del kick-off (decisión del usuario). REG actual: weight decay 0,01 + dropout 0,1
# (ddqn_spec 1.20). Como REG ayudó, se prueban variantes más fuertes:
#
#   R_wd03: weight decay 0,03 + dropout 0,1
#   R_do2:  weight decay 0,01 + dropout 0,2
#   R_both: weight decay 0,03 + dropout 0,2
#
# 1M pasos, semillas 0 y 1, configuración por defecto (máscara del aliado, sin simetría,
# validación reservada). La referencia es REG de la decimocuarta cola (mismas semillas).
#
# **Elección sin tocar `eval`** (protocolo §3c.2): cada final.pt se mide contra los 30 equipos
# `val` con 200 partidas por rival (600 contra los 3 heurísticos por semilla, 1 200 por
# configuración). Regla fijada de antemano: se adopta la variante con mayor tasa media en `val`
# solo si supera a REG actual por más de 4 puntos (≈ 2 errores estándar de la diferencia); si no,
# se mantiene REG actual. La elección queda en $OUT/chosen_reg.txt (opciones del entrenador) y la
# usa la decimosexta cola (kick-offs). Los diagnósticos con `eval` se guardan solo como registro.
# Espera a KICKOFF_QUEUE14_DONE. Resumen: $OUT/kickoff_summary15.md.
#
#   bash scripts/server/kickoff_queue15.sh ~/iapsvgc_runs/kickoff15 ~/iapsvgc_runs ~/iapsvgc_runs/kickoff14
set -u
OUT=${1:-$HOME/iapsvgc_runs/kickoff15}
R=${2:-$HOME/iapsvgc_runs}
Q14=${3:-$HOME/iapsvgc_runs/kickoff14}
mkdir -p "$OUT"/diag
source ~/miniconda3/etc/profile.d/conda.sh; conda activate kuroneko
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
export PYTHONUNBUFFERED=1 PS_RL_STRICT=1
unset RL_MASK_ALLY_DAMAGE RL_SYM_AUG RL_HOLDOUT_VAL RL_TRAIN_SUBSET RL_FOE_SYNTH  # valores por defecto
log() { echo "$(date '+%F %T') $*" >> "$OUT/queue.log"; }
clean_ps() { pkill -f "pokemon-showdown.*start" 2>/dev/null; sleep 5; }
log "esperando a la decimocuarta cola ($Q14)"
until grep -q KICKOFF_QUEUE14_DONE "$Q14/queue.log" 2>/dev/null; do sleep 60; done
clean_ps
log "inicio commit $(git rev-parse --short HEAD)"
TEAM=showdown/rl/data/ddqn_team.txt
PV=$(sed -n 's/.*preview=\([^ ]*\).*/\1/p' "$R/kickoff3/mini_choice.txt" 2>/dev/null)
[ -f "${PV:-}" ] || PV=""
log "preview: ${PV:-ninguno}"

diag() {  # nombre checkpoint grupo partidas [variables...]
  local name=$1 ck=$2 split=$3 games=$4; shift 4
  [ -f "$ck" ] || { log "diag $name: falta $ck"; return; }
  env PS_TEAM_SPLIT=$split "$@" python -m showdown.rl.ddqn.policy_diag --checkpoint "$ck" \
    --games $games --port 8406 --agent-team $TEAM ${PV:+--preview-checkpoint "$PV"} \
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
  diag ${name}_final_val "$OUT/$name/final.pt" val 200
  diag ${name}_final_eval "$OUT/$name/final.pt" eval 100  # solo registro, no elige
}

# 1. referencia: REG actual (decimocuarta cola) medido igual
diag REG_s0_final_val "$Q14/REG_s0/final.pt" val 200
diag REG_s1_final_val "$Q14/REG_s1/final.pt" val 200

# 2. variantes (alternando, para tener una semilla de cada una lo antes posible)
run R_wd03_s0 0 --weight-decay 0.03 --dropout 0.1
run R_do2_s0 0 --weight-decay 0.01 --dropout 0.2
run R_both_s0 0 --weight-decay 0.03 --dropout 0.2
run R_wd03_s1 1 --weight-decay 0.03 --dropout 0.1
run R_do2_s1 1 --weight-decay 0.01 --dropout 0.2
run R_both_s1 1 --weight-decay 0.03 --dropout 0.2

# 3. elección y resumen
python - "$OUT" "$Q14" <<'PY'
import json, math, sys
from pathlib import Path
d, q14 = Path(sys.argv[1]), Path(sys.argv[2])
HEUR = ('greedy_solo', 'greedy_joint', 'expectimax_bayesian')
CONF = {'REG': ('--weight-decay 0.01 --dropout 0.1', 'weight decay 0,01 + dropout 0,1 (actual)'),
        'R_wd03': ('--weight-decay 0.03 --dropout 0.1', 'weight decay 0,03 + dropout 0,1'),
        'R_do2': ('--weight-decay 0.01 --dropout 0.2', 'weight decay 0,01 + dropout 0,2'),
        'R_both': ('--weight-decay 0.03 --dropout 0.2', 'weight decay 0,03 + dropout 0,2')}

def wins(p):
    try:
        r = json.loads(p.read_text())['resultados']
    except Exception:
        return None
    return (sum(r.get(f, {}).get('win', 0) for f in HEUR), sum(sum(r.get(f, {}).values()) for f in HEUR),
            ' / '.join(str(r.get(f, {}).get('win', 0)) for f in HEUR))

L = ['# Resumen de la decimoquinta cola: ajuste de REG en DDQN', '',
     'DDQN 1M, configuración por defecto. Elección con `val` (30 equipos no entrenados), final.pt, '
     '200 partidas por rival; 3 heurísticos. Regla: se adopta la mejor variante solo si supera a REG '
     'actual por más de 4 puntos en `val`. `eval` (100 por rival) es solo registro.', '',
     '| Configuración | `val` semilla 0 | `val` semilla 1 | `val` media | `eval` semilla 0 | `eval` semilla 1 |',
     '|---|---|---|---|---|---|']
score = {}
for k, (flags, desc) in CONF.items():
    v, e = [], []
    for s in (0, 1):
        v.append(wins(d / 'diag' / f'{k}_s{s}_final_val.json'))
        e.append(wins((q14 if k == 'REG' else d) / 'diag' / f'{k}_s{s}_final_eval.json'))
    vv = [x for x in v if x]
    if vv:
        score[k] = sum(x[0] for x in vv) / sum(x[1] for x in vv), sum(x[1] for x in vv)
    f = lambda x: '—' if not x else f'{100 * x[0] / x[1]:.1f} % ({x[2]})'
    m = f'{100 * score[k][0]:.1f} %' if k in score else '—'
    L.append(f'| {desc} | {f(v[0])} | {f(v[1])} | **{m}** | {f(e[0])} | {f(e[1])} |')
choice, why = 'REG', 'sin datos suficientes: se mantiene REG actual'
if 'REG' in score:
    cands = {k: v for k, v in score.items() if k != 'REG'}
    if cands:
        best = max(cands, key=lambda k: cands[k][0])
        diff = cands[best][0] - score['REG'][0]
        p1, n1, p0, n0 = cands[best][0], cands[best][1], score['REG'][0], score['REG'][1]
        se = math.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
        if diff > 0.04:
            choice, why = best, f'{best} supera a REG actual por {100 * diff:+.1f} puntos en `val` (± {100 * se:.1f}): se adopta'
        else:
            why = f'la mejor variante ({best}) queda a {100 * diff:+.1f} puntos de REG actual en `val` (± {100 * se:.1f}), por debajo de +4: se mantiene REG actual'
L += ['', f'**Elección:** {CONF[choice][1]}. Motivo: {why}.', '']
(d / 'chosen_reg.txt').write_text(CONF[choice][0] + '\n', encoding='utf-8')
(d / 'kickoff_summary15.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
PY
log "elección rc=$? ($(cat "$OUT/chosen_reg.txt" 2>/dev/null))"
log "KICKOFF_QUEUE15_DONE"
