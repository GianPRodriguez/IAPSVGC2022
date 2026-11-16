#!/bin/bash
# Décima cola (16-11-2026): validación en vivo con los valores por defecto nuevos (rl_env_spec
# v1.28: máscara de daño al aliado activa, simetría apagada), con máscaras conjuntas (DDQN) y por
# hueco (PPO). Es un control previo al kick-off, no un experimento. Espera a KICKOFF_QUEUE9_DONE.
# Resumen: $OUT/kickoff_summary10.md.
#
#   bash scripts/server/kickoff_queue10.sh ~/iapsvgc_runs/kickoff10 ~/iapsvgc_runs/kickoff9
set -u
OUT=${1:-$HOME/iapsvgc_runs/kickoff10}
Q9=${2:-$HOME/iapsvgc_runs/kickoff9}
mkdir -p "$OUT"
source ~/miniconda3/etc/profile.d/conda.sh; conda activate kuroneko
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
export PYTHONUNBUFFERED=1 PS_RL_STRICT=1
unset RL_MASK_ALLY_DAMAGE RL_SYM_AUG RL_HOLDOUT_VAL RL_TRAIN_SUBSET RL_FOE_SYNTH  # valores por defecto
log() { echo "$(date '+%F %T') $*" >> "$OUT/queue.log"; }
clean_ps() { pkill -f "pokemon-showdown.*start" 2>/dev/null; sleep 5; }
log "esperando a la novena cola ($Q9)"
until grep -q KICKOFF_QUEUE9_DONE "$Q9/queue.log" 2>/dev/null; do sleep 60; done
clean_ps
log "inicio commit $(git rev-parse --short HEAD)"
TEAM=showdown/rl/data/ddqn_team.txt

for kind in joint slot; do
  python -m showdown.rl.live_validation --mask-kind $kind --episodes 25 --port 8396 --device cuda \
    --agent-team $TEAM --out "$OUT/live_validation_$kind.json" > "$OUT/live_validation_$kind.txt" 2>&1
  log "validación en vivo ($kind) rc=$?"; clean_ps
done

python - "$OUT" <<'PY'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
L = ['# Resumen de la décima cola: validación en vivo con los valores por defecto nuevos', '',
     'Máscara de daño al aliado activa y simetría apagada (rl_env_spec v1.28). Acciones legales al '
     'azar, 25 episodios por rival (4 rivales), equipo fijo del agente. Todo lo que no sea el '
     'conteo de pasos, episodios o acciones quitadas por la máscara debería ser 0.', '']
for kind, title in (('joint', 'Máscaras conjuntas (DDQN)'), ('slot', 'Máscaras por hueco (PPO)')):
    L += [f'## {title}', '']
    p = d / f'live_validation_{kind}.json'
    if not p.is_file():
        L += ['Sin datos.', '']
        continue
    r = json.loads(p.read_text())
    c, ms = r['counts'], r.get('ms', {})
    rows = [
        ('Pasos / episodios', f"{c.get('steps', 0)} / {c.get('episodes', 0)}"),
        ('Caídas (excepciones)', c.get('crashes', 0)),
        ('Máscara conjunta vacía', c.get('mask_empty', 0)),
        ('Máscara de entrenamiento ≠ inferencia (DDQN)', c.get('mask_train_vs_infer', 0)),
        ('Pasos en que la máscara del aliado quitó acciones',
         f"{c.get('ally_mask_steps', 0)} de {c.get('ally_mask_checks', 0)} "
         f"({c.get('ally_mask_removed', 0)} acciones conjuntas)"),
        ('La máscara del aliado agregó acciones (no debería)', c.get('ally_mask_added', 0)),
        ('Fallbacks de orden', c.get('order_fallbacks', 0)),
        ('Paridad de la obs / alineación / historial / simetrías (diferencias)',
         f"{c.get('parity_steps_bad', 0)} / {c.get('align_bad', 0)} / {c.get('history_bad', 0)} / {c.get('sym_bad', 0)}"),
        ('Equipo del agente distinto del fijo',
         f"{c.get('agent_team_mismatch', 0)} de {c.get('agent_team_checks', 0)}"),
        ('Rival no validado', c.get('foe_not_validated', 0)),
    ]
    if kind == 'slot':
        rows += [
            ('Máscara del hueco vacía', f"{c.get('slot_mask_empty', 0)} de {c.get('slot_checks', 0)}"),
            ('Par elegido ilegal en la máscara conjunta', c.get('slot_vs_joint_bad', 0)),
            ('Máscara de entrenamiento ≠ inferencia (PPO)', c.get('slot_train_vs_infer', 0)),
        ]
    L += ['| Chequeo | Resultado |', '|---|---|'] + [f'| {a} | {b} |' for a, b in rows]
    if ms:
        L += ['', 'Costo (ms): ' + ', '.join(
            f"{k} mediana {v['p50']:.2f}" for k, v in ms.items() if isinstance(v, dict))]
    if r.get('examples'):
        L += ['', 'Ejemplos de discrepancias: `' + json.dumps(r['examples'], default=str)[:600] + '`']
    L.append('')
(d / 'kickoff_summary10.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
PY
log "resumen rc=$?"
log "KICKOFF_QUEUE10_DONE"
