#!/bin/bash
# Sexta cola (16-11-2026): análisis previos al kick-off de PPO — máscaras por hueco en vivo,
# desglose de jugadas y validación periódica / reanudación en vivo. Espera a KICKOFF_QUEUE5_DONE.
#
#   bash scripts/server/kickoff_queue6.sh ~/iapsvgc_runs/kickoff6 ~/iapsvgc_runs/kickoff2 ~/iapsvgc_runs/kickoff5
set -u
OUT=${1:-$HOME/iapsvgc_runs/kickoff6}
Q2=${2:-$HOME/iapsvgc_runs/kickoff2}
Q5=${3:-$HOME/iapsvgc_runs/kickoff5}
mkdir -p "$OUT"
source ~/miniconda3/etc/profile.d/conda.sh; conda activate kuroneko
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
export PYTHONUNBUFFERED=1 PS_RL_STRICT=1
log() { echo "$(date '+%F %T') $*" >> "$OUT/queue.log"; }
clean_ps() { pkill -f "pokemon-showdown.*start" 2>/dev/null; sleep 5; }
log "esperando a la quinta cola ($Q5)"
until grep -q KICKOFF_QUEUE5_DONE "$Q5/queue.log" 2>/dev/null; do sleep 60; done
clean_ps
log "inicio commit $(git rev-parse --short HEAD)"
TEAM=showdown/rl/data/ddqn_team.txt
PV="$(cat "$Q2/preview_best.txt" 2>/dev/null)"; [ -f "$PV" ] || PV=""

# 3. validación en vivo con máscaras por hueco (como PPO)
python -m showdown.rl.live_validation --mask-kind slot --episodes 25 --port 8350 --device cuda \
  --agent-team $TEAM --out "$OUT/live_validation_slot.json" > "$OUT/live_validation_slot.txt" 2>&1
log "3 validación en vivo (slot) rc=$?"; clean_ps

# 4. desglose de jugadas del piloto de PPO
CK="$Q2/ppo_pilot/final.pt"
python -m showdown.rl.ddqn.policy_diag --agent ppo --checkpoint "$CK" --games 50 --port 8370 \
  --agent-team $TEAM ${PV:+--preview-checkpoint "$PV"} --out "$OUT/policy_diag_ppo.json" \
  > "$OUT/policy_diag_ppo.txt" 2>&1
log "4 desglose de jugadas PPO rc=$?"; clean_ps

# 2. validación periódica + reanudación de PPO en vivo
COMMON="--seed 0 --device cuda --start-ps --expand-omega --num-envs 32 --envs-per-server 4 --log-every 2000 --log-every-sec 30 --suite-games 0 --eval-every 6000 --eval-games 10 --checkpoint-dir $OUT/ppo_resume/ck"
mkdir -p "$OUT/ppo_resume"
python -m showdown.rl.ppo.train $COMMON --phases random:6000,greedy_solo:6000 \
  > "$OUT/ppo_resume/train1.txt" 2>&1
python -m showdown.rl.ppo.train $COMMON --phases random:6000,greedy_solo:12000 \
  --resume "$OUT/ppo_resume/ck/final.pt" > "$OUT/ppo_resume/train2.txt" 2>&1
log "2 validación + reanudación PPO rc=$?"; clean_ps

# resumen
python - "$OUT" <<'PY'
import json, re, sys
from pathlib import Path
d = Path(sys.argv[1])
L = ['# Resumen de la sexta cola (análisis previos al kick-off de PPO)', '']
p = d / 'live_validation_slot.json'
L += ['## 3. Validación en vivo con máscaras por hueco', '']
if p.is_file():
    r = json.loads(p.read_text())
    c = r['counts']
    L += ['| Chequeo | Resultado |', '|---|---|',
          f"| Pasos / episodios | {c.get('steps', 0)} / {c.get('episodes', 0)} |",
          f"| Máscara del hueco vacía | {c.get('slot_mask_empty', 0)} de {c.get('slot_checks', 0)} |",
          f"| Par elegido ilegal en la máscara conjunta | {c.get('slot_vs_joint_bad', 0)} |",
          f"| Máscara de entrenamiento ≠ inferencia (PPO) | {c.get('slot_train_vs_infer', 0)} |",
          f"| Fallbacks de orden | {c.get('order_fallbacks', 0)} |",
          f"| Paridad de la obs / alineación / historial / simetrías (diferencias) | {c.get('parity_steps_bad', 0)} / {c.get('align_bad', 0)} / {c.get('history_bad', 0)} / {c.get('sym_bad', 0)} |",
          f"| Equipo del agente distinto del fijo | {c.get('agent_team_mismatch', 0)} de {c.get('agent_team_checks', 0)} |", '']
else:
    L += ['Sin datos.', '']
L += ['## 4. Desglose de jugadas del piloto de PPO (sin exploración)', '']
p = d / 'policy_diag_ppo.json'
if p.is_file():
    from showdown.rl.ddqn.policy_diag import format_table
    L += [format_table(json.loads(p.read_text())), '']
else:
    L += ['Sin datos.', '']
L += ['## 2. Validación periódica y reanudación de PPO en vivo', '', '```']
for f in ('train1.txt', 'train2.txt'):
    t = d / 'ppo_resume' / f
    if t.is_file():
        L += [l[:200] for l in t.read_text(errors='replace').splitlines()
              if 'validación' in l or 'reanudando' in l or 'ya completada' in l or l.startswith('[ppo] DONE')]
L += ['```', '']
(d / 'kickoff_summary6.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
PY
log "resumen rc=$?"
log "KICKOFF_QUEUE6_DONE"
