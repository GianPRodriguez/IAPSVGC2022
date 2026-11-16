#!/bin/bash
# Cuarta cola previa al kick-off (16-11-2026): generalización a equipos rivales no vistos
# (grupo `eval`) con el mejor checkpoint del mini kick-off, y suite completa de tests.
# Espera a KICKOFF_QUEUE3_DONE. Resumen: $OUT/kickoff_summary4.md.
#
#   bash scripts/server/kickoff_queue4.sh ~/iapsvgc_runs/kickoff4 ~/iapsvgc_runs/kickoff ~/iapsvgc_runs/kickoff3
set -u
OUT=${1:-$HOME/iapsvgc_runs/kickoff4}
Q1=${2:-$HOME/iapsvgc_runs/kickoff}
Q3=${3:-$HOME/iapsvgc_runs/kickoff3}
mkdir -p "$OUT"
source ~/miniconda3/etc/profile.d/conda.sh; conda activate kuroneko
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
export PYTHONUNBUFFERED=1
log() { echo "$(date '+%F %T') $*" >> "$OUT/queue.log"; }
clean_ps() { pkill -f "pokemon-showdown.*start" 2>/dev/null; sleep 5; }
log "esperando a la tercera cola ($Q3)"
until grep -q KICKOFF_QUEUE3_DONE "$Q3/queue.log" 2>/dev/null; do sleep 60; done
clean_ps
log "inicio commit $(git rev-parse --short HEAD)"

# ---------------------------------------------------------------- 1. generalización train / eval
CK=""
for f in "$Q3/mini_kickoff/best.pt" "$Q3/mini_kickoff/final.pt" "$Q1/tune/T1_lr1e-4/best.pt"; do
  [ -f "$f" ] && { CK="$f"; break; }
done
PV=$(sed -n 's/.*preview=\([^ ]*\).*/\1/p' "$Q3/mini_choice.txt" 2>/dev/null)
[ -f "${PV:-}" ] || PV=""
echo "$CK" > "$OUT/checkpoint.txt"
for split in train eval; do
  PS_RL_STRICT=1 PS_TEAM_SPLIT=$split python -m showdown.rl.ddqn.policy_diag --checkpoint "$CK" \
    --games 100 --port 8372 ${PV:+--preview-checkpoint "$PV"} --out "$OUT/gen_$split.json" \
    > "$OUT/gen_$split.txt" 2>&1
  log "1 generalización equipos $split rc=$?"; clean_ps
done

# ---------------------------------------------------------------- 2. suite completa
python -m pytest -q -p no:cacheprovider tests > "$OUT/suite.txt" 2>&1
log "2 suite rc=$? ($(tail -1 "$OUT/suite.txt"))"

# ---------------------------------------------------------------- resumen
python - "$OUT" <<'PY'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
L = ['# Resumen de la cuarta cola previa al kick-off', '',
     '## 1. Generalización: equipos rivales vistos (`train`) frente a no vistos (`eval`)', '',
     f"Checkpoint: `{(d / 'checkpoint.txt').read_text().strip()}`; 100 partidas por rival, sin exploración, "
     'rivales deterministas, equipo fijo del agente.', '',
     '| Rival | Victorias con equipos `train` | Victorias con equipos `eval` | Diferencia |', '|---|---|---|---|']
try:
    tr = json.loads((d / 'gen_train.json').read_text())['resultados']
    ev = json.loads((d / 'gen_eval.json').read_text())['resultados']
    tw = ew = tn = en = 0
    for foe in ('random', 'greedy_solo', 'greedy_joint', 'expectimax_bayesian'):
        a, b = tr.get(foe, {}), ev.get(foe, {})
        na, nb = sum(a.values()) or 1, sum(b.values()) or 1
        wa, wb = a.get('win', 0), b.get('win', 0)
        tw, ew, tn, en = tw + wa, ew + wb, tn + na, en + nb
        L.append(f'| {foe} | {100 * wa / na:.0f} % | {100 * wb / nb:.0f} % | {100 * (wb / nb - wa / na):+.0f} |')
    L.append(f'| **total** | {100 * tw / tn:.1f} % | {100 * ew / en:.1f} % | {100 * (ew / en - tw / tn):+.1f} |')
    L += ['', 'Con 400 partidas por grupo, diferencias de menos de ~7 puntos en el total son ruido. Una caída '
          'clara en `eval` indicaría memorización de los equipos rivales de entrenamiento.']
except Exception as exc:
    L.append(f'| sin datos ({exc!r}) | | | |')
suite = (d / 'suite.txt').read_text(errors='replace').strip().splitlines() if (d / 'suite.txt').is_file() else ['sin datos']
fails = [l for l in suite if l.startswith('FAILED')]
L += ['', '## 2. Suite completa de tests', '', '```', suite[-1]] + fails[:20] + ['```', '']
(d / 'kickoff_summary4.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
PY
log "resumen rc=$?"
log "KICKOFF_QUEUE4_DONE"
