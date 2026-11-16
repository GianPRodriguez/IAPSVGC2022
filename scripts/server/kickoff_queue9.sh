#!/bin/bash
# Novena cola (16-11-2026): ¿cuánto del 92 % del piloto de PPO es memorización de los equipos
# rivales? Mismo diagnóstico que DDQN en la cuarta cola (equipos train frente a eval), con los dos
# pilotos de PPO (con simetría, cola 2; sin simetría, cola 5). Después, la suite completa de tests.
# Espera a KICKOFF_QUEUE8_DONE. Resumen: $OUT/kickoff_summary9.md.
#
# Los pilotos se entrenaron sin la máscara de daño al aliado (activa por defecto desde el
# 16-11-2026): el diagnóstico la apaga para medirlos en su condición de entrenamiento.
#
#   bash scripts/server/kickoff_queue9.sh ~/iapsvgc_runs/kickoff9 ~/iapsvgc_runs ~/iapsvgc_runs/kickoff8
set -u
OUT=${1:-$HOME/iapsvgc_runs/kickoff9}
R=${2:-$HOME/iapsvgc_runs}
Q8=${3:-$HOME/iapsvgc_runs/kickoff8}
mkdir -p "$OUT"/diag
source ~/miniconda3/etc/profile.d/conda.sh; conda activate kuroneko
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
export PYTHONUNBUFFERED=1 PS_RL_STRICT=1
log() { echo "$(date '+%F %T') $*" >> "$OUT/queue.log"; }
clean_ps() { pkill -f "pokemon-showdown.*start" 2>/dev/null; sleep 5; }
log "esperando a la octava cola ($Q8)"
until grep -q KICKOFF_QUEUE8_DONE "$Q8/queue.log" 2>/dev/null; do sleep 60; done
clean_ps
log "inicio commit $(git rev-parse --short HEAD)"
TEAM=showdown/rl/data/ddqn_team.txt
PV="$(cat "$R/kickoff2/preview_best.txt" 2>/dev/null)"; [ -f "$PV" ] || PV=""
log "preview: ${PV:-ninguno}"

# ---------------------------------------------------------------- 1. PPO: equipos train frente a eval
for run in "ppo_sym $R/kickoff2/ppo_pilot/final.pt" "ppo_nosym $R/kickoff5/ppo_pilot_nosym/final.pt"; do
  set -- $run
  name=$1 ck=$2
  [ -f "$ck" ] || { log "$name: falta $ck"; continue; }
  for split in train eval; do
    PS_TEAM_SPLIT=$split RL_MASK_ALLY_DAMAGE=0 python -m showdown.rl.ddqn.policy_diag --agent ppo \
      --checkpoint "$ck" --games 100 --port 8394 --agent-team $TEAM ${PV:+--preview-checkpoint "$PV"} \
      --out "$OUT/diag/${name}_$split.json" > "$OUT/diag/${name}_$split.txt" 2>&1
    log "1 $name equipos $split rc=$?"; clean_ps
  done
done

# ---------------------------------------------------------------- 2. suite completa
OMP_NUM_THREADS=4 python -m pytest -q -p no:cacheprovider tests > "$OUT/suite.txt" 2>&1
log "2 suite rc=$? ($(tail -1 "$OUT/suite.txt"))"

# ---------------------------------------------------------------- resumen
python - "$OUT" <<'PY'
import json, math, sys
from pathlib import Path
d = Path(sys.argv[1])
FOES = ('random', 'greedy_solo', 'greedy_joint', 'expectimax_bayesian')
L = ['# Resumen de la novena cola: memorización en PPO', '',
     'Pilotos de PPO (400k pasos: random 150k, greedy_solo 250k; preview v2; equipo fijo). 100 partidas '
     'por rival y grupo, sin exploración, rivales deterministas. Referencia DDQN (cuarta cola, mini '
     'kick-off 1M): 84,2 % con equipos `train` frente a 50,0 % con `eval`.', '']
for name, desc in (('ppo_sym', 'con simetría (cola 2)'), ('ppo_nosym', 'sin simetría (cola 5)')):
    L += [f'## PPO {desc}', '', '| Rival | Equipos `train` | Equipos `eval` | Diferencia |', '|---|---|---|---|']
    try:
        tr = json.loads((d / 'diag' / f'{name}_train.json').read_text())['resultados']
        ev = json.loads((d / 'diag' / f'{name}_eval.json').read_text())['resultados']
        tw = ew = tn = en = 0
        for foe in FOES:
            a, b = tr.get(foe, {}), ev.get(foe, {})
            na, nb = sum(a.values()) or 1, sum(b.values()) or 1
            wa, wb = a.get('win', 0), b.get('win', 0)
            if foe != 'random':
                tw, ew, tn, en = tw + wa, ew + wb, tn + na, en + nb
            L.append(f'| {foe} | {100 * wa / na:.0f} % | {100 * wb / nb:.0f} % | {100 * (wb / nb - wa / na):+.0f} |')
        pt, pe = tw / tn, ew / en
        se = 100 * math.sqrt(pt * (1 - pt) / tn + pe * (1 - pe) / en)
        L.append(f'| **3 heurísticos** | {100 * pt:.1f} % | {100 * pe:.1f} % | {100 * (pe - pt):+.1f} (± {se:.1f}) |')
    except Exception as exc:
        L.append(f'| sin datos ({exc!r}) | | | |')
    L.append('')
suite = (d / 'suite.txt').read_text(errors='replace').strip().splitlines() if (d / 'suite.txt').is_file() else ['sin datos']
fails = [l for l in suite if l.startswith('FAILED') or l.startswith('ERROR')]
L += ['## Suite completa de tests', '', '```', suite[-1]] + fails[:30] + ['```', '']
(d / 'kickoff_summary9.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
PY
log "resumen rc=$?"
log "KICKOFF_QUEUE9_DONE"
