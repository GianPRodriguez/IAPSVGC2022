#!/bin/bash
# D3: GREEDY_JOINT EN MODO ARQUETIPO NIVEL 3 (protocolo §4.17, decisión del usuario del 27-11-2026). Descriptivo.
# ¿El nivel 3 (clima, campo y núcleos clásicos: lluvia, sol, doble clima, arena, nieve, campos con semillas,
# Trick Room con redirección, Tailwind) ejecuta sus planes? ¿Cuánto les cuesta a PPO v2 y DDQN v1 frente a joint
# normal (nivel 0) y al nivel 2? Sirve para decidir, con el informe de v3, si el nivel 3 entra en el entrenamiento
# (reconsideración de P4).
#
#   0. Espera a que termine la cola v3 (última línea de su queue.log = V3_DONE; un método a la vez).
#   1. Tests y prueba en vivo corta (PPO v2 s0 contra greedy_joint_arq3, 4 partidas con --context); si falla, para.
#   2. PPO v2 s0–s4 y DDQN v1 s0–s4 contra greedy_joint, greedy_joint_arq (nivel 2) y greedy_joint_arq3 (nivel 3):
#      policy_diag sobre eval, 200 partidas por rival, --context, rivales deterministas. De a $JOBS en paralelo, cada
#      evaluación en su puerto.
#   3. Informes por rival (scripts/analysis/archetype_diag.py: victorias por arquetipo, ejecución de Trick Room /
#      Tailwind / lluvia / sol / arena / nieve) y resumen de tasas. Marca: ARQ3_DONE.
#
#   JOBS=4 bash scripts/server/arq3_queue.sh ~/iapsvgc_runs/arq3
set -u
OUT=${1:-$HOME/iapsvgc_runs/arq3}
JOBS=${JOBS:-4}
V3LOG=${V3LOG:-$HOME/iapsvgc_runs/v3/queue.log}
PV=${PV:-$HOME/iapsvgc_runs/preview_v21b/preview_s0/best.pt}
DDQN1=${DDQN1:-$HOME/iapsvgc_runs/kickoff_ddqn_v2}
mkdir -p "$OUT"/diag
source ~/miniconda3/etc/profile.d/conda.sh; conda activate kuroneko
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
PPO2=${PPO2:-$REPO/checkpoints/ppo_v2}
export PYTHONUNBUFFERED=1 PS_RL_STRICT=1
unset RL_MASK_ALLY_DAMAGE RL_SYM_AUG RL_HOLDOUT_VAL RL_TRAIN_SUBSET RL_FOE_SYNTH PREVIEW_FORCE_ORDER PS_JOINT_ARCHETYPE \
  PPO_FOE_CHECKPOINTS PPO_FOE_PREVIEW RL_OBS_ABILITIES RL_OBS_CALC PREVIEW_SAMPLE PREVIEW_TEMPERATURE PS_TEAM_SPLIT \
  RL_OBS_SERIES_MEMORY RL_MASK_PROTECT_STREAK RL_SERIES_BO3 PREVIEW_SERIES_CONDITION PREVIEW_KIND
log() { echo "$(date '+%F %T') $*" >> "$OUT/queue.log"; }
clean_ps() { pkill -f "[p]okemon-showdown.*start" 2>/dev/null; sleep 5; }
kill_port() {
  pkill -f "[p]okemon-showdown start.* $1\$" 2>/dev/null
  pkill -f "[p]s_seeded_start.js.* $1\$" 2>/dev/null
}
TEAM=showdown/rl/data/ddqn_team.txt
FOES=greedy_joint,greedy_joint_arq,greedy_joint_arq3

log "esperando el fin de la cola v3 ($V3LOG)"
# Se ignoran las líneas del vigía de la caché (escribe su cierre después de V3_DONE).
until grep -v "vigía caché" "$V3LOG" 2>/dev/null | tail -n 1 | grep -q " V3_DONE\$"; do sleep 120; done
[ -f "$PV" ] || { log "ERROR: falta el preview $PV"; log "ARQ3_DONE"; exit 1; }
log "inicio commit $(git rev-parse --short HEAD); JOBS=$JOBS"
clean_ps

OMP_NUM_THREADS=4 python -m pytest -q -p no:cacheprovider tests/test_joint_archetype_v3.py tests/test_joint_archetype.py \
  tests/test_greedy_joint.py tests/test_context_diag.py tests/test_rl_foe_validated.py tests/test_ppo_v2.py \
  > "$OUT/tests.txt" 2>&1
rc=$?; log "tests rc=$rc ($(tail -1 "$OUT/tests.txt"))"
[ $rc = 0 ] || { log "ERROR: tests fallan; se detiene"; log "ARQ3_DONE"; exit 1; }

prev_ck() { if [ "$1" = ppo ]; then echo "$PPO2/ppo_s$2/final.pt"; else echo "$DDQN1/ddqn_s$2/final.pt"; fi; }

# Prueba en vivo corta: el nivel 3 juega partidas reales sin errores y deja eventos.
env PS_TEAM_SPLIT=eval python -m showdown.rl.ddqn.policy_diag --agent ppo --checkpoint "$(prev_ck ppo 0)" --port 8470 \
  --agent-team $TEAM --preview-checkpoint "$PV" --foes greedy_joint_arq3 --games 4 --context \
  --out "$OUT/smoke.json" > "$OUT/smoke.txt" 2>&1
rc=$?; kill_port 8470
log "prueba en vivo rc=$rc ($(grep -c '' "$OUT/smoke.txt") líneas)"
[ $rc = 0 ] && [ -s "$OUT/smoke.json" ] || { log "ERROR: la prueba en vivo falla; se detiene"; log "ARQ3_DONE"; exit 1; }

diag_port() {  # puerto nombre familia checkpoint
  local port=$1 name=$2 fam=$3 ck=$4
  [ -s "$OUT/diag/$name.json" ] && { log "diag $name ya hecho (reanudación)"; return; }
  [ -f "$ck" ] || { log "diag $name: falta $ck"; return; }
  local t0=$SECONDS
  env PS_TEAM_SPLIT=eval python -m showdown.rl.ddqn.policy_diag --agent "$fam" --checkpoint "$ck" --port "$port" \
    --agent-team $TEAM --preview-checkpoint "$PV" --foes $FOES --games 200 --context \
    --out "$OUT/diag/$name.json" > "$OUT/diag/$name.txt" 2>&1
  log "diag $name rc=$? puerto=$port $(( (SECONDS - t0) / 60 )) min"
  kill_port "$port"
}
declare -A PORT_PID
free_port() {
  local i p pid
  for ((i = 0; i < JOBS; i++)); do
    p=$((8440 + i)); pid=${PORT_PID[$p]:-}
    if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then echo "$p"; return; fi
  done
}
launch() {
  local p
  while :; do p=$(free_port); [ -n "$p" ] && break; wait -n 2>/dev/null || sleep 5; done
  diag_port "$p" "$@" &
  PORT_PID[$p]=$!
}
for fam in ppo ddqn; do
  for s in 0 1 2 3 4; do
    launch d3_${fam}_s$s $fam "$(prev_ck $fam $s)"
  done
done
wait
log "D3 partidas terminadas"

for foe in greedy_joint greedy_joint_arq greedy_joint_arq3; do
  python scripts/analysis/archetype_diag.py --rivales $foe --out "$OUT/d3_$foe.md" \
    --titulo "D3: PPO v2 y DDQN v1 contra $foe por arquetipo del rival (protocolo §4.17)" \
    "$OUT/diag/d3_ppo_s*.json" "$OUT/diag/d3_ddqn_s*.json" > "$OUT/d3_$foe.err" 2>&1
  log "informe $foe rc=$?"
done
python - "$OUT" > "$OUT/d3_resumen.md" 2>&1 <<'PY'
import sys
from pathlib import Path
from showdown.rl.kickoff_report import _load, rate
out = Path(sys.argv[1])
foes = ('greedy_joint', 'greedy_joint_arq', 'greedy_joint_arq3')
print('# D3: tasa de victorias contra greedy_joint por nivel de arquetipo (eval, 200 partidas por rival y semilla)\n')
print('| Familia | Semilla | nivel 0 | nivel 2 | nivel 3 |')
print('|---|---|---|---|---|')
for fam in ('ppo', 'ddqn'):
    tot = {f: [] for f in foes}
    for s in range(5):
        res = _load(out / 'diag' / f'd3_{fam}_s{s}.json')
        if not res:
            continue
        cells = []
        for f in foes:
            r = rate(res, (f,))
            cells.append(f'{100 * r[0]:.1f} %' if r else '--')
            if r:
                tot[f].append(r[0])
        print(f'| {fam} | s{s} | ' + ' | '.join(cells) + ' |')
    print(f'| **{fam}** | media | ' + ' | '.join(
        f'**{100 * sum(v) / len(v):.1f} %**' if v else '--' for v in tot.values()) + ' |')
PY
log "resumen en $OUT/d3_resumen.md"
log "ARQ3_DONE"
