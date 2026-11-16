#!/bin/bash
# Segunda cola previa al kick-off (16-11-2026): preview PPO v2, confirmación de lr 1e-4 y piloto
# de PPO. Espera a que termine la primera cola (KICKOFF_QUEUE_DONE). Una etapa que falla no
# detiene las siguientes. Resumen: $OUT/kickoff_summary2.md.
#
#   bash scripts/server/kickoff_queue2.sh ~/iapsvgc_runs/kickoff2 ~/iapsvgc_runs/kickoff
set -u
OUT=${1:-$HOME/iapsvgc_runs/kickoff2}
Q1=${2:-$HOME/iapsvgc_runs/kickoff}
mkdir -p "$OUT"/lr_confirm
source ~/miniconda3/etc/profile.d/conda.sh; conda activate kuroneko
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
export PYTHONUNBUFFERED=1 PS_RL_STRICT=1
log() { echo "$(date '+%F %T') $*" >> "$OUT/queue.log"; }
clean_ps() { pkill -f "pokemon-showdown.*start" 2>/dev/null; sleep 5; }
log "esperando a la primera cola ($Q1)"
until grep -q KICKOFF_QUEUE_DONE "$Q1/queue.log" 2>/dev/null; do sleep 60; done
clean_ps
log "inicio commit $(git rev-parse --short HEAD)"
TEAM=showdown/rl/data/ddqn_team.txt

# ---------------------------------------------------------------- A/B. preview PPO v2
pv() {  # nombre flags...
  local name=$1; shift
  python -m showdown.rl.team_preview_ppo.train --seed 0 --device cuda --backend live --num-workers 28 \
    --envs-per-server 4 --start-ps --pilot greedy_joint --updates 200 --eval-every 20 \
    --eval-periodic-episodes 256 --eval-baseline pilot --canonical --own-team $TEAM \
    --checkpoint-dir "$OUT/$name" "$@" > "$OUT/$name.txt" 2>&1
  log "preview $name rc=$?"; clean_ps
}
pv preview_v2_ent02
pv preview_v2_ent005 --ent-coef 0.005
# el mejor de los dos (MEAN_WR final frente al preview heurístico) para el piloto de PPO
BEST_PV=$(python - "$OUT" <<'PY'
import re, sys
from pathlib import Path
d = Path(sys.argv[1]); best = None
for name in ('preview_v2_ent02', 'preview_v2_ent005'):
    rep = d / name / 'reporte.txt'
    if not rep.is_file():
        continue
    m = re.search(r'MEAN_WR:\s*([\d.]+)', rep.read_text())
    if m and (best is None or float(m.group(1)) > best[1]):
        ck = d / name / 'best.pt'
        best = (str(ck if ck.is_file() else d / name / 'final.pt'), float(m.group(1)))
print(best[0] if best else '')
PY
)
echo "${BEST_PV:-(ninguno)}" > "$OUT/preview_best.txt"
log "preview elegido: ${BEST_PV:-(ninguno)}"

# ---------------------------------------------------------------- C. confirmación de lr 1e-4
# Mismo preview y configuración que el tuning y la ablación de la primera cola, para emparejar
# con la base (lr 3e-4) en las semillas 1 y 2 (ablation/struct_sym_s1, _s2).
OLD_PV=""
[ -f "$(cat "$Q1/preview_used.txt" 2>/dev/null)" ] && OLD_PV="--preview-checkpoint $(cat "$Q1/preview_used.txt")"
COMMON="--device cuda --start-ps --expand-omega --num-envs 32 --envs-per-server 4 --log-every 10000 --log-every-sec 120"
TUNE="--phases random:150000,greedy_solo:250000 --eval-every 200000 --eval-games 50 --suite-games 0"
for s in 1 2; do
  python -m showdown.rl.ddqn.train --seed $s $COMMON $TUNE $OLD_PV --lr 1e-4 \
    --checkpoint-dir "$OUT/lr_confirm/lr1e-4_s$s" > "$OUT/lr_confirm/lr1e-4_s$s.txt" 2>&1
  log "lr 1e-4 semilla $s rc=$?"; clean_ps
done

# ---------------------------------------------------------------- D. piloto de PPO
PV_FLAG=""
[ -n "${BEST_PV:-}" ] && PV_FLAG="--preview-checkpoint $BEST_PV"
python -m showdown.rl.ppo.train --seed 0 --device cuda --start-ps --expand-omega --num-envs 32 \
  --envs-per-server 4 --phases random:150000,greedy_solo:250000 --suite-games 100 \
  --log-every 10000 --log-every-sec 120 $PV_FLAG --checkpoint-dir "$OUT/ppo_pilot" \
  > "$OUT/ppo_pilot.txt" 2>&1
log "piloto PPO rc=$?"; clean_ps

python -m showdown.rl.ddqn.kickoff_summary2 "$OUT" "$Q1" > /dev/null 2>&1
log "resumen rc=$?"
log "KICKOFF_QUEUE2_DONE"
