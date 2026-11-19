#!/bin/bash
# KICK-OFF OFICIAL DE DDQN con P5 completo (calculadora real) + preview v2.1 sin val (19-11-2026).
# Decisiones del usuario: kick-off de DDQN con la observación nueva y el preview v2.1 (no el v2 congelado);
# preparado con control intermedio: las semillas 0 y 1 hacen de piloto.
#
# 1. Espera a PREVIEW_V21B_DONE y lee la decisión de preview_v21b.md: si el preview v2.1 sin val
#    "No se adopta", se detiene (decisión del usuario pendiente). Si se adopta, usa la semilla elegida.
# 2. Reinicia el servidor de la calculadora (código nuevo).
# 3. Semillas 0 y 1: 1M pasos, configuración final (REG 0,03 + dropout 0,1, cabeza conjunta, sin
#    simetría, máscara del aliado, validación reservada, equipo fijo, --obs-abilities 1), y por semilla
#    final.pt contra val (control) y el set del kick-off.
# 4. CONTROL INTERMEDIO: Δval (semillas 0 y 1 agrupadas) frente a DDQN REG 0,03 (decimoquinta cola) ≥ −2
#    y calculadora sin fallos (respaldo < 1 %, pedidos sin respuesta < 1 %). Si no, se detiene.
# 5. Semillas 2–4 con el mismo set de evaluación.
# Set del kick-off por semilla (200 partidas por rival): final.pt contra eval (principal), best.pt contra
# eval, final.pt contra sus propios equipos, final.pt contra greedy_joint en modo arquetipo.
# Resumen: $OUT/kickoff_summary16.md (showdown/rl/kickoff_report.py; PPO queda sin datos).
#
#   bash scripts/server/kickoff_ddqn_v2.sh ~/iapsvgc_runs/kickoff_ddqn_v2 ~/iapsvgc_runs/preview_v21b ~/iapsvgc_runs/kickoff15
set -u
OUT=${1:-$HOME/iapsvgc_runs/kickoff_ddqn_v2}
PVB=${2:-$HOME/iapsvgc_runs/preview_v21b}
Q15=${3:-$HOME/iapsvgc_runs/kickoff15}
mkdir -p "$OUT"/diag
source ~/miniconda3/etc/profile.d/conda.sh; conda activate kuroneko
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
export PYTHONUNBUFFERED=1 PS_RL_STRICT=1
unset RL_MASK_ALLY_DAMAGE RL_SYM_AUG RL_HOLDOUT_VAL RL_TRAIN_SUBSET RL_FOE_SYNTH PREVIEW_FORCE_ORDER PS_JOINT_ARCHETYPE \
  PPO_FOE_CHECKPOINTS PPO_FOE_PREVIEW RL_OBS_ABILITIES RL_OBS_CALC PREVIEW_SAMPLE PREVIEW_TEMPERATURE
log() { echo "$(date '+%F %T') $*" >> "$OUT/queue.log"; }
clean_ps() { pkill -f "pokemon-showdown.*start" 2>/dev/null; sleep 5; }
log "esperando al preview v2.1 sin val ($PVB)"
until grep -q PREVIEW_V21B_DONE "$PVB/queue.log" 2>/dev/null; do sleep 60; done

# ---------------------------------------------------------------- 1. preview elegido
DEC=$(grep -o "Elegido por \`val\`: semilla [0-9]*\. \*\*[^*]*\*\*" "$PVB/preview_v21b.md" 2>/dev/null)
log "decisión del preview: ${DEC:-sin resumen}"
case "$DEC" in
  *"Se adopta"*) ;;
  *) log "ERROR: el preview v2.1 sin val no se adopta (o falta el resumen); se detiene para decisión del usuario"
     log "KICKOFF_DDQN_V2_DONE"; exit 1;;
esac
PS_=$(echo "$DEC" | grep -o "semilla [0-9]*" | grep -o "[0-9]*")
PV="$PVB/preview_s$PS_/best.pt"; [ -f "$PV" ] || PV="$PVB/preview_s$PS_/final.pt"
[ -f "$PV" ] || { log "ERROR: falta $PV"; log "KICKOFF_DDQN_V2_DONE"; exit 1; }
log "preview v2.1: $PV"

# ---------------------------------------------------------------- 2. calculadora con el código nuevo
clean_ps
pkill -f "calc_serve[r].js" 2>/dev/null; sleep 2
log "inicio commit $(git rev-parse --short HEAD)"
TEAM=showdown/rl/data/ddqn_team.txt

diag() {  # nombre checkpoint grupo [VARIABLE=valor...] [--opciones...]
  local name=$1 ck=$2 split=$3; shift 3
  [ -f "$ck" ] || { log "diag $name: falta $ck"; return; }
  local envs=() opts=()
  for x in "$@"; do case "$x" in --*) opts+=("$x");; *) envs+=("$x");; esac; done
  env PS_TEAM_SPLIT=$split ${envs[@]+"${envs[@]}"} python -m showdown.rl.ddqn.policy_diag --agent ddqn \
    --checkpoint "$ck" --games 200 --port 8435 --agent-team $TEAM --preview-checkpoint "$PV" \
    ${opts[@]+"${opts[@]}"} --out "$OUT/diag/$name.json" > "$OUT/diag/$name.txt" 2>&1
  log "diag $name rc=$?"; clean_ps
}
seed() {  # semilla
  local s=$1
  python -m showdown.rl.ddqn.train --seed $s --device cuda --start-ps --expand-omega --num-envs 32 --envs-per-server 4 \
    --log-every 10000 --log-every-sec 120 \
    --phases random:150000,greedy_solo:250000,greedy_joint:300000,expectimax_bayesian:300000 \
    --eval-every 250000 --eval-games 50 --suite-games 0 --agent-team $TEAM --preview-checkpoint "$PV" \
    --obs-abilities 1 --lr 1e-4 --batch-size 256 --checkpoint-dir "$OUT/ddqn_s$s" > "$OUT/ddqn_s$s.txt" 2>&1
  log "ddqn_s$s rc=$?"; clean_ps
  diag ddqn_s${s}_final_eval "$OUT/ddqn_s$s/final.pt" eval
  diag ddqn_s${s}_best_eval "$OUT/ddqn_s$s/best.pt" eval
  diag ddqn_s${s}_final_own "$OUT/ddqn_s$s/final.pt" train RL_HOLDOUT_VAL=1
  diag ddqn_s${s}_final_arquetipo "$OUT/ddqn_s$s/final.pt" eval PS_JOINT_ARCHETYPE=1 --foes=greedy_joint --context
}

# ---------------------------------------------------------------- 3. semillas 0 y 1 (piloto)
for s in 0 1; do
  seed $s
  diag ddqn_s${s}_final_val "$OUT/ddqn_s$s/final.pt" val
done

# ---------------------------------------------------------------- 4. control intermedio
python - "$OUT" "$Q15" > "$OUT/control.txt" 2>&1 <<'PY'
import re, sys
from pathlib import Path
from showdown.rl.kickoff_summary24 import _load, diff, pooled
out, q15 = Path(sys.argv[1]), Path(sys.argv[2])
cand = pooled([_load(out / 'diag' / f'ddqn_s{s}_final_val.json') for s in (0, 1)])
ref = pooled([_load(q15 / 'diag' / f'R_wd03_s{s}_final_val.json') for s in (0, 1)])
d = diff(cand, ref)
print('DDQN val', cand, 'referencia', ref, 'Δ', d)
ok = d is not None and d[0] >= -2.0
for s in (0, 1):
    lines = [l for l in (out / f'ddqn_s{s}.txt').read_text(errors='replace').splitlines()
             if 'calculadora real en la observación' in l]
    fb = re.search(r"'fallback_rate': ([0-9.e-]+)", lines[-1]) if lines else None
    ms = re.search(r"'missing_rate': ([0-9.e-]+)", lines[-1]) if lines else None
    print(f'ddqn_s{s}', 'respaldo', fb and fb.group(1), 'sin respuesta', ms and ms.group(1))
    if not (fb and ms) or float(fb.group(1)) >= 0.01 or float(ms.group(1)) >= 0.01:
        ok = False
print('CONTROL', 'OK' if ok else 'FAIL')
sys.exit(0 if ok else 1)
PY
if [ $? != 0 ]; then
  log "CONTROL_FAIL ($(tr '\n' ' ' < "$OUT/control.txt")); se detiene para revisión"
  python -m showdown.rl.kickoff_report "$OUT" > "$OUT/kickoff_report.err" 2>&1
  log "KICKOFF_DDQN_V2_DONE"; exit 1
fi
log "control intermedio cumplido: $(tr '\n' ' ' < "$OUT/control.txt")"

# ---------------------------------------------------------------- 5. semillas 2–4
for s in 2 3 4; do seed $s; done
log "KICKOFF_DDQN_DONE"
python -m showdown.rl.kickoff_report "$OUT" > "$OUT/kickoff_report.err" 2>&1
log "resumen rc=$?"
log "KICKOFF_DDQN_V2_DONE"
