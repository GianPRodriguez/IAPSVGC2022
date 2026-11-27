#!/bin/bash
# F3 DE LA FAMILIA QD 2.0 (protocolo §4.13, réplica): robustez de la conclusión.
#   3a. Desde PPO v2 s4 (segunda mejor val, regla fijada antes de ver nada): control, J 2.0 y K 2.0 (1M pasos).
#   3b. K 2.0 con N1 = N2 = 10 (las del paper) desde s1.
#   Despliegue como F1 (top-10, 1000 val, bloque 30002) y partidas eval comunes (bloque 70001, 2000 por
#   heurístico) de J4, K4, control4, init4, K10, K1 (desplegado de F1) y control1.
# Espera a QD_F2_DONE. Marca: QD_F3_DONE. Resúmenes: $OUT/replica_s4.txt, $OUT/k_n10.txt.
#   bash scripts/server/qd_f3.sh ~/iapsvgc_runs/qd_f3
set -u
OUT=${1:-$HOME/iapsvgc_runs/qd_f3}; MARK=QD_F3_DONE
mkdir -p "$OUT"
REPO=$(cd "$(dirname "$0")/../.." && pwd); cd "$REPO"
source scripts/server/qd_f_common.sh
export PREVIEW_KIND=team_preview_ppo
wait_for "$RUNS/qd_f2" QD_F2_DONE
need "$INIT" "$INIT4" "$CTRL" "$PV" "$F1/deploy/deploy_K.pt"
log "inicio commit $(git rev-parse --short HEAD)"; clean_ps
OMP_NUM_THREADS=4 python -m pytest -q -p no:cacheprovider tests/test_qd_f234.py tests/test_qd_confirm_f1.py \
  tests/test_qd_pga2.py tests/test_qd_ppga2.py tests/test_qd_ppo2.py > "$OUT/tests.txt" 2>&1
rc=$?; log "tests rc=$rc ($(tail -1 "$OUT/tests.txt"))"; [ $rc = 0 ] || { log "ERROR: tests fallan"; log "$MARK"; exit 1; }
COMMONTR="--seed 0 --device cuda --start-ps --num-envs 32 --envs-per-server 4 --preview-checkpoint $PV --require-preview --agent-team $TEAM"
run() {  # nombre módulo init [opciones]
  local name=$1 mod=$2 init=$3; shift 3
  if [ -f "$OUT/$name/summary.json" ]; then log "$name ya hecho (reanudación)"; return; fi
  python -m "$mod" $COMMONTR --init-checkpoint "$init" --checkpoint-dir "$OUT/$name" "$@" > "$OUT/$name.txt" 2>&1
  log "$name rc=$?"; clean_ps; disk
}
run control4 showdown.rl.qd_ppo.train2 "$INIT4" --control
run j4 showdown.rl.qd_pga.train_j2 "$INIT4"
run k4 showdown.rl.qd_ppga.train_k2 "$INIT4"
run k10 showdown.rl.qd_ppga.train_k2 "$INIT" --n1 10 --n2 10
need "$OUT/control4/final.pt" "$OUT/j4/archive.pt" "$OUT/k4/archive.pt" "$OUT/k10/archive.pt"
COMMON="--cfg-checkpoint $CTRL --num-envs 32 --envs-per-server 4 --agent-team $TEAM --preview-checkpoint $PV --require-preview"
if [ ! -f "$OUT/deploy/deploy.json" ]; then
  python -m showdown.rl.qd_ppo.confirm_f1 deploy $COMMON --out "$OUT/deploy" --top 10 --games 1000 --seed-block 30002 \
    --archive J4="$OUT/j4/archive.pt" --archive K4="$OUT/k4/archive.pt" --archive K10="$OUT/k10/archive.pt" \
    > "$OUT/deploy.txt" 2>&1
  log "deploy rc=$?"; clean_ps
fi
need "$OUT/deploy/deploy_J4.pt" "$OUT/deploy/deploy_K4.pt" "$OUT/deploy/deploy_K10.pt"
if [ ! -f "$OUT/confirm/results.json" ]; then
  python -m showdown.rl.qd_ppo.confirm_f1 confirm $COMMON --out "$OUT/confirm" --per-foe 2000 --seed-block 70001 \
    --agent J4="$OUT/deploy/deploy_J4.pt" --agent K4="$OUT/deploy/deploy_K4.pt" --agent control4="$OUT/control4/final.pt" \
    --agent init4="$INIT4" --agent K10="$OUT/deploy/deploy_K10.pt" --agent K1="$F1/deploy/deploy_K.pt" \
    --agent control1="$CTRL" > "$OUT/confirm.txt" 2>&1
  log "confirm rc=$?"; clean_ps
fi
python -m showdown.rl.qd_ppo.confirm_f1 stats --results "$OUT/confirm/results.json" --control control4 --ni J4,K4 \
  --sup J4:expectimax_bayesian --init init4 --out "$OUT/replica_s4" --title "F3a: réplica desde PPO v2 s4 (protocolo §4.13)" \
  > /dev/null 2>&1
log "stats réplica rc=$?"
python -m showdown.rl.qd_ppo.confirm_f1 stats --results "$OUT/confirm/results.json" --control control1 --ni K10,K1 \
  --sup "" --init none --out "$OUT/k_n10" --title "F3b: K 2.0 con N1 = N2 = 10 (protocolo §4.13)" > /dev/null 2>&1
log "stats K10 rc=$?"
log "$MARK"
