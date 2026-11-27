# Común a las colas F2–F4 de la familia QD 2.0 (protocolo §4.12–§4.14). Se carga con `source`.
source ~/miniconda3/etc/profile.d/conda.sh; conda activate kuroneko
export PYTHONUNBUFFERED=1 PS_RL_STRICT=1
unset RL_MASK_ALLY_DAMAGE RL_SYM_AUG RL_HOLDOUT_VAL RL_TRAIN_SUBSET RL_FOE_SYNTH PREVIEW_FORCE_ORDER PS_JOINT_ARCHETYPE \
  PPO_FOE_CHECKPOINTS PPO_FOE_PREVIEW RL_OBS_ABILITIES RL_OBS_CALC PREVIEW_SAMPLE PREVIEW_TEMPERATURE PS_TEAM_SPLIT \
  PREVIEW_PILOT_CHECKPOINT PREVIEW_FOE_PILOT PS_PREVIEW_OWN_TEAM PREVIEW_CHECKPOINT PREVIEW_REQUIRED RL_BD2 PREVIEW_KIND \
  PREVIEW_BD PREVIEW_N_OUTER
RUNS=$HOME/iapsvgc_runs
QI=$RUNS/qd_i2; QJK=$RUNS/qd_jk2; F1=$RUNS/qd_f1
INIT=$RUNS/kickoff_ppo_v2/ppo_s1/final.pt
INIT4=$RUNS/kickoff_ppo_v2/ppo_s4/final.pt
CTRL=$QI/control/final.pt
PV=$RUNS/preview_v21b/preview_s0/best.pt
TEAM=showdown/rl/data/ddqn_team.txt
log() { echo "$(date '+%F %T') $*" >> "$OUT/queue.log"; }
clean_ps() { pkill -f "[p]okemon-showdown.*start" 2>/dev/null; sleep 5; }
disk() { log "  disco: $(df -h ~ | tail -1 | awk '{print $5}')"; }
need() { for f in "$@"; do [ -f "$f" ] || { log "ERROR: falta $f"; log "$MARK"; exit 1; }; done; }
wait_for() { log "esperando $2 en $1"; until grep -q "$2" "$1/queue.log" 2>/dev/null; do sleep 120; done; }
