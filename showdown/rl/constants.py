"""Frozen constants from docs/rl_env_spec.md v1.8."""
from __future__ import annotations

SPEC_VERSION = 'v1.8'

# ObsSpec v2 (rl_env_spec v1.18): OTS completo + estado. Ver showdown/rl/observation.py.
# Antes (v1): 1024 = 128 especie + 256 belief + 128 campo + 128 boosts + 384 movimientos,
# con tipos/especie/item como hash md5 y el segmento belief en cero para DDQN/PPO.
from showdown.rl.team_preview_obs import MON_DIM as OTS_MON_DIM  # noqa: E402

OBS_VERSION = 2
N_OUR_MONS = 4  # activos a, b + 2 de banca (orden = índice de cambio)
N_FOE_MONS = 6  # activos a, b + 4 restantes de su hoja (vistos primero)
MON_DYN_DIM = 58  # estado (40) + incertidumbre SP (6) + historial y traído (12)
MOVE_SLOT_DIM = 81  # por movimiento disponible de nuestros activos (alineado con la acción)
FOE_MOVE_DIM = 43  # por movimiento de la hoja de los activos rivales
FIELD_DIM = 48
PAIR_DIM = 3 * N_OUR_MONS * N_FOE_MONS  # daño us→foe, foe→us, velocidad efectiva
MON_BLOCK_DIM = OTS_MON_DIM + MON_DYN_DIM
OBS_DIM = (
    (N_OUR_MONS + N_FOE_MONS) * MON_BLOCK_DIM
    + 2 * 4 * MOVE_SLOT_DIM
    + 2 * 4 * FOE_MOVE_DIM
    + PAIR_DIM
    + FIELD_DIM
)

# Canonical party slots
PARTY_ORDER = (
    'self_active_1',
    'self_active_2',
    'self_bench_1',
    'self_bench_2',
    'foe_active_1',
    'foe_active_2',
    'foe_bench_1',
    'foe_bench_2',
)
N_PARTY_SLOTS = 8

# Reward Candidate 2 weights
W_WIN = 1.0
W_LOSS = 1.0
W_FAINT = 0.15
W_HP = 0.05
C2_TIE_PENALTY = -0.25
R_CLIP_LO = -1.30
R_CLIP_HI = 1.40

# BD
BD_MIN_TURNS = 4

# Action space v2 (per slot, rl_env_spec v1.18)
# 0-15:  move i (0-3) × target t (0-3)  [foe_a, foe_b, ally, self/field]
# 16-31: MEGA + move i × target t (megaevolucionar SIEMPRE va con un movimiento;
#        nunca con un cambio). El agente aprende cuándo megaevolucionar y con qué.
#        (v1: un único id 21 = mega + movimiento 0 → no podía elegir el movimiento.)
# 32-35: hard switch bench index 0-3
# 36:    pass
# 37:    tera (ActionSpec reserved). Reg M-C: always masked off —
#        see showdown.format_rules.TERASTALLIZATION_ENABLED.
# 38-39: reserved
N_MOVE_SLOTS = 4
N_TARGETS = 4  # foe_a, foe_b, ally, self_or_field
N_SWITCH_SLOTS = 4
N_MOVE_ACTIONS = N_MOVE_SLOTS * N_TARGETS  # 16
MEGA_BASE = N_MOVE_ACTIONS  # 16
SWITCH_BASE = MEGA_BASE + N_MOVE_ACTIONS  # 32
PASS_ID = SWITCH_BASE + N_SWITCH_SLOTS  # 36
TERA_ID = PASS_ID + 1  # 37
RESERVED_ID = PASS_ID + 2  # 38
SLOT_ACTION_N = PASS_ID + 4  # 40
JOINT_ACTION_N = SLOT_ACTION_N * SLOT_ACTION_N  # 1600 (DDQNSpec)
MEGA_ID = MEGA_BASE  # compat: primer id mega (mega + movimiento 0 → rival A)

ILLEGAL_LOGIT = -1.0e9

# Pivot moves (count as offensive for BD1, not hard switches for BD2)
PIVOT_MOVES = frozenset({
    'uturn', 'voltswitch', 'flipturn', 'partingshot', 'teleport',
    'batonpass', 'shedtail',
})

STATUS_PROTECT_SETUP = frozenset({
    'protect', 'detect', 'spikyshield', 'banefulbunker', 'kingsshield',
    'silktrap', 'obstruct', 'wideguard', 'quickguard', 'craftyshield',
    'substitute', 'willowisp', 'thunderwave', 'toxic', 'stunspore',
    'glare', 'spore', 'sleeppowder', 'sing', 'hypnosis', 'yawn',
    'swagger', 'flatter', 'partingshot',  # partingshot also pivot — offensive wins in classify
    'tailwind', 'trickroom', 'reflect', 'lightscreen', 'auroraveil',
    'sunnyday', 'raindance', 'sandstorm', 'snowscape', 'hail',
    'electricterrain', 'grassyterrain', 'mistyterrain', 'psychicterrain',
    'swordsdance', 'nastyplot', 'calmmind', 'dragondance', 'quiverdance',
    'bulkup', 'irondefense', 'amnesia', 'agility', 'rockpolish',
    'helpinghand', 'coaching', 'decorate', 'aromaticmist',
})
