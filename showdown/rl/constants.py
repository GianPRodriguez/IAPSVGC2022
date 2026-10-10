"""Frozen constants from docs/rl_env_spec.md v1.2."""
from __future__ import annotations

SPEC_VERSION = 'v1.2'

# ObsSpec segments (must sum to OBS_DIM)
SEG_SPECIES = 128
SEG_BELIEF = 256
SEG_FIELD = 128
SEG_STATS_VOL = 128
SEG_MOVE_DMG = 384
OBS_DIM = SEG_SPECIES + SEG_BELIEF + SEG_FIELD + SEG_STATS_VOL + SEG_MOVE_DMG  # 1024

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

# Action space (per slot)
# 0-15: move i (0-3) × target t (0-3)  [foe_a, foe_b, ally, self/field]
# 16-19: hard switch bench index 0-3
# 20: pass
# 21: mega flag with move0/target0 (compact); prefer mega via info in decode
# 22: tera flag with move0/target0
# 23: reserved / default
N_MOVE_SLOTS = 4
N_TARGETS = 4  # foe_a, foe_b, ally, self_or_field
N_SWITCH_SLOTS = 4
PASS_ID = 16 + N_SWITCH_SLOTS  # 20
MEGA_ID = 21
TERA_ID = 22
RESERVED_ID = 23
SLOT_ACTION_N = 24
JOINT_ACTION_N = SLOT_ACTION_N * SLOT_ACTION_N  # 576 (DDQNSpec)

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
