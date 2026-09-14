"""Catálogo de efectos de moves: schema NN + scoring 1-ply (expectimax/joint).

Fuente: dex del proyecto (`move_entry` / moves.json), no hardcodear cientos de moves.

- `MoveEffectFeatures` / `to_feature_dict` / `to_dense_vector`: observación estable
  para futuros agentes NN.
- `expected_move_effect_score`: reutiliza `status_effects` (status/flinch/drops
  secundarios) y suma setup, heal/drain/recoil, pivot, volatiles y control de campo.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from showdown.battle_bots.move_mechanics.accuracy import move_accuracy
from showdown.battle_bots.move_mechanics.charge import norm_id
from showdown.battle_bots.move_mechanics.status_effects import (
    BOOST_DROP_UNIT,
    EffectScore,
    expected_effect_score,
    move_entry,
)

# --- Pesos 1-ply (mismo orden de magnitud que STATUS_VALUE / BOOST_DROP_UNIT) ---
BOOST_SELF_UNIT = 7.0
BOOST_SELF_DROP_UNIT = 4.0  # coste self-drop (Draco, Superpower…)
HEAL_UNIT = 40.0
DRAIN_UNIT = 22.0
RECOIL_UNIT = 18.0
HP_COST_UNIT = 28.0  # Belly Drum / Clangorous Soul / Steel Beam fracciones
PIVOT_BONUS = 14.0
FIELD_CONTROL_BONUS = 16.0  # weather / terrain
HAZARD_BONUS = 20.0  # stickyweb / rocks / spikes
VOLATILE_VALUE: Dict[str, float] = {
    'taunt': 38.0,
    'disable': 28.0,
    'substitute': 32.0,
    'encore': 30.0,
    'torment': 18.0,
    'healblock': 22.0,
    'embargo': 12.0,
}

_STAT_KEYS: Tuple[str, ...] = (
    'atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion',
)

_PROTECT_IDS = frozenset({
    'protect', 'detect', 'spikyshield', 'banefulbunker', 'silktrap',
    'burningbulwark', 'obstruct', 'kingsshield', 'maxguard',
})

# HP cost fraction when dex no lo tipa (callbacks PS).
_MANUAL_HP_COST: Dict[str, float] = {
    'bellydrum': 0.50,
    'clangoroussoul': 1.0 / 3.0,
    'steelbeam': 0.50,
}


def _frac_pair(raw: Any) -> float:
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        try:
            a, b = float(raw[0]), float(raw[1])
            return a / b if b else 0.0
        except (TypeError, ValueError):
            return 0.0
    if isinstance(raw, (int, float)):
        return float(raw)
    return 0.0


def _stages_dict(raw: Any) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if not isinstance(raw, dict):
        return out
    for k, v in raw.items():
        sk = str(k).lower().replace('specialattack', 'spa').replace('specialdefense', 'spd')
        sk = {
            'attack': 'atk', 'defense': 'def', 'spatk': 'spa', 'spdef': 'spd',
            'speed': 'spe',
        }.get(sk, sk)
        try:
            out[sk] = int(v)
        except (TypeError, ValueError):
            continue
    return out


def _multihit_bounds(raw: Any) -> Tuple[int, int]:
    if raw is None:
        return (0, 0)
    if isinstance(raw, int):
        return (raw, raw)
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        try:
            return (int(raw[0]), int(raw[1]))
        except (TypeError, ValueError):
            return (0, 0)
    return (0, 0)


@dataclass(frozen=True)
class MoveEffectFeatures:
    """Schema estable de efectos de un move (NN + scoring)."""

    move_id: str = ''
    category: str = ''
    base_power: int = 0
    accuracy: float = 1.0
    priority: int = 0
    target: str = ''
    type: str = ''

    multihit_min: int = 0
    multihit_max: int = 0
    crit_ratio: int = 1
    recoil_frac: float = 0.0
    drain_frac: float = 0.0
    heal_frac: float = 0.0
    hp_cost_frac: float = 0.0

    primary_status: str = ''
    secondary_status_chance: float = 0.0
    flinch_chance: float = 0.0

    foe_boosts: Dict[str, int] = field(default_factory=dict)
    self_boosts: Dict[str, int] = field(default_factory=dict)
    secondary_self_boost_chance: float = 0.0
    secondary_foe_boost_chance: float = 0.0

    self_switch: bool = False
    force_switch: bool = False
    breaks_protect: bool = False
    bypass_sub: bool = False
    ignore_defensive: bool = False

    weather: str = ''
    terrain: str = ''
    side_condition: str = ''
    slot_condition: str = ''
    volatile_status: str = ''

    is_protect: bool = False
    protect_variant: str = ''

    has_callback_only: bool = False

    def to_feature_dict(self) -> Dict[str, float]:
        """Dict denso (float) ordenado por `feature_keys()`."""
        keys = feature_keys()
        vec = self.to_dense_vector()
        return {k: float(v) for k, v in zip(keys, vec)}

    def to_dense_vector(self) -> List[float]:
        f = self
        cat = (f.category or '').lower()
        out: List[float] = [
            float(f.base_power) / 250.0,
            float(f.accuracy),
            float(f.priority) / 5.0,
            1.0 if cat == 'physical' else 0.0,
            1.0 if cat == 'special' else 0.0,
            1.0 if cat == 'status' else 0.0,
            float(f.multihit_min) / 10.0,
            float(f.multihit_max) / 10.0,
            float(f.crit_ratio) / 3.0,
            float(f.recoil_frac),
            float(f.drain_frac),
            float(f.heal_frac),
            float(f.hp_cost_frac),
            float(f.secondary_status_chance),
            float(f.flinch_chance),
            float(f.secondary_self_boost_chance),
            float(f.secondary_foe_boost_chance),
            1.0 if f.self_switch else 0.0,
            1.0 if f.force_switch else 0.0,
            1.0 if f.breaks_protect else 0.0,
            1.0 if f.bypass_sub else 0.0,
            1.0 if f.ignore_defensive else 0.0,
            1.0 if f.is_protect else 0.0,
            1.0 if f.weather else 0.0,
            1.0 if f.terrain else 0.0,
            1.0 if f.side_condition else 0.0,
            1.0 if f.volatile_status else 0.0,
            1.0 if f.has_callback_only else 0.0,
            1.0 if f.primary_status else 0.0,
        ]
        for sk in _STAT_KEYS:
            out.append(float(f.self_boosts.get(sk, 0)) / 6.0)
        for sk in _STAT_KEYS:
            out.append(float(f.foe_boosts.get(sk, 0)) / 6.0)
        for st in ('brn', 'par', 'slp', 'psn', 'tox', 'frz'):
            out.append(1.0 if f.primary_status == st else 0.0)
        for vol in ('taunt', 'disable', 'substitute', 'encore'):
            out.append(1.0 if norm_id(f.volatile_status) == vol else 0.0)
        return out


def feature_keys() -> List[str]:
    keys = [
        'base_power_n', 'accuracy', 'priority_n',
        'cat_physical', 'cat_special', 'cat_status',
        'multihit_min_n', 'multihit_max_n', 'crit_ratio_n',
        'recoil_frac', 'drain_frac', 'heal_frac', 'hp_cost_frac',
        'secondary_status_chance', 'flinch_chance',
        'secondary_self_boost_chance', 'secondary_foe_boost_chance',
        'self_switch', 'force_switch', 'breaks_protect', 'bypass_sub',
        'ignore_defensive', 'is_protect',
        'has_weather', 'has_terrain', 'has_side_condition', 'has_volatile',
        'has_callback_only', 'has_primary_status',
    ]
    for sk in _STAT_KEYS:
        keys.append(f'self_boost_{sk}')
    for sk in _STAT_KEYS:
        keys.append(f'foe_boost_{sk}')
    for st in ('brn', 'par', 'slp', 'psn', 'tox', 'frz'):
        keys.append(f'primary_{st}')
    for vol in ('taunt', 'disable', 'substitute', 'encore'):
        keys.append(f'volatile_{vol}')
    return keys


def _merge_boosts(dst: Dict[str, int], src: Dict[str, int]) -> Dict[str, int]:
    out = dict(dst)
    for k, v in src.items():
        out[k] = int(out.get(k, 0)) + int(v)
    return out


def _classify_root_boosts(
    root_boosts: Dict[str, int],
    *,
    flags: Dict[str, Any],
    category: str,
    base_power: int,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Devuelve (self_boosts, foe_boosts) para entry['boosts'] de raíz."""
    if not root_boosts:
        return {}, {}
    cat = (category or '').lower()
    has_neg = any(v < 0 for v in root_boosts.values())
    has_pos = any(v > 0 for v in root_boosts.values())
    # Status foe drops: Charm / Fake Tears / Tickle (protect+mirror+reflectable).
    if cat == 'status' and base_power <= 0 and has_neg and not has_pos:
        if flags.get('reflectable') or flags.get('mirror'):
            return {}, dict(root_boosts)
    # Setup / Shell Smash (snatch o self-only).
    if cat == 'status' and base_power <= 0:
        return dict(root_boosts), {}
    # Damaging moves raramente tienen boosts de raíz; atribuir a self.
    return dict(root_boosts), {}


@lru_cache(maxsize=1024)
def parse_move_effects(move_name: Optional[str]) -> MoveEffectFeatures:
    """Parsea el dex → features. Cacheado por nombre normalizado."""
    mid = norm_id(move_name)
    if not mid:
        return MoveEffectFeatures()
    entry = move_entry(move_name)
    if not entry:
        return MoveEffectFeatures(move_id=mid, has_callback_only=True)

    acc_raw = entry.get('accuracy', True)
    if acc_raw is True or acc_raw is None:
        acc = 1.0
    else:
        try:
            av = float(acc_raw)
            acc = av / 100.0 if av > 1.0 else av
        except (TypeError, ValueError):
            acc = 1.0

    mh_min, mh_max = _multihit_bounds(entry.get('multihit'))
    flags = entry.get('flags') or {}
    if not isinstance(flags, dict):
        flags = {}

    bp = int(entry.get('basePower') or 0)
    cat = str(entry.get('category') or '')
    foe_boosts: Dict[str, int] = {}
    self_boosts: Dict[str, int] = {}

    root_boosts = _stages_dict(entry.get('boosts'))
    sb, fb = _classify_root_boosts(
        root_boosts, flags=flags, category=cat, base_power=bp,
    )
    self_boosts = _merge_boosts(self_boosts, sb)
    foe_boosts = _merge_boosts(foe_boosts, fb)

    self_block = entry.get('self') or {}
    if not isinstance(self_block, dict):
        self_block = {}
    self_boosts = _merge_boosts(self_boosts, _stages_dict(self_block.get('boosts')))
    sb2 = entry.get('selfBoost') or {}
    if isinstance(sb2, dict):
        self_boosts = _merge_boosts(self_boosts, _stages_dict(sb2.get('boosts')))

    sec_status_chance = 0.0
    flinch_chance = 0.0
    sec_self_boost_chance = 0.0
    sec_foe_boost_chance = 0.0

    def _absorb_secondary(sec: Dict[str, Any]) -> None:
        nonlocal sec_status_chance, flinch_chance
        nonlocal sec_self_boost_chance, sec_foe_boost_chance
        nonlocal foe_boosts, self_boosts
        if not isinstance(sec, dict) or not sec:
            return
        chance = float(sec.get('chance') or 100) / 100.0
        if sec.get('status'):
            sec_status_chance = max(sec_status_chance, chance)
        vol = sec.get('volatileStatus') or sec.get('volatile')
        if vol and norm_id(vol) == 'flinch':
            flinch_chance = max(flinch_chance, chance)
        boosts = _stages_dict(sec.get('boosts'))
        if boosts:
            if any(v < 0 for v in boosts.values()):
                foe_boosts = _merge_boosts(foe_boosts, boosts)
                sec_foe_boost_chance = max(sec_foe_boost_chance, chance)
            elif any(v > 0 for v in boosts.values()):
                # Secondary boosts on target (rare); count as foe positive
                foe_boosts = _merge_boosts(foe_boosts, boosts)
                sec_foe_boost_chance = max(sec_foe_boost_chance, chance)
        self_sec = sec.get('self') if isinstance(sec.get('self'), dict) else {}
        if self_sec:
            sbx = _stages_dict(self_sec.get('boosts'))
            if sbx:
                self_boosts = _merge_boosts(self_boosts, sbx)
                sec_self_boost_chance = max(sec_self_boost_chance, chance)
        if isinstance(sec.get('selfBoost'), dict):
            sbx = _stages_dict((sec.get('selfBoost') or {}).get('boosts'))
            if sbx:
                self_boosts = _merge_boosts(self_boosts, sbx)
                sec_self_boost_chance = max(sec_self_boost_chance, chance)

    sec = entry.get('secondary')
    if isinstance(sec, dict):
        _absorb_secondary(sec)
    secs = entry.get('secondaries')
    if isinstance(secs, list):
        for s in secs:
            if isinstance(s, dict):
                _absorb_secondary(s)

    heal = _frac_pair(entry.get('heal'))
    drain = _frac_pair(entry.get('drain'))
    recoil = _frac_pair(entry.get('recoil'))
    if entry.get('struggleRecoil'):
        recoil = max(recoil, 0.25)
    crash = entry.get('crash')
    if crash:
        recoil = max(recoil, _frac_pair(crash) if not isinstance(crash, bool) else 0.5)

    hp_cost = float(_MANUAL_HP_COST.get(mid, 0.0))
    if self_block.get('damage'):
        dmg = self_block.get('damage')
        if isinstance(dmg, (list, tuple)):
            hp_cost = max(hp_cost, _frac_pair(dmg))
        elif dmg in ('50%', 'half'):
            hp_cost = max(hp_cost, 0.5)

    volatile = str(entry.get('volatileStatus') or '')
    weather = str(entry.get('weather') or '')
    terrain_s = str(entry.get('terrain') or '')
    side = str(entry.get('sideCondition') or '')
    slot = str(entry.get('slotCondition') or '')

    is_protect = mid in _PROTECT_IDS
    if norm_id(volatile) in _PROTECT_IDS or norm_id(volatile) in (
        'protect', 'spikyshield', 'kingsshield', 'banefulbunker',
        'silktrap', 'burningbulwark', 'obstruct', 'maxguard',
    ):
        is_protect = True

    has_cb = bool(
        entry.get('basePowerCallback')
        or entry.get('onBasePower')
        or entry.get('damageCallback')
    )

    return MoveEffectFeatures(
        move_id=mid,
        category=cat,
        base_power=bp,
        accuracy=acc,
        priority=int(entry.get('priority') or 0),
        target=str(entry.get('target') or ''),
        type=str(entry.get('type') or ''),
        multihit_min=mh_min,
        multihit_max=mh_max,
        crit_ratio=int(entry.get('critRatio') or 1),
        recoil_frac=float(recoil),
        drain_frac=float(drain),
        heal_frac=float(heal),
        hp_cost_frac=float(hp_cost),
        primary_status=norm_id(str(entry.get('status') or '')),
        secondary_status_chance=float(sec_status_chance),
        flinch_chance=float(flinch_chance),
        foe_boosts=foe_boosts,
        self_boosts=self_boosts,
        secondary_self_boost_chance=float(sec_self_boost_chance),
        secondary_foe_boost_chance=float(sec_foe_boost_chance),
        self_switch=bool(entry.get('selfSwitch')),
        force_switch=bool(entry.get('forceSwitch')),
        breaks_protect=bool(entry.get('breaksProtect')),
        bypass_sub=bool(flags.get('bypasssub')),
        ignore_defensive=bool(
            entry.get('ignoreDefensive') or entry.get('ignoreEvasion')
        ),
        weather=norm_id(weather),
        terrain=norm_id(terrain_s),
        side_condition=norm_id(side),
        slot_condition=norm_id(slot),
        volatile_status=norm_id(volatile),
        is_protect=is_protect,
        protect_variant=mid if is_protect else '',
        has_callback_only=bool(has_cb and mid == 'acrobatics'),
    )


def _boost_score(stages: Dict[str, int], *, unit_pos: float, unit_neg: float) -> float:
    s = 0.0
    for _k, v in stages.items():
        if v > 0:
            s += unit_pos * float(v)
        elif v < 0:
            s += unit_neg * float(abs(v))
    return s


def expected_move_effect_score(
    move_name: Optional[str],
    *,
    attacker=None,
    defender=None,
    terrain: Optional[str] = None,
    we_outspeed: Optional[bool] = None,
    skip_if_fakeout_flinch: bool = True,
) -> EffectScore:
    """E[efectos] para joint/expectimax: status existentes + setup/heal/pivot/campo."""
    mid = norm_id(move_name)
    acc = move_accuracy(move_name)
    if not mid:
        return EffectScore('', 0.0, acc, False)

    base = expected_effect_score(
        move_name,
        attacker=attacker,
        defender=defender,
        terrain=terrain,
        we_outspeed=we_outspeed,
        skip_if_fakeout_flinch=skip_if_fakeout_flinch,
    )

    feats = parse_move_effects(move_name)
    extra = 0.0

    # Primary foe drops (Charm) — status_effects no lee entry.boosts.
    # Evitar doble-contar si ya entraron vía secondary en base.score:
    # status_effects solo cuenta boosts dentro de secondary specs.
    if feats.foe_boosts:
        if feats.secondary_foe_boost_chance <= 0:
            neg = {k: v for k, v in feats.foe_boosts.items() if v < 0}
            extra += _boost_score(neg, unit_pos=0.0, unit_neg=BOOST_DROP_UNIT)

    # Setup / self-boosts y self-drops
    if feats.self_boosts:
        pos = {k: v for k, v in feats.self_boosts.items() if v > 0}
        neg = {k: v for k, v in feats.self_boosts.items() if v < 0}
        if feats.secondary_self_boost_chance > 0:
            extra += (
                feats.secondary_self_boost_chance
                * _boost_score(pos, unit_pos=BOOST_SELF_UNIT, unit_neg=0.0)
            )
            extra -= (
                feats.secondary_self_boost_chance
                * _boost_score(neg, unit_pos=0.0, unit_neg=BOOST_SELF_DROP_UNIT)
            )
        else:
            extra += _boost_score(pos, unit_pos=BOOST_SELF_UNIT, unit_neg=0.0)
            extra -= _boost_score(neg, unit_pos=0.0, unit_neg=BOOST_SELF_DROP_UNIT)

    if feats.heal_frac > 0:
        extra += HEAL_UNIT * feats.heal_frac
    if feats.drain_frac > 0:
        extra += DRAIN_UNIT * feats.drain_frac
    if feats.recoil_frac > 0:
        extra -= RECOIL_UNIT * feats.recoil_frac
    if feats.hp_cost_frac > 0:
        extra -= HP_COST_UNIT * feats.hp_cost_frac

    if feats.self_switch:
        extra += PIVOT_BONUS
    if feats.force_switch:
        extra += PIVOT_BONUS * 0.75

    vol = feats.volatile_status
    if vol and vol != 'flinch' and vol not in _PROTECT_IDS and vol != 'protect':
        extra += VOLATILE_VALUE.get(vol, 10.0)

    if feats.weather:
        extra += FIELD_CONTROL_BONUS
    if feats.terrain:
        extra += FIELD_CONTROL_BONUS
    side = feats.side_condition
    if side:
        if side in ('stealthrock', 'spikes', 'toxicspikes', 'stickyweb', 'steelrocks'):
            extra += HAZARD_BONUS
        else:
            extra += FIELD_CONTROL_BONUS * 0.75

    total = float(base.score) + float(extra) * float(acc)
    return EffectScore(
        mid,
        total,
        acc,
        is_primary_status=base.is_primary_status,
    )


def is_protect_move(move_name: Optional[str]) -> bool:
    return parse_move_effects(move_name).is_protect


def protect_move_ids() -> frozenset:
    return _PROTECT_IDS
