"""Auditoría barata: flags de inferencia/calc vs lo persistido en Battle.

Corre una vez por batch (sin coste por turno). Objetivo crédito Cursor:
  NEED_FLAGS_ADD=NO  → nada urgente que agregar en Battle/deps.
  NEED_FLAGS_ADD=YES → hay gaps (storage o wiring) listados en el reporte.

NO alimenta NEED_FIX (sync/SP/errors siguen siendo el único gatillo de agente).
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from showdown.battle import Battle, Battler, Pokemon
from showdown import stat_inference as si


# --------------------------------------------------------------------------- #
# Contrato: lo que DamageContext / Spe YA usan y DEBE vivir en Battle/deps
# --------------------------------------------------------------------------- #

# (id, owner_cls, attr)
_REQUIRED_STORAGE: Tuple[Tuple[str, type, str], ...] = (
    ('weather', Battle, 'weather'),
    ('terrain', Battle, 'terrain'),
    ('side_conditions', Battler, 'side_conditions'),
    ('boosts', Pokemon, 'boosts'),
    ('is_helping_hand', Pokemon, 'is_helping_hand'),
    ('is_charged', Pokemon, 'is_charged'),
    ('is_protected', Pokemon, 'is_protected'),
    ('volatile_statuses', Pokemon, 'volatile_statuses'),
    ('abilityPkmn', Pokemon, 'abilityPkmn'),
    ('objectPkmn', Pokemon, 'objectPkmn'),
    ('type1', Pokemon, 'type1'),
    ('type2', Pokemon, 'type2'),
    ('ability_on', Pokemon, 'ability_on'),
    ('just_switched', Pokemon, 'just_switched'),
    ('tera_type', Pokemon, 'tera_type'),
    ('is_terastallized', Pokemon, 'is_terastallized'),
)

# Claves que _field_payload debe emitir cuando el ctx lo pide
_REQUIRED_FIELD_KEYS: Tuple[str, ...] = (
    'isHelpingHand',
    'isCharge',
    'isReflect',
    'isLightScreen',
    'isAuroraVeil',
    'isFriendGuard',
    'isWonderRoom',
    'isMagicRoom',
    'isGravity',
    'isBattery',
    'isPowerSpot',
    'isSteelySpirit',
    'isFlowerGift',
    'isBeadsOfRuin',
    'isSwordOfRuin',
    'isTabletsOfRuin',
    'isVesselOfRuin',
    'weather',
    'terrain',
    'isSpread',
)

# Gaps residuales (vacío si greedy ya cablea side flags).
_KNOWN_GAPS: Tuple[Tuple[str, str, str, str], ...] = ()


@dataclass
class FlagGap:
    flag: str
    priority: str
    storage: str
    detail: str


@dataclass
class FlagAuditResult:
    need_flags_add: bool
    missing_storage: List[str] = field(default_factory=list)
    missing_wire: List[str] = field(default_factory=list)
    gaps: List[FlagGap] = field(default_factory=list)
    ok_storage: List[str] = field(default_factory=list)
    meta_hits: List[str] = field(default_factory=list)

    def flag_line(self) -> str:
        return 'YES' if self.need_flags_add else 'NO'

    def samples(self, limit: int = 12) -> List[str]:
        out: List[str] = []
        for m in self.missing_storage:
            out.append(f'missing_storage: {m}')
        for m in self.missing_wire:
            out.append(f'missing_wire: {m}')
        for g in self.gaps:
            if g.priority == 'low':
                continue
            out.append(f'{g.priority}:{g.flag} → {g.storage} | {g.detail}')
        if self.meta_hits:
            out.append('meta_abilities_relevant: ' + ', '.join(self.meta_hits[:8]))
        return out[:limit]


_ABILITY_HINTS: Dict[str, str] = {
    'flashfire': 'abilityOn',
    'stakeout': 'abilityOn',
    'slowstart': 'abilityOn',
    'plus': 'abilityOn',
    'minus': 'abilityOn',
    'electromorphosis': 'abilityOn',
    'battery': 'ally_side_boosts',
    'powerspot': 'ally_side_boosts',
    'steelyspirit': 'ally_side_boosts',
    'flowergift': 'ally_side_boosts',
    'beadsofruin': 'ruin_field_flags',
    'swordofruin': 'ruin_field_flags',
    'tabletsofruin': 'ruin_field_flags',
    'vesselofruin': 'ruin_field_flags',
}


def _norm_ability(raw: str) -> str:
    return str(raw or '').strip().lower().replace(' ', '').replace('-', '')


def _cls_has_attr(cls: type, attr: str) -> bool:
    if hasattr(cls, attr):
        return True
    try:
        src = inspect.getsource(cls.__init__)
    except (OSError, TypeError):
        return False
    # self.attr = ...
    return f'self.{attr}' in src


def _field_payload_source() -> str:
    try:
        return inspect.getsource(si._field_payload)
    except (OSError, TypeError):
        return ''


def _poke_payload_source() -> str:
    try:
        return inspect.getsource(si._poke_payload)
    except (OSError, TypeError):
        return ''


def scan_meta_ability_hints(abilities: Sequence[str]) -> List[str]:
    """Habilidades del pool que tocan gaps conocidos (señal barata)."""
    hits: List[str] = []
    seen: Set[str] = set()
    for ab in abilities:
        key = _norm_ability(ab)
        gap = _ABILITY_HINTS.get(key)
        if gap and key not in seen:
            seen.add(key)
            hits.append(f'{ab}→{gap}')
    return hits


def collect_pool_abilities(team_metas: Sequence[object]) -> List[str]:
    out: List[str] = []
    for tm in team_metas:
        mons = getattr(tm, 'mons', None) or getattr(tm, 'pokemon', None) or []
        for mon in mons:
            if isinstance(mon, dict):
                ab = mon.get('ability') or mon.get('abilityPkmn')
            else:
                ab = getattr(mon, 'ability', None) or getattr(mon, 'abilityPkmn', None)
            if ab:
                out.append(str(ab))
    return out


def audit_inference_flags(
    *,
    pool_abilities: Optional[Sequence[str]] = None,
) -> FlagAuditResult:
    """Auditoría estática (+hints del pool). Sin batallas / sin calc."""
    missing_storage: List[str] = []
    ok_storage: List[str] = []
    for fid, cls, attr in _REQUIRED_STORAGE:
        label = f'{cls.__name__}.{attr}'
        if _cls_has_attr(cls, attr):
            ok_storage.append(label)
        else:
            missing_storage.append(f'{fid} ({label})')

    src = _field_payload_source()
    missing_wire: List[str] = []
    for key in _REQUIRED_FIELD_KEYS:
        # weather/terrain/isSpread/isWonderRoom están en el dict raíz;
        # side keys solo si el if correspondiente existe.
        if key not in src:
            missing_wire.append(key)

    # abilityOn no debe aparecer en poke payload hoy → gap, no required wire fail
    poke_src = _poke_payload_source()
    gaps: List[FlagGap] = []
    for flag, prio, storage, note in _KNOWN_GAPS:
        # Si algún día se cablea, no reportar como gap
        if flag == 'abilityOn' and 'abilityOn' in poke_src:
            continue
        if flag == 'ally_side_boosts' and (
            'isBattery' in src or 'isPowerSpot' in src or 'isSteelySpirit' in src
        ):
            continue
        if flag == 'ruin_field_flags' and 'OfRuin' in src:
            continue
        if flag == 'gravity_magic_room' and 'isGravity' in src and 'isMagicRoom' in src:
            continue
        if flag == 'live_types' and 'type1' in poke_src:
            continue
        if flag == 'teraType' and 'teraType' in poke_src:
            continue
        gaps.append(FlagGap(flag=flag, priority=prio, storage=storage, detail=note))

    meta_hits = scan_meta_ability_hints(pool_abilities or [])

    actionable = bool(missing_storage or missing_wire) or any(
        g.priority in ('high', 'medium') for g in gaps
    )
    return FlagAuditResult(
        need_flags_add=actionable,
        missing_storage=missing_storage,
        missing_wire=missing_wire,
        gaps=gaps,
        ok_storage=ok_storage,
        meta_hits=meta_hits,
    )


def format_audit_section(result: FlagAuditResult) -> List[str]:
    lines = [
        '--- inference flags (Battle storage / calc wiring) ---',
        f'NEED_FLAGS_ADD: {result.flag_line()}',
        f'flags_storage_ok: {len(result.ok_storage)}',
        f'flags_missing_storage: {len(result.missing_storage)}',
        f'flags_missing_wire: {len(result.missing_wire)}',
        f'flags_gaps_high_med: '
        f'{sum(1 for g in result.gaps if g.priority in ("high", "medium"))}',
    ]
    samples = result.samples(12)
    if samples:
        lines.append('flags_samples:')
        for s in samples:
            lines.append(f'- {s}')
    elif result.need_flags_add is False:
        lines.append(
            '- OK: storage + wiring de DamageContext presentes; sin gaps high/medium.'
        )
    return lines
