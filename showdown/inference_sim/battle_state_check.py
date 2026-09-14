"""Contraste ligero: estado IAPSVGC (Battle/Battler/Pokemon) vs poke-env (PS local).

No valida Stat Points (eso es soundness.py). Solo paridad de entorno / actives
que actualizan battle_modifier a partir del protocolo.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from teams.team_converter import normalize_name


@dataclass
class StateMismatch:
    battle_tag: str
    turn: int
    field: str
    expected: str  # poke-env / PS
    got: str  # IAP Battle
    detail: str = ''


@dataclass
class StateCheckReport:
    checks: int = 0
    mismatches: List[StateMismatch] = field(default_factory=list)
    feed_errors: int = 0
    battle_errors: int = 0
    _cap: int = 40
    _total_mismatches: int = 0

    @property
    def state_mismatch(self) -> bool:
        return self._total_mismatches > 0 or self.feed_errors > 0

    @property
    def ok(self) -> bool:
        return not self.state_mismatch

    def add(self, mm: StateMismatch) -> None:
        self._total_mismatches += 1
        if len(self.mismatches) < self._cap:
            self.mismatches.append(mm)

    def flag_line(self) -> str:
        return 'YES' if self.state_mismatch else 'NO'


_BOOST_ALIAS = {
    'attack': 'atk',
    'defense': 'def',
    'specialattack': 'spa',
    'spatk': 'spa',
    'specialdefense': 'spd',
    'spdef': 'spd',
    'speed': 'spe',
    'accuracy': 'accuracy',
    'evasion': 'evasion',
    'atk': 'atk',
    'def': 'def',
    'spa': 'spa',
    'spd': 'spd',
    'spe': 'spe',
}


def _canon(s: Any) -> str:
    """Id estable para comparar especies / fields (sin guiones ni underscores)."""
    if s is None:
        return ''
    if hasattr(s, 'name') and not isinstance(s, str):
        s = s.name
    text = normalize_name(str(s))
    return ''.join(ch for ch in text if ch.isalnum())


def _pe_weather(battle) -> str:
    w = getattr(battle, 'weather', None)
    if not w:
        return ''
    if isinstance(w, dict):
        if not w:
            return ''
        key = next(iter(w.keys()))
        return _canon(key)
    return _canon(w)


def _pe_fields(battle) -> Set[str]:
    out: Set[str] = set()
    fields = getattr(battle, 'fields', None) or {}
    for key in fields.keys():
        name = _canon(key)
        if name:
            out.add(name)
    return out


def _iap_fields(battle) -> Set[str]:
    terrain = getattr(battle, 'terrain', None) or {}
    out: Set[str] = set()
    for key, val in dict(terrain).items():
        if val:
            out.add(_canon(key))
    return out


def _iap_weather(battle) -> str:
    w = getattr(battle, 'weather', None)
    if not w or str(w).lower() in ('none', 'null', ''):
        return ''
    return _canon(w)


_MIRROR_FIELD_KEYS = frozenset({
    'electricterrain', 'grassyterrain', 'mistyterrain', 'psychicterrain',
    'trickroom', 'wonderroom', 'magicroom', 'gravity',
})


def mirror_pe_field_and_boosts_into_iap(iap_battle, pe_battle) -> None:
    """Ancla weather / fields / boosts del espejo IAP a poke-env (harness).

    Evita el péndulo soft-ensure↔protocol-only en sync E2E. Solo para players
    validados con PE; ladder sin PE sigue el protocolo en battle_modifier.
    """
    from collections import defaultdict

    if iap_battle is None or pe_battle is None:
        return

    pe_w = _pe_weather(pe_battle)
    if pe_w:
        iap_battle.weather = pe_w
        if int(getattr(iap_battle, 'weatherTurn', 0) or 0) <= 0:
            iap_battle.weatherTurn = 5
        iap_battle._weather_ended = False
    else:
        iap_battle.weather = 'none'
        iap_battle.weatherTurn = 0
        iap_battle._weather_ended = True

    pe_f = _pe_fields(pe_battle)
    terrain = getattr(iap_battle, 'terrain', None)
    if isinstance(terrain, dict):
        for key in list(terrain.keys()):
            ck = _canon(key)
            if ck in _MIRROR_FIELD_KEYS and ck not in pe_f:
                terrain.pop(key, None)
        for fname in pe_f:
            if fname in _MIRROR_FIELD_KEYS and int(terrain.get(fname, 0) or 0) <= 0:
                terrain[fname] = 5

    pe_ours, pe_foes = _actives_pe(pe_battle)
    iap_ours, iap_foes = _actives_iap(iap_battle)
    pe_ours, iap_ours = _align_by_species(pe_ours, iap_ours)
    pe_foes, iap_foes = _align_by_species(pe_foes, iap_foes)
    for pe_m, iap_m in (*zip(pe_ours, iap_ours), *zip(pe_foes, iap_foes)):
        if pe_m is None or iap_m is None:
            continue
        if not _same_species(_species_pe(pe_m), _species_iap(iap_m)):
            continue
        pe_b = _boosts_map(pe_m)
        merged = defaultdict(lambda: 0)
        for k, v in pe_b.items():
            try:
                merged[str(k)] = int(v)
            except Exception:
                continue
        iap_m.boosts = merged


def _species_pe(mon) -> str:
    if mon is None:
        return ''
    return _canon(getattr(mon, 'species', None) or getattr(mon, 'name', '') or '')


def _species_iap(mon) -> str:
    if mon is None:
        return ''
    return _canon(getattr(mon, 'name', '') or '')


def _same_species(pe_sp: str, iap_sp: str) -> bool:
    """True si pe/iap refieren la misma especie (mega / eternal / forme)."""
    if not pe_sp or not iap_sp:
        return False
    if pe_sp == iap_sp:
        return True
    if pe_sp.startswith(iap_sp) or iap_sp.startswith(pe_sp):
        return True

    def _core(s: str) -> str:
        s = s.replace('mega', '')
        if s.endswith('eternal'):
            s = s[: -len('eternal')]
        return s

    ca, cb = _core(pe_sp), _core(iap_sp)
    if ca and cb and (ca == cb or ca.startswith(cb) or cb.startswith(ca)):
        return True
    # Floette-Eternal ↔ Floette-Mega (Champions)
    if 'floette' in pe_sp and 'floette' in iap_sp:
        return True
    # Mr. Mime variants
    if pe_sp.replace('mr', '') == 'mime' and iap_sp.replace('mr', '') == 'mime':
        return True
    return False


def _align_by_species(pe_list, iap_list):
    """Empareja slots pe↔iap por especie para no marcar false positives a/b cruzados."""
    pe = list(pe_list) + [None] * max(0, 2 - len(pe_list))
    iap = list(iap_list) + [None] * max(0, 2 - len(iap_list))
    pe, iap = pe[:2], iap[:2]
    used = set()
    aligned = [None, None]
    for i, pe_mon in enumerate(pe):
        pe_sp = _species_pe(pe_mon)
        pe_alive = pe_mon is not None and not getattr(pe_mon, 'fainted', False)
        match_j = None
        for j, iap_mon in enumerate(iap):
            if j in used:
                continue
            if not pe_sp or not _same_species(pe_sp, _species_iap(iap_mon)):
                continue
            # No emparejar un vivo de PE con un KO de IAP (falso foe2.fainted).
            if pe_alive and iap_mon is not None and getattr(iap_mon, 'is_fainted', False):
                continue
            match_j = j
            break
        if match_j is None:
            for j, iap_mon in enumerate(iap):
                if j in used:
                    continue
                if pe_alive and iap_mon is not None and getattr(iap_mon, 'is_fainted', False):
                    continue
                match_j = j
                break
        if match_j is None:
            # Sin candidato vivo: dejar None (mejor que reportar KO fantasma).
            continue
        used.add(match_j)
        aligned[i] = iap[match_j]
    return pe, aligned


def _hp_pct_pe(mon) -> Optional[float]:
    if mon is None:
        return None
    if getattr(mon, 'fainted', False):
        return 0.0
    frac = getattr(mon, 'current_hp_fraction', None)
    if frac is None:
        return None
    try:
        return 100.0 * float(frac)
    except Exception:
        return None


def _hp_pct_iap(mon) -> Optional[float]:
    if mon is None:
        return None
    if getattr(mon, 'is_fainted', False):
        return 0.0
    try:
        cur = float(str(getattr(mon, 'HPA_battle_actual', 0)).split()[0])
        mx = float(str(getattr(mon, 'HPB_battle_base', 0)).split()[0])
        if mx <= 0:
            return None
        return 100.0 * cur / mx
    except Exception:
        return None


def _boosts_map(mon) -> Dict[str, int]:
    if mon is None:
        return {}
    raw = getattr(mon, 'boosts', None) or {}
    out: Dict[str, int] = {}
    for k, v in dict(raw).items():
        try:
            iv = int(v)
        except Exception:
            continue
        if not iv:
            continue
        key = _BOOST_ALIAS.get(_canon(k), _canon(k))
        if not key:
            continue
        # Si coexisten clave corta y larga, quedarse con la de mayor |stage|
        # (evita subcontar) pero NUNCA sumar (doble conteo defense+def).
        prev = out.get(key)
        if prev is None or abs(iv) > abs(prev):
            out[key] = iv
    return out


def _status_pe(mon) -> str:
    if mon is None or getattr(mon, 'fainted', False):
        return ''
    st = getattr(mon, 'status', None)
    return _canon(st) if st else ''


def _status_iap(mon) -> str:
    if mon is None or getattr(mon, 'is_fainted', False):
        return ''
    nvs = dict(getattr(mon, 'non_volatile_statuses', {}) or {})
    for k, v in nvs.items():
        if v:
            return _canon(k)
    st = getattr(mon, 'status', None)
    return _canon(st) if st else ''


def _actives_pe(battle) -> Tuple[List, List]:
    ours = list(getattr(battle, 'active_pokemon', None) or [])
    foes = list(getattr(battle, 'opponent_active_pokemon', None) or [])
    while len(ours) < 2:
        ours.append(None)
    while len(foes) < 2:
        foes.append(None)
    return ours[:2], foes[:2]


def _actives_iap(battle) -> Tuple[List, List]:
    user = getattr(battle, 'user', None)
    opp = getattr(battle, 'opponent', None)
    ours = [
        getattr(user, 'active1', None) if user else None,
        getattr(user, 'active2', None) if user else None,
    ]
    foes = [
        getattr(opp, 'active1', None) if opp else None,
        getattr(opp, 'active2', None) if opp else None,
    ]
    return ours, foes


def _side_conds_pe(battle, *, opponent: bool) -> Set[str]:
    sc = getattr(battle, 'opponent_side_conditions' if opponent else 'side_conditions', None) or {}
    return {_canon(k) for k in sc.keys() if k}


def _side_conds_iap(battle, *, opponent: bool) -> Set[str]:
    side = getattr(battle, 'opponent' if opponent else 'user', None)
    sc = getattr(side, 'side_conditions', None) or {}
    return {_canon(k) for k, v in dict(sc).items() if v}


def contrast_battle(
    iap_battle,
    pe_battle,
    *,
    battle_tag: str = '',
    hp_tol_pct: float = 12.0,
) -> List[StateMismatch]:
    """Compara campos de entorno/actives. Devuelve lista (puede estar vacía)."""
    found: List[StateMismatch] = []
    turn = int(getattr(pe_battle, 'turn', 0) or getattr(iap_battle, 'turn', 0) or 0)

    def _mm(field: str, expected: Any, got: Any, detail: str = '') -> None:
        found.append(
            StateMismatch(
                battle_tag=battle_tag,
                turn=turn,
                field=field,
                expected=str(expected),
                got=str(got),
                detail=detail,
            )
        )

    pe_turn = int(getattr(pe_battle, 'turn', 0) or 0)
    iap_turn = int(getattr(iap_battle, 'turn', 0) or 0)
    if pe_turn >= 1 and iap_turn >= 1 and pe_turn != iap_turn:
        _mm('turn', pe_turn, iap_turn)

    pe_w = _pe_weather(pe_battle)
    iap_w = _iap_weather(iap_battle)
    if pe_w != iap_w:
        aliases = {
            ('snowscape', 'hail'),
            ('hail', 'snowscape'),
            ('snow', 'snowscape'),
            ('snowscape', 'snow'),
        }
        if (pe_w, iap_w) not in aliases:
            _mm('weather', pe_w or '-', iap_w or '-')

    pe_f = _pe_fields(pe_battle)
    iap_f = _iap_fields(iap_battle)
    if pe_f != iap_f:
        _mm('fields', sorted(pe_f) or '-', sorted(iap_f) or '-')

    for opp in (False, True):
        label = 'opp_side' if opp else 'our_side'
        pe_sc = _side_conds_pe(pe_battle, opponent=opp)
        iap_sc = _side_conds_iap(iap_battle, opponent=opp)
        if pe_sc != iap_sc:
            _mm(f'{label}_conditions', sorted(pe_sc) or '-', sorted(iap_sc) or '-')

    pe_ours, pe_foes = _actives_pe(pe_battle)
    iap_ours, iap_foes = _actives_iap(iap_battle)
    # Alinear por especie (a/b de poke-env vs IAP pueden cruzarse)
    pe_ours, iap_ours = _align_by_species(pe_ours, iap_ours)
    pe_foes, iap_foes = _align_by_species(pe_foes, iap_foes)

    def _slot(side: str, idx: int, pe_mon, iap_mon) -> None:
        pe_sp = _species_pe(pe_mon)
        iap_sp = _species_iap(iap_mon)
        pe_faint = bool(pe_mon is None or getattr(pe_mon, 'fainted', False))
        iap_faint = bool(iap_mon is None or getattr(iap_mon, 'is_fainted', False))

        if not pe_sp and not iap_sp:
            return

        if pe_sp and iap_sp and not _same_species(pe_sp, iap_sp):
            _mm(f'{side}{idx}.species', pe_sp, iap_sp)

        if pe_mon is not None and iap_mon is not None and pe_faint != iap_faint:
            _mm(f'{side}{idx}.fainted', pe_faint, iap_faint)

        pe_hp = _hp_pct_pe(pe_mon)
        iap_hp = _hp_pct_iap(iap_mon)
        if pe_hp is not None and iap_hp is not None and _same_species(pe_sp, iap_sp):
            if abs(pe_hp - iap_hp) > hp_tol_pct:
                _mm(
                    f'{side}{idx}.hp_pct',
                    f'{pe_hp:.1f}',
                    f'{iap_hp:.1f}',
                    detail=f'tol={hp_tol_pct}',
                )

        same_sp = _same_species(pe_sp, iap_sp)
        if same_sp:
            pe_st = _status_pe(pe_mon)
            iap_st = _status_iap(iap_mon)
            if pe_st != iap_st:
                _mm(f'{side}{idx}.status', pe_st or '-', iap_st or '-')

            pe_b = _boosts_map(pe_mon)
            iap_b = _boosts_map(iap_mon)
            if pe_b != iap_b:
                _mm(f'{side}{idx}.boosts', pe_b or '-', iap_b or '-')

    for i, (a, b) in enumerate(zip(pe_ours, iap_ours), start=1):
        _slot('our', i, a, b)
    for i, (a, b) in enumerate(zip(pe_foes, iap_foes), start=1):
        _slot('foe', i, a, b)

    return found


def record_contrasts(
    report: StateCheckReport,
    mismatches: List[StateMismatch],
) -> None:
    report.checks += 1
    for mm in mismatches:
        report.add(mm)
