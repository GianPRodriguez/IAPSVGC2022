"""Muestreo bayesiano de perfiles de oponente (θ) desde StatBelief + MetaVGC.

No existe un OpponentStatInferenceEngine aparte: reutiliza `StatBelief`
(observaciones de combate) y el compendio MetaVGC (priors de SP/item/moves).

Champions usa Stat Points, no EV/IV clásicos.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from showdown.battle_bots.greedy_solo.greedy import CalcPoke, _stats_or_default
from showdown.stat_inference import STAT_KEYS, champions_stat
from teams.team_converter import normalize_name

_JSON_PATH = (
    Path(__file__).resolve().parents[2]
    / 'inference_sim'
    / 'metavgc_regmc_teams.json'
)

# species -> list of raw mon dicts with count weight
_MetaBag = List[Tuple[dict, int]]
_index: Dict[str, _MetaBag] = {}
_loaded_path: Optional[Path] = None


def _norm(s: Optional[str]) -> str:
    return normalize_name(s or '') or ''


def _load_meta(path: Optional[Path] = None, *, force: bool = False) -> int:
    global _index, _loaded_path
    p = Path(path) if path is not None else _JSON_PATH
    if not force and _loaded_path == p and _index:
        return sum(len(v) for v in _index.values())
    _index = defaultdict(list)
    _loaded_path = p
    if not p.is_file():
        return 0
    try:
        raw = json.loads(p.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError):
        return 0
    if not isinstance(raw, list):
        return 0
    # Dedup exact mons per species with counts
    buckets: Dict[str, Dict[Tuple, int]] = defaultdict(lambda: defaultdict(int))
    for team in raw:
        for mon in (team.get('mons') or []):
            species = _norm(mon.get('species'))
            if not species:
                continue
            sp = mon.get('sp') or {}
            key = (
                _norm(mon.get('nature')) or 'serious',
                _norm(mon.get('item')),
                _norm(mon.get('ability')),
                tuple(sorted(_norm(m) for m in (mon.get('moves') or []) if m)),
                tuple(int(sp.get(k, 0) or 0) for k in STAT_KEYS),
            )
            buckets[species][key] += 1
    for species, ctr in buckets.items():
        rows: _MetaBag = []
        for key, count in ctr.items():
            nature, item, ability, moves, sp_t = key
            rows.append((
                {
                    'species': species,
                    'nature': nature,
                    'item': item or None,
                    'ability': ability or None,
                    'moves': list(moves),
                    'sp': {STAT_KEYS[i]: sp_t[i] for i in range(len(STAT_KEYS))},
                },
                int(count),
            ))
        rows.sort(key=lambda x: (-x[1], x[0].get('nature') or ''))
        _index[species] = rows
    return sum(len(v) for v in _index.values())


@dataclass(frozen=True)
class OpponentProfile:
    """Perfil θ_j: spread SP + naturaleza + item/ability/moveset meta."""

    species: str
    nature: str
    sp: Tuple[int, int, int, int, int, int]
    item: Optional[str] = None
    ability: Optional[str] = None
    moves: Tuple[str, ...] = ()
    prior_mass: float = 1.0

    def sp_map(self) -> Dict[str, int]:
        return {STAT_KEYS[i]: int(self.sp[i]) for i in range(len(STAT_KEYS))}


@dataclass
class BeliefNode:
    """Nodo de creencia: k perfiles normalizados (Σ P(θ)=1)."""

    species: str
    profiles: List[OpponentProfile] = field(default_factory=list)
    probabilities: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.profiles and not self.probabilities:
            masses = [max(0.0, float(p.prior_mass)) for p in self.profiles]
            total = sum(masses) or 1.0
            self.probabilities = [m / total for m in masses]
        elif self.profiles and self.probabilities:
            total = sum(self.probabilities) or 1.0
            self.probabilities = [p / total for p in self.probabilities]

    def __iter__(self):
        return zip(self.profiles, self.probabilities)


def _compatible(
    sp: Dict[str, int],
    candidates: Dict[str, Set[int]],
) -> bool:
    for i, key in enumerate(STAT_KEYS):
        allowed = candidates.get(key)
        if allowed is None:
            continue
        if int(sp.get(key, 0) or 0) not in allowed:
            return False
    return True


def _contains_revealed(moves: Sequence[str], revealed: Sequence[str]) -> bool:
    if not revealed:
        return True
    have = {_norm(m) for m in moves}
    return all(_norm(m) in have for m in revealed if m)


def sample_belief_node(
    *,
    species: str,
    nature: Optional[str],
    candidates: Optional[Dict[str, Set[int]]],
    known_item: Optional[str] = None,
    known_ability: Optional[str] = None,
    revealed_moves: Optional[Sequence[str]] = None,
    k: int = 6,
    base_stats: Optional[Dict[str, int]] = None,
    ots: Optional['OpponentPublicOTS'] = None,
) -> BeliefNode:
    """
    Construye BeliefNode con hasta k perfiles.

    Con OTS Champions (sin SP): item/ability/nature/moves quedan fijos del OTS;
    MetaVGC solo aporta vectores de SP compatibles con `candidates`.

    Sin OTS completo: comportamiento previo (filtra meta por lo conocido).
    """
    from showdown.battle_bots.expectimax_bayesian.ots import OpponentPublicOTS

    sid = _norm(species)
    # Clavar campos visibles del OTS (nunca SP).
    if ots is not None:
        if ots.nature:
            nature = ots.nature
        if ots.item:
            known_item = ots.item
        if ots.ability:
            known_ability = ots.ability
        if ots.moves:
            revealed_moves = list(ots.moves)

    nid = _norm(nature) or None
    revealed = list(revealed_moves or [])
    ots_locked = bool(
        ots is not None
        and known_item
        and known_ability
        and nid
        and revealed
    )
    _load_meta()
    cands = candidates or {key: set(range(0, 33)) for key in STAT_KEYS}

    scored: List[OpponentProfile] = []
    for mon, count in _index.get(sid, []):
        sp = mon.get('sp') or {}
        if not _compatible(sp, cands):
            continue
        mon_nat = _norm(mon.get('nature')) or 'serious'
        if nid and mon_nat != nid:
            # Con OTS la nature es fija; sin OTS filtramos. Si meta no tiene
            # esa nature, aún podemos reutilizar el vector SP más abajo.
            if not ots_locked:
                continue
            else:
                # Reusar SP de otro nature meta si encaja en candidatos.
                pass
        if not ots_locked:
            if known_item and _norm(mon.get('item')) and _norm(mon.get('item')) != _norm(known_item):
                continue
            if known_ability and _norm(mon.get('ability')) and _norm(mon.get('ability')) != _norm(known_ability):
                continue
        moves = tuple(mon.get('moves') or ())
        if ots_locked:
            moves = tuple(revealed)
            mon_nat = nid or mon_nat
            mass = float(count)
        else:
            if not _contains_revealed(moves, revealed):
                mass = float(count) * 0.15
            else:
                mass = float(count)
        sp_t = tuple(int(sp.get(key, 0) or 0) for key in STAT_KEYS)
        scored.append(OpponentProfile(
            species=sid,
            nature=mon_nat if not ots_locked else (nid or mon_nat),
            sp=sp_t,  # type: ignore[arg-type]
            item=(known_item or mon.get('item')),
            ability=(known_ability or mon.get('ability')),
            moves=moves if moves else tuple(revealed),
            prior_mass=mass,
        ))

    scored.sort(key=lambda p: (-p.prior_mass, p.nature, p.sp))
    # Con OTS: dedup solo por SP (item/moves/nature ya fijos).
    uniq: List[OpponentProfile] = []
    seen: Set[Tuple] = set()
    for p in scored:
        if ots_locked:
            key = tuple(p.sp)
        else:
            key = (p.nature, p.sp, _norm(p.item), tuple(sorted(_norm(m) for m in p.moves)))
        if key in seen:
            continue
        seen.add(key)
        if ots_locked:
            # Reescribir siempre con OTS público (por si el row meta traía otro item).
            p = OpponentProfile(
                species=sid,
                nature=nid or p.nature,
                sp=p.sp,
                item=known_item,
                ability=known_ability,
                moves=tuple(revealed),
                prior_mass=p.prior_mass,
            )
        uniq.append(p)
        if len(uniq) >= max(1, int(k)):
            break

    if not uniq:
        # Fallback: point-estimate SP via meta prior / mediana — OTS fijo si hay.
        try:
            from showdown.meta_sp_prior import preferred_sp_map
            picked = preferred_sp_map(
                species=sid,
                nature=nid or 'serious',
                candidates=cands,
            )
        except Exception:
            picked = {
                key: sorted(cands.get(key) or [0])[len(sorted(cands.get(key) or [0])) // 2]
                for key in STAT_KEYS
                if cands.get(key)
            }
        sp_t = tuple(int(picked.get(key, 0) or 0) for key in STAT_KEYS)
        # Diversificar SP dentro de candidatos (min/mediana/max por stat) si OTS locked
        if ots_locked and k > 1:
            alts: List[Tuple[int, ...]] = [sp_t]
            for key in STAT_KEYS:
                sps = sorted(cands.get(key) or [0])
                if len(sps) < 2:
                    continue
                for q in (0, len(sps) // 2, len(sps) - 1):
                    vec = list(sp_t)
                    vec[STAT_KEYS.index(key)] = int(sps[q])
                    alts.append(tuple(vec))
            seen_sp: Set[Tuple] = set()
            for vec in alts:
                if vec in seen_sp:
                    continue
                seen_sp.add(vec)
                uniq.append(OpponentProfile(
                    species=sid,
                    nature=nid or 'serious',
                    sp=vec,  # type: ignore[arg-type]
                    item=known_item,
                    ability=known_ability,
                    moves=tuple(revealed),
                    prior_mass=1.0,
                ))
                if len(uniq) >= max(1, int(k)):
                    break
        if not uniq:
            uniq = [OpponentProfile(
                species=sid,
                nature=nid or 'serious',
                sp=sp_t,  # type: ignore[arg-type]
                item=known_item,
                ability=known_ability,
                moves=tuple(revealed),
                prior_mass=1.0,
            )]

    return BeliefNode(species=sid, profiles=uniq[: max(1, int(k))])


def profile_to_calc(
    base: CalcPoke,
    profile: OpponentProfile,
    *,
    base_stats: Optional[Dict[str, int]] = None,
) -> CalcPoke:
    """Aplica θ sobre un CalcPoke (stats finales Champions + item/ability)."""
    bases = dict(base_stats or {})
    if not bases:
        # Invertir aproximado: usar stats actuales como finales si no hay bases
        stats = dict(_stats_or_default(base))
    else:
        stats = {}
        for i, key in enumerate(STAT_KEYS):
            b = int(bases.get(key) or 0)
            sp = int(profile.sp[i]) if i < len(profile.sp) else 0
            if b <= 0:
                stats[key] = int(_stats_or_default(base).get(key) or 100)
            else:
                stats[key] = champions_stat(b, sp, profile.nature, key)

    max_hp = int(stats.get('hp') or 100)
    cur = base.cur_hp
    if cur is not None and max_hp > 0:
        # Si cur parece %, escala; si ya es absoluto, clamp
        if int(cur) <= 100 and max_hp > 100:
            cur = max(1, int(round(max_hp * (int(cur) / 100.0))))
        else:
            cur = min(int(cur), max_hp)
    else:
        cur = max_hp

    return CalcPoke(
        species=base.species,
        nature=profile.nature.capitalize() if profile.nature else base.nature,
        item=profile.item if profile.item else base.item,
        ability=profile.ability if profile.ability else base.ability,
        status=base.status,
        boosts=dict(base.boosts or {}),
        stats=stats,
        cur_hp=cur,
        types=base.types,
        ability_on=base.ability_on,
        just_switched=base.just_switched,
        tera_type=base.tera_type,
        is_terastallized=base.is_terastallized,
        is_helping_hand=base.is_helping_hand,
        is_charged=base.is_charged,
        is_protected=base.is_protected,
        has_substitute=base.has_substitute,
        is_redirecting=base.is_redirecting,
        allies_fainted=base.allies_fainted,
        is_two_turn_charging=base.is_two_turn_charging,
        must_recharge=base.must_recharge,
    )


def extract_known_moves(pkmn) -> Tuple[str, ...]:
    """Moves OTS / revelados desde un Pokemon IAP o lista cruda."""
    raw = getattr(pkmn, 'moves', None)
    if not raw:
        return ()
    out = []
    for m in raw:
        if not m:
            continue
        name = str(m).strip()
        if name and name.lower() not in ('none', 'null', '-'):
            out.append(name)
    return tuple(out)


def resolve_opp_move_names(
    profile: Optional[OpponentProfile],
    known_moves: Sequence[str],
) -> Tuple[str, ...]:
    """
    Con OTS/revelados: usar esos 4 (fuente de verdad).
    Sin known: caer al moveset del perfil MetaVGC.
    """
    known = tuple(m for m in known_moves if m)
    if known:
        return known
    if profile is not None and profile.moves:
        return tuple(profile.moves)
    return ()


def belief_nodes_for_foes(
    foes: Sequence[Tuple[int, CalcPoke]],
    *,
    beliefs=None,
    k: int = 6,
    foe_known_moves: Optional[Dict[int, Sequence[str]]] = None,
    ots_sheet: Optional[Dict[str, 'OpponentPublicOTS']] = None,
) -> Dict[int, BeliefNode]:
    """BeliefNode por target id de rival activo. OTS público clava item/moves/nature."""
    from showdown.battle_bots.expectimax_bayesian.ots import lookup_ots

    known_map = foe_known_moves or {}
    out: Dict[int, BeliefNode] = {}
    for tid, foe in foes:
        belief = None
        if beliefs is not None:
            try:
                # beliefs puede ser TeamStatBeliefs o dict species->StatBelief
                if hasattr(beliefs, 'beliefs'):
                    key = _norm(foe.species)
                    belief = beliefs.beliefs.get(key)
                    if belief is None:
                        # try fuzzy
                        for sp, b in (beliefs.beliefs or {}).items():
                            if _norm(sp) == key:
                                belief = b
                                break
                elif isinstance(beliefs, dict):
                    belief = beliefs.get(_norm(foe.species))
            except Exception:
                belief = None

        candidates = None
        nature = foe.nature
        base_stats = None
        if belief is not None:
            candidates = getattr(belief, 'candidates', None)
            nature = getattr(belief, 'nature', None) or nature
            base_stats = getattr(belief, 'base_stats', None)

        ots = lookup_ots(ots_sheet, foe.species)
        revealed = tuple(known_map.get(tid) or ())
        if ots is not None and ots.moves:
            revealed = tuple(ots.moves)
        out[tid] = sample_belief_node(
            species=foe.species,
            nature=(ots.nature if ots and ots.nature else nature),
            candidates=candidates,
            known_item=(ots.item if ots and ots.item else foe.item),
            known_ability=(ots.ability if ots and ots.ability else foe.ability),
            revealed_moves=revealed,
            k=k,
            base_stats=base_stats,
            ots=ots,
        )
    return out


def iter_worlds(
    nodes: Dict[int, BeliefNode],
    foe_order: Sequence[int],
    *,
    max_worlds: int = 24,
) -> List[Tuple[Dict[int, OpponentProfile], float]]:
    """
    Producto cartesiano de perfiles por foe, podado a max_worlds por masa.
    Devuelve (tid->profile, P(world)).
    """
    if not foe_order:
        return [({}, 1.0)]

    worlds: List[Tuple[Dict[int, OpponentProfile], float]] = [({}, 1.0)]
    for tid in foe_order:
        node = nodes.get(tid)
        if node is None or not node.profiles:
            continue
        nxt: List[Tuple[Dict[int, OpponentProfile], float]] = []
        for partial, p_w in worlds:
            for prof, p_t in node:
                merged = dict(partial)
                merged[tid] = prof
                nxt.append((merged, p_w * float(p_t)))
        nxt.sort(key=lambda x: -x[1])
        worlds = nxt[: max(1, int(max_worlds))]

    total = sum(p for _, p in worlds) or 1.0
    return [(w, p / total) for w, p in worlds]
