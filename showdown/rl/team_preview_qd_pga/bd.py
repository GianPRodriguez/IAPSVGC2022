"""Preview BD b1 (speed-control) / b2 (offense vs support) — spec §3."""
from __future__ import annotations

from typing import Sequence, Tuple

from showdown.battle_bots.expectimax_bayesian.team_preview import (
    FAKE_OUT_SPECIES,
    REDIRECT_SPECIES,
    SPEED_CONTROL_SPECIES,
    TAILWIND_SPECIES,
    TRICK_ROOM_SPECIES,
    _nid,
)
from showdown.rl.team_preview_action import decode_preview_action


def preview_bd_from_action(
    action: int,
    our_species: Sequence[str],
) -> Tuple[float, float]:
    """
    b1: TR / TW-speed-control bias.
    b2: aggro leads+bench (Fake Out or non-support) vs redirect/TW/TR support.
    """
    i1, i2, i3, i4 = decode_preview_action(int(action))
    names = list(our_species)
    while len(names) < 6:
        names.append('Pikachu')
    bring_ids = {_nid(names[i]) for i in (i1, i2, i3, i4)}
    lead_ids = [_nid(names[i1]), _nid(names[i2])]
    bench_ids = [_nid(names[i3]), _nid(names[i4])]

    has_tr = bool(bring_ids & TRICK_ROOM_SPECIES)
    has_tw_sc = bool(bring_ids & TAILWIND_SPECIES) or bool(
        bring_ids & SPEED_CONTROL_SPECIES
    )
    b1 = float(max(0.0, min(1.0, 0.5 * float(has_tr) + 0.5 * float(has_tw_sc))))

    support = REDIRECT_SPECIES | TAILWIND_SPECIES | TRICK_ROOM_SPECIES

    def _aggro(nid: str) -> bool:
        if nid in FAKE_OUT_SPECIES:
            return True
        return nid not in support

    n_lead = sum(1 for x in lead_ids if _aggro(x))
    n_bench = sum(1 for x in bench_ids if _aggro(x))
    b2 = float(max(0.0, min(1.0, (n_lead + 0.5 * n_bench) / 3.0)))
    return b1, b2
