"""Unit tests: protocolo Conclusión (seed, ε, ranking, snapshot)."""
from __future__ import annotations

from showdown.inference_sim.conclusion_eval.pipelines import DEFAULT_PIPELINES
from showdown.inference_sim.conclusion_eval.snapshot import (
    TeamRef,
    build_snapshot,
    dedupe_teams,
)
from showdown.inference_sim.conclusion_eval.stats import (
    CrossResult,
    cross_points,
    deterministic_cross_order,
    rank_pipelines,
    wilson_ci,
)
from showdown.inference_sim.team_factory import MonSet, TeamMeta


def _fake_team(name: str, species: str = 'Incineroar') -> TeamMeta:
    mon = MonSet(
        species=species,
        item='Safety Goggles',
        ability='Intimidate',
        nature='Adamant',
        moves=['Fake Out', 'Flare Blitz', 'Knock Off', 'Parting Shot'],
        sp={'hp': 20, 'atk': 20, 'def': 20, 'spa': 0, 'spd': 20, 'spe': 20},
    )
    # 6 mons distintos para dedup key variety
    mons = []
    for i, sp in enumerate(
        [species, 'Rillaboom', 'Amoonguss', 'Indeedee-F', 'Chi-Yu', 'Flutter Mane']
    ):
        mons.append(
            MonSet(
                species=sp if i == 0 else f'{sp}',
                item=f'Item{i}{name[-1:]}',
                ability='Pressure',
                nature='Timid',
                moves=['Protect', 'Substitute', 'Rest', 'Sleep Talk'],
                sp={'hp': 10, 'atk': 10, 'def': 10, 'spa': 10, 'spd': 10, 'spe': 10},
            )
        )
    return TeamMeta(name=name, source_url='', mons=mons)


def test_default_pipelines_count():
    assert len(DEFAULT_PIPELINES) == 11
    ids = [p.id for p in DEFAULT_PIPELINES]
    assert ids[0] == 'ddqn'
    assert 'random' in ids and 'expectimax_bayesian' in ids


def test_cross_points_eps():
    # N=200: 105-95 → empate; 106-94 → victoria
    assert cross_points(105, 200) == (0.5, 0.5)
    assert cross_points(95, 200) == (0.5, 0.5)
    assert cross_points(104, 200) == (0.5, 0.5)
    assert cross_points(106, 200) == (1.0, 0.0)
    assert cross_points(94, 200) == (0.0, 1.0)
    assert cross_points(100, 200) == (0.5, 0.5)


def test_wilson_ci_bounds():
    wr, lo, hi = wilson_ci(112, 200)
    assert wr == 0.56
    assert lo is not None and hi is not None
    assert 0.0 <= lo < wr < hi <= 1.0


def test_deterministic_cross_order_stable():
    a = deterministic_cross_order(11, seed=42)
    b = deterministic_cross_order(11, seed=42)
    c = deterministic_cross_order(11, seed=43)
    assert a == b
    assert len(a) == 55
    assert a[0][0] < a[0][1]
    assert a != c


def test_rank_pipelines_points_and_tiebreak():
    ids = ['alpha', 'beta', 'random']
    crosses = [
        CrossResult(
            i=0, j=1, id_i='alpha', id_j='beta',
            wins_i=120, wins_j=80, valid=200,
        ),
        CrossResult(
            i=0, j=2, id_i='alpha', id_j='random',
            wins_i=150, wins_j=50, valid=200,
        ),
        CrossResult(
            i=1, j=2, id_i='beta', id_j='random',
            wins_i=140, wins_j=60, valid=200,
        ),
    ]
    ranks = rank_pipelines(ids, crosses)
    assert [r.pipeline_id for r in ranks] == ['alpha', 'beta', 'random']
    assert ranks[0].points == 2.0
    assert ranks[1].points == 1.0
    assert ranks[2].points == 0.0


def test_rank_eps_half_point():
    ids = ['a', 'b']
    crosses = [
        CrossResult(
            i=0, j=1, id_i='a', id_j='b',
            wins_i=105, wins_j=95, valid=200,
        ),
    ]
    ranks = rank_pipelines(ids, crosses)
    assert ranks[0].points == 0.5
    assert ranks[1].points == 0.5


def test_build_snapshot_reproducible():
    teams = [_fake_team(f'T{i}', species=f'Mon{i}') for i in range(20)]
    # Unique item keys already in _fake_team via name
    s1 = build_snapshot(seed=7, snapshot_n=8, n_pairs=3, teams=teams)
    s2 = build_snapshot(seed=7, snapshot_n=8, n_pairs=3, teams=teams)
    assert s1.pool_a == s2.pool_a
    assert s1.pool_b == s2.pool_b
    assert s1.pairs == s2.pairs
    assert len(s1.pool_a) == 4 and len(s1.pool_b) == 4
    assert len(s1.pairs) == 3
    # Sin reemplazo: pares únicos
    keys = {(a.name, b.name) for a, b in s1.pairs}
    assert len(keys) == 3
    names_a = {t.name for t in s1.pool_a}
    names_b = {t.name for t in s1.pool_b}
    assert names_a.isdisjoint(names_b)


def test_dedupe_teams():
    t1 = _fake_team('A')
    t2 = _fake_team('B')  # same species/items pattern → same key if items collide
    # Force identical mons
    t2.mons = list(t1.mons)
    t2.name = 'B_dup'
    out = dedupe_teams([t1, t2, _fake_team('C', species='Zapdos')])
    assert len(out) == 2
