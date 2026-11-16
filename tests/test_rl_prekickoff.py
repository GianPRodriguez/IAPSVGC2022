"""Correcciones previas al entrenamiento oficial de DDQN (puntos 1, 6 y 3 del plan).

1. El backend mock ya no congela la observación dentro del episodio.
6. El resto del rival tiene un orden estable por especie (revelar o descartar no mueve bloques).
3. DDQNPlayer ya no se traga las excepciones: las cuenta, las registra y en modo estricto
   las propaga.
"""
import sys

import numpy as np
import pytest

sys.path.insert(0, '.')

from showdown.rl.decision_guard import DecisionStats, strict_mode
from showdown.rl.observation import _foe_slots, build_observation
from showdown.rl.snapshot import MonSnapshot, SideSnapshot


# --------------------------------------------------------------------------- 1
def test_mock_observation_changes_between_steps():
    from showdown.rl.env import MockBattleBackend, VGCEnv

    env = VGCEnv(backend=MockBattleBackend(max_turns=40))
    env.reset(seed=0)
    o0 = build_observation(env._snap)
    snap0 = env._snap
    m1, m2 = env._masks()
    a = (int(np.flatnonzero(m1)[0]), int(np.flatnonzero(m2)[0]))
    env.step(a)
    assert env._snap is not snap0  # snapshot nuevo en cada paso, como el backend real
    o1 = build_observation(env._snap)
    env.step(a)
    o2 = build_observation(env._snap)
    assert not np.array_equal(o0, o1)
    assert not np.array_equal(o1, o2)


def test_mock_step_does_not_mutate_previous_snapshot():
    from showdown.rl.env import MockBattleBackend

    be = MockBattleBackend(max_turns=40)
    s0 = be.reset(seed=1)
    turn0 = s0.turn
    s1, _ = be.step((0, 0))
    assert s0.turn == turn0 and s1.turn == turn0 + 1


# --------------------------------------------------------------------------- 6
def _mon(name, *, seen=True, excluded=False):
    return MonSnapshot(species=name, ots_species=name, hp_pct=100.0, seen=seen, excluded=excluded)


def _side(rest):
    return SideSnapshot(active=[_mon('Incineroar'), _mon('Whimsicott')], bench=[None, None],
                        reserve=list(rest))


def _names(slots):
    return [m.species if m is not None else None for m in slots]


def test_foe_rest_sorted_by_species_regardless_of_seen():
    a = _side([_mon('Garchomp', seen=False), _mon('Amoonguss', seen=True),
               _mon('Rillaboom', seen=False), _mon('Kingambit', seen=True)])
    assert _names(_foe_slots(a))[2:] == ['Amoonguss', 'Garchomp', 'Kingambit', 'Rillaboom']


def test_revealing_a_mon_does_not_move_blocks():
    before = _side([_mon('Garchomp', seen=False), _mon('Amoonguss', seen=False),
                    _mon('Rillaboom', seen=False), _mon('Kingambit', seen=False)])
    after = _side([_mon('Garchomp', seen=True), _mon('Amoonguss', seen=False),
                   _mon('Rillaboom', seen=False), _mon('Kingambit', seen=False)])
    assert _names(_foe_slots(before)) == _names(_foe_slots(after))


def test_excluded_mon_stays_as_zero_block_in_place():
    before = _side([_mon('Garchomp'), _mon('Amoonguss'), _mon('Rillaboom'), _mon('Kingambit')])
    after = _side([_mon('Garchomp'), _mon('Amoonguss', excluded=True), _mon('Rillaboom'),
                   _mon('Kingambit')])
    nb, na = _names(_foe_slots(before)), _names(_foe_slots(after))
    assert na[2] is None                     # Amoonguss (descartado) → bloque en cero
    assert na[3:] == nb[3:]                  # el resto no se desplaza


# --------------------------------------------------------------------------- 3
def test_decision_stats_counts_and_rates(monkeypatch):
    monkeypatch.delenv('PS_RL_STRICT', raising=False)
    st = DecisionStats(agent='ddqn')
    st.record_ok()
    st.record_ok(order_fallback=True)
    st.record_exception(ValueError('obs rota'))
    d = st.as_dict()
    assert d['decisions'] == 3 and d['exceptions'] == 1 and d['order_fallbacks'] == 1
    assert d['by_error'] == {'ValueError': 1} and 'obs rota' in d['last_error']


def test_strict_mode_reraises(monkeypatch):
    monkeypatch.setenv('PS_RL_STRICT', '1')
    assert strict_mode()
    with pytest.raises(KeyError):
        DecisionStats().record_exception(KeyError('x'))


def _bare_ddqn_player():
    from showdown.rl.ddqn.player import DDQNPlayer
    p = object.__new__(DDQNPlayer)  # sin conectar a Showdown
    p.decision_stats = DecisionStats(agent='ddqn')
    p.epsilon = 0.0
    return p


def _double_battle():
    import logging
    from poke_env.battle import DoubleBattle
    return DoubleBattle('battle-test-1', 'tester', logging.getLogger('t'), gen=9)


def test_ddqn_player_counts_exception_instead_of_hiding_it(monkeypatch):
    from poke_env.player.battle_order import DefaultBattleOrder
    from showdown.rl.ddqn import player as mod

    monkeypatch.delenv('PS_RL_STRICT', raising=False)

    def boom(*_a, **_k):
        raise RuntimeError('snapshot roto')

    monkeypatch.setattr(mod, 'snapshot_from_battle', boom)
    monkeypatch.setattr(mod, 'player_stat_beliefs', lambda _p: None)
    p = _bare_ddqn_player()
    order = mod.DDQNPlayer.choose_move(p, _double_battle())
    assert isinstance(order, DefaultBattleOrder)
    assert p.decision_stats.exceptions == 1
    assert p.decision_stats.by_error == {'RuntimeError': 1}


def test_ddqn_player_strict_mode_propagates(monkeypatch):
    from showdown.rl.ddqn import player as mod

    monkeypatch.setenv('PS_RL_STRICT', '1')
    monkeypatch.setattr(mod, 'snapshot_from_battle',
                        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError('snapshot roto')))
    monkeypatch.setattr(mod, 'player_stat_beliefs', lambda _p: None)
    with pytest.raises(RuntimeError):
        mod.DDQNPlayer.choose_move(_bare_ddqn_player(), _double_battle())


# --------------------------------------------------------------------------- B
@pytest.mark.parametrize('g', range(1, 8))
def test_live_validation_mirror_matches_symmetry(g):
    """El espejado de la validación en vivo coincide con la permutación de la obs."""
    from showdown.rl import symmetry as SYM
    from showdown.rl.live_validation import mirror_snapshot
    from tests.test_rl_symmetry import _asym_snap
    s = _asym_snap()
    base = build_observation(s)
    perm = SYM.augment_obs(base[None], np.array([g]))[0]
    assert np.allclose(perm, build_observation(mirror_snapshot(s, g)), atol=1e-6)
    assert build_observation(s) is not None and s.our.active[0].hp_pct == 41.0  # no muta el original


def test_last_moves_skips_called_moves_and_counts_turns():
    from showdown.rl.live_validation import _last_moves

    class B:
        _replay_data = [
            ['', 'turn', '1'],
            ['', 'move', 'p1a: Incineroar', 'Fake Out', 'p2a: Garchomp'],
            ['', 'move', 'p2b: Oranguru', 'Instruct', 'p2a: Garchomp'],
            ['', 'move', 'p2a: Garchomp', 'Earthquake', '', '[from]move: Instruct'],
            ['', 'turn', '2'],
            ['', 'move', 'p2b: Oranguru', 'Psychic', 'p1a: Incineroar'],  # antes de un cambio forzado
        ]
    assert _last_moves(B()) == {'p1: Incineroar': ('fakeout', 1), 'p2: Oranguru': ('psychic', 0)}
