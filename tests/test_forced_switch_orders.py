"""Cambios forzados: el hueco que NO cambia debe enviar `pass`.

Con `default` Showdown lo toma como otro cambio y responde
"[Invalid choice] Can't switch: You sent more switches than Pokémon that need to
switch". poke-env reintenta entonces la misma orden (el bot es determinista) y
solo sale del bucle cuando ``random.random() < 1/1000`` acepta la orden por
defecto: miles de reintentos por cambio forzado, partidas colgadas y resultados
no reproducibles.
"""
import sys
from types import SimpleNamespace

sys.path.insert(0, '.')

import numpy as np
import pytest
from poke_env.player.battle_order import (
    DefaultBattleOrder,
    DoubleBattleOrder,
    PassBattleOrder,
    SingleBattleOrder,
)


class _Mon:
    """Doble mínimo de un Pokémon de poke-env (hashable, como el real)."""

    def __init__(self, name, fainted=False):
        self.species = name
        self.name = name
        self.fainted = fainted
        self.types = (SimpleNamespace(name='NORMAL'),)
        self.current_hp_fraction = 0.0 if fainted else 1.0


def _mon(name, *, fainted=False):
    return _Mon(name, fainted)


def _battle(force_switch, *, reserve):
    foe = _mon('foe')
    return SimpleNamespace(
        force_switch=list(force_switch),
        active_pokemon=[_mon('left', fainted=force_switch[0]),
                        _mon('right', fainted=force_switch[1])],
        opponent_active_pokemon=[foe, None],
        available_switches=[list(reserve), list(reserve)],
    )


def _player(kind):
    if kind == 'joint':
        from showdown.battle_bots.greedy_joint.pokeenv_player import GreedyJointPlayer as cls
    else:
        from showdown.battle_bots.greedy_solo.pokeenv_player import GreedyDamagePlayer as cls
    return cls.__new__(cls)  # sin conexión: solo se usa la lógica de la orden


def _orders(order):
    assert isinstance(order, DoubleBattleOrder)
    return order.first_order, order.second_order


@pytest.mark.parametrize('kind', ['joint', 'solo'])
def test_only_left_slot_switches_right_gets_pass(kind):
    battle = _battle([True, False], reserve=[_mon('bench1'), _mon('bench2')])
    first, second = _orders(_player(kind)._forced_switch_order(battle))
    assert isinstance(first, SingleBattleOrder) and not isinstance(first, (PassBattleOrder, DefaultBattleOrder))
    assert isinstance(second, PassBattleOrder)
    assert not isinstance(second, DefaultBattleOrder)
    assert second.message.endswith('pass')


@pytest.mark.parametrize('kind', ['joint', 'solo'])
def test_only_right_slot_switches_left_gets_pass(kind):
    battle = _battle([False, True], reserve=[_mon('bench1')])
    first, second = _orders(_player(kind)._forced_switch_order(battle))
    assert isinstance(first, PassBattleOrder)
    assert isinstance(second, SingleBattleOrder) and not isinstance(second, PassBattleOrder)


@pytest.mark.parametrize('kind', ['joint', 'solo'])
def test_both_slots_switch_to_different_mons(kind):
    battle = _battle([True, True], reserve=[_mon('bench1'), _mon('bench2')])
    first, second = _orders(_player(kind)._forced_switch_order(battle))
    assert isinstance(first, SingleBattleOrder) and isinstance(second, SingleBattleOrder)
    assert first.order is not second.order  # nunca el mismo reserva dos veces


@pytest.mark.parametrize('kind', ['joint', 'solo'])
def test_no_reserve_left_sends_pass_not_default(kind):
    battle = _battle([True, False], reserve=[])
    first, second = _orders(_player(kind)._forced_switch_order(battle))
    assert isinstance(first, PassBattleOrder) and isinstance(second, PassBattleOrder)


def test_expectimax_inherits_the_fixed_forced_switch():
    from showdown.battle_bots.expectimax_bayesian.pokeenv_player import ExpectimaxBayesianPlayer
    from showdown.battle_bots.greedy_joint.pokeenv_player import GreedyJointPlayer

    assert ExpectimaxBayesianPlayer._forced_switch_order is GreedyJointPlayer._forced_switch_order


# --------------------------------------------------------------------------- #
# Máscara de RL: durante un cambio forzado ajeno solo PASS
# --------------------------------------------------------------------------- #

def test_rl_slot_mask_only_pass_for_the_slot_that_does_not_switch():
    from showdown.rl.constants import PASS_ID
    from showdown.rl.pe_adapter import pe_slot_mask

    move = SimpleNamespace(id='tackle')
    battle = SimpleNamespace(
        force_switch=[True, False],
        available_switches=[[_mon('bench1')], []],
        available_moves=[[], [move]],
        can_mega_evolve=[False, True],
        can_tera=[False, False],
    )
    forced = pe_slot_mask(battle, 0)
    assert forced[PASS_ID] == 1.0 and forced[16] == 1.0       # puede cambiar
    idle = pe_slot_mask(battle, 1)
    assert idle[PASS_ID] == 1.0
    assert idle.sum() == 1.0                                  # solo PASS: sin movimientos ni mega


def test_rl_slot_mask_unchanged_outside_forced_switch():
    from showdown.rl.constants import PASS_ID
    from showdown.rl.pe_adapter import pe_slot_mask

    move = SimpleNamespace(id='tackle')
    battle = SimpleNamespace(
        force_switch=[False, False],
        available_switches=[[_mon('bench1')], [_mon('bench1')]],
        available_moves=[[move], [move]],
        can_mega_evolve=[False, False],
        can_tera=[False, False],
    )
    m = pe_slot_mask(battle, 1)
    assert m[PASS_ID] == 1.0 and m.sum() > 1.0                # movimientos y cambio disponibles


# --------------------------------------------------------------------------- #
# Una sola mega por orden doble
# --------------------------------------------------------------------------- #

def _real_move(move_id):
    from poke_env.battle import Move

    return Move(move_id, gen=9)


def _move_order(move_id, *, mega):
    from poke_env.player.battle_order import SingleBattleOrder

    return SingleBattleOrder(_real_move(move_id), mega=mega, move_target=1)


def test_single_mega_keeps_higher_value_slot():
    from showdown.battle_bots.helpers import enforce_single_mega

    a, b = _move_order('hypervoice', mega=True), _move_order('moonblast', mega=True)
    o1, o2 = enforce_single_mega(a, b, 10.0, 55.0)
    assert (o1.mega, o2.mega) == (False, True)
    o1, o2 = enforce_single_mega(a, b, 80.0, 55.0)
    assert (o1.mega, o2.mega) == (True, False)
    o1, o2 = enforce_single_mega(a, b, 40.0, 40.0)        # empate: se queda el izquierdo
    assert (o1.mega, o2.mega) == (True, False)
    assert o1.message.count('mega') + o2.message.count('mega') == 1


def test_single_mega_leaves_other_cases_untouched():
    from showdown.battle_bots.helpers import enforce_single_mega

    a, b = _move_order('hypervoice', mega=True), _move_order('moonblast', mega=False)
    assert enforce_single_mega(a, b) == (a, b)
    c, d = _move_order('hypervoice', mega=False), _move_order('moonblast', mega=False)
    assert enforce_single_mega(c, d) == (c, d)
    assert enforce_single_mega(PassBattleOrder(), a) == (PassBattleOrder(), a)


def test_rl_mask_blocks_second_mega_after_first_slot_megas():
    from showdown.rl.action_space import SlotAction
    from showdown.rl.pe_adapter import pe_slot_mask

    move = SimpleNamespace(id='tackle')
    battle = SimpleNamespace(
        force_switch=[False, False],
        available_switches=[[], []],
        available_moves=[[move], [move]],
        can_mega_evolve=[True, True],
        can_tera=[False, False],
    )
    assert pe_slot_mask(battle, 1)[21] == 1.0                              # sin mega previa
    assert pe_slot_mask(battle, 1, other=SlotAction('mega'))[21] == 0.0    # el otro ya la usa
    assert pe_slot_mask(battle, 1, other=SlotAction('move'))[21] == 1.0


def test_rl_joint_order_never_has_two_megas():
    from poke_env.player.battle_order import DoubleBattleOrder

    from showdown.rl.pe_adapter import joint_order_from_spec_actions

    battle = SimpleNamespace(
        force_switch=[False, False],
        available_switches=[[], []],
        available_moves=[[_real_move('tackle')], [_real_move('growl')]],
        can_mega_evolve=[True, True],
        can_tera=[False, False],
        valid_orders=None,
    )
    from showdown.rl.action_space import encode_slot_action

    mega = encode_slot_action(kind='mega')
    order, _fallback = joint_order_from_spec_actions(battle, (mega, mega))
    assert isinstance(order, DoubleBattleOrder)
    assert order.message.count('mega') <= 1


# --------------------------------------------------------------------------- #
# BO3: el guard de salas no debe tragarse /confirmready de la sala padre
# --------------------------------------------------------------------------- #

def _guarded_client():
    import asyncio

    from showdown.inference_sim.stale_room_guard import StaleRoomGuardMixin

    sent = []

    class _Client:
        logger = None

        async def send_message(self, message, room='', message_2=None):
            sent.append((message, room))

    class _Player(StaleRoomGuardMixin):
        def __init__(self):
            self._left_rooms = set()
            self.battles = {}
            self.ps_client = _Client()

    p = _Player()
    p._install_stale_room_guards()
    return p, sent, asyncio


def test_guard_forwards_confirmready_to_the_bestof_room():
    p, sent, asyncio = _guarded_client()
    room = 'game-bestof3-gen9championsvgc2026regmcbo3-123'
    asyncio.run(p.ps_client.send_message('/confirmready', room=room))
    assert sent == [('/confirmready', room)]
    assert not p._left_rooms          # la sala padre no se marca como abandonada


def test_guard_still_drops_orders_for_unknown_battle_rooms():
    p, sent, asyncio = _guarded_client()
    asyncio.run(p.ps_client.send_message('/choose move 1', room='battle-gen9-999'))
    assert sent == []                 # sala fantasma: se descarta, como antes
