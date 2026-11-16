"""Opciones de diagnóstico del 16-11-2026: máscara de ataques de daño al aliado y recompensa simétrica."""
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, '.')

from showdown.rl.constants import MEGA_BASE, SLOT_ACTION_N
from showdown.rl.pe_adapter import _filter_ally_damage


def _move(mid, bp, mtype):
    return SimpleNamespace(id=mid, base_power=bp, type=SimpleNamespace(name=mtype))


def _battle(moves, ally_ability):
    me = SimpleNamespace(ability='intimidate')
    ally = SimpleNamespace(ability=ally_ability)
    return SimpleNamespace(available_moves=[moves, moves], active_pokemon=[me, ally])


MOVES = [_move('closecombat', 120, 'FIGHTING'), _move('helpinghand', 0, 'NORMAL'),
         _move('pollenpuff', 90, 'BUG'), _move('thunderbolt', 90, 'ELECTRIC')]


def test_off_by_default_changes_nothing(monkeypatch):
    monkeypatch.delenv('RL_MASK_ALLY_DAMAGE', raising=False)
    m = np.ones(SLOT_ACTION_N, dtype=np.float32)
    assert np.array_equal(_filter_ally_damage(_battle(MOVES, 'intimidate'), 0, m), m)


def test_blocks_damaging_ally_targets_but_keeps_support(monkeypatch):
    monkeypatch.setenv('RL_MASK_ALLY_DAMAGE', '1')
    m = np.ones(SLOT_ACTION_N, dtype=np.float32)
    out = _filter_ally_damage(_battle(MOVES, 'intimidate'), 0, m)
    ally = lambda mi, base=0: out[base + mi * 4 + 2]
    assert ally(0) == 0 and ally(0, MEGA_BASE) == 0      # Close Combat al aliado: bloqueado
    assert ally(1) == 1                                   # Helping Hand (estado): permitido
    assert ally(2) == 1                                   # Pollen Puff (cura al aliado): permitido
    assert ally(3) == 0                                   # Thunderbolt sin absorción: bloqueado
    assert out[0 * 4 + 0] == 1 and out[0 * 4 + 1] == 1    # objetivos rivales intactos


def test_absorbing_ally_ability_keeps_matching_type(monkeypatch):
    monkeypatch.setenv('RL_MASK_ALLY_DAMAGE', '1')
    out = _filter_ally_damage(_battle(MOVES, 'Lightning Rod'), 0, np.ones(SLOT_ACTION_N, dtype=np.float32))
    assert out[3 * 4 + 2] == 1                            # Thunderbolt a un aliado con Lightning Rod
    assert out[0 * 4 + 2] == 0


def _mat(faint, hp):
    from showdown.rl.snapshot import TeamMaterial
    return TeamMaterial(faint_count=faint, hp_pct_total=hp)


def _reward():
    from showdown.rl.reward import RewardCandidate, compute_reward
    return compute_reward(candidate=RewardCandidate.DENSE, prev_our=_mat(0, 400.0), prev_foe=_mat(0, 400.0),
                          cur_our=_mat(0, 300.0), cur_foe=_mat(0, 350.0), finished=False, won=None, tied=False)


def test_reward_default_ignores_own_hp_loss(monkeypatch):
    monkeypatch.delenv('RL_REWARD_HP_SELF', raising=False)
    r = _reward()
    assert abs(r.reward - 0.05 * 0.5) < 1e-9 and r.components['d_hp_self'] == 1.0


def test_reward_symmetric_variant_subtracts_own_hp_loss(monkeypatch):
    monkeypatch.setenv('RL_REWARD_HP_SELF', '1')
    r = _reward()
    assert abs(r.reward - (0.05 * 0.5 - 0.05 * 1.0)) < 1e-9
