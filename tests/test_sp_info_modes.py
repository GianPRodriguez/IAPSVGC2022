"""Modos de información de SP del rival (diagnóstico oracle / inference / none)."""
import sys
from types import SimpleNamespace

sys.path.insert(0, '.')

import pytest

from showdown import stat_inference as si
from showdown.battle import Pokemon
from showdown.inference_sim.validated_greedy_player import SP_INFO_MODES, apply_sp_info_mode

TRUE_SP = {'hp': 2, 'atk': 32, 'def': 0, 'spa': 0, 'spd': 0, 'spe': 32}


def _salamence():
    return Pokemon('salamence', -1, True, 'salamencite', 'intimidate', None, None, None, None, 'jolly')


def _incineroar():
    return Pokemon('incineroar', -1, True, None, 'intimidate', None, None, None, None, 'jolly')


def _narrowing_setup(belief):
    """Acota Spe a SP 10..20 y devuelve (velocidad propia, SP que deben sobrevivir).

    Desde el rango completo una sola observación de orden nunca estrecha (SP 1 y 2
    dan la misma Vel y las podas grandes se descartan por prudencia); con la creencia
    ya acotada, 'el rival actuó primero' contra Vel(SP k) + margen poda solo SP 10..k-1.
    """
    belief.candidates['spe'] = set(range(10, 21))
    base = int(belief.base_stats['spe'])
    stat = lambda sp: si.champions_stat(base, sp, belief.nature, 'spe')
    k = next(k for k in range(11, 15) if stat(k - 1) < stat(k))
    return stat(k) + si._SPEED_COMPARE_MARGIN, set(range(k, 21))


def _iap(beliefs):
    return SimpleNamespace(stat_beliefs=beliefs)


def test_inference_mode_is_production_noop():
    beliefs = si.TeamStatBeliefs()
    apply_sp_info_mode(_iap(beliefs), 'inference', lambda: None)
    assert beliefs.inference_off is False and beliefs.oracle_lookup is None


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        apply_sp_info_mode(_iap(si.TeamStatBeliefs()), 'perfect', lambda: None)
    assert set(SP_INFO_MODES) == {'inference', 'oracle', 'oracle_spe', 'none'}


def test_none_mode_never_narrows_speed():
    si.reset_sp_limits()
    beliefs = si.TeamStatBeliefs()
    apply_sp_info_mode(_iap(beliefs), 'none', lambda: None)
    mon = _incineroar()
    belief = beliefs.belief_for(mon)
    user_speed, survivors = _narrowing_setup(belief)
    before = set(belief.candidates['spe'])
    changed = beliefs.observe_speed_order(
        mon, si.SpeedContext(), user_speed, si.SpeedContext(), opponent_first=True,
    )
    assert changed is False
    assert beliefs.belief_for(mon).candidates['spe'] == before
    assert len(before) > 1


def test_speed_order_still_narrows_in_inference_mode():
    """Control del test anterior: la misma observación sí estrecha en producción."""
    si.reset_sp_limits()
    beliefs = si.TeamStatBeliefs()
    mon = _incineroar()
    belief = beliefs.belief_for(mon)
    user_speed, survivors = _narrowing_setup(belief)
    before = set(belief.candidates['spe'])
    beliefs.observe_speed_order(
        mon, si.SpeedContext(), user_speed, si.SpeedContext(), opponent_first=True,
    )
    assert beliefs.belief_for(mon).candidates['spe'] == survivors != before


def test_oracle_mode_resolves_true_sp_and_survives_mega(monkeypatch):
    si.reset_sp_limits()
    from showdown.inference_sim import soundness

    monkeypatch.setattr(soundness, '_lookup_truth', lambda team, sid: (dict(TRUE_SP), {}))
    beliefs = si.TeamStatBeliefs()
    apply_sp_info_mode(_iap(beliefs), 'oracle', lambda: object())
    mon = _salamence()
    belief = beliefs.belief_for(mon)
    assert all(belief.candidates[s] == {TRUE_SP[s]} for s in si.STAT_KEYS)
    # Megaevolución: cambian las bases y los stats se reabren; el oráculo los repone.
    mega = Pokemon('salamence-mega', -1, True, 'salamencite', 'aerilate', None, None, None, None, 'jolly')
    belief = beliefs.belief_for(mega)
    assert all(belief.candidates[s] == {TRUE_SP[s]} for s in si.STAT_KEYS)


def test_oracle_mode_applies_to_beliefs_created_before_it(monkeypatch):
    """bootstrap_battle crea las creencias del OTS antes de fijar el modo: deben resolverse igual."""
    si.reset_sp_limits()
    from showdown.inference_sim import soundness

    monkeypatch.setattr(soundness, '_lookup_truth', lambda team, sid: (dict(TRUE_SP), {}))
    beliefs = si.TeamStatBeliefs()
    pre = beliefs.belief_for(_salamence())  # creada sin oráculo
    assert len(pre.candidates['atk']) > 1
    apply_sp_info_mode(_iap(beliefs), 'oracle', lambda: object())
    assert all(pre.candidates[s] == {TRUE_SP[s]} for s in si.STAT_KEYS)


def test_oracle_spe_mode_only_resolves_speed(monkeypatch):
    si.reset_sp_limits()
    from showdown.inference_sim import soundness

    monkeypatch.setattr(soundness, '_lookup_truth', lambda team, sid: (dict(TRUE_SP), {}))
    beliefs = si.TeamStatBeliefs()
    apply_sp_info_mode(_iap(beliefs), 'oracle_spe', lambda: object())
    belief = beliefs.belief_for(_salamence())
    assert belief.candidates['spe'] == {TRUE_SP['spe']}
    assert all(len(belief.candidates[s]) > 1 for s in si.STAT_KEYS if s != 'spe')
