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
    assert set(SP_INFO_MODES) == {'inference', 'oracle', 'oracle_spe', 'none', 'spe_min', 'spe_max', 'meta'}


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


def test_greedy_agents_default_to_no_inference():
    """Solo y Joint usan C ('meta', sin inferencia) salvo que se fuerce otro modo; expectimax A."""
    from showdown.inference_sim.validated_greedy_player import (
        ValidatedExpectimaxBayesianPlayer,
        ValidatedGreedyJointPlayer,
        ValidatedGreedyPlayer,
    )

    for cls in (ValidatedGreedyPlayer, ValidatedGreedyJointPlayer):
        p = cls.__new__(cls)
        assert p.effective_sp_info_mode() == 'meta'
        p.sp_info_mode = 'inference'  # forzado (p. ej. --sp-info-a inference)
        assert p.effective_sp_info_mode() == 'inference'
        p.sp_info_mode = None
        assert p.effective_sp_info_mode() == 'meta'
    # Expectimax v1.8: también A por defecto.
    ex = ValidatedExpectimaxBayesianPlayer.__new__(ValidatedExpectimaxBayesianPlayer)
    assert ex.effective_sp_info_mode() == 'none'


def test_robust_e2e_unset_mode_means_agent_default(monkeypatch):
    from showdown.inference_sim import run_robust_e2e as r

    monkeypatch.delenv('PS_SP_INFO_A', raising=False)
    monkeypatch.delenv('PS_SP_INFO_B', raising=False)
    monkeypatch.delenv('PS_B_VALIDATED', raising=False)
    assert r.sp_info_mode('A') is None
    assert r.effective_sp_info_label('A', 'greedy_joint') == 'meta(default)'
    assert r.effective_sp_info_label('A', 'greedy_solo') == 'meta(default)'
    assert r.effective_sp_info_label('A', 'expectimax_bayesian') == 'none(default)'
    assert r.effective_sp_info_label('A', 'ddqn') == 'inference(default)'
    # B validated por defecto: mismo predeterminado que su agente.
    assert r.effective_sp_info_label('B', 'greedy_joint') == 'meta(default)'
    assert r.effective_sp_info_label('B', 'random') == 'n/a(sin wrapper)'
    monkeypatch.setenv('PS_B_VALIDATED', '0')  # --b-raw
    assert r.effective_sp_info_label('B', 'greedy_joint') == 'n/a(sin wrapper)'
    monkeypatch.delenv('PS_B_VALIDATED')
    monkeypatch.setenv('PS_SP_INFO_A', 'inference')
    assert r.effective_sp_info_label('A', 'greedy_joint') == 'inference'


def test_b_validated_by_default_and_b_raw_opt_out(monkeypatch):
    from showdown.inference_sim import run_robust_e2e as r

    monkeypatch.delenv('PS_B_VALIDATED', raising=False)
    for kind in ('greedy_solo', 'greedy_joint', 'expectimax_bayesian', 'ddqn', None):
        assert r.b_validated(kind) is True
    assert r.b_validated('random') is False  # random no tiene versión validated
    monkeypatch.setenv('PS_B_VALIDATED', '1')  # --b-validated (compatibilidad)
    assert r.b_validated('expectimax_bayesian') is True
    monkeypatch.setenv('PS_B_VALIDATED', '0')  # --b-raw
    assert r.b_validated('greedy_joint') is False


@pytest.mark.parametrize('mode,expected', [('spe_min', 0), ('spe_max', 32)])
def test_spe_extreme_modes_fix_speed_only(mode, expected):
    """spe_min/spe_max: Spe rival fija en el extremo (también tras mega), resto como A."""
    si.reset_sp_limits()
    beliefs = si.TeamStatBeliefs()
    pre = beliefs.belief_for(_salamence())  # creada antes del modo (como bootstrap)
    apply_sp_info_mode(_iap(beliefs), mode, lambda: None)
    assert beliefs.inference_off is True
    assert pre.candidates['spe'] == {expected}
    assert all(len(pre.candidates[s]) == 33 for s in si.STAT_KEYS if s != 'spe')
    mega = Pokemon('salamence-mega', -1, True, 'salamencite', 'aerilate', None, None, None, None, 'jolly')
    assert beliefs.belief_for(mega).candidates['spe'] == {expected}


def test_preview_override_maps_to_heuristic_previews():
    from showdown.battle_bots.expectimax_bayesian.pokeenv_player import ExpectimaxBayesianPlayer
    from showdown.battle_bots.greedy_solo.pokeenv_player import GreedyDamagePlayer
    from showdown.inference_sim.validated_greedy_player import PREVIEW_CHOICES, preview_function

    assert {'own', 'greedy', 'expectimax'} <= set(PREVIEW_CHOICES)
    assert preview_function(None) is None and preview_function('own') is None
    assert preview_function('greedy') is GreedyDamagePlayer.teampreview
    assert preview_function('expectimax') is ExpectimaxBayesianPlayer.teampreview
    with pytest.raises(ValueError):
        preview_function('nash')


def test_validated_player_uses_preview_override_independent_of_combat(monkeypatch):
    """El combate sigue siendo el del agente; solo el preview cambia."""
    from showdown.inference_sim import validated_greedy_player as vgp

    calls = []
    monkeypatch.setattr(vgp.ExpectimaxBayesianPlayer, 'teampreview',
                        lambda self, battle: calls.append('expectimax') or '/team 1234')
    monkeypatch.setattr(vgp.GreedyJointPlayer, 'teampreview',
                        lambda self, battle: calls.append('own_joint') or '/team 5612')
    p = vgp.ValidatedGreedyJointPlayer.__new__(vgp.ValidatedGreedyJointPlayer)
    assert p.teampreview(None) == '/team 5612'          # propio (joint)
    p.preview_override = 'expectimax'
    assert p.teampreview(None) == '/team 1234'          # preview de expectimax
    assert type(p).choose_move is vgp.ValidatedGreedyJointPlayer.choose_move  # combate intacto
    assert calls == ['own_joint', 'expectimax']


def test_robust_e2e_preview_choice_env(monkeypatch):
    from showdown.inference_sim import run_robust_e2e as r

    monkeypatch.delenv('PS_PREVIEW_A', raising=False)
    assert r.preview_choice('A') is None
    monkeypatch.setenv('PS_PREVIEW_A', 'own')
    assert r.preview_choice('A') is None
    monkeypatch.setenv('PS_PREVIEW_A', 'expectimax')
    assert r.preview_choice('A') == 'expectimax'
    monkeypatch.setenv('PS_PREVIEW_B', 'bad')
    with pytest.raises(ValueError):
        r.preview_choice('B')


def test_preview_diagnostic_variants():
    """Variantes: sin roles / sin calidad de lead / orden de la hoja."""
    import random
    from showdown.inference_sim.team_factory import active_teams
    from showdown.battle_bots.greedy_solo.team_preview import mon_from_species, best_team_preview_order
    from showdown.battle_bots.expectimax_bayesian import team_preview as ep
    from showdown.inference_sim import validated_greedy_player as vgp

    a, b = random.Random(3).sample(active_teams(), 2)
    us = [mon_from_species(i + 1, m.species) for i, m in enumerate(a.mons)]
    fo = [mon_from_species(i + 1, m.species) for i, m in enumerate(b.mons)]
    assert vgp._sheet_order(us, fo) == [1, 2, 3, 4]
    nolead = vgp._PREVIEW_VARIANTS['expectimax_nolead'](us, fo)
    assert list(nolead[:2]) == sorted(nolead)[:2]  # leads = orden de la hoja (como greedy)
    nr = ep.best_expectimax_preview_with_meta(us, fo, role_weight=0.0)[0].order
    assert tuple(vgp._PREVIEW_VARIANTS['expectimax_noroles'](us, fo)) == tuple(nr)
    assert best_team_preview_order(us, fo).order  # greedy intacto
    # Sin roles ni lead y con pesos greedy, la variante reproduce exactamente a greedy.
    same = vgp._expectimax_variant_order(0.0, 0.0, weights=dict(alpha=1.0, beta=1.0, gamma=0.01))(us, fo)
    assert set(same) == set(best_team_preview_order(us, fo).order)
    for k in ('expectimax_noroles', 'expectimax_nolead', 'sheet', 'expectimax_typing', 'greedy_roles25'):
        assert callable(vgp.preview_function(k))


def test_meta_mode_fixes_all_six_stats_to_loo_prior_spread():
    """Opción C: los 6 stats fijos al reparto del prior (con leave-one-out) y también tras mega."""
    from showdown import meta_sp_prior as msp
    from showdown.inference_sim.validated_greedy_player import apply_prior_leave_one_out

    si.reset_sp_limits()
    beliefs = si.TeamStatBeliefs()
    iap = _iap(beliefs)
    pre = beliefs.belief_for(_salamence())  # creada antes (como en bootstrap)
    team = SimpleNamespace(mons=[SimpleNamespace(species='Salamence', nature='Jolly',
                                                 sp={'hp': 0, 'atk': 32, 'def': 0, 'spa': 0,
                                                     'spd': 2, 'spe': 32})])
    apply_prior_leave_one_out(iap, team)
    apply_sp_info_mode(iap, 'meta', lambda: team)
    assert beliefs.inference_off is True
    full = {s: set(range(33)) for s in si.STAT_KEYS}
    want = msp.preferred_sp_map(species=pre.species, nature=pre.nature, candidates=full,
                                exclude=beliefs.prior_exclude)
    assert all(pre.candidates[s] == {want[s]} for s in si.STAT_KEYS)
    assert sum(want.values()) <= 66 + 2  # reparto real del meta, no SP 32 en cinco stats
    mega = Pokemon('salamence-mega', -1, True, 'salamencite', 'aerilate', None, None, None, None, 'jolly')
    assert all(len(beliefs.belief_for(mega).candidates[s]) == 1 for s in si.STAT_KEYS)
