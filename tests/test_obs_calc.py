"""P5 completo: pedidos a la calculadora real para la observación (showdown/rl/obs_calc.py)."""
from types import SimpleNamespace as NS

import pytest

from showdown.rl import obs_calc as C
from showdown.rl.snapshot import FieldSnapshot, MonSnapshot


def _prof(ability='', types=('Normal',), item='', stats=120):
    st = {k: float(stats) for k in C.STAT_KEYS}
    return NS(ability=ability, types=tuple(types), item=item, smin=dict(st), smax=dict(st))


def test_field_payload_auras_ruin_and_screens():
    f = FieldSnapshot(weather='SunnyDay', terrain='grassyterrain', opp_reflect=True, our_lightscreen=True)
    on = {'fairyaura', 'swordofruin'}
    out = C.field_payload(f, attacker_ours=True, att_ally_prof=_prof('battery'), def_ally_prof=_prof('friendguard'),
                          on_field=on, is_spread=True)
    assert out['weather'] == 'Sun' and out['terrain'] == 'Grassy' and out['isSpread']
    assert out['isFairyAura'] and not out['isDarkAura'] and out['isSwordOfRuin'] and not out['isBeadsOfRuin']
    assert out['defenderSide'] == {'isReflect': True, 'isFriendGuard': True}  # pantalla del rival
    assert out['attackerSide'] == {'isBattery': True}
    out2 = C.field_payload(f, attacker_ours=False, att_ally_prof=None, def_ally_prof=None, on_field=set(),
                           is_spread=False)
    assert out2['defenderSide'] == {'isLightScreen': True} and not out2['isFairyAura']


def test_mega_species_name_and_stats_modes():
    zard = MonSnapshot(species='Charizard', item='charizarditey')
    assert C.calc_species_name(zard, _prof(item='charizarditey')) == 'Charizard-Mega-Y'
    p = NS(smin={k: 100.0 for k in C.STAT_KEYS}, smax={k: 140.0 for k in C.STAT_KEYS})
    assert C._stats(p, 'mid', attacker=True)['atk'] == 120
    assert C._stats(p, 'max', attacker=True)['atk'] == 140 and C._stats(p, 'max', attacker=False)['def'] == 100
    assert C._stats(p, 'min', attacker=True)['atk'] == 100 and C._stats(p, 'min', attacker=False)['def'] == 140


def test_damage_frac_conversion():
    p = _prof(stats=200)
    assert C.damage_frac((50, 70), p, 'mid', 0.9) == pytest.approx(0.5 * 120 * 0.9 / 200)
    assert C.damage_frac((50, 70), p, 'max', 0.9) == pytest.approx(70 / 200)
    assert C.damage_frac((50, 70), p, 'min', 0.9) == pytest.approx(50 / 200)
    assert C.damage_frac(None, p, 'mid', 1.0) is None


def test_real_calc_liquid_voice_in_sun():
    """El caso de la prueba online con la calculadora real: Hyper Voice (Agua por Liquid Voice) en sol."""
    try:
        from showdown.damage_calc_bridge import damage_ranges_batch
    except Exception:
        pytest.skip('sin puente a la calculadora')
    prim = MonSnapshot(species='Primarina', hp_pct=100.0)
    venu = MonSnapshot(species='Venusaur', hp_pct=100.0)
    pp = _prof('liquidvoice', ('Water', 'Fairy'), 'leftovers', 150)
    vp = _prof('chlorophyll', ('Grass', 'Poison'), '', 150)
    reqs = []
    for weather in ('', 'SunnyDay'):
        reqs.append(C.build_request(prim, pp, venu, vp, 'hypervoice', FieldSnapshot(weather=weather),
                                    attacker_ours=True, att_ally_prof=None, def_ally_prof=None, on_field=set(),
                                    is_spread=False, mode='mid'))
    try:
        out = damage_ranges_batch(reqs)
    except Exception as exc:  # pragma: no cover - entorno sin Node
        pytest.skip(f'calculadora no disponible: {exc}')
    if any(r is None for r in out):
        pytest.skip('calculadora no disponible')
    (lo0, hi0), (lo1, hi1) = out
    assert hi1 / hi0 == pytest.approx(0.5, rel=0.05)  # Agua: el sol la reduce a la mitad (no Normal)


# --------------------------------------------------------------------------- integración (parte 2a)
def _snap_from_pool():
    import sys
    sys.path.insert(0, 'tests')
    from test_rl_obs_v2 import _snap
    return _snap()


def test_requests_cover_moves_and_pairs():
    from showdown.rl import observation as O
    snap = _snap_from_pool()
    ours, foes = O._our_slots(snap.our), O._foe_slots(snap.foe)
    reqs = C.requests_for_snapshot(snap, O._profile, ours, foes)
    keys = set(reqs)
    a0, f0 = snap.our.active[0], snap.foe.active[0]
    mv = O._profile(a0).damaging[0]
    for mode in ('mid', 'max', 'min'):
        assert (id(a0), id(f0), mv.mid, mode) in keys
    bench_foe = [m for m in foes[2:] if m is not None][0]
    assert any(k[0] == id(a0) and k[1] == id(bench_foe) and k[3] == 'mid' for k in keys)  # matriz de pares
    assert 80 <= len(reqs) <= 400


def test_observation_uses_calc_table_and_falls_back(monkeypatch):
    """La obs toma el daño de la tabla de la calculadora; sin respuesta usa la aproximación."""
    import numpy as np
    from showdown.rl import observation as O
    snap = _snap_from_pool()
    monkeypatch.setenv('RL_OBS_ABILITIES', '1')
    monkeypatch.setenv('RL_OBS_CALC', '0')
    base = O.build_observation(snap, abilities=True)
    monkeypatch.setenv('RL_OBS_CALC', '1')
    calls = []

    def fake_ranges(reqs):
        calls.append(len(reqs))
        return [(1, 1) for _ in reqs]  # daño mínimo: la obs debe cambiar
    monkeypatch.setattr(C, 'damage_table', lambda reqs, ranges_fn=None: C.__dict__['_real_damage_table'](reqs, fake_ranges))
    O._LAST = (None, None, None)
    alt = O.build_observation(snap, abilities=True)
    assert calls and calls[0] >= 80 and not np.allclose(base, alt)
    st = O.calc_stats()
    assert st['calc_obs'] >= 1


def test_real_calc_observation_end_to_end(monkeypatch):
    """Con la calculadora real (servidor): la obs se arma sin respaldo y sin pedidos fallidos."""
    from showdown.rl import observation as O
    try:
        from showdown.damage_calc_bridge import damage_ranges_batch
        if any(r is None for r in damage_ranges_batch([C.build_request(
                MonSnapshot(species='Pikachu', hp_pct=100.0), _prof(), MonSnapshot(species='Pikachu', hp_pct=100.0),
                _prof(), 'thunderbolt', FieldSnapshot(), attacker_ours=True, att_ally_prof=None, def_ally_prof=None,
                on_field=set(), is_spread=False, mode='mid')])):
            pytest.skip('calculadora no disponible')
    except Exception as exc:
        pytest.skip(f'calculadora no disponible: {exc}')
    monkeypatch.setenv('RL_OBS_ABILITIES', '1')
    monkeypatch.setenv('RL_OBS_CALC', '1')
    before = O.calc_stats()
    O._LAST = (None, None, None)
    obs = O.build_observation(_snap_from_pool(), abilities=True)
    after = O.calc_stats()
    assert obs.shape[0] > 0
    assert after['calc_obs'] == before['calc_obs'] + 1 and after['errors'] == before['errors']
    assert after['missing'] - before['missing'] <= 0.02 * (after['requests'] - before['requests'])


def test_training_tally_aggregates_calc_stats():
    from showdown.rl.train_metrics import FoeStateTally
    t = FoeStateTally()
    t.add({'obs_calc': {'obs': 10, 'calc_obs': 9, 'fallback_obs': 1, 'requests': 2000, 'missing': 4}}, 'random')
    t.add({'checks': 5, 'mismatches': 0, 'feed_errors': 0,
           'obs_calc': {'obs': 10, 'calc_obs': 10, 'fallback_obs': 0, 'requests': 2000, 'missing': 0}}, 'greedy_joint')
    out = t.flush()
    assert out['total']['episodes'] == 1  # el episodio sin espejo no cuenta en el contraste del rival
    assert out['obs_calc']['obs'] == 20 and out['obs_calc']['fallback_rate'] == pytest.approx(0.05)
    assert out['obs_calc']['missing_rate'] == pytest.approx(4 / 4000)


def test_summary26_runs_and_reports_calc_usage(tmp_path):
    import json
    from showdown.rl import kickoff_summary26 as K
    out = tmp_path / 'p5full'
    (out / 'diag').mkdir(parents=True)
    (out / 'ppo_p5_s0.txt').write_text(
        "[ppo] DONE 100.0% | steps 1M/1M | 250 step/s | ETA 0s | elapsed 1h6m\n"
        "[ppo] calculadora real en la observación (P5): {'obs': 1000, 'fallback_rate': 0.001, 'missing_rate': 0.0005}\n",
        encoding='utf-8')
    assert K.main([str(out), str(tmp_path / 'v2'), str(tmp_path / 'v1'), str(tmp_path / 'q15')]) == 0
    md = (out / 'kickoff_summary26.md').read_text(encoding='utf-8')
    assert 'vigesimosexta cola' in md and '1000 obs; respaldo 0.10 %' in md and '250 pasos/s, 1h6m' in md
