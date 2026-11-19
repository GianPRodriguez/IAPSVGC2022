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
