"""Tolerancia de daño estrecha por movimiento (solo donde hay evidencia medida)."""
import json
import sys

sys.path.insert(0, '.')

import pytest

from showdown import stat_inference as si
from showdown.inference_sim import analyze_calibration as ac


@pytest.fixture
def narrow_file(tmp_path, monkeypatch):
    path = tmp_path / 'tol.json'
    path.write_text(json.dumps({'moves': {
        'moonblast': {'abs': {'opp_attacks': {'lo': 6, 'hi': 4, 'n': 100}},
                      'pct': {'opp_defends': {'lo': 5, 'hi': 9, 'n': 100}}},
    }}), encoding='utf-8')
    monkeypatch.setenv('DAMAGE_NARROW_TOL_FILE', str(path))
    monkeypatch.setenv('DAMAGE_NARROW_TOL', '1')
    si.reset_narrow_tolerance_cache()
    yield path
    si.reset_narrow_tolerance_cache()


def test_narrow_tolerance_lookup_by_move_unit_and_side(narrow_file):
    f = si.narrow_damage_tolerance
    assert f('Moonblast', percent=False, opponent_is_attacker=True, is_ko=False) == (6, 4)
    assert f('moon-blast', percent=True, opponent_is_attacker=False, is_ko=False) == (5, 9)
    # sin entrada para esa combinación -> tolerancia amplia (None)
    assert f('Moonblast', percent=False, opponent_is_attacker=False, is_ko=False) is None
    assert f('Hyper Voice', percent=False, opponent_is_attacker=True, is_ko=False) is None
    # los KO nunca usan la estrecha: el daño observado es solo una cota inferior
    assert f('Moonblast', percent=False, opponent_is_attacker=True, is_ko=True) is None


def test_narrow_tolerance_can_be_disabled(narrow_file, monkeypatch):
    monkeypatch.setenv('DAMAGE_NARROW_TOL', '0')
    si.reset_narrow_tolerance_cache()
    assert si.narrow_damage_tolerance(
        'Moonblast', percent=False, opponent_is_attacker=True, is_ko=False) is None


def test_missing_table_means_legacy_tolerance(tmp_path, monkeypatch):
    monkeypatch.setenv('DAMAGE_NARROW_TOL_FILE', str(tmp_path / 'no_existe.json'))
    si.reset_narrow_tolerance_cache()
    assert si.narrow_damage_tolerance(
        'Moonblast', percent=False, opponent_is_attacker=True, is_ko=False) is None
    si.reset_narrow_tolerance_cache()


def test_absolute_match_uses_given_tolerances_and_defaults_to_legacy():
    m = si._rolls_match_absolute
    # observado 60, calc 20-24: residuo bajo 36 > 28 -> la amplia ya lo descarta
    assert not m(20, 24, 60, bilateral=True)
    # observado 40, calc 20-24: residuo 16: pasa con la amplia, no con la estrecha (6)
    assert m(20, 24, 40, bilateral=False)
    assert not m(20, 24, 40, bilateral=False, tol_lo=6)
    # cota superior: min_roll 50 vs observado 30 (residuo 20): amplia 28 pasa, estrecha 4 no
    assert m(50, 60, 30, bilateral=True)
    assert not m(50, 60, 30, bilateral=True, tol_hi=4)
    # los KO nunca aplican cota superior
    assert m(50, 60, 30, bilateral=True, is_ko=True, tol_hi=4)


def test_percent_match_uses_given_tolerances():
    m = si._rolls_match_percent
    hp = 200
    # observado 50%, calc 20-24 HP (10-12%): residuo bajo 38 puntos
    assert not m(20, 24, hp, 50, bilateral=False)                  # amplia (32) ya lo descarta
    assert m(40, 48, hp, 50, bilateral=False)                      # 20-24% : 26 pts -> pasa
    assert not m(40, 48, hp, 50, bilateral=False, tol_lo=10)       # estrecha: no
    assert m(120, 140, hp, 20, bilateral=True)                     # min 60% vs 20%: amplia 44 pasa
    assert not m(120, 140, hp, 20, bilateral=True, tol_hi=12)


# --------------------------------------------------------------------------- #
# Generación y comprobación de la tabla desde un registro de calibración
# --------------------------------------------------------------------------- #

def _row(move, *, lo=0.0, hi=0.0, percent=False, opp_attacks=True, ko=False, true_in=True):
    return {'move': move, 'res_lo': lo, 'res_hi': hi, 'percent': percent,
            'opp_attacks': opp_attacks, 'ko': ko, 'true_in_groups': true_in}


def test_build_tolerances_only_for_well_measured_moves():
    rows = (
        [_row('Moonblast', lo=2.0, hi=0.0) for _ in range(70)]            # limpio
        + [_row('Eruption', lo=40.0) for _ in range(70)]                  # residuo enorme
        + [_row('Dual Wingbeat', lo=1.0) for _ in range(10)]              # pocas muestras
        + [_row('Close Combat', lo=1.0, ko=True) for _ in range(100)]     # KO: no cuenta
    )
    tol = ac.build_move_tolerances(rows, margin=1.5, min_n=60, cap=14)
    assert set(tol) == {'moonblast'}
    t = tol['moonblast']['abs']['opp_attacks']
    assert t == {'lo': 4, 'hi': 1, 'n': 70}                               # ceil(2*1.5)+1, ceil(0)+1


def test_check_violations_counts_only_covered_rows():
    tol = {'moonblast': {'abs': {'opp_attacks': {'lo': 4, 'hi': 1, 'n': 70}}}}
    ok = [_row('Moonblast', lo=3.0, hi=1.0) for _ in range(5)]
    bad = [_row('Moonblast', lo=9.0)]
    other = [_row('Eruption', lo=99.0)]                                   # no cubierto
    res = ac.check_violations(ok + bad + other, tol)
    assert res['covered'] == 6 and res['violations_lo'] == 1 and res['violations_hi'] == 0
    assert ac.coverage(ok + bad + other, tol)['narrow'] == 6
    assert ac.check_violations(ok, tol)['violations_lo'] == 0


def test_move_key_normalizes_names():
    assert ac.move_key('Dual Wingbeat') == 'dualwingbeat'
    assert ac.move_key("King's Shield") == 'kingsshield'


def test_lo_only_mode_leaves_the_upper_bound_wide():
    rows = [_row('Moonblast', lo=2.0, hi=7.0) for _ in range(70)]
    full = ac.build_move_tolerances(rows, margin=1.5, min_n=60, cap=14)
    assert full['moonblast']['abs']['opp_attacks']['hi'] == 12
    only = ac.build_move_tolerances(rows, margin=1.5, min_n=60, cap=14, lo_only=True)
    t = only['moonblast']['abs']['opp_attacks']
    assert t['lo'] == 4 and t['hi'] is None
    # una violación en la cota superior no cuenta cuando esa cota no se estrecha
    res = ac.check_violations([_row('Moonblast', lo=1.0, hi=99.0)], only)
    assert res['violations_hi'] == 0 and res['covered'] == 1


def test_hi_none_keeps_wide_upper_bound_and_no_attacker_bilateral(tmp_path, monkeypatch):
    path = tmp_path / 'tol.json'
    path.write_text(json.dumps({'moves': {
        'moonblast': {'abs': {'opp_attacks': {'lo': 4, 'hi': None, 'n': 70}}}}}), encoding='utf-8')
    monkeypatch.setenv('DAMAGE_NARROW_TOL_FILE', str(path))
    monkeypatch.setenv('DAMAGE_NARROW_TOL', '1')
    si.reset_narrow_tolerance_cache()
    try:
        assert si.narrow_damage_tolerance(
            'Moonblast', percent=False, opponent_is_attacker=True, is_ko=False) == (4, None)
        # hi=None: el cálculo usa la tolerancia amplia en la cota superior
        assert si._rolls_match_absolute(50, 60, 30, bilateral=True, tol_lo=4, tol_hi=None)
    finally:
        si.reset_narrow_tolerance_cache()
