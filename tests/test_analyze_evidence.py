"""analyze_evidence: la regla de acumulación mide riesgo y poder correctamente."""
import sys

sys.path.insert(0, '.')

from showdown.inference_sim import analyze_evidence as ae


def _obs(seq, excluded, *, true_final=100, finals=(98, 99, 100, 101, 102), series=1):
    return {
        'pid': 1, 'series': series, 'seq': seq, 'species': 'x', 'stat': 'atk',
        'percent': False, 'opp_attacks': True, 'true_final': true_final,
        'profile': {'finals': list(finals), 'lo8': list(excluded)},
    }


def _cases(rows):
    return ae.group_by_case(rows, unit='abs', who='opp_attacks')


def test_single_observation_rule_excludes_immediately():
    cases = _cases([_obs(0, [98, 99])])
    r = ae.evaluate_rule(cases, ('lo', 8, 1, 1.0))
    assert r['cases'] == 1 and r['true_excluded'] == 0
    assert abs(r['power'] - 2 / 4) < 1e-9          # 2 de los 4 valores falsos


def test_k2_needs_two_agreeing_observations():
    rows = [_obs(0, [98, 99]), _obs(1, [98])]       # solo 98 repetido
    r = ae.evaluate_rule(_cases(rows), ('lo', 8, 2, 1.0))
    assert abs(r['power'] - 1 / 4) < 1e-9
    # con una sola observación el caso no tiene k=2 observaciones: no se evalúa
    r1 = ae.evaluate_rule(_cases([_obs(0, [98])]), ('lo', 8, 2, 1.0))
    assert r1['cases'] == 0


def test_true_value_excluded_by_outliers_counts_as_risk():
    rows = [_obs(0, [100]), _obs(1, [100]), _obs(2, [])]
    strict = ae.evaluate_rule(_cases(rows), ('lo', 8, 2, 1.0))      # 2 de 3 < 100 %
    assert strict['true_excluded'] == 0
    loose = ae.evaluate_rule(_cases(rows), ('lo', 8, 2, 0.6))       # 2 de 3 >= 60 %
    assert loose['true_excluded'] == 1 and loose['risk'] == 1.0


def test_cases_are_split_by_series_and_species():
    rows = [_obs(0, [98], series=1), _obs(0, [98], series=2)]
    assert len(_cases(rows)) == 2
    rows2 = rows + [dict(_obs(1, [98], series=1), species='y')]
    assert len(_cases(rows2)) == 3


def test_wilson_upper_bound_is_conservative_for_zero_events():
    assert 0.0 < ae.wilson_upper(0, 1000) < 0.005
    assert ae.wilson_upper(0, 10) > ae.wilson_upper(0, 1000)
    assert ae.wilson_upper(5, 5) == 1.0
