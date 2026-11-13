"""Validación Reg M-C de equipos: caché por proceso y contrato del validador."""
import sys

sys.path.insert(0, '.')

import pytest

from showdown.inference_sim import validate_regmc_teams as vr
from showdown.inference_sim.paths import POKEMON_SHOWDOWN_ROOT
from showdown.inference_sim.team_factory import load_scraped_teams, to_showdown_export


@pytest.fixture(autouse=True)
def _clean_cache():
    vr.clear_validation_cache()
    yield
    vr.clear_validation_cache()


def test_validation_is_cached_per_export(monkeypatch):
    teams = load_scraped_teams()[:4]
    calls = []

    def fake_validate(rows, *, format_id):
        calls.append(len(rows))
        return [{'name': r['name'], 'ok': i != 1, 'problems': [] if i != 1 else ['malo']}
                for i, r in enumerate(rows)]

    monkeypatch.setattr(vr, 'validate_exports', fake_validate)
    ok, bad = vr.validate_team_metas(teams, format_id='fmt')
    assert len(ok) == 3 and len(bad) == 1 and bad[0][1] == ['malo']
    # Segunda llamada (p. ej. el siguiente reset de batalla): no vuelve a node.
    ok2, bad2 = vr.validate_team_metas(teams, format_id='fmt')
    assert calls == [4]
    assert [t.name for t in ok2] == [t.name for t in ok]
    assert [t.name for t, _ in bad2] == [t.name for t, _ in bad]
    # Otro formato sí se valida aparte.
    vr.validate_team_metas(teams, format_id='otro')
    assert calls == [4, 4]


def test_missing_result_marks_team_invalid(monkeypatch):
    teams = load_scraped_teams()[:2]
    monkeypatch.setattr(vr, 'validate_exports', lambda rows, *, format_id: [])
    ok, bad = vr.validate_team_metas(teams, format_id='fmt')
    assert ok == [] and all(p == ['missing validation result'] for _t, p in bad)


def test_batch_script_ships_with_repo():
    assert vr._BATCH_JS.is_file()
    assert vr._ensure_batch_script() == vr._BATCH_JS


_PS_BUILT = (POKEMON_SHOWDOWN_ROOT / 'dist' / 'sim' / 'team-validator.js').is_file()


@pytest.mark.skipif(not _PS_BUILT, reason='requiere pokemon-showdown compilado (dist/)')
def test_real_validator_accepts_pool_team_and_rejects_illegal_ones():
    good = to_showdown_export(load_scraped_teams()[0])
    blocks = good.strip().split('\n\n')
    dup = '\n\n'.join([blocks[0], blocks[0]] + blocks[2:])
    banned = 'Mewtwo @ Life Orb\nAbility: Pressure\nLevel: 50\n- Psystrike\n\n' + '\n\n'.join(blocks[1:])
    res = {r['name']: r for r in vr.validate_exports([
        {'name': 'ok', 'text': good},
        {'name': 'dup', 'text': dup},
        {'name': 'banned', 'text': banned},
    ])}
    assert res['ok']['ok'] is True and res['ok']['problems'] == []
    assert res['dup']['ok'] is False and 'Species Clause' in res['dup']['problems'][0]
    assert res['banned']['ok'] is False
