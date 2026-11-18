"""Preview por separado (memorizacion_equipos.md §13): formación forzada, registro y resumen."""
import json
from types import SimpleNamespace as NS

import pytest

from showdown.rl import frozen_preview as FP


def test_forced_order_bypasses_network_and_is_logged(monkeypatch):
    pol = FP.FrozenPreviewPolicy(kind='team_preview_ppo', checkpoint='nope.pt')
    pol._loaded = True  # sin red (_chooser None)
    monkeypatch.setattr(FP, 'mark_selected_in_teampreview', lambda b, o: None)
    monkeypatch.setenv('PREVIEW_FORCE_ORDER', '1356')
    assert pol.teampreview(NS()) == '/team 1356' and pol.last_order == (1, 3, 5, 6)
    monkeypatch.setenv('PREVIEW_FORCE_ORDER', '1123')
    with pytest.raises(ValueError):
        pol.teampreview(NS())
    monkeypatch.delenv('PREVIEW_FORCE_ORDER')
    assert pol.teampreview(NS()) == '/team 1234' and pol.last_order == (1, 2, 3, 4)  # respaldo sin red


def _diag(path, wins, forms=None, ctx=None):
    res = {f: {'win': w, 'loss': 200 - w} for f, w in zip(('greedy_solo', 'greedy_joint', 'expectimax_bayesian'), wins)}
    parts = [{'rival': 'greedy_joint', 'res': 'win', 'formacion': f} for f in (forms or [])]
    rep = {'resultados': res, 'partidas': parts}
    if ctx:
        rep['contexto'] = ctx
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rep), encoding='utf-8')


def test_summary20(tmp_path):
    from showdown.rl import kickoff_summary20 as K
    d, runs = tmp_path / 'q20', tmp_path / 'runs'
    for s in (0, 1):
        _diag(d / 'diag' / f'ddqn_s{s}_v2.json', (130, 130, 130), forms=['1356', '1356', '2456'],
              ctx={'turno_1': {'decisiones': 10, 'protect': 1}})
        _diag(d / 'diag' / f'ddqn_s{s}_moda.json', (120, 120, 120))
        _diag(d / 'diag' / f'ddqn_s{s}_1234.json', (80, 80, 80))
        _diag(d / 'diag' / f'pool_ddqn_s{s}_azar.json', (90, 90, 90))
        _diag(d / 'diag' / f'pool_ddqn_s{s}_fijo.json', (100, 100, 100))
        _diag(runs / 'kickoff19' / 'diag' / f'ddqn_s{s}.json', (40, 40, 40))
        _diag(runs / 'kickoff15' / 'diag' / f'R_wd03_s{s}_final_eval.json', (128, 128, 128))
    assert K.main([str(d), str(runs)]) == 0
    md = (d / 'kickoff_summary20.md').read_text(encoding='utf-8')
    assert '| DDQN | 65.0 % | 60.0 % | 40.0 % | +5.0' in md
    assert '| DDQN | 2 | 1356 (67 %) |' in md
    assert '| turno_1 | 20 |' in md
    assert '**45.0 %** (+25.0' in md and '**50.0 %** (-14.0' in md
    assert '| PPO | — |' in md  # sin datos de PPO
