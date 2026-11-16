"""Al reanudar, el mejor checkpoint previo no se reemplaza por uno peor (bug visto en la sexta cola)."""
import json
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, '.')


@pytest.mark.parametrize('mod', ['showdown.rl.ddqn.train', 'showdown.rl.ppo.train'])
def test_restore_from_same_dir(tmp_path, mod):
    import importlib
    m = importlib.import_module(mod)
    (tmp_path / 'best.json').write_text(json.dumps({'step': 6000, 'mean_wr': 0.4}), encoding='utf-8')
    (tmp_path / 'best.pt').write_bytes(b'x')
    v = SimpleNamespace(best=None)
    rec = m._restore_best(v, tmp_path, str(tmp_path / 'final.pt'))
    assert rec['mean_wr'] == 0.4 and v.best is rec
    # la lógica de los validadores: 0,375 ya no cuenta como mejora
    assert not (v.best is None or 0.375 > float(v.best['mean_wr']))


@pytest.mark.parametrize('mod', ['showdown.rl.ddqn.train', 'showdown.rl.ppo.train'])
def test_restore_from_source_dir_copies_best(tmp_path, mod):
    import importlib
    m = importlib.import_module(mod)
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    src.mkdir(); dst.mkdir()
    (src / 'best.json').write_text(json.dumps({'step': 6000, 'mean_wr': 0.4}), encoding='utf-8')
    (src / 'best.pt').write_bytes(b'weights')
    v = SimpleNamespace(best=None)
    m._restore_best(v, dst, str(src / 'final.pt'))
    assert v.best['step'] == 6000 and (dst / 'best.pt').read_bytes() == b'weights'


def test_no_previous_best(tmp_path):
    from showdown.rl.ddqn.train import _restore_best
    v = SimpleNamespace(best=None)
    assert _restore_best(v, tmp_path, '') is None and v.best is None
