"""Registro del contraste de estado de los rivales heurísticos en el entrenamiento (16-11-2026)."""
import sys
from types import SimpleNamespace

sys.path.insert(0, '.')

from showdown.inference_sim.battle_state_check import StateCheckReport, StateMismatch
from showdown.rl.train_metrics import EpisodeTracker, FoeStateTally


def _backend_with(report):
    from showdown.rl.pokeenv_backend import PokeEnvBattleBackend
    be = PokeEnvBattleBackend.__new__(PokeEnvBattleBackend)
    be._foe = SimpleNamespace(state_report=report)
    be._fs_base = None
    return be


def test_backend_counts_per_episode_and_samples():
    rep = StateCheckReport()
    be = _backend_with(rep)
    rep.checks = 5
    be._fs_base = be._foe_state_counts()           # reset del episodio
    rep.checks += 3
    rep.add(StateMismatch('b1', 4, 'weather', 'raindance', 'none'))
    fs = be.foe_state_episode()
    assert fs['checks'] == 3 and fs['mismatches'] == 1 and fs['feed_errors'] == 0
    assert 'weather' in fs['samples'][0]


def test_backend_handles_rebuilt_foe_and_no_mirror():
    rep = StateCheckReport()
    be = _backend_with(rep)
    be._fs_base = (50, 0, 0, 0)                     # reporte anterior más grande
    rep.checks = 2
    assert be.foe_state_episode()['checks'] == 2
    be._foe = SimpleNamespace()                      # rival random: sin espejo
    assert be.foe_state_episode() is None


def test_tally_flush_by_opponent_and_warns(capsys):
    t = FoeStateTally()
    t.add({'checks': 10, 'mismatches': 0, 'feed_errors': 0}, 'greedy_joint')
    t.add({'checks': 8, 'mismatches': 2, 'feed_errors': 0, 'samples': ['hp: x']}, 'expectimax_bayesian')
    t.add(None, 'random')
    assert 'AVISO' in capsys.readouterr().out
    out = t.flush()
    assert out['by_opp']['greedy_joint']['checks'] == 10
    assert out['by_opp']['expectimax_bayesian']['mismatches'] == 2
    assert out['total']['episodes'] == 2 and abs(out['total']['mismatch_rate'] - 2 / 18) < 1e-9
    assert out['samples'] == ['expectimax_bayesian: hp: x']
    again = t.flush()
    assert again['by_opp'] == {} and again['total']['checks'] == 18   # ventana se reinicia, total no


def test_tracker_records_foe_state_in_episode():
    import showdown.rl.train_metrics as tm
    tm.FOE_STATE = FoeStateTally()
    tr = EpisodeTracker(None)
    tr.start(0, 'greedy_joint')
    rec = tr.end(0, global_step=1, terminated=True, truncated=False, components={'d_win': 1.0},
                 foe_state={'checks': 7, 'mismatches': 0, 'feed_errors': 0})
    assert rec['foe_state'] == {'checks': 7, 'mismatches': 0, 'feed_errors': 0}
    assert tm.FOE_STATE.total['checks'] == 7


def test_lite_info_passes_foe_state():
    from showdown.rl.vec_env import _lite_info
    lite = _lite_info({'engine': {'used_fallback': False, 'foe_state': {'checks': 3}}})
    assert lite['foe_state'] == {'checks': 3}
    assert _lite_info({'engine': 'mock'})['foe_state'] is None
