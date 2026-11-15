"""Prior meta leave-one-out: el prior no debe contar el set del equipo rival."""
import json
import sys
from collections import Counter
from types import SimpleNamespace

sys.path.insert(0, '.')

import pytest

from showdown import meta_sp_prior as msp
from showdown import stat_inference as si
from showdown.inference_sim.validated_greedy_player import apply_prior_leave_one_out

_FULL = {s: set(range(33)) for s in si.STAT_KEYS}


def _meta_mons():
    raw = json.loads(msp._JSON_PATH.read_text(encoding='utf-8'))
    return [m for t in raw for m in t['mons']]


def _unique_mon():
    """Un set cuya combinación especie+naturaleza aparece una sola vez en el compendio."""
    mons = _meta_mons()
    key = lambda m: (msp._norm(m['species']), msp._norm(m['nature']) or 'serious')
    counts = Counter(key(m) for m in mons)
    for m in mons:
        if counts[key(m)] == 1:
            return m
    pytest.skip('compendio sin combinaciones únicas')


def _team(*mons):
    return SimpleNamespace(mons=[
        SimpleNamespace(species=m['species'], nature=m['nature'], sp=dict(m['sp'])) for m in mons
    ])


def _sp(m):
    return tuple(int(m['sp'][s]) for s in si.STAT_KEYS)


def test_unique_set_is_leaked_without_loo_and_hidden_with_it():
    m = _unique_mon()
    kw = dict(species=m['species'], nature=m['nature'], candidates=_FULL)
    leaked = msp.preferred_sp_map(**kw)
    assert tuple(leaked[s] for s in si.STAT_KEYS) == _sp(m)  # sin LOO: devuelve el set real
    joint = msp._compatible_spreads(msp._norm(m['species']), msp._norm(m['nature']), _FULL,
                                    msp.exclusion_from_team(_team(m)))
    assert joint == []  # con LOO ya no queda ningún set observado de esa combinación
    loo = msp.preferred_sp_map(**kw, exclude=msp.exclusion_from_team(_team(m)))
    assert set(loo) == set(si.STAT_KEYS)  # cae a los marginales, no falla


def test_loo_only_discounts_the_excluded_copies():
    m = _unique_mon()
    other = next(x for x in _meta_mons() if msp._norm(x['species']) != msp._norm(m['species']))
    ex = msp.exclusion_from_team(_team(m))
    assert msp.preferred_sp_map(species=other['species'], nature=other['nature'], candidates=_FULL) \
        == msp.preferred_sp_map(species=other['species'], nature=other['nature'], candidates=_FULL,
                                exclude=ex)


def test_loo_lowers_speed_accuracy_over_whole_pool():
    """Medida global: el prior honesto acierta menos la Spe real que el prior con filtración."""
    mons = _meta_mons()
    hit_leak = hit_loo = 0
    for m in mons:
        kw = dict(species=m['species'], nature=m['nature'], candidates=_FULL)
        hit_leak += msp.preferred_sp_map(**kw)['spe'] == m['sp']['spe']
        hit_loo += msp.preferred_sp_map(**kw, exclude=msp.exclusion_from_team(_team(m)))['spe'] \
            == m['sp']['spe']
    assert hit_loo < hit_leak
    assert hit_loo > 0.5 * len(mons)  # el meta sigue siendo informativo


def test_stat_beliefs_use_exclusion_for_existing_and_new_beliefs():
    si.reset_sp_limits()
    m = _unique_mon()
    beliefs = si.TeamStatBeliefs()
    iap = SimpleNamespace(stat_beliefs=beliefs)
    pre = si.StatBelief(msp._norm(m['species']), {s: 100 for s in si.STAT_KEYS}, m['nature'])
    beliefs.beliefs[pre.species] = pre
    assert apply_prior_leave_one_out(iap, _team(m)) is True
    assert pre.prior_exclude == msp.exclusion_from_team(_team(m))
    assert msp._compatible_spreads(pre.species, msp._norm(m['nature']), pre.candidates,
                                   pre.prior_exclude) == []


def test_loo_can_be_disabled(monkeypatch):
    beliefs = si.TeamStatBeliefs()
    monkeypatch.setenv('PS_PRIOR_LOO', '0')
    assert apply_prior_leave_one_out(SimpleNamespace(stat_beliefs=beliefs), _team(_unique_mon())) is False
    assert beliefs.prior_exclude is None


def test_expectimax_worlds_exclude_opponent_set():
    """Sin LOO el set real del rival es un mundo meta; con LOO ya no sale del compendio."""
    from showdown.battle_bots.expectimax_bayesian.belief import sample_belief_node

    m = _unique_mon()
    item = msp._norm(m.get('item'))
    kw = dict(species=m['species'], nature=m['nature'], candidates=None, k=64)
    from_meta = lambda node: {(p.sp, msp._norm(p.item)) for p in node.profiles}
    assert (_sp(m), item) in from_meta(sample_belief_node(**kw))
    honest = sample_belief_node(**kw, exclude=msp.exclusion_from_team(_team(m)))
    assert (_sp(m), item) not in from_meta(honest)
    assert honest.profiles  # el fallback (marginal del prior) sigue dando algún mundo
