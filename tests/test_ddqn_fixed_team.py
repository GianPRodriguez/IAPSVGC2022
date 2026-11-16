"""Equipo fijo del agente DDQN (ddqn_spec 1.14): el agente siempre juega con el mismo equipo
y el rival sortea el suyo del grupo `train` en cada reinicio."""
import sys

import numpy as np
import pytest

sys.path.insert(0, '.')

from showdown.rl.ddqn.train import DEFAULT_AGENT_TEAM


def _team():
    from showdown.inference_sim.team_factory import load_team_file
    return load_team_file(DEFAULT_AGENT_TEAM)


def test_fixed_team_is_valid_and_complete():
    from showdown.inference_sim.team_factory import SP_BUDGET, team_passes_item_clause
    t = _team()
    assert [m.species for m in t.mons] == ['Raichu', 'Rillaboom', 'Gholdengo', 'Arcanine-Hisui',
                                          'Staraptor', 'Primarina']
    assert all(sum(m.sp.values()) == SP_BUDGET and max(m.sp.values()) <= 32 for m in t.mons)
    assert all(len(m.moves) == 4 for m in t.mons)
    assert team_passes_item_clause(t)
    assert t.mons[0].item == 'Raichunite Y' and t.mons[4].item == 'Staraptite'


def test_fixed_team_species_match_poke_env_preview():
    from poke_env.battle.pokemon import Pokemon
    from showdown.inference_sim.team_factory import canonicalize_species
    for m in _team().mons:
        name = canonicalize_species(m.species)
        assert Pokemon(gen=9, species=name).identifies_as(name)


def test_team_from_export_rejects_bad_teams():
    from showdown.inference_sim.team_factory import team_from_export
    good = open(DEFAULT_AGENT_TEAM, encoding='utf-8').read()
    with pytest.raises(ValueError):
        team_from_export(good.split('\n\n', 1)[1])                         # solo 5 Pokémon
    with pytest.raises(ValueError):
        team_from_export(good.replace('32 HP / 1 Def', '32 HP / 40 Def'))  # > 32 SP en un stat
    with pytest.raises(ValueError):
        team_from_export(good.replace('- Protect\n- Fake Out', '- Protect', 1))  # 3 movimientos


def test_envspec_agent_team_reaches_backend_as_teammeta():
    """Llega como TeamMeta (no como texto): si no, los rivales validados caerían al crudo."""
    from showdown.inference_sim.team_factory import TeamMeta
    from showdown.rl import pokeenv_backend as pb
    from showdown.rl.vec_env import EnvSpec, make_env
    env = make_env(EnvSpec(opponent='greedy_joint', agent_team=DEFAULT_AGENT_TEAM))
    our = env.backend._our_team_arg
    assert isinstance(our, TeamMeta) and pb._team_meta(our) is our
    assert env.backend._opp_team_arg is None             # el rival se sortea del grupo train
    assert make_env(EnvSpec(opponent='greedy_joint')).backend._our_team_arg is None


def test_cli_defaults():
    import inspect
    from showdown.rl.ddqn import train as T
    assert T.DEFAULT_AGENT_TEAM.endswith('ddqn_team.txt')
    assert inspect.signature(T.run_training).parameters['agent_team'].default == T.DEFAULT_AGENT_TEAM


def test_fixed_team_keeps_foes_validated(monkeypatch):
    from showdown.inference_sim import validated_greedy_player as vgp
    from showdown.rl import pokeenv_backend as pb

    class Fake:
        def __init__(self, **kw):
            self.kw = kw
    monkeypatch.delenv('PS_RL_FOE_VALIDATED', raising=False)
    monkeypatch.setattr(vgp, 'validated_player_class', lambda kind: Fake)
    foe_meta, _ = pb._default_team_metas(seed=2)
    foe = pb._make_foe_player('greedy_joint', account_configuration=None, battle_format='x',
                              team='t', server_configuration=None, log_level=40,
                              foe_meta=foe_meta, agent_meta=_team())
    assert isinstance(foe, Fake) and foe.kw['opp_team'].name == 'ddqn_team'


def test_foe_team_changes_every_reset_and_comes_from_train(monkeypatch):
    """El rival cambia de equipo en cada reinicio (semilla distinta por episodio) y sale del
    grupo `train`; el equipo del agente no entra en el sorteo."""
    from showdown.inference_sim.team_split import split_keys, team_key
    from showdown.rl import pokeenv_backend as pb
    monkeypatch.delenv('PS_TEAM_SPLIT', raising=False)
    train = split_keys('train')
    seen = set()
    for ep in range(40):
        seed = 5 + 7919 * ep                      # como EnvRunner.reset
        _d0, d1 = pb._default_team_metas(seed=seed)
        assert team_key(d1) in train
        seen.add(team_key(d1))
    assert len(seen) >= 25                        # cambia de equipo casi en cada reinicio
