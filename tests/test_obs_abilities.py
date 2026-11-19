"""P5: corrección de habilidades en la estimación de daño de la observación (rl_env_spec v1.32)."""
import inspect
from types import SimpleNamespace as NS

import pytest

from showdown.rl import observation as O
from showdown.rl import team_preview_obs as P


def _mv(name, ability='', types=()):
    return P._move_infos([name], ability, types)[0][0]


def _prof(ability, types, item=''):
    st = {k: 120.0 for k in ('hp', 'atk', 'def', 'spa', 'spd', 'spe')}
    return NS(ability=ability, types=tuple(types), item=item, smin=dict(st), smax=dict(st))


def _mon():
    return NS(boosts={}, status=None, item='', hp_pct=100.0, fainted=False)


def _field(weather=''):
    return NS(weather=weather, terrain='', trick_room=False,
              our_reflect=False, our_lightscreen=False, our_auroraveil=False,
              opp_reflect=False, opp_lightscreen=False, opp_auroraveil=False)


def test_liquid_voice_makes_sound_moves_water():
    att = _prof('liquidvoice', ('Water', 'Fairy'))
    hv = _mv('Hyper Voice', 'liquidvoice', ('Water', 'Fairy'))
    assert hv.type == 'Normal'  # el perfil (compartido con el preview) no cambia
    assert O.ability_damage_mods(att, _prof('', ('Grass',)), hv)[0] == 'Water'
    mb = _mv('Moonblast', 'liquidvoice', ('Water', 'Fairy'))
    assert O.ability_damage_mods(att, _prof('', ('Grass',)), mb)[0] == 'Fairy'


def test_power_and_defensive_abilities():
    tc = _prof('toughclaws', ('Normal',))
    contact = _mv('Double-Edge', 'toughclaws', ('Normal',))
    assert O.ability_damage_mods(tc, _prof('', ('Normal',)), contact)[1] == pytest.approx(1.3)
    fire = _mv('Flamethrower')
    assert O.ability_damage_mods(_prof('', ('Fire',)), _prof('thickfat', ('Normal',)), fire)[2] == 0.5
    pb = _mv('Return')
    assert O.ability_damage_mods(_prof('parentalbond', ('Normal',)), _prof('', ('Normal',)), pb)[1] == 1.25


def test_hyper_voice_into_venusaur_in_sun(monkeypatch):
    """El caso de la prueba online: la obs decía ×1 (Normal); real ×0,375 (Agua resistida, con sol)."""
    monkeypatch.delenv('RL_OBS_ABILITIES', raising=False)
    att, dfn_p = _prof('liquidvoice', ('Water', 'Fairy')), _prof('chlorophyll', ('Grass', 'Poison'))
    hv = _mv('Hyper Voice', 'liquidvoice', ('Water', 'Fairy'))
    sun = _field('SunnyDay')
    off = O.move_damage_frac(att, _mon(), dfn_p, _mon(), hv, sun, defender_is_ours=False)
    tok = O._OBS_ABILITIES.set(True)
    try:
        on = O.move_damage_frac(att, _mon(), dfn_p, _mon(), hv, sun, defender_is_ours=False)
    finally:
        O._OBS_ABILITIES.reset(tok)
    assert on / off == pytest.approx(0.375, rel=0.08)


def test_default_off_and_scope(monkeypatch):
    monkeypatch.delenv('RL_OBS_ABILITIES', raising=False)
    assert not O.obs_abilities_on()
    monkeypatch.setenv('RL_OBS_ABILITIES', '1')
    assert O.obs_abilities_on()
    assert 'abilities' in inspect.signature(O.build_observation).parameters
    assert 'abilities' in inspect.signature(O.build_observation_ppo).parameters
    assert 'abilities' in inspect.signature(O.build_observation_ddqn).parameters


def test_ppo_checkpoint_keeps_flag_and_old_ones_are_off(tmp_path):
    import torch
    from showdown.rl.ppo.agent import PPOAgent, PPOConfig
    ag = PPOAgent(PPOConfig(device='cpu'))
    assert ag.cfg.obs_abilities is True  # entrenamientos nuevos: con la corrección
    p = tmp_path / 'new.pt'
    ag.save(p)
    b = PPOAgent(PPOConfig(device='cpu'))
    b.load(p, map_location='cpu')
    assert b.cfg.obs_abilities is True
    payload = torch.load(p, map_location='cpu', weights_only=False)
    payload['config'].pop('obs_abilities')  # como un checkpoint de PPO v1
    old = tmp_path / 'old.pt'
    torch.save(payload, old)
    c = PPOAgent(PPOConfig(device='cpu'))
    c.load(old, map_location='cpu')
    assert c.cfg.obs_abilities is False


def test_ddqn_default_and_old_checkpoint(tmp_path):
    import torch
    from showdown.rl.ddqn.agent import DDQNAgent, DDQNConfig
    ag = DDQNAgent(DDQNConfig(device='cpu', replay_device='cpu', replay_capacity=64))
    assert ag.cfg.obs_abilities is True
    p = tmp_path / 'd.pt'
    ag.save(p)
    payload = torch.load(p, map_location='cpu', weights_only=False)
    payload['config'].pop('obs_abilities')
    torch.save(payload, p)
    b = DDQNAgent(DDQNConfig(device='cpu', replay_device='cpu', replay_capacity=64))
    b.load(p, map_location='cpu')
    assert b.cfg.obs_abilities is False


def test_trainer_flags():
    from showdown.rl.ddqn import train as DT
    from showdown.rl.ppo import train as PT
    for mod in (PT, DT):
        assert 'obs_abilities' in inspect.signature(mod.run_training).parameters
        assert '--obs-abilities' in inspect.getsource(mod)


def test_summary25(tmp_path):
    import json
    from showdown.rl import kickoff_summary25 as K

    def w(p, wins, foes=('greedy_solo', 'greedy_joint', 'expectimax_bayesian')):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({'resultados': {f: {'win': x, 'loss': 200 - x} for f, x in zip(foes, wins)}}))
    for s in (0, 1):
        w(tmp_path / 'v2' / 'diag' / f'v1_s{s}_val.json', (170, 170, 170))
        w(tmp_path / 'p5' / 'diag' / f'ppo_p5_s{s}_val.json', (176, 176, 176))
        w(tmp_path / 'q15' / 'diag' / f'R_wd03_s{s}_final_val.json', (150, 150, 150))
        w(tmp_path / 'p5' / 'diag' / f'ddqn_p5_s{s}_val.json', (140, 140, 140))
    assert K.main([str(tmp_path / 'p5'), str(tmp_path / 'v2'), str(tmp_path / 'v1'), str(tmp_path / 'q15')]) == 0
    md = (tmp_path / 'p5' / 'kickoff_summary25.md').read_text(encoding='utf-8')
    assert '| PPO | 85.0 % | 88.0 % | +3.0' in md and '| ok |' in md
    assert '**revisar: pierde más de 2 puntos**' in md


# --------------------------------------------------------------------------- P5: no daño (v1.33)
def _on(monkeypatch):
    monkeypatch.setenv('RL_OBS_ABILITIES', '1')


def test_priority_grassy_glide_and_gale_wings(monkeypatch):
    dex = P._dex()['moves']
    rill, mon = _prof('grassysurge', ('Grass',)), NS(**{**_mon().__dict__, 'item': ''})
    grassy = NS(**{**_field().__dict__, 'terrain': 'grassyterrain'})
    monkeypatch.delenv('RL_OBS_ABILITIES', raising=False)
    assert O.move_priority(rill, mon, 'grassyglide', dex['grassyglide'], grassy) == 0  # v1: sin cambio
    _on(monkeypatch)
    assert O.move_priority(rill, mon, 'grassyglide', dex['grassyglide'], grassy) == 1
    assert O.move_priority(rill, mon, 'grassyglide', dex['grassyglide'], _field()) == 0
    talon = _prof('galewings', ('Fire', 'Flying'))
    assert O.move_priority(talon, mon, 'bravebird', dex['bravebird'], _field()) == 1
    hurt = NS(**{**mon.__dict__, 'hp_pct': 60.0})
    assert O.move_priority(talon, hurt, 'bravebird', dex['bravebird'], _field()) == 0


def test_speed_iron_ball_and_surge_surfer(monkeypatch):
    f = NS(**{**_field().__dict__, 'our_tailwind': False, 'opp_tailwind': False})
    base = O.effective_speed(_prof('', ('Normal',)), _mon(), f, ours=True)[0]
    _on(monkeypatch)
    assert O.effective_speed(_prof('', ('Normal',), item='ironball'), _mon(), f, ours=True)[0] == base * 0.5
    ef = NS(**{**f.__dict__, 'terrain': 'electricterrain'})
    assert O.effective_speed(_prof('surgesurfer', ('Electric',)), _mon(), ef, ours=True)[0] == base * 2.0


def test_ability_immunities_and_sturdy(monkeypatch):
    _on(monkeypatch)
    assert O.move_blocked_by_ability(_prof('bulletproof', ('Grass',)), _mon(), 'sludgebomb')
    assert O.move_blocked_by_ability(_prof('soundproof', ('Normal',)), _mon(), 'hypervoice')
    assert not O.move_blocked_by_ability(_prof('soundproof', ('Normal',)), _mon(), 'moonblast')
    rise = NS(**{**_mon().__dict__, 'volatiles': {'magnetrise': 1.0}})
    assert O.move_blocked_by_ability(_prof('', ('Steel',)), rise, 'earthquake')
    sturdy = _prof('sturdy', ('Rock',))
    assert O.sturdy_survives(sturdy, _mon(), _mv('Surf'))
    assert not O.sturdy_survives(sturdy, NS(**{**_mon().__dict__, 'hp_pct': 50.0}), _mv('Surf'))
    monkeypatch.delenv('RL_OBS_ABILITIES')
    assert not O.sturdy_survives(sturdy, _mon(), _mv('Surf'))  # v1: sin cambio


# --------------------------------------------------------------------------- P5 parte 2b: preview
def test_preview_move_profile_corrections():
    hv_old = P._move_infos(['Hyper Voice'], 'liquidvoice', ('Water', 'Fairy'), abilities=False)[0][0]
    hv_new = P._move_infos(['Hyper Voice'], 'liquidvoice', ('Water', 'Fairy'), abilities=True)[0][0]
    assert hv_old.type == 'Normal' and hv_new.type == 'Water'
    wb = P._move_infos(['Weather Ball'], 'drought', ('Fire',), abilities=True)[0][0]
    assert wb.type == 'Fire' and wb.power == pytest.approx(100.0)
    assert P._move_infos(['Weather Ball'], '', ('Normal',), abilities=True)[0][0].type == 'Normal'
    de_old = P._move_infos(['Double-Edge'], 'toughclaws', ('Normal',), abilities=False)[0][0]
    de_new = P._move_infos(['Double-Edge'], 'toughclaws', ('Normal',), abilities=True)[0][0]
    assert de_new.power == pytest.approx(de_old.power * 1.3)
    # agregados por tipo del preview: Liquid Voice suma potencia de Agua
    agg_old = P._move_infos(['Hyper Voice'], 'liquidvoice', ('Water', 'Fairy'), abilities=False)[1]
    agg_new = P._move_infos(['Hyper Voice'], 'liquidvoice', ('Water', 'Fairy'), abilities=True)[1]
    wi = P.TYPES.index('Water')
    assert agg_old[wi] == 0.0 and agg_new[wi] > 0.0


def test_profile_cache_is_per_version():
    from showdown.rl.team_preview_obs import mon_feat_from_species, mon_profile, obs_abilities_scope
    feat = mon_feat_from_species('Primarina', ability='Liquid Voice', item='Leftovers',
                                 moves=('Hyper Voice', 'Moonblast'))
    with obs_abilities_scope(False):
        old = {m.mid: m.type for m in mon_profile(feat).damaging}
    with obs_abilities_scope(True):
        new = {m.mid: m.type for m in mon_profile(feat).damaging}
    assert old.get('hypervoice') == 'Normal' and new.get('hypervoice') == 'Water'


def test_preview_checkpoint_version(tmp_path):
    import torch
    from showdown.rl.frozen_preview import FrozenPreviewPolicy
    from showdown.rl.team_preview_ppo.agent import PreviewPPOAgent, PreviewPPOConfig
    ag = PreviewPPOAgent(PreviewPPOConfig())
    ag._ensure_torch()
    assert ag.cfg.obs_abilities is True  # preview v2.1 en adelante
    new = ag.save(tmp_path / 'v21.pt')
    pol = FrozenPreviewPolicy(kind='team_preview_ppo', checkpoint=str(new))
    pol._ensure()
    assert pol.obs_abilities is True
    payload = torch.load(new, map_location='cpu', weights_only=False)
    payload['cfg'].pop('obs_abilities')  # como el preview v2 congelado
    torch.save(payload, tmp_path / 'v2.pt')
    pol2 = FrozenPreviewPolicy(kind='team_preview_ppo', checkpoint=str(tmp_path / 'v2.pt'))
    pol2._ensure()
    assert pol2.obs_abilities is False


def test_preview_trainer_flag():
    from showdown.rl.team_preview_ppo import train as T
    assert 'obs_abilities' in inspect.signature(T.train).parameters and '--obs-abilities' in inspect.getsource(T)
