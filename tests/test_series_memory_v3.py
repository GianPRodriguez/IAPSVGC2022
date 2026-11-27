"""PPO v3 / DDQN v2 (protocolo §4.16): memoria de serie BO3, máscara del Protect consecutivo, preview condicionado,
arranque en caliente, simetría ampliada y modo serie del backend."""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from showdown.rl.constants import MEM_DIM, MEM_GLOBAL_DIM, MEM_SLOT_DIM, OBS_DIM, OBS_DIM_MEMORY, obs_dim_for
from showdown.rl.series_memory import GameRecord, SeriesMemory, species_key


def _msgs(text: str):
    return [ln.split('|') for ln in text.strip().split('\n') if ln.startswith('|')]


# Partida sintética: el agente es p2; el rival (p1) abre con Pelipper (lluvia) + Farigiraf, protege en el
# turno 1, activa Trick Room, pone Tailwind, cambia una vez y pierde a Pelipper.
GAME = """
|player|p1|rival|1|
|player|p2|bot|2|
|switch|p1a: Peli|Pelipper, L50, F|100/100
|switch|p1b: Fari|Farigiraf, L50, M|100/100
|switch|p2a: Gholdengo|Gholdengo, L50|189/189
|switch|p2b: Staraptor|Staraptor, L50, M|185/185
|-weather|RainDance|[from] ability: Drizzle|[of] p1a: Peli
|turn|1
|move|p1a: Peli|Protect|p1a: Peli
|move|p1b: Fari|Trick Room|p1b: Fari
|-fieldstart|move: Trick Room|[of] p1b: Fari
|move|p2a: Gholdengo|Shadow Ball|p1b: Fari
|move|p1b: Fari|Psychic|p2b: Staraptor|[from]move: Instruct
|-weather|RainDance|[upkeep]
|turn|2
|switch|p1b: Basculegion|Basculegion, L50, M|100/100
|move|p1a: Peli|Tailwind|p1a: Peli
|-sidestart|p1: rival|move: Tailwind
|move|p1a: Peli|Hurricane|p2a: Gholdengo
|faint|p1a: Peli
|turn|3
|switch|p1a: Archaludon|Archaludon, L50, F|100/100
|turn|4
|win|rival
"""


def test_game_record_parses_foe_actions():
    r = GameRecord.from_messages(_msgs(GAME), our_role='p2', won=False)
    assert r.foe_role == 'p1'
    assert r.brought() == ['archaludon', 'basculegion', 'farigiraf', 'pelipper']
    assert r.leads() == ['farigiraf', 'pelipper']
    assert r.trick_room_turn == 1 and r.tailwind_turn == 2
    assert r.weather == {'rain'} and r.protects == 1
    assert r.switches == 1  # Basculegion entra por Farigiraf; Archaludon es reemplazo tras un KO
    pel = r.mons['pelipper']
    assert pel.fainted and pel.protect_turn1 and pel.protects == 1 and pel.moves == 3 and pel.damaging == 1
    assert r.mons['farigiraf'].moves == 1  # el Psychic llamado por Instruct no cuenta
    assert r.turns == 4 and r.won is False


def test_species_key_strips_mega_forms():
    assert species_key('Charizard-Mega-Y') == species_key('Charizard') == 'charizard'
    assert species_key('Staraptor-Mega') == 'staraptor'
    assert species_key('Floette-Eternal') == 'floetteeternal'
    assert species_key('Meganium') == 'meganium'


def test_series_memory_vector_and_decision():
    mem = SeriesMemory()
    v = mem.vector(['Pelipper', None])
    assert v.shape == (MEM_DIM,) and v[0] == 1.0 and v[1:].sum() == 0.0  # partida 1 sin historial
    mem.add(GameRecord.from_messages(_msgs(GAME), our_role='p2', won=False))
    assert mem.game_index == 2 and mem.foe_wins == 1 and not mem.decided
    v = mem.vector(['Pelipper', 'Farigiraf', 'Kingambit', None, None, None])
    g = v[:MEM_GLOBAL_DIM]
    assert g[1] == 1.0 and g[4] == 0.5 and g[6] == 1.0 and g[8] == 1.0 and g[10] == 1.0 and g[20] == 1.0
    pel = v[MEM_GLOBAL_DIM:MEM_GLOBAL_DIM + MEM_SLOT_DIM]
    assert pel[0] == 1.0 and pel[1] == 1.0 and pel[2] == 1.0 and pel[3] == 1.0 and pel[6] == 1.0
    king = v[MEM_GLOBAL_DIM + 2 * MEM_SLOT_DIM:MEM_GLOBAL_DIM + 3 * MEM_SLOT_DIM]
    assert king[0] == 1.0 and king[1] == 0.0  # hay memoria, pero no lo trajo
    assert v[MEM_GLOBAL_DIM + 3 * MEM_SLOT_DIM:].sum() == 0.0  # bloques vacíos
    mem.add(GameRecord(won=False))
    assert mem.decided and mem.foe_brought() == ['archaludon', 'basculegion', 'farigiraf', 'pelipper']


def _obs_snap():
    from tests.test_rl_obs_v2 import _snap
    return _snap()


def test_observation_memory_block_appended_only_when_on():
    from showdown.rl.observation import build_observation, obs_layout
    snap = _obs_snap()
    base = build_observation(snap, memory=False)
    assert base.shape == (OBS_DIM,)
    mem = SeriesMemory()
    mem.add(GameRecord.from_messages(_msgs(GAME), our_role='p2', won=True))
    snap2 = dataclasses.replace(snap, series_memory=mem)
    full = build_observation(snap2, memory=True)
    assert full.shape == (OBS_DIM_MEMORY,)
    np.testing.assert_allclose(full[:OBS_DIM], build_observation(snap2, memory=False))
    assert 'memory' not in obs_layout()
    a, b = obs_layout(memory=True)['memory']
    assert (a, b) == (OBS_DIM, OBS_DIM_MEMORY)
    assert full[a + 1] == 1.0 and full[a + 3] == 0.5  # partida 2, una victoria nuestra
    # sin memoria adjunta: bloque de "partida 1"
    plain = build_observation(snap, memory=True)
    assert plain[a] == 1.0 and plain[a + 1:].sum() == 0.0


def test_symmetry_extends_to_memory_and_is_involution():
    from showdown.rl import symmetry as SYM
    P = SYM.obs_perms(OBS_DIM_MEMORY)
    base = SYM.obs_perms(OBS_DIM)
    assert P.shape == (8, OBS_DIM_MEMORY)
    np.testing.assert_array_equal(P[:, :OBS_DIM], base)
    x = np.arange(OBS_DIM_MEMORY)
    for g in range(8):
        np.testing.assert_array_equal(x[P[g]][P[g]], x)
    s0 = OBS_DIM + MEM_GLOBAL_DIM
    foe = P[SYM.BIT_FOE]
    np.testing.assert_array_equal(foe[s0:s0 + MEM_SLOT_DIM], np.arange(s0 + MEM_SLOT_DIM, s0 + 2 * MEM_SLOT_DIM))
    np.testing.assert_array_equal(P[SYM.BIT_OUR][OBS_DIM:], np.arange(OBS_DIM, OBS_DIM_MEMORY))
    obs = np.random.default_rng(0).random((5, OBS_DIM_MEMORY)).astype(np.float32)
    assert SYM.augment_obs(obs, np.array([0, 1, 2, 3, 4])).shape == obs.shape


def test_ppo_warm_start_is_function_preserving(tmp_path):
    torch = pytest.importorskip('torch')
    from showdown.rl.ppo.agent import PPOAgent, PPOConfig
    v2 = PPOAgent(PPOConfig(seed=1))
    path = v2.save(tmp_path / 'v2.pt')
    v3 = PPOAgent(PPOConfig(seed=2, series_memory=True))
    info = v3.init_from_previous(path)
    assert v3.obs_dim == OBS_DIM_MEMORY and info['ampliados'] and info['ampliados'][0].startswith('fc1.weight')
    rng = np.random.default_rng(0)
    obs = rng.random((4, OBS_DIM)).astype(np.float32)
    mem = rng.random((4, MEM_DIM)).astype(np.float32)
    with torch.no_grad():
        z1a, _, va = v2.net(torch.as_tensor(obs))
        z1b, _, vb = v3.net(torch.as_tensor(np.concatenate([obs, mem], axis=1)))
    np.testing.assert_allclose(z1a.numpy(), z1b.numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(va.numpy(), vb.numpy(), rtol=1e-5, atol=1e-5)
    # guardar v3 y cargarlo en un agente por defecto: reconstruye la red con memoria
    p3 = v3.save(tmp_path / 'v3.pt')
    fresh = PPOAgent(PPOConfig())
    fresh.load(p3)
    assert fresh.cfg.series_memory and fresh.obs_dim == OBS_DIM_MEMORY
    # un checkpoint v2 cargado en un agente v3 vuelve a la obs sin memoria
    back = PPOAgent(PPOConfig(series_memory=True))
    back.load(path)
    assert not back.cfg.series_memory and back.obs_dim == OBS_DIM


def test_ddqn_warm_start_is_function_preserving(tmp_path):
    torch = pytest.importorskip('torch')
    from showdown.rl.ddqn.agent import DDQNAgent, DDQNConfig
    kw = dict(replay_capacity=64, replay_device='cpu', dropout=0.0)
    v1 = DDQNAgent(DDQNConfig(seed=1, **kw))
    path = v1.save(tmp_path / 'v1.pt')
    v2 = DDQNAgent(DDQNConfig(seed=2, series_memory=True, **kw))
    v2.init_from_previous(path)
    assert v2.obs_dim == OBS_DIM_MEMORY and v2.replay.obs_dim == OBS_DIM_MEMORY
    from showdown.rl.series_options import apply_series_options
    v2b = DDQNAgent(DDQNConfig(seed=3, **kw))
    meta = apply_series_options(v2b, {'series_memory': 1, 'init_from': str(path)})
    assert v2b.cfg.eps_start == v2b.cfg.eps_end and meta['init_info']['eps_start'] == v2b.cfg.eps_end
    rng = np.random.default_rng(0)
    obs = rng.random((3, OBS_DIM)).astype(np.float32)
    mem = rng.random((3, MEM_DIM)).astype(np.float32)
    v1.online.eval()
    v2.online.eval()
    with torch.no_grad():
        qa = v1.online(torch.as_tensor(obs)).numpy()
        qb = v2.online(torch.as_tensor(np.concatenate([obs, mem], axis=1))).numpy()
        qt = v2.target(torch.as_tensor(np.concatenate([obs, mem], axis=1))).numpy()
    np.testing.assert_allclose(qa, qb, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(qb, qt, rtol=1e-5, atol=1e-5)
    p2 = v2.save(tmp_path / 'v2.pt')
    fresh = DDQNAgent(DDQNConfig(**kw))
    fresh.load(p2)
    assert fresh.cfg.series_memory and fresh.obs_dim == OBS_DIM_MEMORY


def _protect_battle(counter: int):
    from showdown.rl.action_space import SLOT_ACTION_N  # noqa: F401
    protect = SimpleNamespace(id='protect', is_protect_counter=True, base_power=0, type=None)
    hit = SimpleNamespace(id='shadowball', is_protect_counter=False, base_power=80, type=None)
    mon = SimpleNamespace(protect_counter=counter, ability='')
    return SimpleNamespace(
        force_switch=[False, False], available_switches=[[], []],
        available_moves=[[protect, hit], [hit]], can_mega_evolve=[False, False], can_tera=[False, False],
        active_pokemon=[mon, SimpleNamespace(protect_counter=0, ability='')], valid_orders=None,
    )


def test_protect_streak_mask(monkeypatch):
    from showdown.rl.pe_adapter import pe_slot_mask
    monkeypatch.delenv('RL_MASK_PROTECT_STREAK', raising=False)
    monkeypatch.setenv('RL_MASK_ALLY_DAMAGE', '0')
    off = pe_slot_mask(_protect_battle(1), 0)
    assert off[0:4].sum() == 4  # apagada por defecto: Protect sigue legal
    monkeypatch.setenv('RL_MASK_PROTECT_STREAK', '1')
    on = pe_slot_mask(_protect_battle(1), 0)
    assert on[0:4].sum() == 0 and on[4:8].sum() == 4  # Protect fuera, el ataque queda
    fresh = pe_slot_mask(_protect_battle(0), 0)
    assert fresh[0:4].sum() == 4  # sin Protect previo no se toca


def test_preview_keep_foe_filter():
    from showdown.rl.frozen_preview import _keep_foe
    mons = [SimpleNamespace(species=s) for s in ('Pelipper', 'Farigiraf', 'Kingambit', 'Charizard')]
    kept = _keep_foe(mons, ['pelipper', 'charizard'])
    assert [m.species for m in kept] == ['Pelipper', 'Charizard']
    assert len(_keep_foe(mons, None)) == 4
    assert len(_keep_foe(mons, ['nadie'])) == 4  # filtro vacío: hoja completa
    assert len(_keep_foe(mons, ['pelipper', 'farigiraf', 'kingambit', 'charizard'])) == 4


def test_series_tally():
    from showdown.rl.train_metrics import SeriesTally
    t = SeriesTally()
    t.add({'partida': 1, 'victorias': 0, 'derrotas': 0, 'decidida': False}, 1.0)
    t.add({'partida': 2, 'victorias': 1, 'derrotas': 0, 'decidida': True}, True)
    t.add({'partida': 1, 'victorias': 0, 'derrotas': 0, 'decidida': False}, -1.0)
    t.add({'partida': 2, 'victorias': 0, 'derrotas': 1, 'decidida': False}, 1.0)
    t.add({'partida': 3, 'victorias': 1, 'derrotas': 1, 'decidida': True}, False)
    t.add(None, 1.0)  # fuera del modo serie: se ignora
    out = t.flush()
    assert out['wr_partida_1'] == 0.5 and out['wr_partida_2'] == 1.0 and out['wr_partida_3'] == 0.0
    assert out['n_series'] == 2 and out['wr_series'] == 0.5
    assert t.flush() == {}


def test_series_options_apply(monkeypatch):
    pytest.importorskip('torch')
    from showdown.rl.ppo.agent import PPOAgent, PPOConfig
    from showdown.rl.series_options import apply_series_options
    for k in ('RL_OBS_SERIES_MEMORY', 'RL_MASK_PROTECT_STREAK', 'RL_SERIES_BO3', 'PREVIEW_SERIES_CONDITION'):
        monkeypatch.delenv(k, raising=False)
    agent = PPOAgent(PPOConfig())
    meta = apply_series_options(agent, {'series_memory': 1, 'mask_protect_streak': 1, 'preview_series': 1})
    import os
    assert meta['series_bo3'] and meta['obs_dim'] == OBS_DIM_MEMORY and agent.obs_dim == OBS_DIM_MEMORY
    assert int(agent.net.encoder.obs_dim) == OBS_DIM_MEMORY
    assert os.environ['RL_OBS_SERIES_MEMORY'] == '1' and os.environ['RL_SERIES_BO3'] == '1'
    assert os.environ['RL_MASK_PROTECT_STREAK'] == '1' and os.environ['PREVIEW_SERIES_CONDITION'] == '1'
    assert obs_dim_for(True) == OBS_DIM_MEMORY


def test_backend_series_defers_opponent_change_until_series_ends():
    pytest.importorskip('poke_env')
    from showdown.rl.pokeenv_backend import PokeEnvBattleBackend
    b = PokeEnvBattleBackend(opponent='random', series=True)
    assert b.series_state()['partida'] == 1
    b._series.add(GameRecord(won=True))
    b.set_opponent('greedy_solo')
    assert b.opponent == 'random' and b._pending_opponent == 'greedy_solo'  # a mitad de serie: diferido
    b._start_series_if_needed()
    assert len(b._series.records) == 1  # la serie sigue (1-0)
    b._series.add(GameRecord(won=True))
    b._start_series_if_needed()
    assert b.opponent == 'greedy_solo' and not b._series.records and b._pending_opponent is None
    assert b.last_series == {'partidas': 2, 'victorias': 2, 'derrotas': 0, 'cortada': False, 'ganada': True}
    off = PokeEnvBattleBackend(opponent='random', series=False)
    assert off.series_state() is None


def test_v3_report_rules(tmp_path):
    import json
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'analysis'))
    import v3_report as R
    d = tmp_path / 'diag'
    d.mkdir()

    def games(wins, foe='greedy_joint'):
        return [{'rival': foe, 'res': 'win' if w else 'loss'} for w in wins]
    # P1: mismas partidas salvo 2 que la máscara convierte en victoria
    off = [True] * 60 + [False] * 40
    on = list(off)
    on[60] = on[61] = True
    (d / 'p1_ppo_s0_off.json').write_text(json.dumps({'partidas': games(off), 'mascara_protect_consecutivo': {
        'por_rival': {'greedy_joint': {'situaciones': 7, 'repitio_protect': 3}}}}))
    (d / 'p1_ppo_s0_on.json').write_text(json.dumps({'partidas': games(on)}))
    p1 = '\n'.join(R.p1(d))
    assert '| ppo | 100 | +2.00 pts' in p1 and '| sí |' in p1 and '7 / 3' in p1

    # V: V gana más partidas 2–3 que C en las 3 semillas
    def series_games(win23):
        out = []
        for sidx in range(40):
            out.append({'rival': 'greedy_solo', 'serie': sidx, 'partida': 1, 'res': 'win'})
            out.append({'rival': 'greedy_solo', 'serie': sidx, 'partida': 2, 'res': 'win' if sidx < win23 else 'loss'})
        return out
    for k in range(3):
        (d / f'v_ppo_C_s{k}.json').write_text(json.dumps({'partidas': series_games(20)}))
        (d / f'v_ppo_V_s{k}.json').write_text(json.dumps({'partidas': series_games(30)}))
    vtxt = '\n'.join(R.v(d))
    assert '50.0 % → 75.0 %' in vtxt and '+25.00 pts' in vtxt and 'confirmada' in vtxt
    assert 'no inferior' in vtxt
    R.main([str(tmp_path)])
    assert (tmp_path / 'v3_report.md').read_text(encoding='utf-8').startswith('# PPO v3 / DDQN v2')
