"""PPO v2 (memorizacion_equipos.md §17): P1 temperatura, P2 arquetipo nivel 2, P3 rival PPO v1."""
import asyncio
import inspect

import numpy as np
import pytest

from showdown.battle_bots.greedy_joint import joint as J


# --------------------------------------------------------------------------- P2: nivel de arquetipo
def test_archetype_level_from_env(monkeypatch):
    for raw, lvl in (('', 0), ('0', 0), ('1', 1), ('true', 1), ('2', 2)):
        monkeypatch.setenv('PS_JOINT_ARCHETYPE', raw)
        assert J._archetype_level() == lvl
        assert J._archetype_mode() == (lvl >= 1)


def test_wrap_archetype_level_sync_and_async(monkeypatch):
    monkeypatch.delenv('PS_JOINT_ARCHETYPE', raising=False)
    seen = []

    class SyncP:
        def choose_move(self, battle):
            seen.append(J._archetype_level())
            return 'ok'

    class AsyncP:
        def choose_move(self, battle):
            return self._run()

        async def _run(self):
            await asyncio.sleep(0)
            seen.append(J._archetype_level())
            return 'ok'

    assert J.wrap_archetype_level(SyncP(), 2).choose_move(None) == 'ok'
    assert asyncio.run(J.wrap_archetype_level(AsyncP(), 2).choose_move(None)) == 'ok'
    assert seen == [2, 2]
    assert J._archetype_level() == 0  # fuera del rival no queda activo


def _iso(left, right):
    return J._utility_score_damage_iso(left, right, foes=[], left_poke=None, right_poke=None)


def test_level2_protect_cover_and_sleep(monkeypatch):
    monkeypatch.setattr(J, '_fo_partner_pressure_bonus', lambda *a, **k: 0.0)
    monkeypatch.setattr(J, '_field_setup_bonus', lambda atom, *a, **k: 250.0)
    prot = J.JointAtom(move_slot=1, target=0, utility='protect', move_name='Protect')
    tw = J.JointAtom(move_slot=2, target=0, utility='speed', move_name='Tailwind')
    spore = J.JointAtom(move_slot=1, target=1, utility='status', move_name='Spore')
    dmg = J.JointAtom(move_slot=2, target=1, utility='damage', move_name='Tackle')
    monkeypatch.setenv('PS_JOINT_ARCHETYPE', '1')
    l1_cover, l1_sleep = _iso(prot, tw), _iso(spore, dmg)
    monkeypatch.setenv('PS_JOINT_ARCHETYPE', '2')
    assert _iso(prot, tw) - l1_cover == pytest.approx(J._ARCHETYPE_PROTECT_COVER_BONUS)
    assert _iso(spore, dmg) - l1_sleep == pytest.approx(J._ARCHETYPE_SLEEP_BONUS)


def test_level1_unchanged_by_level2_code(monkeypatch):
    monkeypatch.setattr(J, '_fo_partner_pressure_bonus', lambda *a, **k: 0.0)
    spore = J.JointAtom(move_slot=1, target=1, utility='status', move_name='Spore')
    dmg = J.JointAtom(move_slot=2, target=1, utility='damage', move_name='Tackle')
    monkeypatch.delenv('PS_JOINT_ARCHETYPE', raising=False)
    base = _iso(spore, dmg)
    monkeypatch.setenv('PS_JOINT_ARCHETYPE', '1')
    assert _iso(spore, dmg) == base


# --------------------------------------------------------------------------- rivales nuevos
def test_new_foes_registered():
    from showdown.rl import pokeenv_backend as B
    from showdown.rl.ddqn.train import parse_phases
    from showdown.rl.ppo import train as PT
    for k in ('greedy_joint_arq', 'ppo_v1'):
        assert k in B.LIVE_OPPONENTS and k in PT.LIVE_OPPONENTS
    assert B.foe_validated('greedy_joint_arq') and not B.foe_validated('ppo_v1')
    assert B._base_kind('greedy_joint_arq') == 'greedy_joint'
    ph = parse_phases('greedy_joint:150000,greedy_joint_arq:150000,ppo_v1:10k')
    assert [p.opponent for p in ph] == ['greedy_joint', 'greedy_joint_arq', 'ppo_v1']


def test_ppo_v1_needs_checkpoints(monkeypatch):
    from showdown.rl import pokeenv_backend as B
    monkeypatch.setenv('PPO_FOE_CHECKPOINTS', ' a.pt, b.pt ,')
    assert B.ppo_foe_checkpoints() == ['a.pt', 'b.pt']
    monkeypatch.delenv('PPO_FOE_CHECKPOINTS')
    with pytest.raises(ValueError):
        B._build_ppo_v1_foe(account_configuration=None, battle_format='x', team='', server_configuration=None,
                            log_level=0)


def test_archetype_foe_tie_uses_base_kind():
    from showdown.rl import pokeenv_backend as B

    class P:
        tie_margin = None
    p = P()
    assert B.apply_foe_tie(p, 'greedy_joint_arq') and p.tie_margin == 8.0


# --------------------------------------------------------------------------- P1: temperatura
def _agent_inputs():
    from showdown.rl.constants import OBS_DIM, SLOT_ACTION_N
    rng = np.random.default_rng(0)
    obs = rng.normal(size=OBS_DIM).astype(np.float32)
    m = np.ones(SLOT_ACTION_N, dtype=np.float32)
    return obs, m, (lambda a1: m)


def test_ppo_temperature_sampling():
    import torch
    from showdown.rl.ppo.agent import PPOAgent, PPOConfig
    torch.manual_seed(0)
    ag = PPOAgent(PPOConfig(device='cpu', seed=0))
    obs, m, m2 = _agent_inputs()
    g = ag.act(obs, m, m2, greedy=True)[:2]
    cold = {ag.act(obs, m, m2, greedy=False, temperature=1e-3)[:2] for _ in range(20)}
    assert cold == {g}  # T → 0 equivale a la más probable
    warm = {ag.act(obs, m, m2, greedy=False, temperature=1.0)[:2] for _ in range(60)}
    assert len(warm) > 1  # T = 1 sortea
    assert 'temperature' in inspect.signature(ag.act).parameters


def test_policy_diag_sample_flags():
    from showdown.rl.ddqn import policy_diag as P
    sig = inspect.signature(P.run).parameters
    assert 'sample' in sig and 'temperature' in sig
    src = inspect.getsource(P.main)
    assert '--sample' in src and '--temperature' in src
    assert P.FOES == ('random', 'greedy_solo', 'greedy_joint', 'expectimax_bayesian')  # semillas iguales
    assert P.EXTRA_FOES == ('greedy_joint_arq', 'ppo_v1', 'greedy_joint_arq3')  # arq3 al final (§4.17)


# --------------------------------------------------------------------------- resumen
def _w(path, wins, foes=('greedy_solo', 'greedy_joint', 'expectimax_bayesian')):
    import json
    path.write_text(json.dumps({'resultados': {f: {'win': w, 'loss': 200 - w} for f, w in zip(foes, wins)}}),
                    encoding='utf-8')


def test_summary24_decisions(tmp_path):
    from showdown.rl import kickoff_summary24 as K
    out, v1 = tmp_path / 'v2', tmp_path / 'v1'
    (out / 'diag').mkdir(parents=True)
    (v1 / 'diag').mkdir(parents=True)
    for s in range(5):
        _w(out / 'diag' / f'v1_s{s}_val.json', (170, 170, 170))          # 85 %
        _w(v1 / 'diag' / f'ppo_s{s}_final_eval.json', (165, 165, 160))
        _w(v1 / 'diag' / f'ppo_s{s}_final_arquetipo.json', (150,), ('greedy_joint',))
        _w(out / 'diag' / f'p1_T1.0_s{s}_val.json', (168, 168, 168))     # −1 → cumple
        _w(out / 'diag' / f'p1_T0.5_s{s}_val.json', (160, 160, 160))     # −5 → no
    for s in (0, 1):
        _w(out / 'diag' / f'v1_s{s}_arq2_val.json', (140,), ('greedy_joint_arq',))
        _w(out / 'diag' / f'p2_s{s}_val.json', (168, 168, 168))           # −1
        _w(out / 'diag' / f'p2_s{s}_arq2_val.json', (160,), ('greedy_joint_arq',))  # +10 → adopta
        _w(out / 'diag' / f'p4_s{s}_val.json', (150, 150, 150))           # −10 → no
        _w(out / 'diag' / f'p3_s{s}_val.json', (170, 170, 170))
        _w(out / 'diag' / f'p3_s{s}_h2h.json', (100,), ('ppo_v1',))       # 50 % → no
    assert K.main([str(out), str(v1)]) == 0
    md = (out / 'kickoff_summary24.md').read_text(encoding='utf-8')
    assert 'se adopta el sorteo con T = 1.0' in md
    assert '| P2: ' in md and '**se adopta** (arquetipo > +4)' in md
    assert 'no (pierde en `val`)' in md and 'no (no alcanza el umbral)' in md
    assert 'Elementos adoptados: P2, P1 (sorteo T = 1.0)' in md


def test_summary24_without_data(tmp_path):
    from showdown.rl import kickoff_summary24 as K
    (tmp_path / 'v2' / 'diag').mkdir(parents=True)
    assert K.main([str(tmp_path / 'v2'), str(tmp_path / 'v1')]) == 0
    assert 'sin datos' in (tmp_path / 'v2' / 'kickoff_summary24.md').read_text(encoding='utf-8')


def test_summary24_h2h_calibration(tmp_path):
    from showdown.rl import kickoff_summary24 as K
    out, v1 = tmp_path / 'v2', tmp_path / 'v1'
    (out / 'diag').mkdir(parents=True)
    (v1 / 'diag').mkdir(parents=True)
    for s in (0, 1):
        _w(out / 'diag' / f'v1_s{s}_val.json', (170, 170, 170))
        _w(out / 'diag' / f'p3_s{s}_val.json', (170, 170, 170))
        _w(out / 'diag' / f'p3_s{s}_h2h.json', (60,), ('ppo_v1',))   # 30 %
        _w(out / 'diag' / f'v1_s{s}_h2h.json', (50,), ('ppo_v1',))   # v1 contra v1: 25 %
    assert K.main([str(out), str(v1)]) == 0
    md = (out / 'kickoff_summary24.md').read_text(encoding='utf-8')
    assert '25.0 % (calibración)' in md and '**se adopta** (contra v1 > 29 %)' in md


def test_divergence_summary(tmp_path):
    import json
    import sys
    sys.path.insert(0, 'scripts/analysis')
    import divergencia_p1 as D
    rep = {'resultados': {'greedy_solo': {'win': 90, 'loss': 10}},
           'divergencia': {'greedy_solo': {'todas_decisiones': 200, 'todas_jugada_distinta': 30,
                                           'todas_hueco1_distinto': 20, 'turno_1_decisiones': 50,
                                           'turno_1_jugada_distinta': 5, 'partidas': 100,
                                           'partidas_con_alguna_distinta': 60}}}
    (tmp_path / 'T0.5_s0_val.json').write_text(json.dumps(rep))
    assert D.main([str(tmp_path)]) == 0
    md = (tmp_path / 'divergencia_p1.md').read_text(encoding='utf-8')
    assert '| 0.5 | 15.0 % | 10.0 % | 10.0 % | 60.0 % | 200 | 90.0 % |' in md


def test_policy_diag_records_divergence():
    import inspect
    from showdown.rl.ddqn import policy_diag as P
    src = inspect.getsource(P.run)
    assert "'divergencia'" in src and 'greedy=True' in src


def test_ppo_v1_foe_uses_its_own_preview():
    import inspect
    from showdown.rl import pokeenv_backend as B
    assert 'PPO_FOE_PREVIEW' in inspect.getsource(B._build_ppo_v1_foe)


def test_preview_divergence_summary(tmp_path):
    import json
    import sys
    sys.path.insert(0, 'scripts/analysis')
    import divergencia_preview as D
    ref = tmp_path / 'q26' / 'diag'
    ref.mkdir(parents=True)
    out = tmp_path / 'p1b' / 'diag'
    out.mkdir(parents=True)
    res = {f: {'win': 180, 'loss': 20} for f in D.HEUR}
    for s in (0, 1):
        (ref / f'ppo_p5_s{s}_val.json').write_text(json.dumps({'resultados': res}))
        games = [{'rival': 'greedy_solo', 'formacion': '1234' if i % 4 else '3456', 'formacion_determinista': '1234'}
                 for i in range(100)]
        (out / f'pv_T1.0_s{s}_val.json').write_text(json.dumps({'resultados': res, 'partidas': games}))
    assert D.main([str(tmp_path / 'p1b'), str(tmp_path / 'q26')]) == 0
    md = (tmp_path / 'p1b' / 'divergencia_preview.md').read_text(encoding='utf-8')
    assert '| Preview sorteado T = 1,0 (combate determinista) | 90.0 % | +0.0' in md and '25.0 %' in md


def test_frozen_preview_sampling_options(monkeypatch):
    from showdown.rl.frozen_preview import load_frozen_preview
    monkeypatch.setenv('PREVIEW_SAMPLE', '1')
    monkeypatch.setenv('PREVIEW_TEMPERATURE', '0.5')
    pol = load_frozen_preview(kind='team_preview_ppo', checkpoint='no_existe.pt', allow_missing=True)
    assert pol.greedy is False and pol.temperature == 0.5
    monkeypatch.delenv('PREVIEW_SAMPLE')
    assert load_frozen_preview(kind='team_preview_ppo', checkpoint='no_existe.pt').greedy is True
