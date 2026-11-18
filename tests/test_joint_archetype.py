"""Modo arquetipo de greedy_joint (PS_JOINT_ARCHETYPE, memorizacion_equipos.md §14)."""
from types import SimpleNamespace as NS

from showdown.battle_bots.greedy_joint import joint as J


def _call(move, *, partner_dmg=0.0, tr=False, tw=False, our_spe=50.0, foe_spe=100.0, monkeypatch=None):
    monkeypatch.setattr(J, '_living_foes', lambda foes: [(0, 'f')])
    monkeypatch.setattr(J, '_mean_effective_spe', lambda pokes, tailwind=False: our_spe if 'f' not in pokes else foe_spe)
    atom = NS(move_name=move)
    partner = NS(total_damage_pct=partner_dmg, has_guaranteed_ko=False)
    return J._field_setup_bonus(atom, partner, left_poke='a', right_poke='b', foes=[(0, 'f')],
                                trick_room=tr, our_tailwind=tw, opp_tailwind=False)


def test_default_mode_unchanged(monkeypatch):
    monkeypatch.delenv('PS_JOINT_ARCHETYPE', raising=False)
    assert _call('Tailwind', partner_dmg=0.0, monkeypatch=monkeypatch) == 0.0  # sin daño del compañero: nada
    assert _call('Tailwind', partner_dmg=30.0, monkeypatch=monkeypatch) == J._FIELD_SETUP_BONUS


def test_archetype_mode_prioritizes_setup(monkeypatch):
    monkeypatch.setenv('PS_JOINT_ARCHETYPE', '1')
    big = J._ARCHETYPE_SETUP_BONUS
    assert _call('Tailwind', monkeypatch=monkeypatch) == big
    assert _call('Tailwind', tw=True, monkeypatch=monkeypatch) == 0.0
    assert _call('Trick Room', monkeypatch=monkeypatch) == big              # más lento: lo activa
    assert _call('Trick Room', our_spe=120.0, monkeypatch=monkeypatch) == 0.0  # más rápido: no
    assert _call('Trick Room', tr=True, monkeypatch=monkeypatch) == 0.0       # ya activo: no lo deshace
    assert _call('Icy Wind', partner_dmg=30.0, monkeypatch=monkeypatch) == J._SPEED_PARTNER_BONUS


def test_policy_diag_foes_flag():
    import inspect
    from showdown.rl.ddqn import policy_diag as P
    assert 'foes' in inspect.signature(P.run).parameters and '--foes' in inspect.getsource(P.main)


def test_ppo_ent_coef_flag():
    import inspect
    from showdown.rl.ppo import train as T
    assert 'ent_coef' in inspect.signature(T.run_training).parameters and '--ent-coef' in inspect.getsource(T)


def test_summary21_runs_without_data(tmp_path):
    from showdown.rl import kickoff_summary21 as K
    (tmp_path / 'q21' / 'diag').mkdir(parents=True)
    assert K.main([str(tmp_path / 'q21'), str(tmp_path)]) == 0
    md = (tmp_path / 'q21' / 'kickoff_summary21.md').read_text(encoding='utf-8')
    assert 'Entropía de PPO' in md and '| 0,003 |' in md
