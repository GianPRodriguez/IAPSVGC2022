"""Extensiones opcionales de G v8 / H 2.3 (protocolo §4.19): arquetipos (A), serie BO3 (S) y hoja con campo (F).

Calculadora falsa (sin Node), como test_mcts_puct_s3c.
"""
from __future__ import annotations

from showdown.rl.mcts_puct import actions as A
from showdown.rl.mcts_puct import arch_ext as X
from showdown.rl.mcts_puct.openloop import prepare_root
from showdown.rl.mcts_puct.s3 import S3Model
from showdown.rl.snapshot import BattleSnapshot, MonSnapshot, SideSnapshot


def _mon(sp, types, moves, spe, ability='', item=''):
    st = {'hp': 180, 'atk': 150, 'def': 120, 'spa': 120, 'spd': 120, 'spe': spe}
    return MonSnapshot(species=sp, hp_pct=100.0, types=types, moves=moves, stats=st, ability=ability, item=item)


def _snap():
    our = SideSnapshot(
        active=[_mon('Pelipper', ('Water', 'Flying'), ('raindance', 'hurricane', 'tailwind', 'protect'), 65,
                     ability='Damp'),
                _mon('Basculegion', ('Water', 'Ghost'), ('wavecrash', 'lastrespects', 'aquajet', 'protect'), 78,
                     ability='Swift Swim')],
        bench=[_mon('Archaludon', ('Steel', 'Dragon'), ('electroshot', 'dracometeor', 'flashcannon', 'protect'), 85,
                    ability='Stamina'), None])
    foe = SideSnapshot(
        active=[_mon('Hatterene', ('Psychic', 'Fairy'), ('trickroom', 'expandingforce', 'dazzlinggleam', 'protect'),
                     29, ability='Magic Bounce'),
                _mon('Indeedee-F', ('Psychic', 'Normal'), ('followme', 'psychic', 'helpinghand', 'protect'), 85,
                     ability='Psychic Surge')],
        bench=[_mon('Ursaluna', ('Ground', 'Normal'), ('facade', 'headlongrush', 'protect', 'earthquake'), 50,
                    ability='Guts'), None])
    return BattleSnapshot(turn=1, our=our, foe=foe)


def _build(snap):
    root = prepare_root(snap)
    return root, S3Model.build(root, ranges_fn=lambda reqs: [(30, 30)] * len(reqs), phase='3c')


def test_default_off_scores_unchanged(monkeypatch):
    for k in ('MCTS_ARCH', 'MCTS_SERIES', 'MCTS_LEAF_FIELD'):
        monkeypatch.delenv(k, raising=False)
    root, model = _build(_snap())
    base = model.slot_scores(root, 0)
    assert base.get(0 * 4 + 3, 0.0) == 0.0          # Rain Dance: S3 no puntúa el clima
    monkeypatch.setenv('MCTS_ARCH', '1')
    arch = model.slot_scores(root, 0)
    assert arch[0 * 4 + 3] == base.get(0 * 4 + 3, 0.0) + X.ARCH_FIELD_W          # Rain Dance (plan lluvia)
    assert arch[2 * 4 + 3] == base.get(2 * 4 + 3, 0.0) + X.ARCH_SPEED_W          # Tailwind (plan tailwind)
    assert all(arch[k] == v for k, v in base.items() if k not in (3, 11))


def test_arch_applies_to_foe_plan_through_flipped_view(monkeypatch):
    monkeypatch.setenv('MCTS_ARCH', '1')
    root, model = _build(_snap())
    view = A.flip_view(root)                          # el rival actúa: Hatterene con Trick Room
    monkeypatch.delenv('MCTS_ARCH')
    off = model.slot_scores(view, 0).get(3, 0.0)
    monkeypatch.setenv('MCTS_ARCH', '1')
    on = model.slot_scores(view, 0).get(3, 0.0)
    assert on == off + X.ARCH_SPEED_W
    root.field.weather = 'raindance'                  # ya llueve: Rain Dance sin bono
    assert model.slot_scores(root, 0).get(3, 0.0) == 0.0


def test_series_bonus_only_for_foe_moves(monkeypatch):
    monkeypatch.setenv('MCTS_SERIES', '1')
    monkeypatch.delenv('MCTS_ARCH', raising=False)
    root, model = _build(_snap())
    view = A.flip_view(root)
    off = model.slot_scores(view, 0)
    tok = X.SERIES_MOVES.set({'hatterene': {'dazzlinggleam'}, 'pelipper': {'hurricane'}})
    try:
        on = model.slot_scores(view, 0)
        ours = model.slot_scores(root, 0)
    finally:
        X.SERIES_MOVES.reset(tok)
    for t in range(4):
        k = 2 * 4 + t
        if k in off:
            assert on[k] == off[k] + X.SERIES_MOVE_W
    assert ours == {k: v for k, v in model.slot_scores(root, 0).items()}   # nuestro lado no cambia


def test_field_advantage_sign():
    root, model = _build(_snap())
    assert X.field_advantage(model, root) == 0.0
    root.field.trick_room = True                      # TR favorece al rival más lento… salvo que nosotros seamos lentos
    adv_tr = X.field_advantage(model, root)
    root.field.trick_room = False
    root.field.our_tailwind = True
    assert X.field_advantage(model, root) >= 0.0
    assert adv_tr <= 0.0


def test_plan_and_seen_formations():
    s = _snap()
    ours = [m for m in s.our.active + s.our.bench if m is not None] + [
        _mon('Rillaboom', ('Grass',), ('fakeout', 'grassyglide', 'woodhammer', 'uturn'), 85, ability='Grassy Surge')]
    foes = [m for m in s.foe.active + s.foe.bench if m is not None] + [
        _mon('Torkoal', ('Fire',), ('eruption', 'heatwave', 'protect', 'solarbeam'), 20, ability='Drought')]
    assert X.plan_formation(foes, ours)[:2] in ((1, 2), (2, 1))   # Hatterene (TR) + Indeedee (Follow Me)
    seen = X.seen_formations(foes, ['Hatterene', 'Indeedee-F', 'Ursaluna', 'Torkoal'])
    assert len(seen) == 6 and all(sorted(f) == [1, 2, 3, 4] for f in seen)
    assert X.seen_formations(foes, ['Hatterene', 'Pikachu', 'Ursaluna', 'Torkoal']) == []


def test_preview_series_restricts_foe_formations(monkeypatch):
    from showdown.rl.mcts_puct.search_preview import SearchPreviewConfig, choose_formation
    monkeypatch.delenv('MCTS_ARCH', raising=False)
    s = _snap()
    extra = [_mon('Rillaboom', ('Grass',), ('fakeout', 'grassyglide'), 85, ability='Grassy Surge'),
             _mon('Incineroar', ('Fire', 'Dark'), ('fakeout', 'flareblitz'), 60, ability='Intimidate'),
             _mon('Gholdengo', ('Steel', 'Ghost'), ('makeitrain', 'shadowball'), 84)]
    ours = [m for m in s.our.active + s.our.bench if m is not None] + extra
    foes = [m for m in s.foe.active + s.foe.bench if m is not None] + [
        _mon('Torkoal', ('Fire',), ('eruption', 'heatwave'), 20, ability='Drought'),
        _mon('Kingambit', ('Dark', 'Steel'), ('kowtowcleave', 'suckerpunch'), 50),
        _mon('Dragonite', ('Dragon', 'Flying'), ('extremespeed', 'tailwind'), 80)]
    cfg = SearchPreviewConfig(n_rollouts=1, turns=1, deadline_sec=60)
    rf = lambda reqs: [(20, 20)] * len(reqs)  # noqa: E731
    _o, info = choose_formation(ours, foes, cfg, ranges_fn=rf)
    assert not info.fallback and info.n_foe == 12
    _o, info = choose_formation(ours, foes, cfg, ranges_fn=rf, foe_brought=['Hatterene', 'Indeedee-F', 'Ursaluna',
                                                                            'Torkoal'])
    assert not info.fallback and info.n_foe == 6
    monkeypatch.setenv('MCTS_ARCH', '1')
    _o, info = choose_formation(ours, foes, cfg, ranges_fn=rf)
    assert not info.fallback and info.n_ours >= 18 and info.n_foe >= 12


def test_series_memory_cycle():
    from types import SimpleNamespace
    from showdown.rl.mcts_puct import player as P
    import os
    os.environ['MCTS_SERIES'] = '1'
    try:
        mv = lambda used: SimpleNamespace(current_pp=8 if used else 16, max_pp=16)  # noqa: E731
        hat = SimpleNamespace(species='hatterene', moves={'trickroom': mv(True), 'psychic': mv(False)})
        ind = SimpleNamespace(species='indeedeef', moves={'followme': mv(True)})
        urs = SimpleNamespace(species='ursaluna', moves={'facade': mv(False)})
        tor = SimpleNamespace(species='torkoal', moves={'eruption': mv(True)})
        battle = SimpleNamespace(opponent_username='rival', opponent_team={'a': hat, 'b': ind, 'c': urs, 'd': tor},
                                 opponent_active_pokemon=[hat, ind])
        pl = SimpleNamespace(series_games=[])
        assert P.series_brought(pl, battle) is None                     # partida 1
        P.series_update(pl, battle)
        battle.opponent_active_pokemon = [urs, tor]
        P.series_update(pl, battle)
        assert P.series_moves(pl, battle)['hatterene'] == {'trickroom'}
        pl.series_games = [(10, True)]                                  # partida 2
        assert sorted(P.series_brought(pl, battle)) == ['hatterene', 'indeedeef', 'torkoal', 'ursaluna']
        pl.series_games = []                                            # serie nueva: memoria nueva
        assert P.series_brought(pl, battle) is None and P.series_moves(pl, battle) is None
    finally:
        os.environ.pop('MCTS_SERIES', None)
