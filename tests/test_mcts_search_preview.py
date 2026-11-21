"""Preview por búsqueda con S3 (showdown/rl/mcts_puct/search_preview.py). Calculadora falsa."""
from __future__ import annotations

from showdown.rl.mcts_puct.search_preview import (
    SearchPreviewConfig,
    choose_formation,
    formations,
    top_combos,
)
from showdown.rl.snapshot import MonSnapshot


def _mon(sp, types, moves, spe=80, ability=''):
    st = {'hp': 180, 'atk': 150, 'def': 120, 'spa': 120, 'spd': 120, 'spe': spe}
    return MonSnapshot(species=sp, hp_pct=100.0, types=types, moves=moves, stats=st, ability=ability)


def _team_a():
    return [
        _mon('Pikachu', ('Electric',), ('thunderbolt',), spe=90),
        _mon('Amoonguss', ('Grass', 'Poison'), ('pollenpuff', 'spore'), spe=30),
        _mon('Kingambit', ('Dark', 'Steel'), ('kowtowcleave', 'suckerpunch', 'protect'), spe=50),
        _mon('Incineroar', ('Fire', 'Dark'), ('fakeout', 'flareblitz', 'partingshot'), spe=60, ability='Intimidate'),
        _mon('Garchomp', ('Dragon', 'Ground'), ('dragonclaw', 'rockslide', 'protect'), spe=102),
        _mon('Gholdengo', ('Steel', 'Ghost'), ('makeitrain', 'shadowball', 'protect'), spe=84),
    ]


def _team_b():
    return [
        _mon('Rillaboom', ('Grass',), ('woodhammer', 'fakeout', 'grassyglide'), spe=85, ability='Grassy Surge'),
        _mon('Urshifu', ('Fighting', 'Water'), ('surgingstrikes', 'closecombat', 'protect'), spe=97),
        _mon('Flutter Mane', ('Ghost', 'Fairy'), ('moonblast', 'shadowball', 'protect'), spe=135),
        _mon('Tornadus', ('Flying',), ('bleakwindstorm', 'tailwind', 'protect'), spe=111),
        _mon('Chien-Pao', ('Dark', 'Ice'), ('iciclecrash', 'suckerpunch', 'protect'), spe=135),
        _mon('Farigiraf', ('Normal', 'Psychic'), ('psychic', 'trickroom', 'protect'), spe=60),
    ]


def _by_attacker(strong: str, big=400, small=20):
    def f(reqs):
        return [((big, big) if strong.lower() in str(r['attacker'].get('species', '')).lower() else (small, small))
                for r in reqs]
    return f


def test_top_combos_and_formations():
    a, b = _team_a(), _team_b()
    combos = top_combos(a, b, 3)
    assert len(combos) == 3 and all(len(c) == 4 and len(set(c)) == 4 for c in combos)
    forms = formations(combos[0])
    assert len(forms) == 6 and len(set(forms)) == 6
    assert all(sorted(f) == sorted(combos[0]) for f in forms)


def test_choose_formation_returns_valid_order_and_info():
    order, info = choose_formation(_team_a(), _team_b(), SearchPreviewConfig(n_rollouts=2, turns=2),
                                   ranges_fn=lambda reqs: [(30, 40)] * len(reqs))
    assert info.fallback == '' and len(order) == 4 and len(set(order)) == 4
    assert all(1 <= i <= 6 for i in order)
    assert info.n_ours == 18 and info.n_foe == 12 and info.rollouts == 18 * 12 * 2
    assert info.calc_requests > 0 and info.calc_missing == 0 and order in info.values


def test_search_puts_dominant_attacker_in_lead():
    """Si un Pokémon propio arrasa (calculadora falsa), debe salir de lead, no quedar en banca."""
    a, b = _team_a(), _team_b()
    combos = top_combos(a, b, 3)
    strong_idx = next(i for i in combos[0])  # un miembro de la mejor combinación
    strong = a[strong_idx - 1].species
    order, info = choose_formation(a, b, SearchPreviewConfig(n_rollouts=3, turns=2), ranges_fn=_by_attacker(strong))
    assert info.fallback == ''
    assert strong_idx in order[:2], (strong, order)


def test_deadline_falls_back_to_greedy():
    order, info = choose_formation(_team_a(), _team_b(), SearchPreviewConfig(deadline_sec=0.0),
                                   ranges_fn=lambda reqs: [(30, 40)] * len(reqs))
    assert info.fallback and order == info.greedy_order


def test_maximin_variant_runs():
    order, info = choose_formation(_team_a(), _team_b(),
                                   SearchPreviewConfig(n_rollouts=2, turns=2, decision='maximin'),
                                   ranges_fn=lambda reqs: [(30, 40)] * len(reqs))
    assert info.fallback == '' and len(order) == 4


def test_player_routes_search_preview(monkeypatch):
    from showdown.rl.mcts_puct import player as P

    called = {}

    def fake(self, battle):
        called['yes'] = True
        return '/team 1234'

    monkeypatch.setattr(P.MCTSPuctPlayer, '_search_preview', fake)
    monkeypatch.setenv('MCTS_PREVIEW', 'search')
    assert P.MCTSPuctPlayer.teampreview(P.MCTSPuctPlayer.__new__(P.MCTSPuctPlayer), None) == '/team 1234'
    assert called.get('yes')
