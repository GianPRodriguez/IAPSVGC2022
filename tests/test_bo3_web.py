"""Tests sin red de la web de retos BO3 (scripts/online/bo3_web.py)."""
from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts' / 'online'))
import bo3_web as w  # noqa: E402


class FakeBattle:
    def __init__(self, opp, finished=False, won=None, tag='b'):
        self.opponent_username = opp
        self.finished = finished
        self.won = won
        self.battle_tag = tag
        self.turn = 5
        self.opponent_team = {}


class FakeClient:
    def __init__(self, player):
        self.player = player
        self.sent = []
        self.logged_in = asyncio.Event()
        self.logged_in.set()

    async def challenge(self, user, fmt, team):
        self.sent.append(('challenge', user, fmt))
        self.player.on_challenge(user)

    async def send_message(self, msg, room=None):
        self.sent.append(('msg', msg))


class FakePlayer:
    """Simula Showdown: según ``behaviour[uid]`` el rival acepta, rechaza, ignora o no existe."""

    def __init__(self, behaviour=None):
        self.username = 'BotUser'
        self.agents = {'ppo': object(), 'ddqn': object()}
        self.assign = {}
        self.pending_to = set()
        self.battles = {}
        self.next_team = 'team'
        self.behaviour = behaviour or {}
        self.ps_client = FakeClient(self)
        self.logger = types.SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None,
                                            exception=lambda *a, **k: None)

    def on_challenge(self, user):
        uid = w.to_id(user)
        mode = self.behaviour.get(uid, 'accept')
        loop = asyncio.get_running_loop()
        if mode == 'missing':
            return
        self.pending_to.add(uid)
        if mode == 'accept':
            def start():
                self.pending_to.discard(uid)
                self.battles[f'battle-{uid}-1'] = FakeBattle(user, tag=f'battle-{uid}-1')
            loop.call_later(0.02, start)
        elif mode == 'reject':
            loop.call_later(0.02, lambda: self.pending_to.discard(uid))


def test_to_id_and_valid_username():
    assert w.to_id('Velasco Alvarado!') == 'velascoalvarado'
    assert w.valid_username('  Ash K.  ') == 'Ash K.'
    for bad in ('', '   ', 'x' * 19, '<script>', '...', 'a|b'):
        assert w.valid_username(bad) is None


def test_series_score_counts_only_new_battles_of_that_user():
    battles = {
        'old': FakeBattle('Ash', True, True),
        'g1': FakeBattle('Ash', True, True),
        'g2': FakeBattle('ash', False),
        'other': FakeBattle('Misty', True, False),
    }
    tags, wins, losses, done = w.series_score(battles, 'ash', {'old'})
    assert sorted(tags) == ['g1', 'g2'] and (wins, losses, done) == (1, 0, False)
    battles['g2'].finished, battles['g2'].won = True, True
    assert w.series_score(battles, 'ash', {'old'})[1:] == (2, 0, True)


def test_series_score_ends_after_three_games_or_two_losses():
    b = {'1': FakeBattle('a', True, True), '2': FakeBattle('a', True, False), '3': FakeBattle('a', True, None)}
    assert w.series_score(b, 'a', ())[3] is True  # tres partidas terminadas (con empate)
    b = {'1': FakeBattle('a', True, False), '2': FakeBattle('a', True, False)}
    assert w.series_score(b, 'a', ())[1:] == (0, 2, True)


def _service(player, **kw):
    kw.setdefault('accept_timeout', 0.5)
    kw.setdefault('poll', 0.01)
    return w.Bo3Service(player, out_dir=Path(kw.pop('out_dir')), **kw)


def test_submit_validations(tmp_path):
    svc = _service(FakePlayer(), out_dir=tmp_path, max_active=1, max_queue=1)
    assert svc.submit('Ash', 'ppo')[0] is False  # aún no conectado
    svc.loop = types.SimpleNamespace(call_soon_threadsafe=lambda f, x: f(x))
    svc.queue = asyncio.Queue()
    assert svc.submit('Ash', 'mcts') == (False, 'Agente no válido.')
    assert svc.submit('bot user', 'ppo')[0] is False  # el propio bot
    assert svc.submit('<x>', 'ppo')[0] is False
    assert svc.submit('Ash', 'ppo') == (True, 'ok')
    assert svc.submit('ASH', 'ddqn')[0] is False  # misma persona con serie activa
    assert svc.submit('Misty', 'ddqn') == (True, 'ok')
    assert svc.submit('Brock', 'ppo')[0] is False  # capacidad: 1 activa + 1 en cola
    assert svc.status('ash')[0].state == 'en_cola' and svc.status('Misty')[1] == 2


def test_service_flow_accept_reject_missing_and_concurrent(tmp_path):
    player = FakePlayer({'misty': 'reject', 'brock': 'missing', 'gary': 'ignore'})

    async def scenario():
        svc = _service(player, out_dir=tmp_path, max_active=3)
        task = asyncio.create_task(svc.run())
        await asyncio.sleep(0.01)
        for user, agent in (('Ash', 'ppo'), ('Misty', 'ddqn'), ('Gary', 'ppo'), ('Dawn', 'ddqn')):
            assert svc.submit(user, agent) == (True, 'ok')
        await asyncio.sleep(1.5)
        # Ash y Dawn juegan a la vez, cada uno con su agente
        assert svc.series['ash'].state == 'jugando' and svc.series['dawn'].state == 'jugando'
        assert player.assign == {'ash': 'ppo', 'dawn': 'ddqn'}
        assert svc.series['misty'].state == 'no_acepto' and 'Rechazó' in svc.series['misty'].msg
        assert svc.series['gary'].state == 'no_acepto'
        assert ('msg', '/cancelchallenge Gary') in player.ps_client.sent
        # Ash gana 2-0
        player.battles['battle-ash-1'].finished = True
        player.battles['battle-ash-1'].won = True
        player.battles['battle-ash-2'] = FakeBattle('Ash', True, True, tag='battle-ash-2')
        await asyncio.sleep(0.1)
        assert (svc.series['ash'].state, svc.series['ash'].wins) == ('terminada', 2)
        assert 'ash' not in player.assign and 'dawn' in player.assign
        task.cancel()

    asyncio.run(scenario())
    saved = list(tmp_path.glob('bo3_ash_*.json'))
    assert len(saved) == 1 and '"victorias": 2' in saved[0].read_text(encoding='utf-8')


def test_missing_user_detected_without_waiting_full_timeout(tmp_path, monkeypatch):
    player = FakePlayer({'brock': 'missing'})
    t = [0.0]
    monkeypatch.setattr(w.time, 'time', lambda: t[0])

    async def scenario():
        svc = _service(player, out_dir=tmp_path, accept_timeout=120)
        task = asyncio.create_task(svc.run())
        await asyncio.sleep(0.01)
        svc.submit('Brock', 'ppo')
        await asyncio.sleep(0.05)
        t[0] = 16.0
        await asyncio.sleep(0.05)
        assert svc.series['brock'].state == 'no_acepto' and 'no registró' in svc.series['brock'].msg
        task.cancel()

    asyncio.run(scenario())


def test_max_active_holds_queue(tmp_path):
    player = FakePlayer()

    async def scenario():
        svc = _service(player, out_dir=tmp_path, max_active=1)
        task = asyncio.create_task(svc.run())
        await asyncio.sleep(0.01)
        svc.submit('Ash', 'ppo')
        svc.submit('Misty', 'ddqn')
        await asyncio.sleep(0.3)
        assert svc.series['ash'].state == 'jugando' and svc.series['misty'].state == 'en_cola'
        assert svc.status('misty')[1] == 1
        player.battles['battle-ash-1'].finished = True
        player.battles['battle-ash-1'].won = False
        player.battles['battle-ash-2'] = FakeBattle('Ash', True, False, tag='battle-ash-2')
        await asyncio.sleep(0.3)
        assert svc.series['ash'].state == 'terminada' and svc.series['misty'].state == 'jugando'
        task.cancel()

    asyncio.run(scenario())


def test_router_delegates_by_opponent():
    pytest.importorskip('poke_env')
    from poke_env.ps_client import AccountConfiguration

    class Agent:
        def __init__(self, name):
            self.name = name

        def choose_move(self, battle):
            return f'move-{self.name}'

        def teampreview(self, battle):
            return f'/team {self.name}'

    Router = w._make_router_class()
    r = Router(agents={'ppo': Agent('ppo'), 'ddqn': Agent('ddqn')},
               account_configuration=AccountConfiguration('BotTestRouter', None),
               battle_format=w.FORMAT, start_listening=False)
    r.assign = {'ash': 'ppo', 'misty': 'ddqn'}
    assert r.choose_move(FakeBattle('Ash')) == 'move-ppo'
    assert r.teampreview(FakeBattle('Misty')) == '/team ddqn'
    assert r.teampreview(FakeBattle('Nadie')) == '/team 1234'

    asyncio.run(r._update_challenges(['', 'updatechallenges',
                                      '{"challengesFrom":{},"challengeTo":{"to":"Ash K","format":"x"}}']))
    assert r.pending_to == {'ashk'}
    asyncio.run(r._update_challenges(['', 'updatechallenges', '{"challengesFrom":{},"challengeTo":null}']))
    assert r.pending_to == set()


def test_render_page_escapes_and_refreshes():
    s = w.Series(user='<b>x', uid='bx', agent='ppo', state='retando')
    page = w.render_page(w.AGENTS, status=(s, 0), bot='Bot<1>', user='<b>x')
    assert '<b>x' not in page
    assert '&lt;b&gt;x' in page and 'Bot&lt;1&gt;' in page
    assert 'http-equiv="refresh"' in page
    s.state = 'terminada'
    assert 'http-equiv="refresh"' not in w.render_page(w.AGENTS, status=(s, 0))
