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

    async def accept_challenge(self, user, team):
        self.sent.append(('accept', user))
        uid = w.to_id(user)
        if self.player.incoming.pop(uid, None) == w.FORMAT:
            tag = f'battle-{uid}-in'
            self.player.battles[tag] = FakeBattle(user, tag=tag)


class FakePlayer:
    """Simula Showdown: según ``behaviour[uid]`` el rival acepta, rechaza, ignora o no existe."""

    def __init__(self, behaviour=None):
        self.username = 'BotUser'
        self.agents = {'ppo': object(), 'ddqn': object()}
        self.assign = {}
        self.pending_to = set()
        self.incoming = {}
        self.popups = []
        self.format_names = {w.FORMAT: '[Gen 9 Champions] VGC 2026 Reg M-C (Bo3)'}
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
        if mode in ('locked', 'notfound'):
            text = ('You are locked and cannot challenge unlocked users.' if mode == 'locked'
                    else f"The user '{user}' was not found.")
            loop.call_later(0.02, lambda: self.popups.append((w.time.time(), text)))
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


def test_summarize_web_totals_and_replay_links(tmp_path):
    import json
    import summarize_web as sw
    (tmp_path / 'replays').mkdir()
    (tmp_path / 'replays' / 'bot - battle-x-111.html').write_text('r', encoding='utf-8')
    games = [{'sala': 'battle-x-111', 'turnos': 7, 'ganó': True}, {'sala': 'battle-x-222', 'turnos': 9, 'ganó': True}]
    for name, d in (('bo3_ash_20261127_101500.json', {'agente': 'PPO v2', 'rival': 'Ash', 'estado': 'terminada',
                                                       'serie': 'ganada', 'victorias': 2, 'derrotas': 0,
                                                       'partidas': games}),
                    ('bo3_misty_20261127_090000.json', {'agente': 'PPO v2', 'rival': 'Misty', 'estado': 'expirada',
                                                         'serie': 'perdida', 'victorias': 0, 'derrotas': 1,
                                                         'partidas': games[:1]})):
        (tmp_path / name).write_text(json.dumps(d), encoding='utf-8')
    md = sw.summarize(sw.load_series(tmp_path), tmp_path)
    assert '| PPO v2 | 2 | 1 | 0 | 1 | 2–1 | 67 % |' in md
    assert md.index('Misty') < md.index('| Ash')  # orden por fecha
    assert '[P1](replays/bot%20-%20battle-x-111.html) P2' in md


def test_router_observes_challenge_pms_popups_and_formats():
    pytest.importorskip('poke_env')
    from poke_env.ps_client import AccountConfiguration
    Router = w._make_router_class()
    r = Router(agents={}, account_configuration=AccountConfiguration('BotObs', None),
               battle_format=w.FORMAT, start_listening=False)
    # Mensajes reales capturados de Showdown (27-11-2026)
    r.observe(f'|pm| BotObs| Deep1ODDQN|/challenge {w.FORMAT}|{w.FORMAT}|||')
    assert r.pending_to == {'deep1oddqn'}
    r.observe('|pm| Deep1ODDQN| BotObs|/challenge')
    assert r.pending_to == set()
    r.observe(f'|pm|+Ash K| BotObs|/challenge {w.FORMAT}|{w.FORMAT}|||')
    assert r.incoming == {'ashk': w.FORMAT}
    r.observe('|pm| BotObs|+Ash K|/challenge')
    assert r.incoming == {}
    r.observe('|popup|You are locked and cannot challenge unlocked users.||If this user is your friend, ask.')
    assert 'locked' in r.popups[-1][1] and '||' not in r.popups[-1][1]
    r.observe('|formats|,1|S/V Singles|[Gen 9] Random Battle,f|[Gen 9 Champions] VGC 2026 Reg M-C (Bo3),e')
    assert r.format_names[w.FORMAT] == '[Gen 9 Champions] VGC 2026 Reg M-C (Bo3)'
    assert r.format_names['gen9randombattle'] == '[Gen 9] Random Battle'


def test_locked_switches_to_inbound_and_accepts_only_registered_users(tmp_path):
    player = FakePlayer({'ash': 'locked'})

    async def scenario():
        svc = _service(player, out_dir=tmp_path, inbound_timeout=0.6)
        task = asyncio.create_task(svc.run())
        await asyncio.sleep(0.01)
        svc.submit('Ash', 'ppo')
        await asyncio.sleep(0.1)
        assert svc.inbound_only and svc.series['ash'].state == 'esperando_reto'
        # Un desconocido reta al bot: se ignora
        player.incoming['stranger'] = w.FORMAT
        # Ash reta con otro formato: se le avisa y no se acepta
        player.incoming['ash'] = 'gen9randombattle'
        await asyncio.sleep(0.05)
        assert 'otro formato' in svc.series['ash'].msg and ('accept', 'Ash') not in player.ps_client.sent
        player.incoming['ash'] = w.FORMAT
        await asyncio.sleep(0.05)
        assert ('accept', 'Ash') in player.ps_client.sent and svc.series['ash'].state == 'jugando'
        assert player.assign['ash'] == 'ppo' and 'stranger' in player.incoming
        # La siguiente serie ya no intenta retar
        n_challenges = sum(1 for x in player.ps_client.sent if x[0] == 'challenge')
        svc.submit('Misty', 'ddqn')
        await asyncio.sleep(0.05)
        assert svc.series['misty'].state == 'esperando_reto'
        assert sum(1 for x in player.ps_client.sent if x[0] == 'challenge') == n_challenges
        await asyncio.sleep(0.8)
        assert svc.series['misty'].state == 'no_acepto' and 'No llegó' in svc.series['misty'].msg
        task.cancel()

    asyncio.run(scenario())


def test_not_found_popup_shown_to_user(tmp_path):
    player = FakePlayer({'nadie': 'notfound'})

    async def scenario():
        svc = _service(player, out_dir=tmp_path)
        task = asyncio.create_task(svc.run())
        await asyncio.sleep(0.01)
        svc.submit('Nadie', 'ppo')
        await asyncio.sleep(0.1)
        s = svc.series['nadie']
        assert s.state == 'no_acepto' and 'was not found' in s.msg and not svc.inbound_only
        task.cancel()

    asyncio.run(scenario())


def test_render_inbound_instructions():
    s = w.Series(user='Ash', uid='ash', agent='ppo', state='esperando_reto')
    page = w.render_page(w.AGENTS, status=(s, 0), bot='teamofabrexuxa',
                         fmt_label='[Gen 9 Champions] VGC 2026 Reg M-C (Bo3)', inbound=True)
    assert 'Rétalo tú' in page and 'VGC 2026 Reg M-C (Bo3)' in page and 'http-equiv="refresh"' in page


def test_router_locked_login_counts_as_logged_in():
    pytest.importorskip('poke_env')
    from poke_env.ps_client import AccountConfiguration
    Router = w._make_router_class()
    r = Router(agents={}, account_configuration=AccountConfiguration('BotLock', None),
               battle_format=w.FORMAT, start_listening=False)
    r.observe('|updateuser| Guest 123|0|2|{}')
    assert not r.ps_client.logged_in.is_set() and not r.locked
    r.observe('|updateuser|‽BotLock|1|266|{}')
    assert r.ps_client.logged_in.is_set() and r.locked
