"""Página web mínima: una persona elige agente (PPO o DDQN) y escribe su usuario de Showdown; el agente
la reta a un BO3 en el servidor oficial con la cuenta exclusiva de la web del ``.env``
(``WEB_PS_USERNAME``; si falta, ``PS_USERNAME``).

Un solo proceso y **una sola conexión** a Showdown (los agentes comparten la cuenta): ``RouterPlayer`` recibe
todas las batallas y delega el preview y cada decisión en el agente asignado a ese rival. Los agentes,
creados sin conexión propia y con el equipo fijo:

- **PPO v2**, semilla s1 (``checkpoints/ppo_v2/ppo_s1/final.pt``, la elegida por ``val``): combate con sorteo
  T = 0,5, el modo contra personas (memorizacion_equipos §19).
- **DDQN v1** (``logs/online_ddqn/``), igual que ``ddqn_online_bo3.py``: combate determinista.
- Ambos: preview v2.1b con sorteo T = 1,0 y observación P5 con la calculadora real (``node`` + ``../damage-calc``).

- Varios jugadores distintos a la vez (``--max-active`` series en curso); una serie activa por usuario.
- Showdown admite un reto saliente pendiente a la vez: los retos se envían en serie; si no se acepta en
  ``--accept-timeout`` s se cancela (``/cancelchallenge``) y se pasa al siguiente.
- Showdown bloquea (*lock*) las IP de centros de datos como la EC2 del Learner Lab (27-11-2026): un usuario
  bloqueado no puede retar, pero sí aceptar. Al primer popup "locked" el bot pasa a modo inverso: la página
  pide a la persona que rete ella al bot en el formato BO3 y el bot acepta solo a quien se registró
  (``--inbound-timeout`` min); ``--inbound-only`` empieza directamente así. Los demás popups (usuario no
  encontrado, etc.) se muestran en la página.
- Timer de Showdown activado en cada partida, para que una persona ausente no deje colgada la serie.
- Cada serie terminada se guarda en ``logs/online_web/bo3_<usuario>_<fecha>.json`` (+ replays).

    python scripts/online/bo3_web.py --check          # solo carga checkpoints, preview y calculadora
    python scripts/online/bo3_web.py --port 8000      # http://localhost:8000
"""
from __future__ import annotations

import argparse
import asyncio
import html
import json
import logging
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import parse_qs, quote, urlparse

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ppo_online_bo3 import _read_env  # noqa: E402

OUT = REPO / 'logs' / 'online_web'
FORMAT = 'gen9championsvgc2026regmcbo3'
AGENTS = {'ppo': 'PPO v2', 'ddqn': 'DDQN v1'}  # clave del formulario -> nombre visible

ACTIVE_STATES = ('en_cola', 'retando', 'esperando_reto', 'jugando')
STATE_TEXT = {
    'en_cola': 'En cola: el reto se enviará en cuanto el bot termine el reto anterior.',
    'retando': 'Reto enviado: acéptalo en Showdown.',
    'esperando_reto': 'Esperando tu reto.',
    'jugando': 'Serie en curso.',
    'terminada': 'Serie terminada.',
    'no_acepto': 'El reto no se aceptó.',
    'expirada': 'La serie superó el tiempo máximo y se dejó de seguir.',
    'error': 'Error del bot.',
}

LOG = logging.getLogger('bo3web')

_USER_RE = re.compile(r'^[A-Za-z0-9 _.\-]{1,18}$')


def to_id(name: Optional[str]) -> str:
    """ID de Showdown: minúsculas, solo letras y números."""
    return re.sub(r'[^a-z0-9]', '', (name or '').lower())


def valid_username(name: Optional[str]) -> Optional[str]:
    """Nombre limpio si parece un usuario válido de Showdown (1–18 caracteres), si no ``None``."""
    name = (name or '').strip()
    return name if _USER_RE.match(name) and to_id(name) else None


def parse_formats(items: Iterable[str]) -> Dict[str, str]:
    """``|formats|`` de Showdown -> {id: nombre visible} (p. ej. '[Gen 9] Random Battle')."""
    out: Dict[str, str] = {}
    for item in items:
        name = item.split(',', 1)[0].strip()
        if name and to_id(name):
            out[to_id(name)] = name
    return out


def series_score(battles: Dict[str, Any], uid: str, ignore: Iterable[str]) -> Tuple[List[str], int, int, bool]:
    """Partidas nuevas contra ``uid`` (las de ``ignore`` ya existían), victorias, derrotas y si acabó el BO3."""
    ignore = set(ignore)
    tags: List[str] = []
    wins = losses = 0
    all_finished = True
    for tag, b in battles.items():
        if tag in ignore or to_id(getattr(b, 'opponent_username', None)) != uid:
            continue
        tags.append(tag)
        if getattr(b, 'finished', False):
            if b.won:
                wins += 1
            elif b.won is False:
                losses += 1
        else:
            all_finished = False
    done = wins >= 2 or losses >= 2 or (len(tags) >= 3 and all_finished)
    return tags, wins, losses, done


@dataclass
class Series:
    user: str
    uid: str
    agent: str
    state: str = 'en_cola'
    msg: str = ''
    t_created: float = field(default_factory=time.time)
    t_challenge: Optional[float] = None
    t_start: Optional[float] = None
    t_end: Optional[float] = None
    wins: int = 0
    losses: int = 0
    tags: List[str] = field(default_factory=list)
    ignore: Set[str] = field(default_factory=set)


# --------------------------------------------------------------------------- #
# Jugador de poke-env: una conexión, varios agentes
# --------------------------------------------------------------------------- #

def _make_router_class():
    import orjson
    from poke_env.player import Player
    from poke_env.player.battle_order import DefaultBattleOrder

    class RouterPlayer(Player):
        """Delega cada batalla en el agente asignado al rival (``assign[uid] = clave``)."""

        def __init__(self, *args, agents: Dict[str, Any], **kwargs):
            super().__init__(*args, **kwargs)
            self.agents = agents
            self.assign: Dict[str, str] = {}
            self.pending_to: Set[str] = set()       # retos salientes pendientes (IDs)
            self.incoming: Dict[str, str] = {}      # retos entrantes pendientes: id -> formato
            self.popups: List[Tuple[float, str]] = []
            self.format_names: Dict[str, str] = {}
            self.ots: Dict[str, List[str]] = {}      # sala -> líneas |showteam| (hojas abiertas)
            self.locked = False                     # sesión bloqueada (‽): no puede retar, sí aceptar
            # poke-env solo registra los popups y los PM de retos: se observan antes de su manejo normal.
            self._orig_handle_message = self.ps_client._handle_message
            self.ps_client._handle_message = self._observe_then_handle

        async def _observe_then_handle(self, message: str):
            try:
                self.observe(message)
            except Exception as exc:  # noqa: BLE001 - un mensaje raro no debe tumbar la conexión
                self.logger.warning('mensaje no reconocido (%s): %.200s', exc, message)
            await self._orig_handle_message(message)

        def observe(self, message: str) -> None:
            """Popups, nombres de formatos y PM ``/challenge`` (Showdown avisa así de los retos)."""
            me = to_id(self.username)
            room = message.split('\n', 1)[0][1:].strip() if message.startswith('>') else ''
            for line in message.split('\n'):
                parts = line.split('|')
                if len(parts) < 2:
                    continue
                if parts[1] == 'updateuser' and len(parts) > 2 and parts[2].strip().startswith('‽') \
                        and to_id(parts[2]) == me:
                    # "‽nombre": sesión iniciada pero bloqueada (Showdown marca la IP de la EC2 como proxy).
                    # poke-env solo reconoce " nombre"; un usuario bloqueado puede jugar y aceptar retos.
                    if not self.ps_client.logged_in.is_set():
                        self.logger.warning('sesión iniciada como %s pero bloqueada (lock): solo se aceptan '
                                            'retos', parts[2].strip())
                    self.locked = True
                    self.ps_client.logged_in.set()
                elif parts[1] == 'popup':
                    self.popups.append((time.time(), '|'.join(parts[2:]).replace('||', ' ').strip()))
                    del self.popups[:-50]
                elif parts[1] == 'showteam' and room:
                    # Hojas abiertas: poke-env las procesa sin pasarlas a parse_message, así que no quedaban
                    # en el replay guardado (27-11-2026 se diagnosticó por error una serie como cerrada).
                    self.ots.setdefault(room, []).append(line)
                    battle = self._battles.get(room)
                    if battle is not None:
                        battle._replay_data.append(parts)
                elif parts[1] == 'formats':
                    self.format_names.update(parse_formats(parts[2:]))
                elif parts[1] == 'pm' and len(parts) >= 5 and parts[4].startswith('/challenge'):
                    sender, receiver = to_id(parts[2]), to_id(parts[3])
                    other = receiver if sender == me else sender
                    fmt = parts[4][len('/challenge'):].strip()
                    if not fmt:  # "/challenge" vacío: ya no hay reto entre ambos (aceptado, rechazado o cancelado)
                        self.pending_to.discard(other)
                        self.incoming.pop(other, None)
                    elif sender == me:
                        self.pending_to.add(receiver)
                    elif receiver == me:
                        self.incoming[sender] = fmt

        def agent_for(self, battle):
            key = self.assign.get(to_id(getattr(battle, 'opponent_username', None)))
            return self.agents.get(key) if key else None

        def choose_move(self, battle):
            agent = self.agent_for(battle)
            if agent is None:
                self.logger.warning('batalla sin agente asignado: %s', battle.battle_tag)
                return DefaultBattleOrder()
            return agent.choose_move(battle)

        def teampreview(self, battle):
            agent = self.agent_for(battle)
            if agent is None:
                return '/team 1234'
            return agent.teampreview(battle)

        async def _update_challenges(self, split_message):
            try:
                data = orjson.loads(split_message[2])
                tos: Set[str] = set()
                ct = data.get('challengeTo')
                if isinstance(ct, dict) and ct.get('to'):
                    tos.add(to_id(ct['to']))
                cts = data.get('challengesTo')  # forma alternativa (varios retos)
                if isinstance(cts, dict):
                    tos |= {to_id(k) for k in cts}
                self.pending_to = tos
            except Exception as exc:  # noqa: BLE001 - un mensaje raro no debe tumbar la conexión
                self.logger.warning('updatechallenges no reconocido (%s): %s', exc, split_message)
            await super()._update_challenges(split_message)

    return RouterPlayer


# --------------------------------------------------------------------------- #
# Cola de retos y seguimiento de series
# --------------------------------------------------------------------------- #

class Bo3Service:
    def __init__(self, player, *, fmt: str = FORMAT, accept_timeout: float = 120.0,
                 series_timeout: float = 90 * 60.0, max_active: int = 4, max_queue: int = 10,
                 out_dir: Path = OUT, poll: float = 1.0, inbound_timeout: float = 300.0):
        self.player = player
        # Showdown bloquea (lock) las IP de centros de datos: un usuario bloqueado no puede retar, pero sí
        # aceptar. Tras el primer "locked" la persona reta al bot y el bot acepta (``inbound_only``).
        self.inbound_only = False
        self.inbound_timeout = inbound_timeout
        self.fmt = fmt
        self.accept_timeout = accept_timeout
        self.series_timeout = series_timeout
        self.max_active = max_active
        self.max_queue = max_queue
        self.out_dir = out_dir
        self.poll = poll
        self.series: Dict[str, Series] = {}
        self.lock = threading.Lock()
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.queue: Optional[asyncio.Queue] = None
        self.order: List[str] = []  # uids en cola, en orden

    # -- lado web (hilo del servidor HTTP) -------------------------------- #
    def submit(self, user: str, agent: str) -> Tuple[bool, str]:
        name = valid_username(user)
        if name is None:
            return False, 'Usuario no válido (1–18 caracteres: letras, números, espacio, punto, guion).'
        if agent not in self.player.agents:
            return False, 'Agente no válido.'
        uid = to_id(name)
        if uid == to_id(self.player.username):
            return False, 'Ese es el usuario del bot.'
        if self.queue is None or self.loop is None:
            return False, 'El bot aún se está conectando; prueba en unos segundos.'
        with self.lock:
            cur = self.series.get(uid)
            if cur is not None and cur.state in ACTIVE_STATES:
                return False, f'{name} ya tiene una serie en curso.'
            if sum(s.state in ACTIVE_STATES for s in self.series.values()) >= self.max_active + self.max_queue:
                return False, 'Hay demasiadas series en curso; prueba en unos minutos.'
            s = Series(user=name, uid=uid, agent=agent)
            self.series[uid] = s
            self.order.append(uid)
        self.loop.call_soon_threadsafe(self.queue.put_nowait, s)
        return True, 'ok'

    def status(self, user: str) -> Optional[Tuple[Series, int]]:
        """Serie más reciente del usuario y su posición en la cola (0 si no está en cola)."""
        uid = to_id(user)
        with self.lock:
            s = self.series.get(uid)
            pos = self.order.index(uid) + 1 if s is not None and s.state == 'en_cola' and uid in self.order else 0
        return (s, pos) if s is not None else None

    # -- lado poke-env (bucle de asyncio de poke-env) ---------------------- #
    def _n_playing(self) -> int:
        return sum(s.state in ('retando', 'esperando_reto', 'jugando') for s in self.series.values())

    def format_label(self) -> str:
        return getattr(self.player, 'format_names', {}).get(self.fmt, self.fmt)

    def _popup_since(self, t: float) -> Optional[str]:
        for ts, text in reversed(getattr(self.player, 'popups', [])):
            if ts >= t:
                return text
        return None

    async def _watchdog(self, grace: float = 90.0) -> None:
        """Termina el proceso (systemd lo reinicia) si no hay sesión en ``grace`` s o se cae el websocket."""
        t0 = time.time()
        while True:
            await asyncio.sleep(10)
            client = self.player.ps_client
            listening = getattr(client, '_listening_coroutine', None)
            if listening is not None and listening.done():
                LOG.error('se cerró la conexión con Showdown: salgo para que systemd reinicie')
                os._exit(3)
            if not client.logged_in.is_set() and time.time() - t0 > grace:
                LOG.error('sin sesión en Showdown tras %d s: salgo para que systemd reinicie', int(grace))
                os._exit(3)

    async def run(self, exit_on_disconnect: bool = False) -> None:
        self.loop = asyncio.get_running_loop()
        self.queue = asyncio.Queue()
        if exit_on_disconnect:
            self.loop.create_task(self._watchdog())
        await self.player.ps_client.logged_in.wait()
        if getattr(self.player, 'locked', False):
            self.inbound_only = True
        LOG.info('conectado como %s%s; esperando retos', self.player.username,
                 ' (bloqueado: la gente reta al bot)' if self.inbound_only else '')
        while True:
            s = await self.queue.get()
            try:
                await self._challenge(s)
            except Exception as exc:  # noqa: BLE001
                LOG.exception('error retando a %s', s.user)
                self._finish(s, 'error', f'{type(exc).__name__}: {exc}')

    async def _challenge(self, s: Series) -> None:
        while self._n_playing() >= self.max_active:
            await asyncio.sleep(self.poll)
        with self.lock:
            if s.uid in self.order:
                self.order.remove(s.uid)
            s.state = 'retando'
        s.ignore = set(self.player.battles)
        s.t_challenge = time.time()
        self.player.assign[s.uid] = s.agent
        if self.inbound_only:
            self._start_inbound(s)
            return
        LOG.info('%s reta a %s con %s', self.player.username, s.user, s.agent)
        await self.player.ps_client.challenge(s.user, self.fmt, self.player.next_team)

        seen_pending = False
        while True:
            tags, _w, _l, _d = series_score(self.player.battles, s.uid, s.ignore)
            if tags:
                break
            waited = time.time() - s.t_challenge
            popup = self._popup_since(s.t_challenge)
            if popup is not None:
                if 'locked' in popup.lower():
                    LOG.warning('Showdown bloquea los retos salientes (%s): desde ahora la gente reta al bot', popup)
                    self.inbound_only = True
                    self._start_inbound(s)
                    return
                self._finish(s, 'no_acepto', f'Showdown: {popup}')
                return
            if s.uid in self.player.pending_to:
                seen_pending = True
            elif seen_pending:
                # El reto desapareció: aceptado (la partida llega en segundos) o rechazado.
                await asyncio.sleep(8 * self.poll)
                if series_score(self.player.battles, s.uid, s.ignore)[0]:
                    break
                self._finish(s, 'no_acepto', 'Rechazó o canceló el reto.')
                return
            elif waited > 15:
                self._finish(s, 'no_acepto', 'Showdown no registró el reto: ¿el usuario existe, está conectado '
                                              'y acepta retos?')
                return
            if waited > self.accept_timeout:
                await self.player.ps_client.send_message(f'/cancelchallenge {s.user}')
                self._finish(s, 'no_acepto', f'No se aceptó en {int(self.accept_timeout)} s.')
                return
            await asyncio.sleep(self.poll)

        s.state = 'jugando'
        s.t_start = time.time()
        asyncio.get_running_loop().create_task(self._watch(s))

    def _start_inbound(self, s: Series) -> None:
        s.state = 'esperando_reto'
        s.t_challenge = time.time()
        LOG.info('esperando el reto de %s (%s)', s.user, s.agent)
        asyncio.get_running_loop().create_task(self._inbound(s))

    async def _inbound(self, s: Series) -> None:
        """Espera el reto de ``s.uid`` en el formato correcto, lo acepta y sigue la serie."""
        try:
            accepted_at: Optional[float] = None
            while not series_score(self.player.battles, s.uid, s.ignore)[0]:
                now = time.time()
                fmt = self.player.incoming.get(s.uid)
                if fmt is not None and fmt != self.fmt:
                    s.msg = f'Tu reto es de otro formato ({fmt}); usa {self.format_label()}.'
                elif fmt is not None and (accepted_at is None or now - accepted_at > 20):
                    s.msg = ''
                    accepted_at = now
                    LOG.info('aceptando el reto de %s', s.user)
                    await self.player.ps_client.accept_challenge(s.user, self.player.next_team)
                if now - s.t_challenge > self.inbound_timeout:
                    self._finish(s, 'no_acepto', f'No llegó tu reto en {int(self.inbound_timeout // 60)} min.')
                    return
                await asyncio.sleep(self.poll)
            s.state = 'jugando'
            s.msg = ''
            s.t_start = time.time()
            await self._watch(s)
        except Exception as exc:  # noqa: BLE001
            self._finish(s, 'error', f'{type(exc).__name__}: {exc}')

    async def _watch(self, s: Series) -> None:
        try:
            while True:
                s.tags, s.wins, s.losses, done = series_score(self.player.battles, s.uid, s.ignore)
                if done:
                    self._finish(s, 'terminada', '')
                    return
                if time.time() - (s.t_start or time.time()) > self.series_timeout:
                    self._finish(s, 'expirada', f'Más de {int(self.series_timeout // 60)} min.')
                    return
                await asyncio.sleep(3 * self.poll)
        except Exception as exc:  # noqa: BLE001
            self._finish(s, 'error', f'{type(exc).__name__}: {exc}')

    def _finish(self, s: Series, state: str, msg: str) -> None:
        with self.lock:
            s.state = state
            s.msg = msg
            s.t_end = time.time()
            if s.uid in self.order:
                self.order.remove(s.uid)
        self.player.assign.pop(s.uid, None)
        LOG.info('serie %s (%s): %s %d-%d %s', s.user, s.agent, state, s.wins, s.losses, msg)
        if s.tags:
            self._save(s)

    def _save(self, s: Series) -> None:
        games = []
        for tag in s.tags:
            b = self.player.battles.get(tag)
            if b is None:
                continue
            ots = getattr(self.player, 'ots', {}).get(tag, [])
            games.append({'sala': tag, 'turnos': b.turn, 'ganó': b.won, 'hojas_abiertas': bool(ots),
                          'hojas': ots,
                          'equipo_rival': sorted(m.species for m in b.opponent_team.values())})
        agent = self.player.agents.get(s.agent)
        summary = {
            'agente': AGENTS.get(s.agent, s.agent), 'rival': s.user, 'formato': self.fmt, 'estado': s.state,
            'nota': s.msg, 'victorias': s.wins, 'derrotas': s.losses,
            'serie': 'ganada' if s.wins > s.losses else ('perdida' if s.losses > s.wins else 'empate/incompleta'),
            'checkpoint': getattr(agent, 'checkpoint_path', None), 'partidas': games,
            'decisiones_acumuladas_agente': (agent.decision_stats.as_dict()
                                             if hasattr(agent, 'decision_stats') else None),
            'espera_aceptar_s': round((s.t_start or s.t_end or 0) - (s.t_challenge or 0), 1),
            'duracion_min': round(((s.t_end or 0) - (s.t_start or s.t_end or 0)) / 60, 1),
        }
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime('%Y%m%d_%H%M%S')
            (self.out_dir / f'bo3_{s.uid}_{stamp}.json').write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        except OSError as exc:
            LOG.warning('no se pudo guardar el resumen: %s', exc)


# --------------------------------------------------------------------------- #
# Web
# --------------------------------------------------------------------------- #

_CSS = """
:root{color-scheme:light dark;--bg:#f6f7f9;--fg:#1d2330;--card:#fff;--mut:#667;--acc:#2f6fde;--err:#c0392b}
@media (prefers-color-scheme:dark){:root{--bg:#14171c;--fg:#e6e8ec;--card:#1e232b;--mut:#9aa;--acc:#6e9cf0;--err:#ff7b6b}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}
main{max-width:420px;margin:40px auto;padding:0 16px}.card{background:var(--card);border-radius:12px;padding:20px;
box-shadow:0 1px 3px #0002;margin-bottom:16px}h1{font-size:1.3rem;margin:0 0 16px}label{display:block;margin:12px 0 4px;
font-weight:600}select,input{width:100%;padding:10px;font:inherit;border:1px solid #8884;border-radius:8px;
background:var(--bg);color:var(--fg)}button{margin-top:16px;width:100%;padding:12px;font:inherit;font-weight:600;
border:0;border-radius:8px;background:var(--acc);color:#fff;cursor:pointer}.mut{color:var(--mut);font-size:.9rem}
.err{color:var(--err)}.big{font-size:1.6rem;font-weight:700}
"""


def render_page(agents: Dict[str, str], *, status: Optional[Tuple[Series, int]] = None,
                error: str = '', user: str = '', bot: str = '', fmt_label: str = '',
                inbound: bool = False, inbound_min: int = 5) -> str:
    esc = html.escape
    refresh = ''
    status_html = ''
    if status is not None:
        s, pos = status
        if s.state in ACTIVE_STATES:
            refresh = '<meta http-equiv="refresh" content="5">'
        line = STATE_TEXT.get(s.state, s.state)
        if s.state == 'en_cola' and pos:
            line += f' (posición {pos})'
        if s.state == 'retando':
            line += f' Busca el reto de <b>{esc(bot)}</b> en Showdown.'
        if s.state == 'esperando_reto':
            line = (f'<b>Rétalo tú:</b> en Showdown busca al usuario <b>{esc(bot)}</b>, pulsa <i>Challenge</i>, '
                    f'elige el formato <b>{esc(fmt_label)}</b> y tu equipo. El bot acepta en segundos '
                    f'(tienes {inbound_min} min).')
        score = (f'<p class="big">{esc(agents.get(s.agent, s.agent))} {s.wins} – {s.losses} {esc(s.user)}</p>'
                 if s.state in ('jugando', 'terminada', 'expirada') else '')
        status_html = (f'<div class="card"><h1>{esc(s.user)} vs {esc(agents.get(s.agent, s.agent))}</h1>'
                       f'{score}<p>{line}</p>'
                       + (f'<p class="mut">{esc(s.msg)}</p>' if s.msg else '') + '</div>')
    options = ''.join(f'<option value="{esc(k)}">{esc(v)}</option>' for k, v in agents.items())
    err_html = f'<p class="err">{esc(error)}</p>' if error else ''
    how = (f'Conéctate a play.pokemonshowdown.com con ese usuario. Después de Enviar, <b>reta tú</b> a '
           f'<b>{esc(bot)}</b> en el formato <b>{esc(fmt_label)}</b> con tu equipo.' if inbound else
           f'Conéctate a play.pokemonshowdown.com con ese usuario: te llegará un reto de <b>{esc(bot)}</b> '
           f'que debes aceptar en 2 minutos.')
    return f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">{refresh}<title>Reto BO3</title>
<style>{_CSS}</style></head><body><main>{status_html}
<div class="card"><h1>Reta a un agente (BO3)</h1>
<form method="post" action="/retar">
<label for="agente">Agente</label><select id="agente" name="agente">{options}</select>
<label for="usuario">Tu usuario de Showdown</label>
<input id="usuario" name="usuario" maxlength="18" required autocomplete="off" value="{esc(user)}">
<button type="submit">Enviar</button>{err_html}
<p class="mut">{how}</p>
</form></div></main></body></html>"""


def make_handler(service: Bo3Service):
    class Handler(BaseHTTPRequestHandler):
        server_version = 'bo3web'

        def log_message(self, fmt, *args):  # silencioso: poke-env ya registra lo importante
            pass

        def _send(self, code: int, body: str, ctype: str = 'text/html; charset=utf-8',
                  headers: Optional[Dict[str, str]] = None) -> None:
            data = body.encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == '/salud':
                self._send(200, json.dumps({'ok': True, 'conectado': service.queue is not None}),
                           'application/json')
                return
            if url.path != '/':
                self._send(404, 'no encontrado', 'text/plain; charset=utf-8')
                return
            user = (parse_qs(url.query).get('u') or [''])[0][:18]
            st = service.status(user) if user else None
            self._send(200, render_page(AGENTS_AVAILABLE(service), status=st, user=user, **_page_kw(service)))

        def do_POST(self):
            if urlparse(self.path).path != '/retar':
                self._send(404, 'no encontrado', 'text/plain; charset=utf-8')
                return
            n = int(self.headers.get('Content-Length') or 0)
            if n > 2000:
                self._send(413, 'demasiado grande', 'text/plain; charset=utf-8')
                return
            form = parse_qs(self.rfile.read(n).decode('utf-8', 'replace'))
            user = (form.get('usuario') or [''])[0].strip()
            agent = (form.get('agente') or [''])[0].strip()
            ok, msg = service.submit(user, agent)
            if ok:
                self._send(303, '', headers={'Location': '/?u=' + quote(user)})
            else:
                self._send(400, render_page(AGENTS_AVAILABLE(service), error=msg, user=user[:18],
                                            **_page_kw(service)))

    return Handler


def _page_kw(service: Bo3Service) -> Dict[str, Any]:
    return {'bot': service.player.username, 'fmt_label': service.format_label(),
            'inbound': service.inbound_only, 'inbound_min': int(service.inbound_timeout // 60)}


def AGENTS_AVAILABLE(service: Bo3Service) -> Dict[str, str]:  # noqa: N802
    return {k: v for k, v in AGENTS.items() if k in service.player.agents}


# --------------------------------------------------------------------------- #
# Arranque
# --------------------------------------------------------------------------- #

def build_agents(args, user: str) -> Tuple[Dict[str, Any], str]:
    """Crea PPO y DDQN sin conexión, igual que sus scripts online. Devuelve (agentes, equipo)."""
    from poke_env.ps_client import AccountConfiguration
    import ddqn_online_bo3 as dd
    from showdown.inference_sim.team_factory import load_team_file, to_showdown_export
    from showdown.rl.ddqn.player import DDQNPlayer
    from showdown.rl.ppo.player import PPOPlayer

    os.environ['PPO_DEVICE'] = 'cpu'
    os.environ['DDQN_DEVICE'] = 'cpu'
    os.environ['PREVIEW_DEVICE'] = 'cpu'
    os.environ.pop('PS_RL_STRICT', None)  # un error en un turno no debe tumbar la serie (se cuenta)
    team = to_showdown_export(load_team_file(REPO / args.team))
    common = dict(account_configuration=AccountConfiguration(user, None), battle_format=args.format,
                  team=team, start_listening=False, log_level=logging.WARNING)

    # DDQN v1: preview con sorteo T=1 contra personas y calculadora real (comprobada por _setup).
    ns = argparse.Namespace(checkpoint=args.ddqn_checkpoint, preview=args.ddqn_preview, team=args.team,
                            preview_greedy=False, preview_temperature=1.0)
    ck, pv, _team = dd._setup(ns)
    ddqn = DDQNPlayer(checkpoint=str(ck), preview_checkpoint=str(pv), epsilon=0.0, required=True, **common)

    # PPO v2 (semilla s1, la elegida por val): mismo preview v2.1b (sorteo T=1, PREVIEW_SAMPLE ya fijado por
    # _setup) y P5 con la calculadora real; combate con sorteo T=0,5, el modo contra personas (§19).
    pck = Path(args.ppo_checkpoint).resolve()
    ppv = Path(args.ppo_preview).resolve() if args.ppo_preview else pv
    for p in (pck, ppv):
        if not p.is_file():
            raise FileNotFoundError(f'falta {p}')
    ppo = PPOPlayer(checkpoint=str(pck), preview_checkpoint=str(ppv), greedy=False,
                    temperature=args.ppo_temperature, **common)
    if not ppo.checkpoint_path or not getattr(ppo.agent.cfg, 'obs_abilities', False):
        raise RuntimeError(f'{pck.name} no es PPO v2 (sin checkpoint u obs_abilities=False)')
    for name, a in (('ppo', ppo), ('ddqn', ddqn)):
        a._frozen_preview._ensure()
        if not a._frozen_preview.available:
            raise RuntimeError(f'preview de {name} no disponible')
    print(f'[web] agentes listos: PPO v2={pck.parent.name}/{pck.name} (T={args.ppo_temperature}, preview '
          f'{ppv.name}), DDQN v1={ck.name} (preview {pv.name}); previews con sorteo T=1',
          flush=True)
    return {'ppo': ppo, 'ddqn': ddqn}, team


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--format', default=FORMAT)
    ap.add_argument('--team', default='showdown/rl/data/ddqn_team.txt')
    ap.add_argument('--ppo-checkpoint', default=str(REPO / 'checkpoints/ppo_v2/ppo_s1/final.pt'))
    ap.add_argument('--ppo-preview', default=None, help='por defecto, el mismo preview v2.1b de DDQN v1')
    ap.add_argument('--ppo-temperature', type=float, default=0.5, help='sorteo del combate de PPO v2')
    ap.add_argument('--ddqn-checkpoint', default=str(REPO / 'logs/online_ddqn/ddqn_v1_s0_final.pt'))
    ap.add_argument('--ddqn-preview', default=str(REPO / 'logs/online_ddqn/preview_v21b_s0.pt'))
    ap.add_argument('--max-active', type=int, default=4, help='series jugándose a la vez')
    ap.add_argument('--max-queue', type=int, default=10, help='series esperando turno de reto')
    ap.add_argument('--accept-timeout', type=float, default=120.0, help='segundos para aceptar el reto')
    ap.add_argument('--series-timeout', type=float, default=90.0, help='minutos máximos por serie')
    ap.add_argument('--inbound-timeout', type=float, default=5.0,
                    help='minutos para que la persona rete al bot cuando Showdown bloquea los retos salientes')
    ap.add_argument('--inbound-only', action='store_true',
                    help='no intentar retar: la persona siempre reta al bot (IP bloqueada por Showdown)')
    ap.add_argument('--check', action='store_true', help='solo carga los agentes, sin conectarse')
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [web] %(message)s')
    LOG.setLevel(logging.INFO)
    env = _read_env(REPO / '.env')
    # Cuenta exclusiva de la web (WEB_PS_*); si falta, la cuenta por defecto (PS_*).
    user = (os.environ.get('WEB_PS_USERNAME') or env.get('WEB_PS_USERNAME')
            or os.environ.get('PS_USERNAME') or env.get('PS_USERNAME'))
    pwd = (os.environ.get('WEB_PS_PASSWORD') or env.get('WEB_PS_PASSWORD')
           or os.environ.get('PS_PASSWORD') or env.get('PS_PASSWORD'))
    if not user or not pwd:
        print('[web] faltan WEB_PS_USERNAME / WEB_PS_PASSWORD (o PS_*) en .env', flush=True)
        return 2
    try:
        agents, team = build_agents(args, user)
    except Exception as exc:  # noqa: BLE001
        print(f'[web] ERROR: {exc}', flush=True)
        return 2
    if args.check:
        print('[web] comprobación completa; listo para servir (sin --check)', flush=True)
        return 0

    from poke_env.concurrency import POKE_LOOP
    from poke_env.ps_client import AccountConfiguration, ShowdownServerConfiguration

    (OUT / 'replays').mkdir(parents=True, exist_ok=True)
    RouterPlayer = _make_router_class()
    player = RouterPlayer(
        agents=agents,
        account_configuration=AccountConfiguration(user, pwd),
        server_configuration=ShowdownServerConfiguration,
        battle_format=args.format,
        team=team,
        max_concurrent_battles=max(50, 3 * (args.max_active + args.max_queue)),
        save_replays=str(OUT / 'replays'),
        start_timer_on_battle_start=True,
        accept_open_team_sheet=True,
        log_level=logging.WARNING,  # en INFO poke-env registra todo el tráfico, incluido el token de login
    )
    pe_log = logging.getLogger(user)  # los agentes internos comparten este logger: un solo handler
    pe_log.handlers = pe_log.handlers[:1]
    service = Bo3Service(player, fmt=args.format, accept_timeout=args.accept_timeout,
                         series_timeout=args.series_timeout * 60, max_active=args.max_active,
                         max_queue=args.max_queue, inbound_timeout=args.inbound_timeout * 60)
    service.inbound_only = bool(args.inbound_only)
    asyncio.run_coroutine_threadsafe(service.run(exit_on_disconnect=True), POKE_LOOP)
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(service))
    print(f'[web] {user} sirviendo en http://{args.host}:{args.port} ({args.format})', flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
