"""Puente Streamlit ↔ run.AsyncDownload (challenge BO3 + telemetría de recompensa)."""
from __future__ import annotations

import multiprocessing
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple

from environs import Env

REPO_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_DIR = REPO_ROOT / 'teams' / 'teams' / 'ui_uploads'

FORMAT_FIXED = 'gen9championsvgc2026regmcbo3'

SERVER_OFFICIAL = 'sim.smogon.com:8000'
SERVER_LOCAL_DEFAULT = 'localhost:8000'


@dataclass
class RewardPoint:
    turn_label: str
    delta: float
    total: float


@dataclass
class BattleSession:
    queue: Any
    thread: Any
    rewards: List[RewardPoint] = field(default_factory=list)
    status: str = 'starting'
    opponent: str = ''
    error: Optional[str] = None
    finished: bool = False
    won: Optional[bool] = None
    log_lines: List[str] = field(default_factory=list)


def load_bot_credentials() -> Tuple[str, str, str]:
    """PS_USERNAME / PS_PASSWORD / TEAM_DIR desde .env (credenciales del bot)."""
    env = Env()
    env.read_env(str(REPO_ROOT / '.env'))
    user = (env('PS_USERNAME', '') or '').strip()
    password = (env('PS_PASSWORD', '') or '').strip()
    team_dir = (env('TEAM_DIR', 'gen8/vgc2022') or 'gen8/vgc2022').strip().strip('/')
    return user, password, team_dir


def websocket_for_choice(choice: str, local_host: str = SERVER_LOCAL_DEFAULT) -> str:
    key = (choice or 'oficial').strip().lower()
    if key in ('local', 'localhost', 'ps local', 'servidor local'):
        return local_host.strip() or SERVER_LOCAL_DEFAULT
    return SERVER_OFFICIAL


def save_uploaded_team(export_text: str, *, stem: str = 'streamlit_upload') -> str:
    """
    Guarda export Showdown bajo teams/teams/ui_uploads/.

    Returns:
        Nombre de archivo relativo a TEAM_DIR... en realidad devolvemos path
        relativo a teams/teams/ como ``ui_uploads/stem.txt`` para load_team
        cuando TEAM_DIR se ajusta, o guardamos con team_dir override.

    Contrato con run.parse_configs: config.team_name = f\"{team_dir}/{teamname}\".
    Guardamos en ui_uploads y usamos team_dir='ui_uploads', teamname=stem.txt.
    """
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe = ''.join(c if c.isalnum() or c in '-_' else '_' for c in stem)[:40] or 'upload'
    fname = f'{safe}.txt'
    path = UPLOAD_DIR / fname
    text = (export_text or '').strip() + '\n'
    path.write_text(text, encoding='utf-8')
    return fname


def start_challenge(
    *,
    human_username: str,
    team_export: str,
    battle_bot: str,
    server_choice: str,
    local_uri: str = SERVER_LOCAL_DEFAULT,
    pokemon_mode: str = FORMAT_FIXED,
) -> BattleSession:
    """Lanza CHALLENGE_USER: bot (.env) invita a ``human_username``."""
    bot_user, _bot_pass, _team_dir = load_bot_credentials()
    if not bot_user:
        raise RuntimeError(
            'Faltan PS_USERNAME / PS_PASSWORD en el archivo .env del proyecto. '
            'Son las credenciales de la cuenta del bot.'
        )
    human = (human_username or '').strip()
    if not human:
        raise ValueError('Escribe tu usuario de Pokémon Showdown.')
    if not (team_export or '').strip():
        raise ValueError('Sube o pega el equipo (export de Showdown) que usará el agente.')

    fname = save_uploaded_team(team_export, stem=f'vs_{human}')
    uri = websocket_for_choice(server_choice, local_uri)
    os.environ['WEBSOCKET_URI'] = uri
    os.environ['TEAM_DIR'] = 'ui_uploads'
    os.environ['BATTLE_BOT'] = (battle_bot or 'random').strip().lower()
    os.environ['POKEMON_MODE'] = pokemon_mode

    # Import tardío: respeta WEBSOCKET_URI / TEAM_DIR ya puestos
    from run import AsyncDownload
    import constants

    q = multiprocessing.Queue()
    worker = AsyncDownload(
        q,
        human,  # oponente a retar
        None,
        fname,
        constants.CHALLENGE_USER,
        1,
        pokemon_mode=pokemon_mode,
        battle_bot=battle_bot,
    )
    session = BattleSession(queue=q, thread=worker, opponent=human, status='challenging')
    session.log_lines.append(f'Bot «{bot_user}» invita a «{human}» ({pokemon_mode}) vía {uri}')
    session.log_lines.append(f'Agente: {battle_bot} · equipo: ui_uploads/{fname}')
    worker.start()
    return session


def poll_session(session: BattleSession, *, max_items: int = 32) -> BattleSession:
    """Drena la cola de la batalla y actualiza recompensas / estado."""
    q = session.queue
    for _ in range(max_items):
        try:
            item = q.get_nowait()
        except Exception:
            break
        if not item:
            continue
        code = item[0]
        if code == 0:
            session.status = 'en_curso'
            session.log_lines.append('BO3 en curso — acepta la invitación en Showdown.')
        elif code == 1:
            # team preview payload
            names = item[1:5] if len(item) >= 5 else []
            session.log_lines.append('Team Preview: ' + ', '.join(str(n) for n in names))
        elif code == 2 and len(item) >= 12:
            # Formato IAP UI: [2, t1,c1,t2,c2,t3,c3,t4,c4, delta, c5, total, c6]
            # delta=index 9, total=index 11 (no 10/12: esos son colores hex).
            try:
                delta = float(item[9])
                total = float(item[11])
            except (TypeError, ValueError):
                delta, total = 0.0, 0.0
            tag = str(item[1] or '').strip() if len(item) > 1 else ''
            n = len(session.rewards) + 1
            label = tag if tag in ('Fin juego', '(wait)') else f'Turno {n}'
            session.rewards.append(
                RewardPoint(turn_label=label, delta=delta, total=total)
            )
            arrow = '↑' if delta > 0 else ('↓' if delta < 0 else '→')
            session.log_lines.append(
                f'{arrow} Recompensa ({label}): {delta:+.1f} · acumulada: {total:.1f}'
            )
            session.status = 'en_curso'
        elif code == -1:
            # [-1, bot_won, bot_name, opponent_name]
            bot_won = bool(item[1]) if len(item) > 1 else False
            session.finished = True
            session.won = bot_won
            session.status = 'victoria_bot' if bot_won else 'victoria_humana'
            opp = item[3] if len(item) > 3 else session.opponent
            session.log_lines.append(
                'Serie terminada: '
                + ('ganó el agente.' if bot_won else f'ganaste tú ({opp}).')
            )
        elif code == -2:
            session.finished = True
            session.error = 'El oponente rechazó la invitación.'
            session.status = 'error'
        elif code == -3:
            session.finished = True
            session.error = 'Equipo inválido o vacío.'
            session.status = 'error'
        elif code == -4:
            session.finished = True
            session.error = 'Usuario de Showdown no encontrado.'
            session.status = 'error'
        elif code == -5:
            session.finished = True
            session.error = 'No se pudo iniciar sesión (revisa .env PS_USERNAME/PS_PASSWORD).'
            session.status = 'error'
        elif code == -6:
            session.finished = True
            session.error = 'Error al iniciar o jugar la batalla.'
            session.status = 'error'
        else:
            session.log_lines.append(f'Evento cola: {item!r}')

    if session.thread is not None and not session.thread.is_alive() and not session.finished:
        # Hilo terminó sin código claro
        if session.rewards:
            session.status = 'finalizada'
        elif session.status == 'challenging':
            time.sleep(0.05)
            if not session.thread.is_alive() and not session.error:
                session.status = 'finalizada'
        session.finished = session.finished or (not session.thread.is_alive())

    return session


def reward_series(session: BattleSession) -> Tuple[List[str], List[float]]:
    labels = [p.turn_label for p in session.rewards]
    totals = [p.total for p in session.rewards]
    return labels, totals
