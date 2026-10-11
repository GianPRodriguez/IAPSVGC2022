"""Vista Jugar: el bot invita a tu usuario Showdown (BO3 Reg M-C)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from showdown.ui.battle_bridge import (
    FORMAT_FIXED,
    SERVER_LOCAL_DEFAULT,
    load_bot_credentials,
    poll_session,
    reward_series,
    start_challenge,
)
from showdown.ui.content.agents_es import AGENT_LABELS, agent_options

st.set_page_config(page_title='Jugar · IAPSVGC', page_icon='⚔️', layout='wide')
st.title('Jugar contra el agente')
st.markdown(
    """
1. Elige el agente y el servidor (oficial o local).  
2. Pega **tu** usuario de Showdown.  
3. Sube el **equipo que usará el agente** (export Showdown).  
4. Pulsa **Invitar** — acepta el desafío en el cliente de Showdown.  
5. Mientras juegas, verás si la **recompensa interna** del bot sube o baja.
"""
)

bot_user, bot_pass, _ = load_bot_credentials()
if not bot_user or not bot_pass:
    st.error(
        'Configura `PS_USERNAME` y `PS_PASSWORD` en el `.env` del proyecto '
        '(cuenta del bot). Sin eso no se puede enviar la invitación.'
    )
    st.stop()

st.caption(f'Cuenta del bot (desde .env): **{bot_user}** · Formato: `{FORMAT_FIXED}`')

opts = agent_options()
agent_key = st.selectbox(
    'Agente',
    [k for k, _ in opts],
    format_func=lambda k: AGENT_LABELS[k],
)

server = st.radio(
    'Servidor Showdown',
    ['Oficial (sim.smogon.com)', 'Local (localhost)'],
    horizontal=True,
)
local_uri = SERVER_LOCAL_DEFAULT
if 'Local' in server:
    local_uri = st.text_input('URI WebSocket local', value=SERVER_LOCAL_DEFAULT)
    st.info(
        'Asegúrate de tener Pokémon Showdown local en marcha con '
        '`node pokemon-showdown start --no-security`.'
    )

human = st.text_input('Tu usuario de Showdown (te enviarán la invitación)', placeholder='MiUsuario')

team_file = st.file_uploader('Equipo del agente (archivo .txt export Showdown)', type=['txt'])
team_paste = st.text_area(
    '…o pega aquí el export del equipo',
    height=160,
    placeholder='Pokémon1 @ Item\nAbility: …\n…',
)
team_text = ''
if team_file is not None:
    team_text = team_file.read().decode('utf-8', errors='replace')
elif team_paste.strip():
    team_text = team_paste

if 'battle_session' not in st.session_state:
    st.session_state.battle_session = None

col_a, col_b = st.columns([1, 1])
with col_a:
    start = st.button('Invitar a BO3', type='primary', use_container_width=True)
with col_b:
    refresh = st.button('Actualizar estado / recompensa', use_container_width=True)

if start:
    try:
        choice = 'local' if 'Local' in server else 'oficial'
        st.session_state.battle_session = start_challenge(
            human_username=human,
            team_export=team_text,
            battle_bot=agent_key,
            server_choice=choice,
            local_uri=local_uri,
        )
        st.success('Invitación enviada. Ábrela en Showdown y acepta el BO3.')
    except Exception as exc:
        st.error(str(exc))

session = st.session_state.battle_session
if session is not None and (refresh or start or not session.finished):
    poll_session(session)
    st.session_state.battle_session = session

if session is not None:
    st.subheader('Estado')
    st.write(f'**{session.status}** · vs `{session.opponent}` · agente `{agent_key}`')
    if session.error:
        st.error(session.error)
    if session.finished and session.won is not None:
        if session.won:
            st.warning('El agente ganó la serie BO3.')
        else:
            st.success('¡Ganaste la serie BO3!')

    st.subheader('Recompensa del agente (sube = le va mejor)')
    labels, totals = reward_series(session)
    if totals:
        # Streamlit acepta dict columna → valores
        chart = {'recompensa_acumulada': totals}
        st.line_chart(chart)
        last = session.rewards[-1]
        if last.delta > 0:
            st.caption(f'Último cambio: **mejoró** ({last.delta:+.1f}). Total: {last.total:.1f}')
        elif last.delta < 0:
            st.caption(f'Último cambio: **empeoró** ({last.delta:+.1f}). Total: {last.total:.1f}')
        else:
            st.caption(f'Último cambio: sin variación. Total: {last.total:.1f}')
        st.caption(
            'Escala interna Spec C2 ×100 (no es Elo). Sube si el agente hace daño/KO '
            'o gana el juego; baja si pierde Pokémon o la partida.'
        )
    else:
        st.caption('Aún no hay turnos con recompensa (espera a que empiece el combate).')

    with st.expander('Registro'):
        for line in session.log_lines[-40:]:
            st.text(line)

    if not session.finished:
        time.sleep(0.8)
        st.rerun()
else:
    st.info('Configura los campos y pulsa **Invitar a BO3**.')
