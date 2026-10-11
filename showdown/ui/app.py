"""
IAPSVGC Lab — interfaz educativa y de uso (Streamlit).

Arranque (desde la raíz del repo)::

    streamlit run showdown/ui/app.py

Requisitos: ``pip install streamlit`` (también en requirements-ui.txt).
Credenciales del bot: ``PS_USERNAME`` / ``PS_PASSWORD`` en ``.env``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Raíz del repo en sys.path (imports run / showdown)
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

st.set_page_config(
    page_title='IAPSVGC Lab',
    page_icon='🎮',
    layout='wide',
    initial_sidebar_state='expanded',
)

st.title('IAPSVGC Lab')
st.markdown(
    """
Laboratorio local para **aprender** cómo piensan los agentes de VGC,
**jugar** un BO3 contra ellos en Pokémon Showdown (Reglamento M-C),
y **practicar** en el laboratorio (con advertencia de tiempo).

Usa el menú de la izquierda:

1. **Aprender** — analogías con el ajedrez, sin jerga técnica  
2. **Jugar** — pegas tu usuario; el bot te invita a un BO3  
3. **Laboratorio** — práctica agente vs bots en Showdown  

Público: jugadores de VGC y estudiantes. Todo el texto está en español.
"""
)

st.info(
    'Formato fijo: **gen9championsvgc2026regmcbo3** (VGC Reg M-C, Open Team Sheets, BO3). '
    'La cuenta del bot se lee del archivo `.env` del proyecto (no hace falta pegar su contraseña aquí).'
)

with st.expander('Glosario rápido (5 palabras)', expanded=False):
    st.markdown(
        """
- **Agente / bot:** programa que elige movimientos por ti en la batalla.  
- **Movimiento legal:** acción que Showdown permite en ese turno.  
- **BO3:** al mejor de 3 partidas (el primero en ganar 2).  
- **Recompensa:** número interno que sube si al bot le va bien y baja si le va mal (no es Elo).  
- **Export de equipo:** texto que copias desde el teambuilder de Showdown.
"""
    )

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader('Aprender')
    st.write('Lee cómo decide cada agente, comparado con el ajedrez.')
with col2:
    st.subheader('Jugar')
    st.write('Invitación a tu usuario Showdown. Tú juegas; el agente usa el equipo que subas.')
with col3:
    st.subheader('Laboratorio')
    st.write('Sesiones de práctica vs otros bots. Consume tiempo y CPU: léete la advertencia.')

st.caption('IAPSVGC2022 · Streamlit · v1 (random, greedy solo/joint, expectimax)')
