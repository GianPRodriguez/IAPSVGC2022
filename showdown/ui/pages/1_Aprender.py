"""Vista educativa: agentes v1 con analogía de ajedrez."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from showdown.ui.content.agents_es import AGENTS_V1, AGENT_CHESS, AGENT_LABELS, agent_options

st.set_page_config(page_title='Aprender · IAPSVGC', page_icon='📚', layout='wide')
st.title('Aprender: ¿cómo piensa cada agente?')
st.markdown(
    'Cada ficha usa una **analogía con el ajedrez**. No hace falta saber de '
    'inteligencia artificial: basta con imaginar cómo elegiría jugadas un jugador humano.'
)

modo = st.radio(
    '¿Para quién es esta lectura?',
    ['Jugador de VGC (simple)', 'Estudiante / tesis (un poco más de detalle)'],
    horizontal=True,
)
es_tesis = 'Estudiante' in modo

keys = [k for k, _ in agent_options()]
tabs = st.tabs([AGENT_LABELS[k] for k in keys])

for tab, key in zip(tabs, keys):
    data = AGENT_CHESS[key]
    with tab:
        st.header(data['titulo'])
        st.caption(data['subtitulo'])
        for p in data['parrafos']:
            st.write(p)
        st.success('**Úsalo cuando:** ' + data['usa_cuando'])
        if es_tesis:
            st.markdown('**Nota de tesis (opcional)**')
            notas = {
                'random': 'Línea base uniforme sobre acciones legales; sin modelo de rival.',
                'greedy_solo': 'Heurística de daño por slot; sin búsqueda multi-turno.',
                'greedy_joint': 'Argmax sobre pares de acciones legales (coordinación 2v2).',
                'expectimax_bayesian': 'Árbol expectimax + creencias; coste computacional mayor.',
            }
            st.code(notas[key], language=None)

st.divider()
st.subheader('Comparar dos agentes')
c1, c2 = st.columns(2)
with c1:
    a = st.selectbox('Agente A', keys, format_func=lambda k: AGENT_LABELS[k], key='cmp_a')
with c2:
    b = st.selectbox(
        'Agente B',
        keys,
        index=min(1, len(keys) - 1),
        format_func=lambda k: AGENT_LABELS[k],
        key='cmp_b',
    )
if a != b:
    left, right = st.columns(2)
    with left:
        st.markdown(f'### {AGENT_LABELS[a]}')
        st.write(AGENT_CHESS[a]['parrafos'][0])
    with right:
        st.markdown(f'### {AGENT_LABELS[b]}')
        st.write(AGENT_CHESS[b]['parrafos'][0])
else:
    st.warning('Elige dos agentes distintos para comparar.')

st.caption('v1: aleatorio, greedy solo, greedy joint, expectimax. PPO / QD-PPO / MCTS llegarán en otra versión de la UI.')
