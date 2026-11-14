"""Vista educativa v2: los 11 agentes (A–K) como pipelines preview → combate."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from showdown.ui.content.agents_v2_es import (
    AGENT_LETTERS,
    AGENTS_V2,
    FAMILIES,
    PREVIEWS,
    agent_title,
    letters_by_family,
)

st.set_page_config(page_title='Aprender · IAPSVGC', page_icon='📚', layout='wide')
st.title('Aprender: ¿cómo piensa cada agente?')
st.markdown(
    'Hay **11 agentes**, nombrados de la **A** a la **K**. Cada uno toma dos '
    'decisiones distintas en una partida de VGC:'
)
c_prev, c_comb = st.columns(2)
with c_prev:
    st.info(
        '**1. Team Preview** — antes de empezar, ve los 6 Pokémon de cada lado y '
        'elige qué 4 lleva y cuáles 2 salen primero.'
    )
with c_comb:
    st.info(
        '**2. Combate** — en cada turno decide qué hacen sus dos Pokémon activos '
        '(atacar, protegerse, cambiar…).'
    )
st.caption(
    'Las fichas usan analogías con el ajedrez: no hace falta saber de '
    'inteligencia artificial.'
)

modo = st.radio(
    '¿Para quién es esta lectura?',
    ['Jugador de VGC (simple)', 'Estudiante / tesis (un poco más de detalle)'],
    horizontal=True,
)
es_tesis = 'Estudiante' in modo

# ---------------------------------------------------------------------------
# Vista general
# ---------------------------------------------------------------------------
st.subheader('Los 11 de un vistazo')
filas = []
for k in AGENT_LETTERS:
    a = AGENTS_V2[k]
    filas.append(
        {
            'Agente': k,
            'Nombre': a['nombre'],
            'Familia': FAMILIES[a['familia']]['nombre'],
            'Team Preview': PREVIEWS[a['preview']]['nombre'],
            'Combate': a['combate'],
            '¿Entrena?': a['entrena'],
            'Velocidad al jugar': a['velocidad'],
        }
    )
st.dataframe(filas, hide_index=True, width='stretch')

# ---------------------------------------------------------------------------
# Fichas por familia
# ---------------------------------------------------------------------------
st.divider()
st.subheader('Ficha de cada agente')

fam_keys = list(FAMILIES)
fam_tabs = st.tabs([FAMILIES[f]['nombre'] for f in fam_keys])


def _ficha(letter: str) -> None:
    a = AGENTS_V2[letter]
    prev = PREVIEWS[a['preview']]
    st.header(agent_title(letter))
    st.caption(a['subtitulo'])
    st.write(a['analogia'])

    col_p, col_c = st.columns(2)
    with col_p:
        st.markdown(f'**Team Preview — {prev["nombre"]}**')
        st.write(prev['como'])
    with col_c:
        st.markdown(f'**Combate — {a["combate"]}**')
        st.write(a['como_juega'])

    m1, m2 = st.columns(2)
    m1.metric('¿Entrena?', a['entrena'])
    m2.metric('Velocidad al jugar', a['velocidad'])

    st.success('**Punto fuerte:** ' + a['fuerte'])
    st.warning('**Límite:** ' + a['debil'])

    if es_tesis:
        st.markdown('**Nota de tesis**')
        st.code(
            f'pipeline: {a["id"]}  (preview={a["preview"]}, combate={a["id"]})\n'
            f'{a["nota_tesis"]}\n'
            f'Preview entrena: {prev["entrena"]}\n'
            f'Spec: {a["spec"]}',
            language=None,
        )


for tab, fam in zip(fam_tabs, fam_keys):
    with tab:
        st.markdown(FAMILIES[fam]['resumen'])
        letras = letters_by_family(fam)
        sub_tabs = st.tabs([agent_title(k) for k in letras])
        for sub, k in zip(sub_tabs, letras):
            with sub:
                _ficha(k)

# ---------------------------------------------------------------------------
# Comparar
# ---------------------------------------------------------------------------
st.divider()
st.subheader('Comparar dos agentes')
c1, c2 = st.columns(2)
with c1:
    x = st.selectbox('Agente 1', AGENT_LETTERS, format_func=agent_title, key='cmp_a')
with c2:
    y = st.selectbox(
        'Agente 2', AGENT_LETTERS, index=4, format_func=agent_title, key='cmp_b'
    )
if x != y:
    left, right = st.columns(2)
    for col, k in ((left, x), (right, y)):
        a = AGENTS_V2[k]
        with col:
            st.markdown(f'### {agent_title(k)}')
            st.caption(FAMILIES[a['familia']]['nombre'])
            st.markdown(f'**Preview:** {PREVIEWS[a["preview"]]["nombre"]}')
            st.write(a['como_juega'])
            st.markdown(f'**Entrena:** {a["entrena"]} · **Velocidad:** {a["velocidad"]}')
    if AGENTS_V2[x]['preview'] == AGENTS_V2[y]['preview']:
        st.info(
            'Ambos usan el **mismo Team Preview**: la diferencia entre ellos está '
            'solo en cómo juegan los turnos.'
        )
else:
    st.warning('Elige dos agentes distintos para comparar.')

# ---------------------------------------------------------------------------
# Previews
# ---------------------------------------------------------------------------
st.divider()
with st.expander('¿Por qué varios agentes comparten el mismo Team Preview?', expanded=False):
    st.markdown(
        """
En las familias de tesis, el Team Preview se **entrena primero y luego se congela**.
Después se entrenan o ajustan los agentes de combate con ese preview fijo. Así,
si un agente gana más, se sabe que es por su forma de combatir y no por el preview.

| Team Preview | Agentes que lo usan |
|---|---|
| Aleatorio | A |
| Por tipos (greedy) | B, C |
| Tipos + roles | D |
| Masked PPO | E, F |
| Payoff Nash | G, H |
| QD-PGA | I, J, K |
"""
    )

st.caption(
    'Aprender v2: los 11 agentes A–K. En Jugar y Laboratorio siguen disponibles '
    'solo A–D (heurísticos). Figuras de entrenamiento: menú Figuras y tune.'
)
