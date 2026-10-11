"""Laboratorio: práctica agente vs bots en PS (v1 heurísticos)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from showdown.ui.content.agents_es import AGENT_LABELS, agent_options

st.set_page_config(page_title='Laboratorio · IAPSVGC', page_icon='🧪', layout='wide')
st.title('Laboratorio (práctica en Showdown)')

st.warning(
    """
**Advertencia — sección de laboratorio**

- Usa **CPU**, red y un servidor Pokémon Showdown (**local** con `--start-ps`).  
- Puede tardar **varios minutos** (sobre todo con Expectimax).  
- En v1 esto es **práctica / evaluación** entre bots (no entrena redes PPO/QD).  
- El aprendizaje por gradiente se integrará en una versión posterior de la UI.
"""
)

st.markdown(
    'Lanza series BO3 cruzadas con el harness robusto (`run_robust_e2e`) '
    'para ver cómo se comporta el agente frente a otro bot en PS local.'
)

opts = agent_options()
keys = [k for k, _ in opts]
agent = st.selectbox('Agente A', keys, format_func=lambda k: AGENT_LABELS[k])
opponent = st.selectbox(
    'Rival (bot B)',
    keys,
    index=0,
    format_func=lambda k: AGENT_LABELS[k],
)
# --teams N (par): N/2 matchups × 2 BO3 cruzados. 4 = humo razonable
n_teams = st.select_slider(
    'Tamaño de práctica (equipos MetaVGC, debe ser par)',
    options=[2, 4, 6, 8],
    value=4,
)

with st.expander('Controles avanzados (ocultos / experimentales)', expanded=False):
    st.caption('Reservado para seeds, checkpoints RL y presupuestos. No activos en v1.')
    st.text_input('Seed (próximamente)', value='0', disabled=True)
    st.number_input('Pasos de entrenamiento RL (próximamente)', value=0, disabled=True)

run = st.button('Iniciar práctica en PS local', type='primary')

if run:
    if agent == opponent:
        st.warning('Elige agentes distintos para A y B (o acepta un espejo consciente).')
    log_box = st.empty()
    prog = st.progress(0, text='Arrancando…')
    cmd = [
        sys.executable,
        '-m',
        'showdown.inference_sim.run_robust_e2e',
        '--teams',
        str(int(n_teams)),
        '--player-a',
        agent,
        '--player-b',
        opponent,
        '--start-ps',
        '--workers',
        '1',
    ]
    log_lines = [f'Comando: {" ".join(cmd)}']
    log_box.code('\n'.join(log_lines), language=None)
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
        )
        assert proc.stdout is not None
        for i, line in enumerate(proc.stdout):
            log_lines.append(line.rstrip())
            if len(log_lines) > 100:
                log_lines = log_lines[-100:]
            log_box.code('\n'.join(log_lines), language=None)
            prog.progress(min(0.95, 0.05 + i * 0.01), text='Práctica en curso…')
        code = proc.wait(timeout=7200)
        prog.progress(1.0, text='Listo')
        report = _ROOT / 'logs' / 'robust_e2e_report.txt'
        if report.is_file():
            st.subheader('Reporte')
            st.text(report.read_text(encoding='utf-8', errors='replace')[-5000:])
        if code == 0:
            st.success('Práctica terminada. Revisa el reporte arriba o en logs/robust_e2e_report.txt')
        else:
            st.error(f'Proceso terminó con código {code}. Revisa el log.')
    except Exception as exc:
        st.error(str(exc))

st.caption(
    'PS local: el harness puede arrancarlo con `--start-ps` '
    '(comando canónico con `--no-security`).'
)
