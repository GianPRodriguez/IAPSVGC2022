"""Galería de figuras reales de tune/train + pipelines preview → combate."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

st.set_page_config(page_title='Figuras · IAPSVGC', page_icon='📈', layout='wide')

FIGURES = _ROOT / 'logs' / 'figures'
LEDGER = FIGURES / 'hp_tune_ledger.jsonl'

FAMILY_PIPELINES = [
    {
        'familia': 'Aprendizaje',
        'preview': 'Team Preview Masked PPO',
        'combate': ['DDQN', 'PPO'],
        'nota': 'Un checkpoint de preview se reutiliza en ambos combates.',
    },
    {
        'familia': 'Planificación',
        'preview': 'Team Preview Payoff Nash',
        'combate': ['MCTS+PUCT', 'MCTS–Nash'],
        'nota': 'Sin pesos de combate; se congela N_sim tras el barrido.',
    },
    {
        'familia': 'Diversidad (QD)',
        'preview': 'Team Preview QD-PGA',
        'combate': ['QD-PPO', 'QD-PGA', 'QD-PPGA'],
        'nota': 'Mismo preview QD-PGA para las tres variantes.',
    },
]

LATEST_LABELS = {
    'ddqn_hp_tune_latest.svg': 'DDQN · HP tune',
    'ppo_hp_tune_latest.svg': 'PPO · HP tune',
    'team_preview_ppo_hp_tune_latest.svg': 'Preview PPO · HP tune',
    'team_preview_nash_hp_tune_latest.svg': 'Preview Nash · HP tune',
    'team_preview_qd_pga_hp_tune_latest.svg': 'Preview QD-PGA · HP tune',
    'qd_ppo_hp_tune_latest.svg': 'QD-PPO · HP tune',
    'qd_pga_hp_tune_latest.svg': 'QD-PGA · HP tune',
    'qd_ppga_hp_tune_latest.svg': 'QD-PPGA · HP tune',
    'mcts_puct_sims_sweep_latest.svg': 'MCTS+PUCT · sims sweep',
    'mcts_puct_nash_sims_sweep_latest.svg': 'MCTS–Nash · sims sweep',
    'hp_tune_ledger_all.svg': 'Ledger histórico (todos)',
}


st.title('Figuras y tune')
st.markdown(
    """
Figuras generadas al cerrar un **tune / sweep / train**
(`logs/figures/*_latest.svg`) y el flujo **preview congelado → combate**.
Para la maqueta pedagógica con Chart.js: `docs/thesis_figures_mock/`.
"""
)

tab_gal, tab_pipe, tab_ledger, tab_cli = st.tabs(
    ['Galería SVG', 'Pipelines preview', 'Ledger JSONL', 'CLIs']
)

with tab_gal:
    if not FIGURES.is_dir():
        st.warning(f'No existe `{FIGURES.relative_to(_ROOT)}`. Corré un tune o regenerá plots.')
    else:
        latest = sorted(FIGURES.glob('*_latest.svg')) + (
            [FIGURES / 'hp_tune_ledger_all.svg']
            if (FIGURES / 'hp_tune_ledger_all.svg').is_file()
            else []
        )
        # de-dupe ledger if also matched *_latest
        seen = set()
        uniq = []
        for p in latest:
            if p.resolve() in seen or not p.is_file():
                continue
            seen.add(p.resolve())
            uniq.append(p)

        if not uniq:
            st.info('Aún no hay `*_latest.svg`. Regenerá con el CLI de la pestaña CLIs.')
        else:
            cols = st.columns(2)
            for i, path in enumerate(uniq):
                label = LATEST_LABELS.get(path.name, path.name)
                with cols[i % 2]:
                    st.subheader(label)
                    st.caption(str(path.relative_to(_ROOT)))
                    st.image(str(path), use_container_width=True)

with tab_pipe:
    st.markdown(
        'El combate **no** aprende el `/team`: primero se entrena el preview, '
        'se congela, y el MDP de turnos practica bajo esa política.'
    )
    for row in FAMILY_PIPELINES:
        with st.container(border=True):
            st.markdown(f'### {row["familia"]}')
            st.markdown(
                f'**Preview:** `{row["preview"]}` → **Combate:** '
                + ', '.join(f'`{c}`' for c in row['combate'])
            )
            st.caption(row['nota'])
    st.info(
        'Fallback sin checkpoint: `/team 1234` (orden de exportación). '
        'Útil para smoke; no para Conclusión.'
    )

with tab_ledger:
    if not LEDGER.is_file():
        st.warning(f'No hay ledger en `{LEDGER.relative_to(_ROOT)}`.')
    else:
        rows = []
        with LEDGER.open(encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        st.caption(f'{len(rows)} entradas · append-only')
        if rows:
            # mostrar columnas útiles si existen
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.info('Ledger vacío.')

with tab_cli:
    st.code(
        'python -m showdown.inference_sim.run_plot_tune_history --all\n'
        'python -m showdown.inference_sim.run_plot_tune_history --agent ddqn\n'
        'python -m showdown.inference_sim.run_ddqn_hp_tune --seed 0 --max-trials 6',
        language='bat',
    )
    st.markdown(
        'Mock HTML (pestañas Tune / Figuras reales / Pipelines)::\n\n'
        '`docs/thesis_figures_mock/index.html` — servir con '
        '`py -3 -m http.server 8765` desde la raíz del repo.'
    )

st.caption('IAPSVGC Lab · figuras desde logs/figures')
