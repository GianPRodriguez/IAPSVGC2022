# IAPSVGC Lab (Streamlit)

Interfaz en español para **aprender**, **jugar** (invitación BO3) y **laboratorio**.

## Arranque

Desde la raíz del repo:

```bat
pip install -r requirements.txt -r requirements-ui.txt
streamlit run showdown/ui/app.py
```

## Requisitos

| Pieza | Uso |
|-------|-----|
| `.env` con `PS_USERNAME` / `PS_PASSWORD` | Cuenta del bot (invitar / aceptar) |
| Cliente Showdown | Aceptar el BO3 cuando te inviten |
| PS local opcional | `node pokemon-showdown start --no-security` |

Formato fijo: `gen9championsvgc2026regmcbo3`.

## Páginas

1. **Aprender (v2)** — fichas básicas de los **11 agentes A–K** (preview → combate), con analogías de ajedrez, tabla resumen y comparador.  
2. **Jugar** — pegas tu usuario; subes el equipo del agente; el bot invita. Gráfica de recompensa.  
3. **Laboratorio** — práctica en PS (advertencia de tiempo). Controles RL avanzados ocultos (v1).

## Agentes

- **Aprender (v2):** los 11 pipelines A–K. Textos en `content/agents_v2_es.py`
  (pares según `showdown/inference_sim/conclusion_eval/pipelines.py`).
- **Jugar / Laboratorio (v1):** solo A–D — `random`, `greedy_solo`, `greedy_joint`,
  `expectimax_bayesian` (`content/agents_es.py`).

| Letra | Team Preview | Combate |
|---|---|---|
| A | random | random |
| B | greedy_solo | greedy_solo |
| C | greedy_solo | greedy_joint |
| D | expectimax_bayesian | expectimax_bayesian |
| E | team_preview_ppo | ddqn |
| F | team_preview_ppo | ppo |
| G | team_preview_nash | mcts_puct |
| H | team_preview_nash | mcts_puct_nash |
| I | team_preview_qd_pga | qd_ppo |
| J | team_preview_qd_pga | qd_pga |
| K | team_preview_qd_pga | qd_ppga |
