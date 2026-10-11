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

1. **Aprender** — analogías con ajedrez (random, greedy solo/joint, expectimax).  
2. **Jugar** — pegas tu usuario; subes el equipo del agente; el bot invita. Gráfica de recompensa.  
3. **Laboratorio** — práctica en PS (advertencia de tiempo). Controles RL avanzados ocultos (v1).

## v1 agentes

`random`, `greedy_solo`, `greedy_joint`, `expectimax_bayesian`.
