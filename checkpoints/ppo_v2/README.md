# Checkpoints de PPO v2

Copia de los checkpoints del kick-off oficial de PPO v2 y del control de la familia QD (26-11-2026).
Los originales están en el servidor, en `~/iapsvgc_runs/kickoff_ppo_v2/` y `~/iapsvgc_runs/qd_i2/control/`;
los md5 coinciden.

## Receta

PPO v1 + P5 completo (calculadora real en la observación, `--obs-abilities 1`) + preview v2.1 sin `val`
(`preview_v21b/preview_s0/best.pt`, el mismo de DDQN v1). 1M pasos por semilla, equipo fijo
`showdown/rl/data/ddqn_team.txt`, commit `3c6e1b3` (memorizacion_equipos.md §19, ppo_spec 1.17).

## Contenido

| Carpeta | Qué es | `val` | `eval` (3 heur.) | md5 de `final.pt` |
|---|---|---|---|---|
| `ppo_s0/` | PPO v2, semilla 0 | 83,0 % | 84,2 % | `eaeb26d9c05f58ea05e16c5bb7c509f2` |
| `ppo_s1/` | PPO v2, semilla 1: **la elegida por `val`**, arranque de I/J/K | 91,0 % | 79,3 % | `abd9b25c4134b62c4271a4013cc8042b` |
| `ppo_s2/` | PPO v2, semilla 2 | 85,2 % | 85,3 % | `6ca8ca9e5d38bf543161f2a6ec58a9da` |
| `ppo_s3/` | PPO v2, semilla 3 | 86,8 % | 86,7 % | `0962578446b6dd59344a405248d97857` |
| `ppo_s4/` | PPO v2, semilla 4: arranque de la réplica F3 | 88,3 % | 87,0 % | `f0e4466cd3cfb67f78e4bc59ad5f0b4e` |
| `control_s1_plus1M/` | **Control** de I/J/K: PPO v2 s1 + 1M pasos con el mosaico de PPO v2 (`train2 --control`, commit `7814cb4`) | 85,3 % (300 partidas) | 82,2 % (F1, 12 000 partidas) | `d5b927763db8691b455777496df95b18` |

Las columnas `val` y `eval` de las semillas vienen de `policy_diag`, con 200 partidas por rival. La media de
las cinco en `eval` es 84,5 % (± 3,8). La cifra del control sale de la confirmación F1 (protocolo §5.10).

Archivos por carpeta:
- `final.pt`: el modelo final, que es el que se evalúa.
- `train_meta.json`: configuración.
- `reporte.txt`: reporte del entrenamiento.
- `best.json`: el mejor checkpoint según `val` durante el entrenamiento. El `best.pt` no se sube.
- `summary.json` (solo el control).

## Uso

```python
from showdown.rl.ppo.agent import PPOAgent, PPOConfig
agent = PPOAgent(PPOConfig(device='cpu'))
agent.load('checkpoints/ppo_v2/ppo_s1/final.pt')   # observación P5: RL_OBS_ABILITIES=1
```

Para jugar como en la tesis: preview `team_preview_ppo` v2.1b, equipo `ddqn_team.txt` y
`RL_OBS_ABILITIES=1`, que se toma de `agent.cfg.obs_abilities`.
