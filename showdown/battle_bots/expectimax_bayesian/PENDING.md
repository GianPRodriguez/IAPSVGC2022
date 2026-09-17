# Pendientes expectimax (versión estable)

## Cambios voluntarios — tune autónomo (0 créditos Cursor)
```bat
python -m showdown.inference_sim.run_expectimax_tune --hours 10 --teams 100 --seed 10
```
- Fase CORE: `enable` + `max_switch_hybrids`∈{1,2} + `switch_v_scale` bajo.
- Acepta si no empeora (pair WR / WR_raw); si empeora → revert.
- Fase SECONDARY solo si CORE dejó voluntary ON.
- Overrides en `ACTIVE_OVERRIDES.json` (defaults del dataclass siguen OFF).
- Al final: `logs/expectimax_tune_summary.txt` — abrir agente **solo** si
  `NEED_FIX_ANY: YES` (exit code 2).

Dry-run candidatos: `python -m showdown.inference_sim.run_expectimax_tune --dry-run`
