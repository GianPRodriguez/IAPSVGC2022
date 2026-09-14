# ExpectimaxConfig — historial de parámetros

Evolución de knobs hacia el comportamiento deseado (p.ej. superar `greedy_joint`).

- Changelog machine-readable: `CONFIG_HISTORY.jsonl` (misma carpeta)
- Runs E2E con snapshot: `logs/expectimax_config_runs.jsonl` (auto al batch robusto)

Registrar un cambio tras editar `ExpectimaxConfig`:

```bat
python -m showdown.battle_bots.expectimax_bayesian.config_history "nota" --changed k=v,... --result "contexto"
```

Ver config actual:

```bat
python -m showdown.battle_bots.expectimax_bayesian.config_history --show
```

## baseline — pre-tuning (código inicial documentado)

Valores de referencia antes del primer ajuste anti-`greedy_joint`:

| param | value |
|---|---|
| switch_min_stay_p_ko | 0.35 |
| threat_weight | 0.75 |
| loss_weight | 1000 |
| save_weight | 220 |
| syn_weight | 0.35 |
| (resto) | ver `BASELINE_PRE_TUNING` en `config_history.py` |

Contexto: robust E2E ~40–60 vs `greedy_joint` (hipótesis: hiperdefensa / pivots).

## 2026-09-14 — anti-joint: menos threat/pivot panic, mas sinergia

- **UTC:** `2026-09-14T01:23:01.204884+00:00`
- **Resultado / contexto:** baseline ~40-60 vs greedy_joint; hipotesis hiperdefensa
- **Cambios:**
  - `loss_weight`: 1000.0 → 650
  - `save_weight`: 220.0 → 140
  - `switch_min_stay_p_ko`: 0.35 → 0.5
  - `syn_weight`: 0.35 → 0.55
  - `threat_weight`: 0.75 → 0.55

```
--- expectimax_config ---
deadline_ms: 700.0
phase1_budget_ms: 50.0
phase2_end_ms: 680.0
k_profiles: 5
max_worlds: 16
max_our_pairs: 36
soft_ko_bonus: 1000.0
enable_voluntary_switch: True
max_switch_hybrids: 12
loss_weight: 650.0
save_weight: 140.0
chip_soft_cap: 30.0
switch_min_stay_p_ko: 0.5
overkill_waste_weight: 140.0
ko_free_switch_bonus: 55.0
entry_punish_weight: 1.15
threat_weight: 0.55
hp_rel_weight: 0.15
syn_weight: 0.55
condemned_chip_discount: 0.7
condemned_lasting_boost: 1.35
threat_relief_weight: 1.0
entry_ability_weight: 1.0
```
