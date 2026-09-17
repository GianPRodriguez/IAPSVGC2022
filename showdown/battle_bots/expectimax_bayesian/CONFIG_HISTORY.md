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

## 2026-09-15 — save_weight: un poco mas salvar mon en derrotas C2

- **UTC:** `2026-09-15T12:05:23.475871+00:00`
- **Resultado / contexto:** from 140; next validate --teams 100 expectimax vs greedy_joint
- **Cambios:**
  - `save_weight`: 140.0 → 170

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
save_weight: 170.0
chip_soft_cap: 30.0
switch_min_stay_p_ko: 0.5
overkill_waste_weight: 140.0
ko_free_switch_bonus: 55.0
entry_punish_weight: 1.15
threat_weight: 0.48
hp_rel_weight: 0.15
syn_weight: 0.55
condemned_chip_discount: 0.7
condemned_lasting_boost: 1.35
threat_relief_weight: 1.0
entry_ability_weight: 1.0
```

## 2026-09-15 — revert save_weight 170→140 (WR 35.4%% vs 44.9%% en teams=100)

- **UTC:** `2026-09-15T12:59:10.517095+00:00`
- **Resultado / contexto:** teams=100 save=170 regressed; restore baseline with threat=0.48
- **Cambios:**
  - `save_weight`: 170.0 → 140
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
threat_weight: 0.48
hp_rel_weight: 0.15
syn_weight: 0.55
condemned_chip_discount: 0.7
condemned_lasting_boost: 1.35
threat_relief_weight: 1.0
entry_ability_weight: 1.0
```

## 2026-09-15 — condemned_chip_discount: menos chip a mon muerto

- **UTC:** `2026-09-15T15:09:32.178690+00:00`
- **Resultado / contexto:** ronda1/3; next --teams 100 expectimax vs greedy_joint
- **Cambios:**
  - `condemned_chip_discount`: 0.7 → 0.8

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
threat_weight: 0.48
hp_rel_weight: 0.15
syn_weight: 0.55
condemned_chip_discount: 0.8
condemned_lasting_boost: 1.35
threat_relief_weight: 1.0
entry_ability_weight: 1.0
```

## 2026-09-15 — deadline_ms 700→800 (solo presupuesto busqueda)

- **UTC:** `2026-09-15T15:31:10.761009+00:00`
- **Resultado / contexto:** ronda2/3 tras condemned_chip=0.80 WR49.5%; next teams=100
- **Cambios:**
  - `deadline_ms`: 700.0 → 800
  - `phase2_end_ms`: 680.0 → 780

```
--- expectimax_config ---
deadline_ms: 800.0
phase1_budget_ms: 50.0
phase2_end_ms: 780.0
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
threat_weight: 0.48
hp_rel_weight: 0.15
syn_weight: 0.55
condemned_chip_discount: 0.8
condemned_lasting_boost: 1.35
threat_relief_weight: 1.0
entry_ability_weight: 1.0
```

## 2026-09-15 — revert deadline 800→700 (WR 36.5%% vs 49.5%%)

- **UTC:** `2026-09-15T15:49:01.379782+00:00`
- **Resultado / contexto:** ronda2: mas tiempo de busqueda empeoro; keep condemned_chip=0.80
- **Cambios:**
  - `deadline_ms`: 700.0 → 700
  - `phase2_end_ms`: 680.0 → 680

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
threat_weight: 0.48
hp_rel_weight: 0.15
syn_weight: 0.55
condemned_chip_discount: 0.8
condemned_lasting_boost: 1.35
threat_relief_weight: 1.0
entry_ability_weight: 1.0
```

## 2026-09-15 — switch_min_stay_p_ko 0.50→0.45 (pivot un poco antes)

- **UTC:** `2026-09-15T15:50:26.665226+00:00`
- **Resultado / contexto:** ronda3/3; keep condemned_chip=0.80 deadline=700; next teams=100
- **Cambios:**
  - `switch_min_stay_p_ko`: 0.35 → 0.45

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
switch_min_stay_p_ko: 0.45
overkill_waste_weight: 140.0
ko_free_switch_bonus: 55.0
entry_punish_weight: 1.15
threat_weight: 0.48
hp_rel_weight: 0.15
syn_weight: 0.55
condemned_chip_discount: 0.8
condemned_lasting_boost: 1.35
threat_relief_weight: 1.0
entry_ability_weight: 1.0
```

## 2026-09-15 — revert switch_min_stay 0.45→0.50 (46.9%% < 49.5%% con stay=0.50)

- **UTC:** `2026-09-15T16:13:21.237792+00:00`
- **Resultado / contexto:** mejor config: condemned_chip=0.80 threat=0.48 save=140 deadline=700 stay=0.50
- **Cambios:**
  - `switch_min_stay_p_ko`: 0.35 → 0.5

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
threat_weight: 0.48
hp_rel_weight: 0.15
syn_weight: 0.55
condemned_chip_discount: 0.8
condemned_lasting_boost: 1.35
threat_relief_weight: 1.0
entry_ability_weight: 1.0
```

## 2026-09-15 — estructura: pi_opp post-calc con dano real + moves F1/F2; knobs greedy=0.82 control=0.12

- **UTC:** `2026-09-15T16:40:14.814799+00:00`
- **Resultado / contexto:** baseline teams100 WR39.4%% NEED_FIX inf golisopod-mega; structural vs joint-like threat
- **Cambios:**
  - `opp_policy_control_weight`: 0.12 → 0.12
  - `opp_policy_greedy_weight`: 0.82 → 0.82
  - `opp_policy_use_real_damage`: True → True

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
threat_weight: 0.48
hp_rel_weight: 0.15
syn_weight: 0.55
condemned_chip_discount: 0.8
condemned_lasting_boost: 1.35
threat_relief_weight: 1.0
entry_ability_weight: 1.0
opp_policy_greedy_weight: 0.82
opp_policy_control_weight: 0.12
opp_policy_use_real_damage: True
```

## 2026-09-15 — pi_opp: split targets + switch mass

- **UTC:** `2026-09-15T18:09:43.804176+00:00`
- **Resultado / contexto:** after div log teams100 WR47.5 noise; 91pct phase2 diverge
- **Cambios:**
  - `opp_policy_greedy_weight`: 0.75 → 0.75
  - `opp_policy_split_targets`: True → True
  - `opp_policy_switch_weight`: 0.1 → 0.1

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
threat_weight: 0.48
hp_rel_weight: 0.15
syn_weight: 0.55
condemned_chip_discount: 0.8
condemned_lasting_boost: 1.35
threat_relief_weight: 1.0
entry_ability_weight: 1.0
opp_policy_greedy_weight: 0.75
opp_policy_control_weight: 0.12
opp_policy_switch_weight: 0.1
opp_policy_use_real_damage: True
opp_policy_split_targets: True
```

## 2026-09-15 — REVERT pi_opp split+switch (WR 33%% << 47.5%%)

- **UTC:** `2026-09-15T18:35:48.535206+00:00`
- **Resultado / contexto:** teams100 after targets+switch: 32-65 NEED_FIX=NO; restore prior pi_opp; keep divergence log
- **Cambios:**
  - `opp_policy_greedy_weight`: 0.75 → 0.82
  - `opp_policy_split_targets`: False → False
  - `opp_policy_switch_weight`: 0.0 → 0.0

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
threat_weight: 0.48
hp_rel_weight: 0.15
syn_weight: 0.55
condemned_chip_discount: 0.8
condemned_lasting_boost: 1.35
threat_relief_weight: 1.0
entry_ability_weight: 1.0
opp_policy_greedy_weight: 0.82
opp_policy_control_weight: 0.12
opp_policy_switch_weight: 0.0
opp_policy_use_real_damage: True
opp_policy_split_targets: False
```

## 2026-09-15 — phase2 gate: keep joint unless V_bayes(p2)>=V(p1)

- **UTC:** `2026-09-15T19:00:14.428030+00:00`
- **Resultado / contexto:** next teams=100 vs greedy_joint; diverge was 91pct
- **Cambios:**
  - `phase2_gate_enabled`: True → True
  - `phase2_gate_margin`: 0.0 → 0.0

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
threat_weight: 0.48
hp_rel_weight: 0.15
syn_weight: 0.55
condemned_chip_discount: 0.8
condemned_lasting_boost: 1.35
threat_relief_weight: 1.0
entry_ability_weight: 1.0
opp_policy_greedy_weight: 0.82
opp_policy_control_weight: 0.12
opp_policy_switch_weight: 0.0
opp_policy_use_real_damage: True
opp_policy_split_targets: False
phase2_gate_enabled: True
phase2_gate_margin: 0.0
```

## 2026-09-15 — phase2 gate margin 0→80 (0 era tautologico)

- **UTC:** `2026-09-15T19:20:24.537184+00:00`
- **Resultado / contexto:** teams100 margin0: WR44 gate_rate=0%%; re-run with margin=80
- **Cambios:**
  - `phase2_gate_margin`: 80.0 → 80.0

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
threat_weight: 0.48
hp_rel_weight: 0.15
syn_weight: 0.55
condemned_chip_discount: 0.8
condemned_lasting_boost: 1.35
threat_relief_weight: 1.0
entry_ability_weight: 1.0
opp_policy_greedy_weight: 0.82
opp_policy_control_weight: 0.12
opp_policy_switch_weight: 0.0
opp_policy_use_real_damage: True
opp_policy_split_targets: False
phase2_gate_enabled: True
phase2_gate_margin: 80.0
```

## 2026-09-15 — ABLATION: enable_phase2=False (joint-only techo)

- **UTC:** `2026-09-15T19:55:01.153054+00:00`
- **Resultado / contexto:** next teams=100 vs greedy_joint; if WR 48-52 then surgical p2 else V-calibration
- **Cambios:**
  - `enable_phase2`: False → False

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
threat_weight: 0.48
hp_rel_weight: 0.15
syn_weight: 0.55
condemned_chip_discount: 0.8
condemned_lasting_boost: 1.35
threat_relief_weight: 1.0
entry_ability_weight: 1.0
opp_policy_greedy_weight: 0.82
opp_policy_control_weight: 0.12
opp_policy_switch_weight: 0.0
opp_policy_use_real_damage: True
opp_policy_split_targets: False
phase2_gate_enabled: True
phase2_gate_margin: 80.0
enable_phase2: False
```

## 2026-09-15 — phase2 surgical: switch/speed/prio only (after ablation WR56%%)

- **UTC:** `2026-09-15T20:09:29.055984+00:00`
- **Resultado / contexto:** ablation joint-only 56%%; surgical niches + margin80; next teams=100
- **Cambios:**
  - `enable_phase2`: True → True
  - `phase2_gate_margin`: 80.0 → 80.0
  - `phase2_surgical_only`: True → True

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
threat_weight: 0.48
hp_rel_weight: 0.15
syn_weight: 0.55
condemned_chip_discount: 0.8
condemned_lasting_boost: 1.35
threat_relief_weight: 1.0
entry_ability_weight: 1.0
opp_policy_greedy_weight: 0.82
opp_policy_control_weight: 0.12
opp_policy_switch_weight: 0.0
opp_policy_use_real_damage: True
opp_policy_split_targets: False
phase2_gate_enabled: True
phase2_gate_margin: 80.0
enable_phase2: True
phase2_surgical_only: True
```

## 2026-09-15 — outcome calibration (#3): V vs W/L logging

- **UTC:** `2026-09-15T20:42:26.325184+00:00`
- **Resultado / contexto:** surgical 51.5%% < ablation 56%%; next teams=100 with outcome stats
- **Cambios:**
  - `enable_phase2`: True → True
  - `phase2_surgical_only`: True → True

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
threat_weight: 0.48
hp_rel_weight: 0.15
syn_weight: 0.55
condemned_chip_discount: 0.8
condemned_lasting_boost: 1.35
threat_relief_weight: 1.0
entry_ability_weight: 1.0
opp_policy_greedy_weight: 0.82
opp_policy_control_weight: 0.12
opp_policy_switch_weight: 0.0
opp_policy_use_real_damage: True
opp_policy_split_targets: False
phase2_gate_enabled: True
phase2_gate_margin: 80.0
enable_phase2: True
phase2_surgical_only: True
```

## 2026-09-15 — V calib: max_delta_v=320 switch_v_scale=0.65 + showteam soft + BO3 short SKIP

- **UTC:** `2026-09-15T21:16:49.028696+00:00`
- **Resultado / contexto:** next teams=100; ablation was 56%%; fix g1 false WIN and showteam WARN
- **Cambios:**
  - `phase2_max_delta_v`: 320.0 → 320.0
  - `switch_v_scale`: 0.65 → 0.65

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
threat_weight: 0.48
hp_rel_weight: 0.15
syn_weight: 0.55
condemned_chip_discount: 0.8
condemned_lasting_boost: 1.35
threat_relief_weight: 1.0
entry_ability_weight: 1.0
opp_policy_greedy_weight: 0.82
opp_policy_control_weight: 0.12
opp_policy_switch_weight: 0.0
opp_policy_use_real_damage: True
opp_policy_split_targets: False
phase2_gate_enabled: True
phase2_gate_margin: 80.0
phase2_max_delta_v: 320.0
enable_phase2: True
phase2_surgical_only: True
switch_v_scale: 0.65
```

## 2026-09-15 — DEFAULT: enable_phase2=False (joint-only; phase2 empeoraba WR)

- **UTC:** `2026-09-15T21:56:49.548076+00:00`
- **Resultado / contexto:** default off; re-validate teams 100 vs greedy_joint
- **Cambios:**
  - `enable_phase2`: False → False

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
threat_weight: 0.48
hp_rel_weight: 0.15
syn_weight: 0.55
condemned_chip_discount: 0.8
condemned_lasting_boost: 1.35
threat_relief_weight: 1.0
entry_ability_weight: 1.0
opp_policy_greedy_weight: 0.82
opp_policy_control_weight: 0.12
opp_policy_switch_weight: 0.0
opp_policy_use_real_damage: True
opp_policy_split_targets: False
phase2_gate_enabled: True
phase2_gate_margin: 80.0
phase2_max_delta_v: 320.0
enable_phase2: False
phase2_surgical_only: True
switch_v_scale: 0.65
```

## 2026-09-15 — ABLATION: enable_voluntary_switch=False (vs solo; pivots sospechosos)

- **UTC:** `2026-09-15T22:55:40.868214+00:00`
- **Resultado / contexto:** ablate then --teams 50 vs greedy_solo
- **Cambios:**
  - `enable_voluntary_switch`: True → False

```
--- expectimax_config ---
deadline_ms: 700.0
phase1_budget_ms: 50.0
phase2_end_ms: 680.0
k_profiles: 5
max_worlds: 16
max_our_pairs: 36
soft_ko_bonus: 1000.0
enable_voluntary_switch: False
max_switch_hybrids: 12
loss_weight: 650.0
save_weight: 140.0
chip_soft_cap: 30.0
switch_min_stay_p_ko: 0.5
overkill_waste_weight: 140.0
ko_free_switch_bonus: 55.0
entry_punish_weight: 1.15
threat_weight: 0.48
hp_rel_weight: 0.15
syn_weight: 0.55
condemned_chip_discount: 0.8
condemned_lasting_boost: 1.35
threat_relief_weight: 1.0
entry_ability_weight: 1.0
opp_policy_greedy_weight: 0.82
opp_policy_control_weight: 0.12
opp_policy_switch_weight: 0.0
opp_policy_use_real_damage: True
opp_policy_split_targets: False
phase2_gate_enabled: True
phase2_gate_margin: 80.0
phase2_max_delta_v: 320.0
enable_phase2: False
phase2_surgical_only: True
switch_v_scale: 0.65
```

## 2026-09-15 — ABLATION #2: threat=0.20 save=50 (alinear KO vs solo)

- **UTC:** `2026-09-15T23:08:07.220285+00:00`
- **Resultado / contexto:** then --teams 50 vs greedy_solo
- **Cambios:**
  - `save_weight`: 220.0 → 50
  - `threat_weight`: 0.75 → 0.2

```
--- expectimax_config ---
deadline_ms: 700.0
phase1_budget_ms: 50.0
phase2_end_ms: 680.0
k_profiles: 5
max_worlds: 16
max_our_pairs: 36
soft_ko_bonus: 1000.0
enable_voluntary_switch: False
max_switch_hybrids: 12
loss_weight: 650.0
save_weight: 50.0
chip_soft_cap: 30.0
switch_min_stay_p_ko: 0.5
overkill_waste_weight: 140.0
ko_free_switch_bonus: 55.0
entry_punish_weight: 1.15
threat_weight: 0.2
hp_rel_weight: 0.15
syn_weight: 0.55
condemned_chip_discount: 0.8
condemned_lasting_boost: 1.35
threat_relief_weight: 1.0
entry_ability_weight: 1.0
opp_policy_greedy_weight: 0.82
opp_policy_control_weight: 0.12
opp_policy_switch_weight: 0.0
opp_policy_use_real_damage: True
opp_policy_split_targets: False
phase2_gate_enabled: True
phase2_gate_margin: 80.0
phase2_max_delta_v: 320.0
enable_phase2: False
phase2_surgical_only: True
switch_v_scale: 0.65
```

## 2026-09-15 — Bayes puro: phase1=solo + enable_phase2 + surgical KO/speed; switches off

- **UTC:** `2026-09-15T23:30:50.369957+00:00`
- **Resultado / contexto:** validate --teams 40 vs greedy_solo
- **Cambios:**
  - `enable_phase2`: True → True

```
--- expectimax_config ---
deadline_ms: 700.0
phase1_budget_ms: 50.0
phase2_end_ms: 680.0
k_profiles: 5
max_worlds: 16
max_our_pairs: 36
soft_ko_bonus: 1000.0
enable_voluntary_switch: False
max_switch_hybrids: 12
loss_weight: 650.0
save_weight: 50.0
chip_soft_cap: 30.0
switch_min_stay_p_ko: 0.5
overkill_waste_weight: 140.0
ko_free_switch_bonus: 55.0
entry_punish_weight: 1.15
threat_weight: 0.2
hp_rel_weight: 0.15
syn_weight: 0.55
condemned_chip_discount: 0.8
condemned_lasting_boost: 1.35
threat_relief_weight: 1.0
entry_ability_weight: 1.0
opp_policy_greedy_weight: 0.82
opp_policy_control_weight: 0.12
opp_policy_switch_weight: 0.0
opp_policy_use_real_damage: True
opp_policy_split_targets: False
phase2_gate_enabled: True
phase2_gate_margin: 80.0
phase2_max_delta_v: 320.0
enable_phase2: True
phase2_surgical_only: True
switch_v_scale: 0.65
```

## 2026-09-15 — Bayes paso1: gate_margin=200 max_delta_v=200 (casi no diverger)

- **UTC:** `2026-09-15T23:54:05.168765+00:00`
- **Resultado / contexto:** teams 40 vs greedy_solo
- **Cambios:**
  - `phase2_gate_margin`: 200.0 → 200
  - `phase2_max_delta_v`: 200.0 → 200

```
--- expectimax_config ---
deadline_ms: 700.0
phase1_budget_ms: 50.0
phase2_end_ms: 680.0
k_profiles: 5
max_worlds: 16
max_our_pairs: 36
soft_ko_bonus: 1000.0
enable_voluntary_switch: False
max_switch_hybrids: 12
loss_weight: 650.0
save_weight: 50.0
chip_soft_cap: 30.0
switch_min_stay_p_ko: 0.5
overkill_waste_weight: 140.0
ko_free_switch_bonus: 55.0
entry_punish_weight: 1.15
threat_weight: 0.2
hp_rel_weight: 0.15
syn_weight: 0.55
condemned_chip_discount: 0.8
condemned_lasting_boost: 1.35
threat_relief_weight: 1.0
entry_ability_weight: 1.0
opp_policy_greedy_weight: 0.82
opp_policy_control_weight: 0.12
opp_policy_switch_weight: 0.0
opp_policy_use_real_damage: True
opp_policy_split_targets: False
phase2_gate_enabled: True
phase2_gate_margin: 200.0
phase2_max_delta_v: 200.0
enable_phase2: True
phase2_surgical_only: True
switch_v_scale: 0.65
```

## 2026-09-16 — Paso1 FAIL 23% (timeouts); revert gate. Paso2: surgical solo speed/prio

- **UTC:** `2026-09-16T00:11:14.425561+00:00`
- **Resultado / contexto:** teams40 next
- **Cambios:**
  - `phase2_gate_margin`: 80.0 → 80
  - `phase2_max_delta_v`: 320.0 → 320

```
--- expectimax_config ---
deadline_ms: 700.0
phase1_budget_ms: 50.0
phase2_end_ms: 680.0
k_profiles: 5
max_worlds: 16
max_our_pairs: 36
soft_ko_bonus: 1000.0
enable_voluntary_switch: False
max_switch_hybrids: 12
loss_weight: 650.0
save_weight: 50.0
chip_soft_cap: 30.0
switch_min_stay_p_ko: 0.5
overkill_waste_weight: 140.0
ko_free_switch_bonus: 55.0
entry_punish_weight: 1.15
threat_weight: 0.2
hp_rel_weight: 0.15
syn_weight: 0.55
condemned_chip_discount: 0.8
condemned_lasting_boost: 1.35
threat_relief_weight: 1.0
entry_ability_weight: 1.0
opp_policy_greedy_weight: 0.82
opp_policy_control_weight: 0.12
opp_policy_switch_weight: 0.0
opp_policy_use_real_damage: True
opp_policy_split_targets: False
phase2_gate_enabled: True
phase2_gate_margin: 80.0
phase2_max_delta_v: 320.0
enable_phase2: True
phase2_surgical_only: True
switch_v_scale: 0.65
```

## 2026-09-16 — Paso2 FAIL 41.7%. Paso3: V threat=syn=save=0

- **UTC:** `2026-09-16T00:23:45.809204+00:00`
- **Resultado / contexto:** teams40 vs solo
- **Cambios:**
  - `save_weight`: 220.0 → 0
  - `syn_weight`: 0.35 → 0
  - `threat_weight`: 0.75 → 0

```
--- expectimax_config ---
deadline_ms: 700.0
phase1_budget_ms: 50.0
phase2_end_ms: 680.0
k_profiles: 5
max_worlds: 16
max_our_pairs: 36
soft_ko_bonus: 1000.0
enable_voluntary_switch: False
max_switch_hybrids: 12
loss_weight: 650.0
save_weight: 0.0
chip_soft_cap: 30.0
switch_min_stay_p_ko: 0.5
overkill_waste_weight: 140.0
ko_free_switch_bonus: 55.0
entry_punish_weight: 1.15
threat_weight: 0.0
hp_rel_weight: 0.15
syn_weight: 0.0
condemned_chip_discount: 0.8
condemned_lasting_boost: 1.35
threat_relief_weight: 1.0
entry_ability_weight: 1.0
opp_policy_greedy_weight: 0.82
opp_policy_control_weight: 0.12
opp_policy_switch_weight: 0.0
opp_policy_use_real_damage: True
opp_policy_split_targets: False
phase2_gate_enabled: True
phase2_gate_margin: 80.0
phase2_max_delta_v: 320.0
enable_phase2: True
phase2_surgical_only: True
switch_v_scale: 0.65
```

## 2026-09-16 — Paso3 FAIL 38.5%. Paso4: pi_opp greedy-only control=0

- **UTC:** `2026-09-16T00:44:20.697026+00:00`
- **Resultado / contexto:** teams40 vs solo
- **Cambios:**
  - `opp_policy_control_weight`: 0.15 → 0
  - `opp_policy_greedy_weight`: 0.75 → 1.0

```
--- expectimax_config ---
deadline_ms: 700.0
phase1_budget_ms: 50.0
phase2_end_ms: 680.0
k_profiles: 5
max_worlds: 16
max_our_pairs: 36
soft_ko_bonus: 1000.0
enable_voluntary_switch: False
max_switch_hybrids: 12
loss_weight: 650.0
save_weight: 0.0
chip_soft_cap: 30.0
switch_min_stay_p_ko: 0.5
overkill_waste_weight: 140.0
ko_free_switch_bonus: 55.0
entry_punish_weight: 1.15
threat_weight: 0.0
hp_rel_weight: 0.15
syn_weight: 0.0
condemned_chip_discount: 0.8
condemned_lasting_boost: 1.35
threat_relief_weight: 1.0
entry_ability_weight: 1.0
opp_policy_greedy_weight: 1.0
opp_policy_control_weight: 0.0
opp_policy_switch_weight: 0.0
opp_policy_use_real_damage: True
opp_policy_split_targets: False
phase2_gate_enabled: True
phase2_gate_margin: 80.0
phase2_max_delta_v: 320.0
enable_phase2: True
phase2_surgical_only: True
switch_v_scale: 0.65
```

## 2026-09-16 — Paso4 FAIL 40% (S=0). Paso5: k=3 worlds=6 pairs=16

- **UTC:** `2026-09-16T00:49:55.780316+00:00`
- **Resultado / contexto:** teams40 vs solo final
- **Cambios:**
  - `k_profiles`: 5 → 3
  - `max_our_pairs`: 36 → 16
  - `max_worlds`: 16 → 6

```
--- expectimax_config ---
deadline_ms: 700.0
phase1_budget_ms: 50.0
phase2_end_ms: 680.0
k_profiles: 3
max_worlds: 6
max_our_pairs: 16
soft_ko_bonus: 1000.0
enable_voluntary_switch: False
max_switch_hybrids: 12
loss_weight: 650.0
save_weight: 0.0
chip_soft_cap: 30.0
switch_min_stay_p_ko: 0.5
overkill_waste_weight: 140.0
ko_free_switch_bonus: 55.0
entry_punish_weight: 1.15
threat_weight: 0.0
hp_rel_weight: 0.15
syn_weight: 0.0
condemned_chip_discount: 0.8
condemned_lasting_boost: 1.35
threat_relief_weight: 1.0
entry_ability_weight: 1.0
opp_policy_greedy_weight: 1.0
opp_policy_control_weight: 0.0
opp_policy_switch_weight: 0.0
opp_policy_use_real_damage: True
opp_policy_split_targets: False
phase2_gate_enabled: True
phase2_gate_margin: 80.0
phase2_max_delta_v: 320.0
enable_phase2: True
phase2_surgical_only: True
switch_v_scale: 0.65
```

## 2026-09-16 — Hipotesis B: opp_policy_mirror_solo=True (pi_opp Dirac greedy)

- **UTC:** `2026-09-16T02:53:24.826477+00:00`
- **Resultado / contexto:** teams40 vs solo; stop if improves
- **Cambios:**
  - `opp_policy_mirror_solo`: True → True

```
--- expectimax_config ---
deadline_ms: 700.0
phase1_budget_ms: 50.0
phase2_end_ms: 680.0
k_profiles: 3
max_worlds: 6
max_our_pairs: 16
soft_ko_bonus: 1000.0
enable_voluntary_switch: False
max_switch_hybrids: 12
loss_weight: 650.0
save_weight: 0.0
chip_soft_cap: 30.0
switch_min_stay_p_ko: 0.5
overkill_waste_weight: 140.0
ko_free_switch_bonus: 55.0
entry_punish_weight: 1.15
threat_weight: 0.0
hp_rel_weight: 0.15
syn_weight: 0.0
condemned_chip_discount: 0.8
condemned_lasting_boost: 1.35
threat_relief_weight: 1.0
entry_ability_weight: 1.0
opp_policy_greedy_weight: 1.0
opp_policy_control_weight: 0.0
opp_policy_switch_weight: 0.0
opp_policy_use_real_damage: True
opp_policy_split_targets: False
opp_policy_mirror_solo: True
phase2_gate_enabled: True
phase2_gate_margin: 80.0
phase2_max_delta_v: 320.0
enable_phase2: True
phase2_surgical_only: True
switch_v_scale: 0.65
```

## 2026-09-16 — B FAIL 20.5%. Hipotesis A: v_solo_isomorphic + mirror off

- **UTC:** `2026-09-16T03:05:18.396398+00:00`
- **Resultado / contexto:** teams40 vs solo
- **Cambios:**
  - `opp_policy_mirror_solo`: False → False
  - `v_solo_isomorphic`: True → True

```
--- expectimax_config ---
deadline_ms: 700.0
phase1_budget_ms: 50.0
phase2_end_ms: 680.0
k_profiles: 3
max_worlds: 6
max_our_pairs: 16
soft_ko_bonus: 1000.0
enable_voluntary_switch: False
max_switch_hybrids: 12
loss_weight: 650.0
save_weight: 0.0
chip_soft_cap: 30.0
switch_min_stay_p_ko: 0.5
overkill_waste_weight: 140.0
ko_free_switch_bonus: 55.0
entry_punish_weight: 1.15
threat_weight: 0.0
hp_rel_weight: 0.15
syn_weight: 0.0
condemned_chip_discount: 0.8
condemned_lasting_boost: 1.35
threat_relief_weight: 1.0
entry_ability_weight: 1.0
opp_policy_greedy_weight: 1.0
opp_policy_control_weight: 0.0
opp_policy_switch_weight: 0.0
opp_policy_use_real_damage: True
opp_policy_split_targets: False
opp_policy_mirror_solo: False
v_solo_isomorphic: True
phase2_gate_enabled: True
phase2_gate_margin: 80.0
phase2_max_delta_v: 320.0
enable_phase2: True
phase2_surgical_only: True
switch_v_scale: 0.65
```

## 2026-09-16 — A=52.5% leve. Hipotesis C: phase2_retarget_only

- **UTC:** `2026-09-16T03:14:33.641639+00:00`
- **Resultado / contexto:** teams40 vs solo
- **Cambios:**
  - `phase2_retarget_only`: True → True

```
--- expectimax_config ---
deadline_ms: 700.0
phase1_budget_ms: 50.0
phase2_end_ms: 680.0
k_profiles: 3
max_worlds: 6
max_our_pairs: 16
soft_ko_bonus: 1000.0
enable_voluntary_switch: False
max_switch_hybrids: 12
loss_weight: 650.0
save_weight: 0.0
chip_soft_cap: 30.0
switch_min_stay_p_ko: 0.5
overkill_waste_weight: 140.0
ko_free_switch_bonus: 55.0
entry_punish_weight: 1.15
threat_weight: 0.0
hp_rel_weight: 0.15
syn_weight: 0.0
condemned_chip_discount: 0.8
condemned_lasting_boost: 1.35
threat_relief_weight: 1.0
entry_ability_weight: 1.0
opp_policy_greedy_weight: 1.0
opp_policy_control_weight: 0.0
opp_policy_switch_weight: 0.0
opp_policy_use_real_damage: True
opp_policy_split_targets: False
opp_policy_mirror_solo: False
v_solo_isomorphic: True
phase2_retarget_only: True
phase2_gate_enabled: True
phase2_gate_margin: 80.0
phase2_max_delta_v: 320.0
enable_phase2: True
phase2_surgical_only: True
switch_v_scale: 0.65
```

## 2026-09-16 — C FAIL 42.5%. Keep A (v_solo_isomorphic); retarget off

- **UTC:** `2026-09-16T03:25:05.473202+00:00`
- **Resultado / contexto:** best of B/A/C = A at 52.5%
- **Cambios:**
  - `phase2_retarget_only`: False → False

```
--- expectimax_config ---
deadline_ms: 700.0
phase1_budget_ms: 50.0
phase2_end_ms: 680.0
k_profiles: 3
max_worlds: 6
max_our_pairs: 16
soft_ko_bonus: 1000.0
enable_voluntary_switch: False
max_switch_hybrids: 12
loss_weight: 650.0
save_weight: 0.0
chip_soft_cap: 30.0
switch_min_stay_p_ko: 0.5
overkill_waste_weight: 140.0
ko_free_switch_bonus: 55.0
entry_punish_weight: 1.15
threat_weight: 0.0
hp_rel_weight: 0.15
syn_weight: 0.0
condemned_chip_discount: 0.8
condemned_lasting_boost: 1.35
threat_relief_weight: 1.0
entry_ability_weight: 1.0
opp_policy_greedy_weight: 1.0
opp_policy_control_weight: 0.0
opp_policy_switch_weight: 0.0
opp_policy_use_real_damage: True
opp_policy_split_targets: False
opp_policy_mirror_solo: False
v_solo_isomorphic: True
phase2_retarget_only: False
phase2_gate_enabled: True
phase2_gate_margin: 80.0
phase2_max_delta_v: 320.0
enable_phase2: True
phase2_surgical_only: True
switch_v_scale: 0.65
```

## 2026-09-16 — A+C: Top-K=3 joint phase1, EV leaf, anti-timeout worlds=4

- **UTC:** `2026-09-16T04:15:51.699347+00:00`
- **Resultado / contexto:** validate --seed 42 --teams 100 vs greedy_joint
- **Cambios:**
  - `deadline_ms`: 700.0 → 550
  - `max_our_pairs`: 36 → 9
  - `max_worlds`: 16 → 4
  - `phase1_use_joint`: True → True
  - `topk_per_side`: 3 → 3
  - `use_topk_abstraction`: True → True

```
--- expectimax_config ---
deadline_ms: 550.0
phase1_budget_ms: 40.0
phase2_end_ms: 500.0
k_profiles: 2
max_worlds: 4
topk_per_side: 3
max_our_pairs: 9
soft_ko_bonus: 1000.0
enable_voluntary_switch: False
max_switch_hybrids: 0
loss_weight: 650.0
save_weight: 0.0
chip_soft_cap: 30.0
switch_min_stay_p_ko: 0.5
overkill_waste_weight: 140.0
ko_free_switch_bonus: 55.0
entry_punish_weight: 1.15
threat_weight: 0.0
hp_rel_weight: 0.15
syn_weight: 0.0
condemned_chip_discount: 0.8
condemned_lasting_boost: 1.35
threat_relief_weight: 1.0
entry_ability_weight: 1.0
opp_policy_greedy_weight: 1.0
opp_policy_control_weight: 0.0
opp_policy_switch_weight: 0.0
opp_policy_use_real_damage: True
opp_policy_split_targets: False
opp_policy_mirror_solo: False
v_solo_isomorphic: True
phase2_retarget_only: False
phase1_use_joint: True
use_topk_abstraction: True
phase2_gate_enabled: True
phase2_gate_margin: 80.0
phase2_max_delta_v: 320.0
enable_phase2: True
phase2_surgical_only: True
switch_v_scale: 0.65
```

## 2026-09-16 — Cuevas teacher: niche Protect/speed + switches<=4; RecursionError logger fix

- **UTC:** `2026-09-16T04:58:24.907897+00:00`
- **Resultado / contexto:** --seed 42 --teams 100 vs greedy_joint
- **Cambios:**
  - `enable_niche_pairs`: True → True
  - `enable_voluntary_switch`: True → True
  - `max_our_pairs`: 36 → 12
  - `max_switch_hybrids`: 12 → 4

```
--- expectimax_config ---
deadline_ms: 550.0
phase1_budget_ms: 40.0
phase2_end_ms: 500.0
k_profiles: 2
max_worlds: 4
topk_per_side: 3
max_our_pairs: 12
soft_ko_bonus: 1000.0
enable_voluntary_switch: True
max_switch_hybrids: 4
enable_niche_pairs: True
loss_weight: 650.0
save_weight: 0.0
chip_soft_cap: 30.0
switch_min_stay_p_ko: 0.5
overkill_waste_weight: 140.0
ko_free_switch_bonus: 55.0
entry_punish_weight: 1.15
threat_weight: 0.0
hp_rel_weight: 0.15
syn_weight: 0.0
condemned_chip_discount: 0.8
condemned_lasting_boost: 1.35
threat_relief_weight: 1.0
entry_ability_weight: 1.0
opp_policy_greedy_weight: 1.0
opp_policy_control_weight: 0.0
opp_policy_switch_weight: 0.0
opp_policy_use_real_damage: True
opp_policy_split_targets: False
opp_policy_mirror_solo: False
v_solo_isomorphic: True
phase2_retarget_only: False
phase1_use_joint: True
use_topk_abstraction: True
phase2_gate_enabled: True
phase2_gate_margin: 80.0
phase2_max_delta_v: 320.0
enable_phase2: True
phase2_surgical_only: True
switch_v_scale: 0.65
```

## 2026-09-16 — Cuevas FAIL seed42 teams100: 35% WR pair19% CLEAR loss; revert niches, keep A+C + logger fix

- **UTC:** `2026-09-16T05:35:27.411977+00:00`
- **Resultado / contexto:** back to A+C baseline
- **Cambios:**
  - `enable_niche_pairs`: False → False
  - `enable_voluntary_switch`: True → False
  - `max_switch_hybrids`: 12 → 0

```
--- expectimax_config ---
deadline_ms: 550.0
phase1_budget_ms: 40.0
phase2_end_ms: 500.0
k_profiles: 2
max_worlds: 4
topk_per_side: 3
max_our_pairs: 12
soft_ko_bonus: 1000.0
enable_voluntary_switch: False
max_switch_hybrids: 0
enable_niche_pairs: False
loss_weight: 650.0
save_weight: 0.0
chip_soft_cap: 30.0
switch_min_stay_p_ko: 0.5
overkill_waste_weight: 140.0
ko_free_switch_bonus: 55.0
entry_punish_weight: 1.15
threat_weight: 0.0
hp_rel_weight: 0.15
syn_weight: 0.0
condemned_chip_discount: 0.8
condemned_lasting_boost: 1.35
threat_relief_weight: 1.0
entry_ability_weight: 1.0
opp_policy_greedy_weight: 1.0
opp_policy_control_weight: 0.0
opp_policy_switch_weight: 0.0
opp_policy_use_real_damage: True
opp_policy_split_targets: False
opp_policy_mirror_solo: False
v_solo_isomorphic: True
phase2_retarget_only: False
phase1_use_joint: True
use_topk_abstraction: True
phase2_gate_enabled: True
phase2_gate_margin: 80.0
phase2_max_delta_v: 320.0
enable_phase2: True
phase2_surgical_only: True
switch_v_scale: 0.65
```

## 2026-09-16 — Paso1 fundacional: indifference eps=40 + surgical_only=False (Top-K libre)

- **UTC:** `2026-09-16T05:46:15.252601+00:00`
- **Resultado / contexto:** next --seed 42 --teams 40 vs greedy_joint; target div>=15% WR>=45%
- **Cambios:**
  - `phase2_gate_mode`: 'indifference' → 'indifference'
  - `phase2_indifference_eps`: 40.0 → 40
  - `phase2_surgical_only`: False → False

```
--- expectimax_config ---
deadline_ms: 550.0
phase1_budget_ms: 40.0
phase2_end_ms: 500.0
k_profiles: 2
max_worlds: 4
topk_per_side: 3
max_our_pairs: 12
soft_ko_bonus: 1000.0
enable_voluntary_switch: False
max_switch_hybrids: 0
enable_niche_pairs: False
loss_weight: 650.0
save_weight: 0.0
chip_soft_cap: 30.0
switch_min_stay_p_ko: 0.5
overkill_waste_weight: 140.0
ko_free_switch_bonus: 55.0
entry_punish_weight: 1.15
threat_weight: 0.0
hp_rel_weight: 0.15
syn_weight: 0.0
condemned_chip_discount: 0.8
condemned_lasting_boost: 1.35
threat_relief_weight: 1.0
entry_ability_weight: 1.0
opp_policy_greedy_weight: 1.0
opp_policy_control_weight: 0.0
opp_policy_switch_weight: 0.0
opp_policy_use_real_damage: True
opp_policy_split_targets: False
opp_policy_mirror_solo: False
v_solo_isomorphic: True
phase2_retarget_only: False
phase1_use_joint: True
use_topk_abstraction: True
phase2_gate_enabled: True
phase2_gate_mode: 'indifference'
phase2_gate_margin: 80.0
phase2_indifference_eps: 40.0
phase2_max_delta_v: 320.0
enable_phase2: True
phase2_surgical_only: False
switch_v_scale: 0.65
```

## 2026-09-16 — Paso1 OK fundacional: indifference+TopK libre

- **UTC:** `2026-09-16T06:01:13.431689+00:00`
- **Resultado / contexto:** seed42 teams40 vs joint: WR55% pairA4/B2/draw14 div_rate=28.4% gate=27.9% wr_with_div=54.4%; confirm --teams 100
- **Cambios:**
  - `phase2_gate_mode`: 'indifference' → 'indifference'
  - `phase2_indifference_eps`: 40.0 → 40
  - `phase2_surgical_only`: False → False

```
--- expectimax_config ---
deadline_ms: 550.0
phase1_budget_ms: 40.0
phase2_end_ms: 500.0
k_profiles: 2
max_worlds: 4
topk_per_side: 3
max_our_pairs: 12
soft_ko_bonus: 1000.0
enable_voluntary_switch: False
max_switch_hybrids: 0
enable_niche_pairs: False
loss_weight: 650.0
save_weight: 0.0
chip_soft_cap: 30.0
switch_min_stay_p_ko: 0.5
overkill_waste_weight: 140.0
ko_free_switch_bonus: 55.0
entry_punish_weight: 1.15
threat_weight: 0.0
hp_rel_weight: 0.15
syn_weight: 0.0
condemned_chip_discount: 0.8
condemned_lasting_boost: 1.35
threat_relief_weight: 1.0
entry_ability_weight: 1.0
opp_policy_greedy_weight: 1.0
opp_policy_control_weight: 0.0
opp_policy_switch_weight: 0.0
opp_policy_use_real_damage: True
opp_policy_split_targets: False
opp_policy_mirror_solo: False
v_solo_isomorphic: True
phase2_retarget_only: False
phase1_use_joint: True
use_topk_abstraction: True
phase2_gate_enabled: True
phase2_gate_mode: 'indifference'
phase2_gate_margin: 80.0
phase2_indifference_eps: 40.0
phase2_max_delta_v: 320.0
enable_phase2: True
phase2_surgical_only: False
switch_v_scale: 0.65
```

## 2026-09-16 — CONFIRM teams100: fundacional cumplido (div+WR)

- **UTC:** `2026-09-16T06:40:27.910803+00:00`
- **Resultado / contexto:** seed42 teams100 vs joint: WR52.1% pairA11/B10/draw25 UNCLEAR div_rate=79.5% gate=5.7% wr_with_div=52.5%; no further steps
- **Cambios:**
  - `phase2_gate_mode`: 'indifference' → 'indifference'
  - `phase2_indifference_eps`: 40.0 → 40
  - `phase2_surgical_only`: False → False

```
--- expectimax_config ---
deadline_ms: 550.0
phase1_budget_ms: 40.0
phase2_end_ms: 500.0
k_profiles: 2
max_worlds: 4
topk_per_side: 3
max_our_pairs: 12
soft_ko_bonus: 1000.0
enable_voluntary_switch: False
max_switch_hybrids: 0
enable_niche_pairs: False
loss_weight: 650.0
save_weight: 0.0
chip_soft_cap: 30.0
switch_min_stay_p_ko: 0.5
overkill_waste_weight: 140.0
ko_free_switch_bonus: 55.0
entry_punish_weight: 1.15
threat_weight: 0.0
hp_rel_weight: 0.15
syn_weight: 0.0
condemned_chip_discount: 0.8
condemned_lasting_boost: 1.35
threat_relief_weight: 1.0
entry_ability_weight: 1.0
opp_policy_greedy_weight: 1.0
opp_policy_control_weight: 0.0
opp_policy_switch_weight: 0.0
opp_policy_use_real_damage: True
opp_policy_split_targets: False
opp_policy_mirror_solo: False
v_solo_isomorphic: True
phase2_retarget_only: False
phase1_use_joint: True
use_topk_abstraction: True
phase2_gate_enabled: True
phase2_gate_mode: 'indifference'
phase2_gate_margin: 80.0
phase2_indifference_eps: 40.0
phase2_max_delta_v: 320.0
enable_phase2: True
phase2_surgical_only: False
switch_v_scale: 0.65
```

## 2026-09-16 — VGC modeling: Knock Off mega-stone exception + Fake Out deny (mega/outspeed) + foe mega/Drought projection

- **UTC:** `2026-09-16T07:20:59.622421+00:00`
- **Resultado / contexto:** scenario Charizard+Sylveon: joint+expectimax Fake Out->Charizard + WH->Sylveon; Knock Off no bonus on Charizardite
- **Cambios:**
  - `enable_fakeout_inject`: None → True
  - `vgc_utility`: None → 'added'

```
--- expectimax_config ---
deadline_ms: 550.0
phase1_budget_ms: 40.0
phase2_end_ms: 500.0
k_profiles: 2
max_worlds: 4
topk_per_side: 3
max_our_pairs: 12
soft_ko_bonus: 1000.0
enable_voluntary_switch: False
max_switch_hybrids: 0
enable_niche_pairs: False
loss_weight: 650.0
save_weight: 0.0
chip_soft_cap: 30.0
switch_min_stay_p_ko: 0.5
overkill_waste_weight: 140.0
ko_free_switch_bonus: 55.0
entry_punish_weight: 1.15
threat_weight: 0.0
hp_rel_weight: 0.15
syn_weight: 0.0
condemned_chip_discount: 0.8
condemned_lasting_boost: 1.35
threat_relief_weight: 1.0
entry_ability_weight: 1.0
opp_policy_greedy_weight: 1.0
opp_policy_control_weight: 0.0
opp_policy_switch_weight: 0.0
opp_policy_use_real_damage: True
opp_policy_split_targets: False
opp_policy_mirror_solo: False
v_solo_isomorphic: True
phase2_retarget_only: False
phase1_use_joint: True
use_topk_abstraction: True
phase2_gate_enabled: True
phase2_gate_mode: 'indifference'
phase2_gate_margin: 80.0
phase2_indifference_eps: 40.0
phase2_max_delta_v: 320.0
enable_phase2: True
phase2_surgical_only: False
switch_v_scale: 0.65
```

## 2026-09-16 — VGC: joint greedy puro; FO deny=amenaza calc; mega-Y projection; Knock Off BP via calc

- **UTC:** `2026-09-16T08:06:06.804244+00:00`
- **Resultado / contexto:** joint Flare Blitz; expectimax should Fake Out when Heat Wave threatens partner
- **Cambios:**
  - `fakeout_deny`: None → 'threat_scaled'
  - `joint_vgc_utility`: None → False

```
--- expectimax_config ---
deadline_ms: 550.0
phase1_budget_ms: 40.0
phase2_end_ms: 500.0
k_profiles: 2
max_worlds: 4
topk_per_side: 3
max_our_pairs: 12
soft_ko_bonus: 1000.0
enable_voluntary_switch: False
max_switch_hybrids: 0
enable_niche_pairs: False
loss_weight: 650.0
save_weight: 0.0
chip_soft_cap: 30.0
switch_min_stay_p_ko: 0.5
overkill_waste_weight: 140.0
ko_free_switch_bonus: 55.0
entry_punish_weight: 1.15
threat_weight: 0.0
hp_rel_weight: 0.15
syn_weight: 0.0
condemned_chip_discount: 0.8
condemned_lasting_boost: 1.35
threat_relief_weight: 1.0
entry_ability_weight: 1.0
opp_policy_greedy_weight: 1.0
opp_policy_control_weight: 0.0
opp_policy_switch_weight: 0.0
opp_policy_use_real_damage: True
opp_policy_split_targets: False
opp_policy_mirror_solo: False
v_solo_isomorphic: True
phase2_retarget_only: False
phase1_use_joint: True
use_topk_abstraction: True
phase2_gate_enabled: True
phase2_gate_mode: 'indifference'
phase2_gate_margin: 80.0
phase2_indifference_eps: 40.0
phase2_max_delta_v: 320.0
enable_phase2: True
phase2_surgical_only: False
switch_v_scale: 0.65
```

## 2026-09-16 — FO deny threat-scaled: max_delta_v off (320 cortaba soft_KO)

- **UTC:** `2026-09-16T08:52:44.179975+00:00`
- **Cambios:**
  - `phase2_max_delta_v`: 0.0 → 0

```
--- expectimax_config ---
deadline_ms: 550.0
phase1_budget_ms: 40.0
phase2_end_ms: 500.0
k_profiles: 2
max_worlds: 4
topk_per_side: 3
max_our_pairs: 12
soft_ko_bonus: 1000.0
enable_voluntary_switch: False
max_switch_hybrids: 0
enable_niche_pairs: False
loss_weight: 650.0
save_weight: 0.0
chip_soft_cap: 30.0
switch_min_stay_p_ko: 0.5
overkill_waste_weight: 140.0
ko_free_switch_bonus: 55.0
entry_punish_weight: 1.15
threat_weight: 0.0
hp_rel_weight: 0.15
syn_weight: 0.0
condemned_chip_discount: 0.8
condemned_lasting_boost: 1.35
threat_relief_weight: 1.0
entry_ability_weight: 1.0
opp_policy_greedy_weight: 1.0
opp_policy_control_weight: 0.0
opp_policy_switch_weight: 0.0
opp_policy_use_real_damage: True
opp_policy_split_targets: False
opp_policy_mirror_solo: False
v_solo_isomorphic: True
phase2_retarget_only: False
phase1_use_joint: True
use_topk_abstraction: True
phase2_gate_enabled: True
phase2_gate_mode: 'indifference'
phase2_gate_margin: 80.0
phase2_indifference_eps: 40.0
phase2_max_delta_v: 0.0
enable_phase2: True
phase2_surgical_only: False
switch_v_scale: 0.65
```

## 2026-09-16 — cobertura 75%: K=7 pairs=49 deadline=1600; quitar inyeccion FO (100% mapeado en SEARCH_COVERAGE_PRESETS)

- **UTC:** `2026-09-16T13:17:28.896356+00:00`
- **Cambios:**
  - `deadline_ms`: 700.0 → 1600
  - `enable_fakeout_inject`: None → False
  - `max_our_pairs`: 36 → 49
  - `phase1_budget_ms`: 50.0 → 80
  - `phase2_end_ms`: 680.0 → 1520
  - `topk_per_side`: 7 → 7

```
--- expectimax_config ---
deadline_ms: 1600.0
phase1_budget_ms: 80.0
phase2_end_ms: 1520.0
k_profiles: 2
max_worlds: 4
topk_per_side: 7
max_our_pairs: 49
soft_ko_bonus: 1000.0
enable_voluntary_switch: False
max_switch_hybrids: 0
enable_niche_pairs: False
loss_weight: 650.0
save_weight: 0.0
chip_soft_cap: 30.0
switch_min_stay_p_ko: 0.5
overkill_waste_weight: 140.0
ko_free_switch_bonus: 55.0
entry_punish_weight: 1.15
threat_weight: 0.0
hp_rel_weight: 0.15
syn_weight: 0.0
condemned_chip_discount: 0.8
condemned_lasting_boost: 1.35
threat_relief_weight: 1.0
entry_ability_weight: 1.0
opp_policy_greedy_weight: 1.0
opp_policy_control_weight: 0.0
opp_policy_switch_weight: 0.0
opp_policy_use_real_damage: True
opp_policy_split_targets: False
opp_policy_mirror_solo: False
v_solo_isomorphic: True
phase2_retarget_only: False
phase1_use_joint: True
use_topk_abstraction: True
phase2_gate_enabled: True
phase2_gate_mode: 'indifference'
phase2_gate_margin: 80.0
phase2_indifference_eps: 40.0
phase2_max_delta_v: 0.0
enable_phase2: True
phase2_surgical_only: False
switch_v_scale: 0.65
```

## 2026-09-17 — A/B voluntary switch small weight vs joint seed10

- **UTC:** `2026-09-17T07:03:29.745130+00:00`
- **Cambios:**
  - `enable_voluntary_switch`: True → True
  - `max_switch_hybrids`: 12 → 4
  - `switch_v_scale`: 0.25 → 0.25

```
--- expectimax_config ---
deadline_ms: 1600.0
phase1_budget_ms: 80.0
phase2_end_ms: 1520.0
k_profiles: 2
max_worlds: 4
topk_per_side: 7
max_our_pairs: 49
soft_ko_bonus: 1000.0
enable_voluntary_switch: True
max_switch_hybrids: 4
enable_niche_pairs: False
loss_weight: 650.0
save_weight: 0.0
chip_soft_cap: 30.0
switch_min_stay_p_ko: 0.5
overkill_waste_weight: 140.0
ko_free_switch_bonus: 55.0
entry_punish_weight: 1.15
threat_weight: 0.0
hp_rel_weight: 0.15
syn_weight: 0.0
condemned_chip_discount: 0.8
condemned_lasting_boost: 1.35
threat_relief_weight: 1.0
entry_ability_weight: 1.0
opp_policy_greedy_weight: 1.0
opp_policy_control_weight: 0.0
opp_policy_switch_weight: 0.0
opp_policy_use_real_damage: True
opp_policy_split_targets: False
opp_policy_mirror_solo: False
v_solo_isomorphic: True
phase2_retarget_only: False
phase1_use_joint: True
use_topk_abstraction: True
phase2_gate_enabled: True
phase2_gate_mode: 'indifference'
phase2_gate_margin: 80.0
phase2_indifference_eps: 40.0
phase2_max_delta_v: 0.0
enable_phase2: True
phase2_surgical_only: False
switch_v_scale: 0.25
```

## 2026-09-17 — REVERT voluntary A/B: teams40 seed10 empeoro (pair 50→33 WR 48.7→45); keep off

- **UTC:** `2026-09-17T07:31:30.519050+00:00`
- **Cambios:**
  - `enable_voluntary_switch`: True → False
  - `max_switch_hybrids`: 12 → 0
  - `switch_v_scale`: 0.65 → 0.65

```
--- expectimax_config ---
deadline_ms: 1600.0
phase1_budget_ms: 80.0
phase2_end_ms: 1520.0
k_profiles: 2
max_worlds: 4
topk_per_side: 7
max_our_pairs: 49
soft_ko_bonus: 1000.0
enable_voluntary_switch: False
max_switch_hybrids: 0
enable_niche_pairs: False
loss_weight: 650.0
save_weight: 0.0
chip_soft_cap: 30.0
switch_min_stay_p_ko: 0.5
overkill_waste_weight: 140.0
ko_free_switch_bonus: 55.0
entry_punish_weight: 1.15
threat_weight: 0.0
hp_rel_weight: 0.15
syn_weight: 0.0
condemned_chip_discount: 0.8
condemned_lasting_boost: 1.35
threat_relief_weight: 1.0
entry_ability_weight: 1.0
opp_policy_greedy_weight: 1.0
opp_policy_control_weight: 0.0
opp_policy_switch_weight: 0.0
opp_policy_use_real_damage: True
opp_policy_split_targets: False
opp_policy_mirror_solo: False
v_solo_isomorphic: True
phase2_retarget_only: False
phase1_use_joint: True
use_topk_abstraction: True
phase2_gate_enabled: True
phase2_gate_mode: 'indifference'
phase2_gate_margin: 80.0
phase2_indifference_eps: 40.0
phase2_max_delta_v: 0.0
enable_phase2: True
phase2_surgical_only: False
switch_v_scale: 0.65
```
