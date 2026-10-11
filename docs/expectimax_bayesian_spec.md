# Expectimax-Bayesian Agent Spec — 1-ply anytime Top-K×Top-K (VGC Gen 9)

| Campo | Valor |
|-------|--------|
| **Documento** | **Contrato exclusivo de implementación — Expectimax-Bayesian** |
| **Versión** | **1.1** (alineado con `docs/rl_env_spec.md` **v1.10** §6.h.4) |
| **Estado** | **FROZEN — implementado en `showdown/battle_bots/expectimax_bayesian/`** |
| **Algoritmo** | Expectimax 1-ply bayesiano anytime; filtro Greedy-Joint Top-K×Top-K; gate vs phase1 |
| **Rol** | Techo heurístico de la escalera; fuente BC; oponente de suite WR |
| **Code root** | `showdown/battle_bots/expectimax_bayesian/` |
| **Id CLI / mosaic** | `expectimax_bayesian` |
| **Player poke-env** | `ExpectimaxBayesianPlayer` (extiende `GreedyJointPlayer`) |
| **Deps** | Sin `torch`. Calc + belief SP; config en `ExpectimaxConfig` |

> **Alcance.** Fuente de verdad para **Expectimax-Bayesian**. Escalera:
> [`heuristic_agents_spec.md`](heuristic_agents_spec.md). Ancla Joint:
> [`greedy_joint_spec.md`](greedy_joint_spec.md). Env RL / non-goals de reward:
> [`rl_env_spec.md`](rl_env_spec.md) §8.
>
> **Arquitectura (código).** **A:** phase1=joint; phase2 solo sobre Top-K×Top-K.
> **C:** daño = esperanza (`expected_damage`), sin ramificar los 16 rolls en el árbol.
>
> **Tune vs contrato.** El historial fino de knobs está en `CONFIG_HISTORY.md` /
> `.jsonl`. El **contrato de tesis** (defaults + cobertura activa + gate) es
> **este** archivo. Overlays en `ACTIVE_OVERRIDES.json` deben snapshotearse en
> reportes E2E.
>
> **Inmutabilidad.** Bump **1.1+** aquí + `rl_env_spec` §6.h juntos.

---

## 1. Por qué existe (lectura no técnica)

Greedy-Joint elige un buen par “optimista” de daño y trucos de un turno.
Expectimax se pregunta: *¿sigue siendo bueno ese par si el daño es aleatorio,
si el rival tiene uno u otro set, y si el rival responde de cierta forma?*
Para no quedarse sin tiempo, primero pide a Joint una lista corta de candidatos
(Top-K por lado) y solo ahí hace la media bajo incertidumbre.

**Analogía:** Joint propone los mejores planes del compás; Expectimax los
re-evalúa con “¿qué pasa si la moneda del daño y el rival no cooperan?”, y solo
cambia de plan si la media no empeora de más (gate).

---

## 2. Delimitación de decisión

| Ítem | Contrato |
|------|----------|
| Horizonte | \(T=1\) anytime (deadline de pared) |
| Espacio phase2 | Subconjunto Top-K×Top-K del espacio joint |
| Aprendizaje | **Ninguno** (sí hay knobs de diseño / tune offline) |
| Fallback | Si timeout/error → acción Joint (`used_fallback`) |
| Reward RL | **Prohibido** usar \(V(u)\) live como \(\Phi(s)\) en \(R_t\) |

---

## 3. Team Preview (contrato afinado)

Módulo: `showdown/battle_bots/expectimax_bayesian/team_preview.py`.  
Entradas IAP / poke-env: `main.py` / `pokeenv_player.py` → `best_expectimax_preview_with_meta`.

### 3.1 Intuición

Greedy solo mira tipado+BST. Expectimax añade: (1) **roles de campo**
reconocibles por especie en preview, (2) **calidad de leads** (los dos que salen
primero deben tipar y, si pueden, traer redirect / Fake Out / Tailwind),
(3) un sesgo defensivo ligeramente mayor (\(\beta>\alpha\)).

**No** usa OTS ni moves ocultos: solo nombre/especie visible en Team Preview.

### 3.2 Búsqueda (exhaustiva)

| Ítem | Contrato |
|------|----------|
| Combinaciones | \(\binom{6}{4}\) brings × \(\binom{4}{2}\) leads = **hasta 90** órdenes |
| Top-K | **No** — se puntúan todos |
| Empate | Mayor `total`, luego mayor BST |
| Salida | `PreviewScore.order` = `(lead1, lead2, bench1, bench2)` 1-based |
| Pool \(<4\) | Relleno defensivo; score 0; meta `pool < 4` |
| API | `best_expectimax_preview_with_meta` (+ wrapper `best_expectimax_preview_order`) |
| Meta | `PreviewDecisionMeta`: top líneas, `why_lines`, conteos |

### 3.3 Fórmula de score (congelada)

Sea \(B\) el score tipado+BST de Solo (`score_combination`):

\[
B = \alpha\cdot\mathrm{off} + \beta\cdot\mathrm{def} + \gamma\cdot\mathrm{BST}
\]

\[
S = B + R_{\mathrm{roles}} + w_{\mathrm{lead}}\cdot Q_{\mathrm{lead}}
\]

| Símbolo | Constante código | Valor |
|---------|------------------|-------|
| \(\alpha\) | `PREVIEW_ALPHA` | **1.05** (greedy: 1.0) |
| \(\beta\) | `PREVIEW_BETA` | **1.25** (greedy: 1.0) |
| \(\gamma\) | `PREVIEW_GAMMA` | **0.008** (greedy: 0.01) |
| \(w_{\mathrm{lead}}\) | `LEAD_QUALITY_WEIGHT` | **0.45** |

### 3.4 Roles \(R_{\mathrm{roles}}\)

Presencia (≥1 especie del set). Matching vía `_nid` =
`normalize_name(name)` **sin** `-`/`_` (p.ej. `Indeedee-F` → `indeedeef`).
Los sets abajo están en esa forma compacta; **no** apilan por cantidad:

| Rol | Constante | Bonus | Set de especies (ids) |
|-----|-----------|-------|------------------------|
| Redirect | `ROLE_BONUS_REDIRECT` | 28 | `amoonguss`, `indeedee`, `indeedeef`, `sableye`, `clefairy` |
| Tailwind | `ROLE_BONUS_TAILWIND` | 32 | `tornadus`, `whimsicott`, `kilowattrel`, `suicune`, `pelipper`, `emolga`, `talonflame`, `braviary`, `braviaryhisui`, `zapdos` |
| Trick Room | `ROLE_BONUS_TRICK_ROOM` | 26 | `indeedee`, `indeedeef`, `hatenna`, `hattrem`, `hatterene`, `slowking`, `slowkinggalar`, `porygon2`, `dusknoir`, `bronzong`, `oranguru` |
| Speed control | `ROLE_BONUS_SPEED_CONTROL` | 18 | `incineroar`, `rillaboom`, `urshifu`, `urshifurapidstrike`, `landorustherian`, `landorus`, `tinglu`, `greattusk` |
| Fake Out | `ROLE_BONUS_FAKE_OUT` | 14 | `incineroar`, `scrafty`, `kangaskhan`, `persian`, `meowth`, `sneasler`, `baxcalibur`, `farigiraf` |

**Diversidad de buckets** (`ROLE_DIVERSITY_BONUS = 12`) si al menos **2** de:

1. redirect  
2. speed_axis = Tailwind **∨** Trick Room  
3. disrupt = speed control **∨** Fake Out  

### 3.5 Calidad de lead \(Q_{\mathrm{lead}}\)

Sobre los **2** leads (no la banca):

\[
Q_{\mathrm{lead}} = \sum_{\ell\in\mathrm{leads}}\big(
1.35\,\mathrm{def}(\ell) + 1.15\,\mathrm{off}(\ell)
+ 10\cdot\mathbf{1}[\ell\in\mathrm{Redirect}\cup\mathrm{FO}\cup\mathrm{TW}]
\big)
\]

Constantes: `LEAD_DEF_SCALE=1.35`, `LEAD_OFF_SCALE=1.15`, `LEAD_ROLE_BONUS=10`.

### 3.6 Contraste con Greedy preview

| | Greedy-Solo/Joint | Expectimax |
|--|-------------------|------------|
| Pesos tipado | \(\alpha=\beta=1\), \(\gamma=0.01\) | §3.3 |
| Roles | No | Sí (§3.4) |
| Lead quality | No (solo orden por score combo) | Sí (§3.5) |
| Búsqueda | \(\binom{6}{4}\times\binom{4}{2}\) | Igual (exhaustiva) |

### 3.7 Non-goals (preview)

- Inferir moves/item/SP desde OTS en el score de preview.  
- Top-K / anytime en preview (eso es combate phase2).  
- Compartir pesos \(\alpha,\beta,\gamma\) con Greedy (son deliberadamente distintos).

Cambiar sets, pesos o fórmula ⇒ bump **1.2+** de este spec.

---


## 4. Pipeline de combate

```
phase1  →  greedy_joint          (phase1_use_joint = True)
         ↓
Top-K × Top-K  (cobertura activa; ver §5)
         ↓
phase2  →  expectimax bayesiano anytime
         ↓
gate    →  aceptar phase2 vs V del par phase1 (§6)
         ↓
fallback → Joint si timeout / error / gate rechaza
```

### 4.1 Mundos y rival

| Módulo | Rol |
|--------|-----|
| `belief.py` | Hipótesis SP / moves conocidos → `iter_worlds` |
| `chance.py` | \(E[\mathrm{damage}]\), \(P(\mathrm{KO})\); `N_ROLLS=16`; `SOFT_KO_BONUS=1000` |
| `opp_policy.py` | Distribución aproximada de acciones rivales |
| `speed_order.py` | Quién actúa antes / factor de carrera |
| `utility.py` | Bonos de campo / posicionamiento / switches (si habilitados) |
| `residual.py` | Proyección de residuales en \(V\) |

Utilidad conceptual de un par propio \(u\):

\[
V(u) \;\approx\;
\mathbb{E}_{\omega,\,o}\big[
  U_{\mathrm{damage/KO}}(u,\omega,o)
  + U_{\mathrm{field}}
  - U_{\mathrm{threat/loss}}
\big]
\]

### 4.2 Switches voluntarios

| Knob | Default v1.0 |
|------|----------------|
| `enable_voluntary_switch` | **False** |
| `max_switch_hybrids` | **0** |

Tune autónomo puede activarlos vía `ACTIVE_OVERRIDES.json`
(`run_expectimax_tune`); el default del dataclass sigue OFF.

---

## 5. Cobertura de búsqueda (congelada)

`ACTIVE_SEARCH_COVERAGE = '75'` aplicada por `apply_search_coverage`.

| Preset | `topk_per_side` | `max_our_pairs` | `deadline_ms` | `phase1_budget_ms` | `phase2_end_ms` |
|--------|-----------------|-----------------|---------------|--------------------|-----------------|
| `50` | 5 | 25 | 1000 | 50 | 950 |
| **`75` (activo)** | **7** | **49** | **1600** | **80** | **1520** |
| `100` | 10 | 100 | 3000 | 100 | 2900 |

---

## 6. `ExpectimaxConfig` — defaults de contrato v1.0

Campos del dataclass (efectivos = defaults + `ACTIVE_OVERRIDES.json` + kwargs):

| Campo | Default |
|-------|---------|
| `deadline_ms` | 1600.0 |
| `phase1_budget_ms` | 80.0 |
| `phase2_end_ms` | 1520.0 |
| `k_profiles` | 2 |
| `max_worlds` | 4 |
| `topk_per_side` | 7 |
| `max_our_pairs` | 49 |
| `soft_ko_bonus` | 1000.0 (`SOFT_KO_BONUS`) |
| `enable_voluntary_switch` | False |
| `max_switch_hybrids` | 0 |
| `enable_niche_pairs` | False |
| `loss_weight` | 650.0 |
| `save_weight` | 0.0 |
| `chip_soft_cap` | 30.0 |
| `switch_min_stay_p_ko` | 0.50 |
| `overkill_waste_weight` | 140.0 |
| `ko_free_switch_bonus` | 55.0 |
| `entry_punish_weight` | 1.15 |
| `threat_weight` | 0.0 |
| `hp_rel_weight` | 0.15 |
| `syn_weight` | 0.0 |
| `condemned_chip_discount` | 0.80 |
| `condemned_lasting_boost` | 1.35 |
| `threat_relief_weight` | 1.0 |
| `entry_ability_weight` | 1.0 |
| `opp_policy_greedy_weight` | 1.0 |
| `opp_policy_control_weight` | 0.0 |
| `opp_policy_switch_weight` | 0.0 |
| `opp_policy_use_real_damage` | True |
| `opp_policy_split_targets` | False |
| `opp_policy_mirror_solo` | False |
| `v_solo_isomorphic` | True |
| `phase2_retarget_only` | False |
| `phase1_use_joint` | True |
| `use_topk_abstraction` | True |
| `phase2_gate_enabled` | True |
| `phase2_gate_mode` | `'indifference'` |
| `phase2_gate_margin` | 80.0 |
| `phase2_indifference_eps` | 40.0 |
| `phase2_max_delta_v` | 0.0 |
| `enable_phase2` | True |
| `phase2_surgical_only` | False |
| `switch_v_scale` | 0.65 |

### Gate phase2 (lectura)

| Modo | Semántica |
|------|-----------|
| `'beat'` | Exigir \(V_{\mathrm{bayes}} \ge V_{p1} + \mathrm{margin}\) |
| **`'indifference'` (activo)** | Aceptar si \(V_{\mathrm{bayes}} \ge V_{p1} - \varepsilon\) con \(\varepsilon=40\) |
| `'off'` | Argmax libre de phase2 |

Diseño: no divergir a peor bajo ruido de estimación; Joint sigue siendo ancla.

---

## 7. Integración experimental

| Uso | Contrato |
|-----|----------|
| Suite offline WR | **Obligatorio**; 200 BO1 |
| Curriculum PPO | Fase opcional / peso bajo en \(\Omega\) |
| BC | Replays internos preferidos (junto a Joint) |
| Robust E2E | `--player-a expectimax_bayesian` (validated) |
| Tune | `python -m showdown.inference_sim.run_expectimax_tune` |
| Snapshot config | `config_history` + logs E2E |

Checklist de reproducibilidad:

1. Cobertura (`75` salvo declaración).
2. Dump de `ExpectimaxConfig` efectiva (+ overrides).
3. \(N\), BO1/BO3, seeds/teams.
4. No usar \(V\) en \(R_t\).

---

## 8. Mapa de código

| Pieza | Ruta |
|-------|------|
| Bot IAP | `main.py` |
| Núcleo | `expectimax.py` (`ExpectimaxConfig`, `best_expectimax_bayesian_action`) |
| Belief / chance / opp | `belief.py`, `chance.py`, `opp_policy.py` |
| Speed / utility | `speed_order.py`, `utility.py`, `residual.py` |
| Preview | `team_preview.py` |
| OTS helpers | `ots.py` |
| Poke-env | `pokeenv_player.py` |
| Historial knobs | `CONFIG_HISTORY.md`, `CONFIG_HISTORY.jsonl`, `config_history.py` |
| Overlay | `ACTIVE_OVERRIDES.json` (opcional) |
| Pendientes tune | `PENDING.md` |

---

## 9. Fortalezas y límites (tesis)

| Fortaleza | Límite |
|-----------|--------|
| Modelo de incertidumbre y rival | Sigue siendo 1-ply |
| Ancla Joint + gate | Anytime: el wall-clock acota candidatos |
| Techo heurístico / BC | Knobs sensibles; no es equilibrio Nash |
| Preview con roles VGC | No sustituye MCTS/Nash (§6.c/§6.d) |

---

## 10. Non-goals

- Multi-ply / PUCT / Nash como política primaria.
- \(V(s)\) como potencial Ng en RewardSpec.
- Optimalidad de teoría de juegos.
- Sustituir el historial de tune (`CONFIG_HISTORY`) por silencio en reportes.

---

## Changelog

| Versión | Fecha | Notas |
|---------|-------|-------|
| **1.1** | 2026-10-11 | **Team Preview §3** afinado: fórmula \(S\), pesos, roles+diversidad, lead quality; constantes nombradas; tests; drop `riesnther`; **fix** `_nid` strip `-`/`_` para formes (`Indeedee-F`→`indeedeef`). Combate / `ExpectimaxConfig` sin cambio semántico. |
| **1.0** | 2026-10-10 | Contrato FROZEN: cobertura 75%, phase1=joint, phase2 Top-K×Top-K, gate indifference \(\varepsilon=40\), voluntary switch OFF, tabla `ExpectimaxConfig`. |

---

*Fin del contrato Expectimax-Bayesian v1.0.*
