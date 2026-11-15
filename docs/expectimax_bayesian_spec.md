# Expectimax-Bayesian Agent Spec — 1-ply anytime Top-K×Top-K (VGC Gen 9)

| Campo | Valor |
|-------|--------|
| **Documento** | **Contrato exclusivo de implementación — Expectimax-Bayesian** |
| **Versión** | **1.9** (alineado con `docs/rl_env_spec.md` **v1.15** §6.h.4) |
| **Estado** | **FROZEN — implementado en `showdown/battle_bots/expectimax_bayesian/`** |
| **Algoritmo** | Expectimax 1-ply bayesiano anytime; filtro Greedy-Joint Top-K×Top-K; gate vs phase1 |
| **Rol** | Techo heurístico de la escalera; fuente BC; oponente de suite WR |
| **Code root** | `showdown/battle_bots/expectimax_bayesian/` |
| **CLI** | id `expectimax_bayesian`; `--player-a expectimax_bayesian` |
| **Training** | **No** |
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
> **Inmutabilidad.** Bump **1.5+** aquí + `rl_env_spec` §6.h juntos.
>
> **Capítulo Team Preview.** §4 es el techo heurístico de preview (roles +
> calidad de leads); combate phase2 es ortogonal pero comparte el nombre del bot.

---

## 1. Por qué existe (lectura no técnica)

Greedy-Joint elige un buen par “optimista” de daño y trucos de un turno.
Expectimax se pregunta: *¿sigue siendo bueno si el daño es aleatorio, si el
rival tiene uno u otro set, y si responde de cierta forma?* Para no quedarse sin
tiempo, pide a Joint una lista corta (Top-K) y solo ahí promedia bajo
incertidumbre.

En **Team Preview** va un paso más allá del tipado puro: premia roles de campo
visibles por especie (Tailwind, Trick Room, Fake Out, redirect) y la calidad de
los dos leads — sin leer moves OTS ocultos.

**Analogía (combate):** Joint propone planes del compás; Expectimax pregunta
“¿y si la moneda del daño y el rival no cooperan?”.
**Analogía (preview):** no solo “quién pega a quién”, sino “¿el once trae el
control de velocidad y la apertura de turno que el formato exige?”.

| Término | Significado aquí |
|---------|------------------|
| Expectimax | Promedia sobre azar (daño / mundos), no asume el peor caso puro |
| Bayesiano (aquí) | Varios mundos de set/SP con pesos de creencia |
| Top-K×Top-K | Solo re-evalúa los mejores candidatos de Joint |
| Gate | Solo cambia el plan Joint si la media no empeora de más |
| Roles preview | TW / TR / FO / redirect / speed-control por especie |
| Anytime | Respeta un deadline de pared; si no llega → Joint |

---

## 2. Glosario y delimitación

| Ítem | Contrato |
|------|----------|
| Horizonte | \(T=1\) anytime (deadline de pared) |
| Espacio phase2 | Subconjunto Top-K×Top-K del espacio joint |
| Aprendizaje | **Ninguno** (sí hay knobs de diseño / tune offline) |
| Fallback | Si timeout/error → acción Joint (`used_fallback`) |
| Reward RL | **Prohibido** usar \(V(u)\) live como \(\Phi(s)\) en \(R_t\) |

---

## 3. Algoritmo paso a paso

### Preview

1. Enumerar los 90 órdenes máximos (bring de cuatro × par de leads).
2. Calcular base de tipado+BST, roles presentes y calidad específica de leads.
3. Sumar \(S=B+R_{\mathrm{roles}}+w_{\mathrm{lead}}Q_{\mathrm{lead}}\).
4. Elegir mayor total y usar BST como desempate.

### Combate: búsqueda y belief

1. Phase1 ejecuta Greedy-Joint y ordena candidatos baratos.
2. Conservar Top-K por lado y hasta `max_our_pairs` pares.
3. `belief.py` genera mundos de set/SP con sus pesos; `opp_policy.py` aproxima
   respuestas rivales.
4. Para cada par propio, promediar utilidad sobre mundos y respuestas. Los 16
   rolls de daño se resumen en esperanza/probabilidad, no en 16 ramas.
5. Respetar el deadline anytime; si hay timeout/error, volver a Joint.
6. Aplicar el gate: phase2 solo sustituye el ancla si no empeora más que la
   tolerancia activa.

Así, «bayesiano» significa ponderar hipótesis, no aprender una red; «expectimax»
significa esperanza sobre azar, no minimización del peor caso.

---

## 4. Team Preview

Módulo: `showdown/battle_bots/expectimax_bayesian/team_preview.py`.  
Entradas IAP / poke-env: `main.py` / `pokeenv_player.py` → `best_expectimax_preview_with_meta`.

> **v1.7 (2026-11-15): el preview por defecto es tipado + BST de Greedy** (α=β=1, γ=0,01,
> `ROLE_WEIGHT=0`, `LEAD_QUALITY_WEIGHT=0`), con las mismas elecciones que Greedy-Solo/Joint.
> En el diseño factorial preview × combate, el preview de roles (§4.1–4.6) restaba
> **−3,8 pts** con combate joint (roles 1,6–2,2; calidad de lead 0,3–1,3; pesos ~0,9). Ningún
> componente mejoraba sobre tipado + BST. Con este preview, el combate de expectimax da
> **52,8 %** contra joint validado (antes 49,7 %). Las secciones §4.1–4.6 describen el preview
> **legacy**, que se conserva en `LEGACY_PREVIEW_KW` y como `--preview-a expectimax_legacy`.
> Leer los roles de los movimientos del team sheet tampoco ayuda (−1,5 / −2,4 sobre greedy).
> Desde v1.7, Greedy-Solo, Greedy-Joint y Expectimax eligen el mismo preview (verificado en
> 1200/1200 enfrentamientos).
> Diagnóstico completo en [`expectimax_diagnostico.md`](expectimax_diagnostico.md).

### 4.1 Intuición

Greedy solo mira tipado+BST. Expectimax añade: (1) **roles de campo**
reconocibles por especie en preview, (2) **calidad de leads** (los dos que salen
primero deben tipar y, si pueden, traer redirect / Fake Out / Tailwind),
(3) un sesgo defensivo ligeramente mayor (\(\beta>\alpha\)).

**No** usa OTS ni moves ocultos: solo nombre/especie visible en Team Preview.

### 4.2 Búsqueda (exhaustiva)

| Parámetro | Valor | Propósito | Cómo influye |
|---|---:|---|---|
| Combinaciones | hasta 90 | Cubrir bring × leads | Evalúa \(\binom64\binom42\) completo |
| Top-K preview | no | Evitar aproximación innecesaria | Puntúa todos los órdenes |
| Empate | `total`, luego BST | Dar orden estable | BST solo decide tras igual total |
| Salida | cuatro slots 1-based | Cumplir `/team` | Dos primeros son leads |
| Pool \(<4\) | relleno; score 0 | Tolerar input incompleto | Marca meta `pool < 4` |
| API | `best_expectimax_preview_with_meta` | Exponer decisión y explicación | El wrapper devuelve solo orden |
| Meta | `PreviewDecisionMeta` | Auditar el porqué | Conserva top líneas, razones y conteos |

### 4.3 Fórmula de score (congelada)

Sea \(B\) el score tipado+BST de Solo (`score_combination`):

\[
B = \alpha\cdot\mathrm{off} + \beta\cdot\mathrm{def} + \gamma\cdot\mathrm{BST}
\]

\[
S = B + R_{\mathrm{roles}} + w_{\mathrm{lead}}\cdot Q_{\mathrm{lead}}
\]

| Parámetro | Valor | Propósito | Cómo influye |
|---|---:|---|---|
| `PREVIEW_ALPHA` \(\alpha\) | 1.05 | Ponderar cobertura ofensiva | Sube ligeramente su aporte frente a Greedy |
| `PREVIEW_BETA` \(\beta\) | 1.25 | Priorizar resistencia | Da mayor peso defensivo que ofensivo |
| `PREVIEW_GAMMA` \(\gamma\) | 0.008 | Usar BST como señal débil | Reduce su dominio frente a tipado/roles |
| `LEAD_QUALITY_WEIGHT` | 0.45 | Separar leads de banca | Escala la calidad de apertura dentro del total |
| `ROLE_DIVERSITY_BONUS` | 12 | Premiar variedad funcional | Se suma al cubrir al menos dos buckets |
| `LEAD_DEF_SCALE` | 1.35 | Ponderar defensa de leads | Favorece aperturas resistentes |
| `LEAD_OFF_SCALE` | 1.15 | Ponderar ataque de leads | Favorece presión inicial |
| `LEAD_ROLE_BONUS` | 10 | Premiar rol inmediato | Suma por lead redirect/FO/TW |

### 4.4 Roles \(R_{\mathrm{roles}}\)

Presencia (≥1 especie del set). Matching vía `_nid` =
`normalize_name(name)` **sin** `-`/`_` (p.ej. `Indeedee-F` → `indeedeef`).
Los sets abajo están en esa forma compacta; **no** apilan por cantidad:

| Parámetro | Valor | Propósito | Cómo influye |
|---|---:|---|---|
| `ROLE_BONUS_REDIRECT` | 28 | Reconocer redirect visible por especie | Suma una vez si el bring contiene `amoonguss`, `indeedee`, `indeedeef`, `sableye` o `clefairy` |
| `ROLE_BONUS_TAILWIND` | 32 | Reconocer Tailwind | Suma una vez para `tornadus`, `whimsicott`, `kilowattrel`, `suicune`, `pelipper`, `emolga`, `talonflame`, `braviary`, `braviaryhisui` o `zapdos` |
| `ROLE_BONUS_TRICK_ROOM` | 26 | Reconocer Trick Room | Suma una vez para `indeedee`, `indeedeef`, `hatenna`, `hattrem`, `hatterene`, `slowking`, `slowkinggalar`, `porygon2`, `dusknoir`, `bronzong` u `oranguru` |
| `ROLE_BONUS_SPEED_CONTROL` | 18 | Reconocer control de velocidad | Suma una vez para `incineroar`, `rillaboom`, `urshifu`, `urshifurapidstrike`, `landorustherian`, `landorus`, `tinglu` o `greattusk` |
| `ROLE_BONUS_FAKE_OUT` | 14 | Reconocer Fake Out | Suma una vez para `incineroar`, `scrafty`, `kangaskhan`, `persian`, `meowth`, `sneasler`, `baxcalibur` o `farigiraf` |

**Diversidad de buckets** (`ROLE_DIVERSITY_BONUS = 12`) si al menos **2** de:

1. redirect  
2. speed_axis = Tailwind **∨** Trick Room  
3. disrupt = speed control **∨** Fake Out  

### 4.5 Calidad de lead \(Q_{\mathrm{lead}}\)

Sobre los **2** leads (no la banca):

\[
Q_{\mathrm{lead}} = \sum_{\ell\in\mathrm{leads}}\big(
1.35\,\mathrm{def}(\ell) + 1.15\,\mathrm{off}(\ell)
+ 10\cdot\mathbf{1}[\ell\in\mathrm{Redirect}\cup\mathrm{FO}\cup\mathrm{TW}]
\big)
\]

Constantes: `LEAD_DEF_SCALE=1.35`, `LEAD_OFF_SCALE=1.15`, `LEAD_ROLE_BONUS=10`.

### 4.6 Contraste con Greedy preview

| | Greedy-Solo/Joint | Expectimax |
|--|-------------------|------------|
| Pesos tipado | \(\alpha=\beta=1\), \(\gamma=0.01\) | §3.3 |
| Roles | No | Sí (§3.4) |
| Lead quality | No (solo orden por score combo) | Sí (§3.5) |
| Búsqueda | \(\binom{6}{4}\times\binom{4}{2}\) | Igual (exhaustiva) |

### 4.7 Non-goals (preview)

- Inferir moves/item/SP desde OTS en el score de preview.  
- Top-K / anytime en preview (eso es combate phase2).  
- Compartir pesos \(\alpha,\beta,\gamma\) con Greedy (son deliberadamente distintos).

Cambiar sets, pesos o fórmula ⇒ bump **1.2+** de este spec.

---


## 5. Combate / turnos

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

### 5.1 Mundos y rival

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

### 5.2 Switches voluntarios

| Knob | Default v1.0 |
|------|----------------|
| `enable_voluntary_switch` | **False** |
| `max_switch_hybrids` | **0** |

Tune autónomo puede activarlos vía `ACTIVE_OVERRIDES.json`
(`run_expectimax_tune`); el default del dataclass sigue OFF.

---

## 6. Cobertura de búsqueda (congelada)

`ACTIVE_SEARCH_COVERAGE = '75'` aplicada por `apply_search_coverage`.

| Parámetro | Valor | Propósito | Cómo influye |
|---|---:|---|---|
| Preset `50` | K=5; pares=25; deadline=1000 ms; phase1=50; phase2_end=950 | Cobertura barata | Evalúa menos pares y termina antes |
| Preset **`75` activo** | K=7; pares=49; deadline=1600 ms; phase1=80; phase2_end=1520 | Balance contractual | Amplía candidatos sin usar cobertura total |
| Preset `100` | K=10; pares=100; deadline=3000 ms; phase1=100; phase2_end=2900 | Cobertura alta | Evalúa más pares con mayor latencia |

---

## 7. `ExpectimaxConfig` — defaults de contrato

Campos del dataclass (efectivos = defaults + `ACTIVE_OVERRIDES.json` + kwargs):

| Parámetro | Valor | Propósito | Cómo influye |
|---|---:|---|---|
| `deadline_ms` | 1600.0 | Acotar latencia total | Al vencer, activa salida/fallback anytime |
| `phase1_budget_ms` | 80.0 | Reservar tiempo al ancla Joint | Limita generación y ranking inicial |
| `phase2_end_ms` | 1520.0 | Marcar fin de reevaluación | Deja margen antes del deadline total |
| `k_profiles` | 4 | Limitar perfiles de belief | C235 (antes 2); más perfiles = más diversidad y costo |
| `max_worlds` | 12 | Acotar mundos bayesianos | C235 (antes 4); trunca hipótesis evaluadas |
| `topk_per_side` | 7 | Filtrar átomos por activo | Define el ancho Top-K×Top-K |
| `max_our_pairs` | 49 | Acotar pares propios | Limita evaluaciones phase2 |
| `soft_ko_bonus` | 250.0 | Valorar probabilidad de KO | C235 (antes 1000); KO esperado pesa sin aplastar el daño |
| `enable_voluntary_switch` | False | Mantener búsqueda default en moves | Excluye híbridos con cambio voluntario |
| `max_switch_hybrids` | 0 | Limitar híbridos move/switch | Cero elimina esos candidatos |
| `enable_niche_pairs` | False | Controlar expansión especializada | No añade pares fuera del filtro normal |
| `loss_weight` | 650.0 | Penalizar pérdida propia | Reduce valor de líneas con riesgo de KO propio |
| `save_weight` | 0.0 | Ponderar rescate por switch | Cero desactiva esa recompensa |
| `chip_soft_cap` | 30.0 | Saturar valor de chip | Rendimientos decrecen sobre el umbral |
| `switch_min_stay_p_ko` | 0.50 | Umbral para considerar huida | Con switch OFF no expande candidatos default |
| `overkill_waste_weight` | 140.0 | Penalizar daño desperdiciado | Baja pares que exceden HP sin beneficio |
| `ko_free_switch_bonus` | 55.0 | Valorar posición tras KO | Premia la entrada libre resultante |
| `entry_punish_weight` | 1.15 | Ponderar castigo a entradas | Amplifica hazards/posición de switch |
| `threat_weight` | 0.0 | Ponderar amenaza absoluta | Cero desactiva ese término |
| `hp_rel_weight` | 0.15 | Valorar HP relativo | Añade una señal posicional débil |
| `syn_weight` | 0.0 | Ponderar sinergia adicional | Cero desactiva el término |
| `condemned_chip_discount` | 0.80 | Descontar chip en objetivo condenado | Reduce valor redundante antes de un KO probable |
| `condemned_lasting_boost` | 1.35 | Valorar efectos persistentes | Los amplifica si el target ya cae |
| `threat_relief_weight` | 1.0 | Valorar eliminar amenaza | Usa el alivio a escala completa |
| `entry_ability_weight` | 1.0 | Valorar habilidad al entrar | Usa su contribución a escala completa |
| `opp_policy_greedy_weight` | 1.0 | Mezclar respuesta rival greedy | Es el componente activo del modelo rival |
| `opp_policy_control_weight` | 0.0 | Mezclar respuestas de control | Cero las excluye |
| `opp_policy_switch_weight` | 0.0 | Mezclar switches rivales | Cero los excluye |
| `opp_policy_use_real_damage` | True | Basar rival en calc real | Ranking rival usa daño calculado |
| `opp_policy_split_targets` | False | Controlar diversidad de targets | No fuerza reparto de foco |
| `opp_policy_mirror_solo` | False | Controlar isomorfismo con Solo | Usa política rival propia |
| `v_solo_isomorphic` | True | Mantener escala compatible | Alinea evaluación individual cuando aplica |
| `v_iso_threat_weight` | 0.0 | Restar amenaza rival en V isomorfo | w=0.1/0.5/1.0 empeoran (−1.3/−3.4/−5.0 pts) |
| `v_iso_syn_weight` | 0.0 | Sumar sinergias del Joint | w=0.5/1.0 empeoran (−7/−8 pts) |
| `v_tempo_weight` | 1.0 | Crédito de tempo: P(KO)·P(actuar antes)·amenaza del rival | C235 (antes 0.0) |
| `phase2_retarget_only` | False | Restringir cambios a target | False permite variar el par completo |
| `phase1_use_joint` | True | Elegir ancla robusta | Los candidatos base vienen de Joint |
| `use_topk_abstraction` | True | Reducir espacio | Solo phase2 de candidatos filtrados |
| `phase2_gate_enabled` | True | Proteger contra regresión | Phase2 debe superar el criterio |
| `phase2_gate_mode` | `'beat'` | Exigir mejora sobre Joint | C235 (antes `'indifference'`) |
| `phase2_gate_margin` | 40.0 | Umbral del modo `beat` | C235 (antes 80); exige \(V_{bayes}\ge V_{p1}+40\) |
| `phase2_indifference_eps` | 40.0 | Tolerancia del modo `indifference` | Solo influye al seleccionar ese modo |
| `phase2_max_delta_v` | 0.0 | Acotar delta adicional | No concede margen extra |
| `enable_phase2` | True | Activar Expectimax | False degradaría a phase1 |
| `phase2_surgical_only` | False | Restringir reevaluación | False permite toda cobertura Top-K |
| `switch_v_scale` | 0.65 | Escalar valor de switches | Atenúa su utilidad si overlays los activan |

### Gate phase2 (lectura)

| Modo | Semántica |
|------|-----------|
| **`'beat'` (activo, C235)** | Exigir \(V_{\mathrm{bayes}} \ge V_{p1} + \mathrm{margin}\) con margin = 40 |
| `'indifference'` | Aceptar si \(V_{\mathrm{bayes}} \ge V_{p1} - \varepsilon\) con \(\varepsilon=40\) |
| `'off'` | Argmax libre de phase2 |

Diseño: no divergir a peor bajo ruido de estimación; Joint sigue siendo ancla.

---

## 8. Entrenamiento

**No entrena — política fija.** El agente no aprende por gradientes, reward ni
replay. `run_expectimax_tune` y `ACTIVE_OVERRIDES.json` son calibración offline
de knobs humanos, no entrenamiento online de una política. Para la tesis se debe
snapshotear la configuración efectiva para que el baseline sea reproducible.

---

## 9. Pipeline de Conclusión e integración experimental

Combina **preview Expectimax-Bayesian + combate Expectimax-Bayesian**. Id
canónico: `expectimax_bayesian`, con combat y preview homónimos, grupo
`heuristic`. Véase
[`conclusion_eval_protocol.md`](conclusion_eval_protocol.md).

| Uso | Contrato |
|-----|----------|
| Suite offline WR | **Obligatorio**; 200 BO1 |
| Curriculum PPO | Fase opcional / peso bajo en \(\Omega\) |
| BC | Replays internos preferidos (junto a Joint) |
| Robust E2E | `--player-a expectimax_bayesian` (validated); el rival B también va validated por defecto (2026-11-14). Toda comparación de expectimax se hace contra B validado; `--b-raw` solo para reproducir corridas antiguas |
| Tune | `python -m showdown.inference_sim.run_expectimax_tune` |
| Snapshot config | `config_history` + logs E2E |

Checklist de reproducibilidad:

1. Cobertura (`75` salvo declaración).
2. Dump de `ExpectimaxConfig` efectiva (+ overrides).
3. \(N\), BO1/BO3, seeds/teams.
4. No usar \(V\) en \(R_t\).
5. B validado (predeterminado; línea `b_validated=YES` del reporte).
6. Para atribuir la ventaja entre preview y combate, diseño factorial con `--preview-a/b` (`own`/`greedy`/`expectimax`): preview y combate se evalúan en conjunto y por separado. Resultados previos con B crudo favorecen a A ~2 pts: p. ej. expectimax base 50,3 % vs joint crudo pero 47,7 % vs joint validado.

---

## 10. CLI / código / tests

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
| Robust smoke | `python -m showdown.inference_sim.run_robust_e2e --teams 4 --player-a expectimax_bayesian --player-b random` |
| Tune offline | `python -m showdown.inference_sim.run_expectimax_tune` |
| Figuras tune | `logs/figures/expectimax_tune_latest.svg` + ledger `hp_tune_ledger.jsonl` |
| Tests | `tests/test_expectimax_bayesian.py`; `tests/test_expectimax_team_preview.py`; `tests/test_conclusion_eval.py` |

---

## 11. Limitaciones y lectura cruzada

| Fortaleza | Límite |
|-----------|--------|
| Modelo de incertidumbre y rival | Sigue siendo 1-ply |
| Ancla Joint + gate | Anytime: el wall-clock acota candidatos |
| Techo heurístico / BC | Knobs sensibles; no es equilibrio Nash |
| Preview con roles VGC | No sustituye MCTS/Nash (§6.c/§6.d) |

---

### 11.1 Non-goals

- Multi-ply / PUCT / Nash como política primaria.
- \(V(s)\) como potencial Ng en RewardSpec.
- Optimalidad de teoría de juegos.
- Sustituir el historial de tune (`CONFIG_HISTORY`) por silencio en reportes.
- Terastallization (Reg M-C: inexistente; `format_rules.allow_tera` → false).

---

### 11.2 Lectura cruzada y referencias

- Ballard, B. W. (1983). The *-minimax search procedure for trees containing
  chance nodes. *Artificial Intelligence*, 21(3), 327–350.
- Russell, S., & Norvig, P. (2020). *Artificial intelligence: A modern approach*
  (4th ed.). Pearson. (expectimax / incertidumbre).
- [`greedy_joint_spec.md`](greedy_joint_spec.md);
  [`heuristic_agents_spec.md`](heuristic_agents_spec.md);
  [`rl_env_spec.md`](rl_env_spec.md) §6.h / §8.
- Preview aprendido: [`team_preview_masked_ppo_spec.md`](team_preview_masked_ppo_spec.md),
  [`team_preview_nash_spec.md`](team_preview_nash_spec.md).

---

## Changelog

| Versión | Fecha | Notas |
|---------|-------|-------|
| **1.9.1** | 2026-11-15 | **Opcional, sin cambio por defecto:** `tie_sample_margin` (0 = determinista). Después de la puerta, muestrea al azar entre los pares con V en [V(jugado) − margen, V(jugado)]; no considera los pares que la puerta rechazó. `meta.tie_cands` y `meta.tie_changed`; traza `PS_DIVERSITY_LOG`. Medido: cuesta ~−7 pts aun con margen 2 (V tiene muchos empates que el orden de los pares desempata bien); **no usar**. Alternativa `phase1_tie_margin` (sorteo en la fase 1 con el puntaje de joint), pendiente de medir. Ver `docs/diversidad_baselines.md`. |
| **1.9** | 2026-11-15 | **Cambios voluntarios por defecto** (`enable_voluntary_switch=True`, `max_switch_hybrids=8`, `v_iso_switch_weight=0.25`). Fase 2c: +0,65 pts frente a v1.8 contra joint validado (2728 frente a 2695 / 5100; McNemar +0,82 pts, p = 0,001; t pareado 2,9), tras dos arreglos: `switch_v` no entraba en V isomorfo, y se ofrecían cambios al lado atrapado (miles de órdenes rechazadas, `switch_sides`). Reajustes de C235 (puerta, `soft_ko_bonus`, tempo), tope de 3 s y costo por cambiar no superan a esta versión. La opción C (`meta`) rinde igual (−0,20 pts, p = 0,72), pero reduce los mundos a uno solo; se **mantienen los mundos bayesianos** (modo `none`). Confirmación (C1/C2): contra joint v1.8 50,67 % (sin diferencia demostrada; contra joint con la regla anterior ~52 % en ambos bloques: la opción C de joint le resta ~1,3 pts en promedio, exploratorio), contra solo 53,02 % (confirmada, ventaja pequeña); v1.9 frente a v1.8 +0,04 pts (no se replica). Ver `docs/protocolo_evaluacion.md` §5.1. Techos (exploratorio, 9101–9117): oráculo de SP +0,30 pts, solo Velocidad +0,10, modelo del rival perfecto −0,57, con amenaza 0,1/0,3 +0,08; ninguno significativo. **Congelada:** el límite es el diseño 1-ply, no la información ni el modelo del rival (`expectimax_diagnostico.md`, "Techos"). |
| **1.8** | 2026-11-15 | **Inferencia de SP apagada por defecto (modo A)**, como en Greedy-Solo/Joint (`ValidatedExpectimaxBayesianPlayer.default_sp_info_mode='none'`). Con v1.7 contra joint validado: A 2695/5100 = B 2695/5100, con juegos idénticos en las 17 semillas y la mitad de tiempo; el oráculo completo da 2722/5100 = 53,4 % (+0,5, IC pareado aprox. −0,3..+1,3, no significativo). Los mundos siguen usando el prior leave-one-out. Traza de diagnóstico `PS_VTERMS_LOG` + `analyze_vterms` (Fase 1, sin efecto en las decisiones). `--sp-info-a inference` reactiva B. |
| **1.7** | 2026-11-15 | **Preview = tipado + BST de Greedy.** El diseño factorial (`--preview-a/b`) separó preview y combate: el combate de expectimax supera al de joint (+2,8; 52,8 % con preview greedy) y su preview de roles resta (−3,8; 46,2 % con combate joint). La atribución: roles 1,6–2,2 (también al 25 %), calidad de lead 0,3–1,3, pesos ~0,9 n.s. Preview legacy en `LEGACY_PREVIEW_KW`. Arreglo del desempate por punto flotante en el ranking (redondeo a 1e-9). Opción 2 probada y **descartada**: roles leídos de los movimientos del team sheet sobre greedy restan −1,5 (10 %) y −2,4 (25 %); el código (`role_moves`, `greedy_moveroles10/25`) queda solo para reproducir. Con v1.7 el preview de expectimax elige lo mismo que Greedy-Solo/Joint (1200/1200 enfrentamientos): B, C y D comparten preview. **Confirmación:** expectimax v1.7 completo contra joint validado da 2695/5100 = 52,8 % (IC 51,5–54,2), con victorias por semilla idénticas a las del factorial con preview greedy (antes 49,7 %). |
| **1.6** | 2026-11-14 | **Mundos sin filtración (leave-one-out):** `sample_belief_node` descuenta los sets del equipo rival del compendio MetaVGC (mismo pool que la evaluación) vía `TeamStatBeliefs.prior_exclude`, que fija el wrapper validated en cada juego; el fallback `preferred_sp_map` también. Antes el set real del rival estaba entre los 4 repartos más frecuentes de su especie en ~97 % de los casos (LOO ~92 %). `PS_PRIOR_LOO=0` lo desactiva. Resultados previos de expectimax pueden estar levemente inflados. |
| **1.5** | 2026-11-14 | **Versión estable C235:** `v_tempo_weight` 0→1.0, gate `indifference`→`beat` margin 80→40, `soft_ko_bonus` 1000→250, `k_profiles` 2→4, `max_worlds` 4→12. Servidor, pool 300 teams, 5100 BO3/config, IC emparejado por semilla: cribado (7001–7017) +0.7 n.s.; confirmación con semillas nuevas (8001–8017) vs greedy_joint sin inferencia de SP 52.5% vs 50.3% (+2.3 [+0.7, +3.9]); combinado 52.4% vs 50.9% (10 200 BO3). Vs joint validado (`PS_B_VALIDATED=1`) 48.3% vs 47.7% (+0.6 n.s.). Base vs greedy_solo 49.6%, vs random 99.7%. Ideas por separado ≈0 o peores; techo del diseño alcanzado. |
| **1.4** | 2026-10-11 | Estructura dual-audience; búsqueda/belief y propósito de knobs; entrenamiento y Conclusión. Semántica intacta. |
| **1.3** | 2026-10-11 | Redacción dual (glosario preview/combate, refs Ballard/AIMA). Semántica intacta. |
| **1.2** | 2026-10-11 | **Reg M-C:** Tera **off** (non-goal de formato); Mega sin cambio. |
| **1.1** | 2026-10-11 | **Team Preview §3** afinado: fórmula \(S\), pesos, roles+diversidad, lead quality; constantes nombradas; tests; drop `riesnther`; **fix** `_nid` strip `-`/`_` para formes (`Indeedee-F`→`indeedeef`). Combate / `ExpectimaxConfig` sin cambio semántico. |
| **1.0** | 2026-10-10 | Contrato FROZEN: cobertura 75%, phase1=joint, phase2 Top-K×Top-K, gate indifference \(\varepsilon=40\), voluntary switch OFF, tabla `ExpectimaxConfig`. |

---

*Fin del contrato Expectimax-Bayesian v1.5.*
