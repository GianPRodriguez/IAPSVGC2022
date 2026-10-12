# Greedy-Solo Agent Spec — Max-daño factored \(T=1\) (VGC Gen 9)

| Campo | Valor |
|-------|--------|
| **Documento** | **Contrato exclusivo de implementación — Greedy-Solo** |
| **Versión** | **1.4** (alineado con `docs/rl_env_spec.md` **v1.15** §6.h.2) |
| **Estado** | **FROZEN — implementado en `showdown/battle_bots/greedy_solo/`** |
| **Algoritmo** | Greedy factored: \(\arg\max_{a_i} s_i\) por activo + calc de daño |
| **Rol** | Primer escalón “inteligente” de la escalera; curriculum early RL |
| **Code root** | `showdown/battle_bots/greedy_solo/` |
| **CLI** | id `greedy_solo`; `--player-a greedy_solo` |
| **Training** | **No** |
| **Player poke-env** | `GreedyDamagePlayer` (`pokeenv_player.py`) |
| **Validated** | `ValidatedGreedyPlayer` (`inference_sim/validated_greedy_player.py`) |
| **Deps** | Sin `torch`. Calc vía `calc_async` / puente `@smogon/calc` del proyecto |

> **Alcance.** Fuente de verdad para **Greedy-Solo**. Escalera/glosario:
> [`heuristic_agents_spec.md`](heuristic_agents_spec.md). Env RL:
> [`rl_env_spec.md`](rl_env_spec.md) — este bot **no** optimiza Cand. 1/2.
>
> **Contraste con Joint.** Solo elige por lado (factored). Joint busca el mejor
> par \((a_1,a_2)\). Ver [`greedy_joint_spec.md`](greedy_joint_spec.md).
>
> **Inmutabilidad.** Bump **1.5+** aquí + `rl_env_spec` §6.h en el mismo cambio.
>
> **Capítulo Team Preview.** §4 es la heurística tipada compartida con Greedy-Joint
> (mismo módulo de código).

---

## 1. Por qué existe (lectura no técnica)

Cada uno de tus dos Pokémon activos elige “el golpe que más duele **ahora**”,
como si el compañero no planificara con él. Ya usa tipado, clima, pantallas y
una calculadora de daño realista. Es barato de explicar y de ejecutar: el primer
rival que demuestra que el daño importa.

En **Team Preview** elige el bring de 4 y los leads mirando cobertura tipada y
resistencia frente a los tipos visibles del rival (sin roles de campo: eso es
Expectimax).

**Analogía (combate):** dos piezas de ajedrez que cada una captura lo mejor a su
alcance, sin coordinar un mate en dos.
**Analogía (preview):** armar el once mirando “quién pega a quién” en la tabla de
tipos, no el plan de Trick Room o Fake Out.

| Término | Significado aquí |
|---------|------------------|
| Greedy / \(T=1\) | Maximiza un puntaje del turno (o preview) actual |
| Factored | Cada activo elige solo; no busca el mejor *par* |
| Calc | Calculadora de daño (`@smogon/calc` / puente del repo) |
| Bring | Los 4 Pokémon elegidos de la plantilla de 6 |
| Leads | Los 2 que salen al campo primero |
| Mega | ON por defecto si está disponible (Reg M-C) |

---

## 2. Glosario y delimitación

| Ítem | Contrato |
|------|----------|
| Formato | Gen 9 VGC dobles |
| Horizonte | \(T=1\) (greedy; sin simulación de turnos futuros) |
| Espacio | Factored: decisión por activo, no producto cartesiano |
| Aprendizaje | **Ninguno** |
| Switches voluntarios en núcleo | **No** (prioridad a moves); forced switch sí |
| Belief rival | Opcional vía `apply_belief_stats` cuando el harness alimenta SP |

---

## 3. Algoritmo paso a paso

1. En preview, enumerar cada bring de cuatro y cada par de leads.
2. Puntuar cobertura ofensiva, resistencia defensiva y BST; conservar el mejor.
3. En combate, construir los moves y targets legales de cada activo.
4. Consultar el calc en un batch, aplicando la forma Mega pendiente cuando procede.
5. Puntuar daño, KO y mecánicas; elegir el máximo **por activo**.
6. Aplicar el micro-ajuste de foco y serializar. No busca el producto completo.

La idea de aprendizaje es «greedy»: convertir cada opción en un número y tomar
el máximo actual. La garantía de ingeniería es más estrecha: \(T=1\), política
factored y sin switches voluntarios en el núcleo.

---

## 4. Team Preview

Módulo: `showdown/battle_bots/greedy_solo/team_preview.py`.  
Compartido por **Greedy-Joint** ([`greedy_joint_spec.md`](greedy_joint_spec.md) §3).  
Wiring: `main.py` / `pokeenv_player.py` → `best_team_preview_order`.

### 4.1 Intuición

Elige el bring de 4 y el par de leads que maximizan cobertura tipada ofensiva +
resistencia defensiva frente a los tipos visibles del rival, con un empate débil
por BST. **No** modela roles de campo (eso es Expectimax).

### 4.2 Búsqueda

| Parámetro | Valor | Propósito | Cómo influye |
|---|---:|---|---|
| Combinaciones | hasta 90 | Cubrir bring × leads | Evalúa \(\binom64\binom42\) sin muestreo |
| Top-K | no | Mantener exactitud del espacio pequeño | No descarta órdenes antes de puntuar |
| Pool \(<4\) | relleno defensivo; score 0 | Evitar crash en input incompleto | Devuelve una salida segura sin afirmar calidad |
| API | `best_team_preview_order` / `score_combination` | Separar decisión y scoring | Facilita tests unitarios |
| Salida | `(lead1, lead2, bench1, bench2)` 1-based | Cumplir `/team` | Las dos primeras posiciones son leads |
| `lead_preference` | reservado; sin efecto | Compatibilidad futura | No altera ninguna decisión v1.4 |

### 4.3 Fórmula

\[
S = \alpha\cdot\mathrm{off} + \beta\cdot\mathrm{def} + \gamma\cdot\mathrm{BST}
\]

| Parámetro | Valor | Propósito | Cómo influye |
|---|---:|---|---|
| `PREVIEW_ALPHA` \(\alpha\) | 1.0 | Ponderar cobertura ofensiva | Cada punto ofensivo entra sin escala |
| `PREVIEW_BETA` \(\beta\) | 1.0 | Ponderar resistencia | Equilibra ataque y defensa |
| `PREVIEW_GAMMA` \(\gamma\) | 0.01 | Desempate suave por BST | Evita que BST domine al tipado |
| `SCORE_EPS` | \(10^{-9}\) | Tolerancia numérica | Scores dentro del margen se tratan como empate |

- \(\mathrm{off}\): suma de `offensive_coverage_score` por mon del bring vs tipos foe  
  (SE \(\ge 2\) suma el mult.; si \(0<\mathrm{mult}<2\), aporta \(0.25\cdot\mathrm{mult}\)).  
- \(\mathrm{def}\): suma de `defensive_resistance_score` (immune/resist suma; SE resta).  
- \(\mathrm{BST}\): suma de BST base de los 4 (dex vía `getPkmnInfo` + `resolve_species_name`).

**Propiedad:** \(\mathrm{off}/\mathrm{def}/\mathrm{BST}\) **no dependen del orden lead/bench**
dentro del mismo conjunto de 4. Por tanto, entre leads de un mismo bring, el
desempate no usa tipado de lead: queda el **primer** orden encontrado en la
iteración `itertools` (tras empate de \(S\) y BST). Expectimax añade
\(Q_{\mathrm{lead}}\) precisamente para romper esa indiferencia.

### 4.4 Empates

1. Mayor \(S\) (margen `SCORE_EPS`).  
2. Si \(|S-S'|<\varepsilon\): mayor BST.  
3. Si BST igual: conservar incumbent (orden de exploración estable).

### 4.5 Contraste

| | Random | Greedy | Expectimax |
|--|--------|--------|------------|
| Tipado+BST | No | Sí (§3.3) | Sí (pesos distintos) |
| Roles | No | No | Sí |
| Lead quality | No | No (indiferente en el mismo 4) | Sí |

### 4.6 Non-goals (preview)

- Roles TW/TR/FO/redirect.  
- OTS / moves ocultos.  
- Usar `lead_preference` (reservado).  
- Compartir \(\alpha,\beta,\gamma\) con Expectimax.

---


## 5. Combate / turnos

### 5.1 Función central

`best_double_action(...)` en `greedy.py`:

1. Un **solo** batch de requests al calc para ambos activos.
2. Si `canMega` en un lado, el calc usa stats/ability de la forme mega pendiente (`apply_pending_mega`).
3. Por cada activo, puntuar cada (move, target) y tomar el argmax.
4. Micro-ajuste de split de foco entre rivales (no es joint completo).

### 5.2 Puntaje por hit (contrato)

\[
\mathrm{score} \approx
\mathrm{damage\_pct}
+ \mathbf{1}[\mathrm{KO\,garantizado}]\cdot \texttt{KO\_BONUS}
+ U_{\mathrm{efectos/accuracy/carga}}
- C_{\mathrm{PowerHerb/recharge}}
\]

| Parámetro | Valor | Propósito | Cómo influye |
|---|---:|---|---|
| `KO_BONUS` | 1000.0 | Priorizar un KO garantizado | Domina diferencias ordinarias de daño porcentual |
| Horizonte | \(T=1\) | Mantener decisión barata | No atribuye valor al turno siguiente |
| Mega | ON si `canMega` | Usar la mecánica Reg M-C | El calc evalúa stats/ability de la forma pendiente |
| Tera | OFF | Respetar Reg M-C | `format_rules.allow_tera` impide emitirla |
| Switch voluntario | OFF | Conservar política max-daño | Solo `forceSwitch` invoca selección de banco |

Reglas cualitativas congeladas:

| Regla | Contrato |
|-------|----------|
| Status (no dañinos) | No van al calc de daño; score de daño **0** (efectos solo vía `move_mechanics` si aplica) |
| Tera | **Off** en Reg M-C (la mecánica no existe). Código legado gated por `format_rules.allow_tera` → siempre false |
| Mega | **Por defecto ON** si `canMega` (no solo en KO); mecánica especial activa del formato |
| Fake Out vs Ghost | Gate de tipado / Scrappy / Mind's Eye (`type_chart`) |
| Priority fail | Respeta bloqueos de habilidad vía `move_mechanics` |

### 5.3 Force switch

`best_forced_switch_slot`: elige banco vivo por matchup tipado / supervivencia (no uniforme).

### 5.4 Belief (`apply_belief_stats`)

| Situación | Stats usadas |
|-----------|--------------|
| Belief estrecho / resuelto | Point-estimate |
| Bulk (HP/Def/SpD) abierto | Extremo **alto** (rival más duro) |
| Ofensa (Atk/SpA/Spe) abierta | Extremo **alto** (más amenaza) |
| Sin belief | Stats del estado Battle / defaults |

---

## 6. Entrenamiento

**No entrena — política fija.** No ajusta parámetros con partidas, reward o
gradientes. El belief puede cambiar la entrada estimada del rival, pero no los
pesos de la política. En la tesis funciona como baseline interpretable y
oponente fijo de curriculum.

---

## 7. Pipeline de Conclusión

Combina **preview Greedy-Solo + combate Greedy-Solo**. Id canónico:
`greedy_solo`, `combat='greedy_solo'`, `preview='greedy_solo'`, grupo
`heuristic`. Protocolo: [`conclusion_eval_protocol.md`](conclusion_eval_protocol.md).

---

## 8. CLI / código / tests

| Uso | Contrato |
|-----|----------|
| Curriculum | Tras `random`, antes de `greedy_joint` |
| Suite offline | Opcional; Joint/Expectimax son obligatorios en §7 `rl_env` |
| Robust E2E | `--player-a greedy_solo` / `--player-b greedy_solo` |
| Validación | Sync Battle + soundness OTS en `ValidatedGreedyPlayer` |

---

| Pieza | Ruta / comando |
|-------|------|
| Bot IAP | `main.py` |
| Núcleo greedy | `greedy.py` (`best_double_action`, `CalcPoke`, `KO_BONUS`) |
| Calc batch | `calc_async.py` |
| Preview | `team_preview.py` |
| Tipado | `type_chart.py`, `species_names.py` |
| Poke-env | `pokeenv_player.py` (`GreedyDamagePlayer`) |
| Mecánicas compartidas | `showdown/battle_bots/move_mechanics/` |
| Robust smoke | `python -m showdown.inference_sim.run_robust_e2e --teams 4 --player-a greedy_solo --player-b random` |
| Tests | `tests/test_greedy_team_preview.py`; tests de heurística/calc del repo |

---

## 9. Limitaciones y lectura cruzada

| Fortaleza | Límite |
|-----------|--------|
| Interpretable y barato | Ignora planes Protect+ataque / redirect conjuntos |
| Fuerte vs Random | Puede sobre-apilar daño en el mismo target |
| Calc realista | Sin horizonte \(T>1\); sin switches voluntarios tácticos |

---

### 9.1 Non-goals

- Producto \(A_1\times A_2\) (eso es Joint).
- Expectimax / mundos bayesianos.
- Optimizar RewardSpec Cand. 1/2.
- Sustituir MCTS (§6.c/§6.d).
- Terastallization (Reg M-C: inexistente).

---

### 9.2 Lectura cruzada y referencias

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction*
  (2nd ed.). MIT Press. (políticas greedy / \(T=1\)).
- Documentación de tipado y daño VGC / Smogon calc (herramienta de dominio).
- [`heuristic_agents_spec.md`](heuristic_agents_spec.md);
  [`greedy_joint_spec.md`](greedy_joint_spec.md);
  [`expectimax_bayesian_spec.md`](expectimax_bayesian_spec.md) §3 (preview con roles).

---

## Changelog

| Versión | Fecha | Notas |
|---------|-------|-------|
| **1.4** | 2026-10-11 | Estructura dual-audience, tablas de propósito/influencia, entrenamiento y Conclusión. Semántica intacta. |
| **1.3** | 2026-10-11 | Redacción dual (glosario preview/combate, refs). Semántica intacta. |
| **1.2** | 2026-10-11 | **Reg M-C:** Terastallization **off** (non-goal); Mega sigue default ON. Kill-switch `format_rules.allow_tera`. |
| **1.1** | 2026-10-11 | **Team Preview §3** afinado: constantes, fórmula off/def/BST, indiferencia de lead en el mismo bring, contraste Random/Expectimax; tests `test_greedy_team_preview.py`. Combate sin cambio semántico. |
| **1.0** | 2026-10-10 | Contrato FROZEN: factored max-daño, `KO_BONUS=1000`, Tera-solo-si-KO (superseded 1.2), Mega default, preview \(\alpha=\beta=1,\gamma=0.01\). |

---

*Fin del contrato Greedy-Solo v1.4.*
