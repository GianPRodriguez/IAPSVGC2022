# Greedy-Solo Agent Spec — Max-daño factored \(T=1\) (VGC Gen 9)

| Campo | Valor |
|-------|--------|
| **Documento** | **Contrato exclusivo de implementación — Greedy-Solo** |
| **Versión** | **1.2** (alineado con `docs/rl_env_spec.md` **v1.14** §6.h.2) |
| **Estado** | **FROZEN — implementado en `showdown/battle_bots/greedy_solo/`** |
| **Algoritmo** | Greedy factored: \(\arg\max_{a_i} s_i\) por activo + calc de daño |
| **Rol** | Primer escalón “inteligente” de la escalera; curriculum early RL |
| **Code root** | `showdown/battle_bots/greedy_solo/` |
| **Id CLI / mosaic** | `greedy_solo` |
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
> **Inmutabilidad.** Bump **1.2+** aquí + `rl_env_spec` §6.h en el mismo cambio.

---

## 1. Por qué existe (lectura no técnica)

Cada uno de tus dos Pokémon activos elige “el golpe que más duele **ahora**”,
como si el compañero no planificara con él. Ya usa tipado, clima, pantallas y
una calculadora de daño realista. Es barato de explicar y de ejecutar: el primer
rival que demuestra que el daño importa.

**Analogía:** dos piezas de ajedrez que cada una captura lo mejor a su alcance,
sin coordinar un mate en dos.

---

## 2. Delimitación de decisión

| Ítem | Contrato |
|------|----------|
| Formato | Gen 9 VGC dobles |
| Horizonte | \(T=1\) (greedy; sin simulación de turnos futuros) |
| Espacio | Factored: decisión por activo, no producto cartesiano |
| Aprendizaje | **Ninguno** |
| Switches voluntarios en núcleo | **No** (prioridad a moves); forced switch sí |
| Belief rival | Opcional vía `apply_belief_stats` cuando el harness alimenta SP |

---

## 3. Team Preview (contrato afinado)

Módulo: `showdown/battle_bots/greedy_solo/team_preview.py`.  
Compartido por **Greedy-Joint** ([`greedy_joint_spec.md`](greedy_joint_spec.md) §3).  
Wiring: `main.py` / `pokeenv_player.py` → `best_team_preview_order`.

### 3.1 Intuición

Elige el bring de 4 y el par de leads que maximizan cobertura tipada ofensiva +
resistencia defensiva frente a los tipos visibles del rival, con un empate débil
por BST. **No** modela roles de campo (eso es Expectimax).

### 3.2 Búsqueda

| Ítem | Contrato |
|------|----------|
| Combinaciones | \(\binom{6}{4}\) brings × \(\binom{4}{2}\) leads = **hasta 90** |
| Top-K | **No** — exhaustivo |
| Pool \(<4\) | Relleno defensivo; score 0 |
| API | `best_team_preview_order` / `score_combination` |
| Salida | `(lead1, lead2, bench1, bench2)` 1-based → `order_to_team_string` |
| `lead_preference` | Parámetro **reservado**; sin efecto en v1.1 |

### 3.3 Fórmula

\[
S = \alpha\cdot\mathrm{off} + \beta\cdot\mathrm{def} + \gamma\cdot\mathrm{BST}
\]

| Constante | Valor |
|-----------|-------|
| `PREVIEW_ALPHA` \(\alpha\) | **1.0** |
| `PREVIEW_BETA` \(\beta\) | **1.0** |
| `PREVIEW_GAMMA` \(\gamma\) | **0.01** |
| `SCORE_EPS` | \(10^{-9}\) |

- \(\mathrm{off}\): suma de `offensive_coverage_score` por mon del bring vs tipos foe  
  (SE \(\ge 2\) suma el mult.; si \(0<\mathrm{mult}<2\), aporta \(0.25\cdot\mathrm{mult}\)).  
- \(\mathrm{def}\): suma de `defensive_resistance_score` (immune/resist suma; SE resta).  
- \(\mathrm{BST}\): suma de BST base de los 4 (dex vía `getPkmnInfo` + `resolve_species_name`).

**Propiedad:** \(\mathrm{off}/\mathrm{def}/\mathrm{BST}\) **no dependen del orden lead/bench**
dentro del mismo conjunto de 4. Por tanto, entre leads de un mismo bring, el
desempate no usa tipado de lead: queda el **primer** orden encontrado en la
iteración `itertools` (tras empate de \(S\) y BST). Expectimax añade
\(Q_{\mathrm{lead}}\) precisamente para romper esa indiferencia.

### 3.4 Empates

1. Mayor \(S\) (margen `SCORE_EPS`).  
2. Si \(|S-S'|<\varepsilon\): mayor BST.  
3. Si BST igual: conservar incumbent (orden de exploración estable).

### 3.5 Contraste

| | Random | Greedy | Expectimax |
|--|--------|--------|------------|
| Tipado+BST | No | Sí (§3.3) | Sí (pesos distintos) |
| Roles | No | No | Sí |
| Lead quality | No | No (indiferente en el mismo 4) | Sí |

### 3.6 Non-goals (preview)

- Roles TW/TR/FO/redirect.  
- OTS / moves ocultos.  
- Usar `lead_preference` (reservado).  
- Compartir \(\alpha,\beta,\gamma\) con Expectimax.

---


## 4. Política de combate

### 4.1 Función central

`best_double_action(...)` en `greedy.py`:

1. Un **solo** batch de requests al calc para ambos activos.
2. Si `canMega` en un lado, el calc usa stats/ability de la forme mega pendiente (`apply_pending_mega`).
3. Por cada activo, puntuar cada (move, target) y tomar el argmax.
4. Micro-ajuste de split de foco entre rivales (no es joint completo).

### 4.2 Puntaje por hit (contrato)

\[
\mathrm{score} \approx
\mathrm{damage\_pct}
+ \mathbf{1}[\mathrm{KO\,garantizado}]\cdot \texttt{KO\_BONUS}
+ U_{\mathrm{efectos/accuracy/carga}}
- C_{\mathrm{PowerHerb/recharge}}
\]

| Knob | Valor congelado |
|------|-----------------|
| `KO_BONUS` | **1000.0** |

Reglas cualitativas congeladas:

| Regla | Contrato |
|-------|----------|
| Status (no dañinos) | No van al calc de daño; score de daño **0** (efectos solo vía `move_mechanics` si aplica) |
| Tera | **Off** en Reg M-C (la mecánica no existe). Código legado gated por `format_rules.allow_tera` → siempre false |
| Mega | **Por defecto ON** si `canMega` (no solo en KO); mecánica especial activa del formato |
| Fake Out vs Ghost | Gate de tipado / Scrappy / Mind's Eye (`type_chart`) |
| Priority fail | Respeta bloqueos de habilidad vía `move_mechanics` |

### 4.3 Force switch

`best_forced_switch_slot`: elige banco vivo por matchup tipado / supervivencia (no uniforme).

### 4.4 Belief (`apply_belief_stats`)

| Situación | Stats usadas |
|-----------|--------------|
| Belief estrecho / resuelto | Point-estimate |
| Bulk (HP/Def/SpD) abierto | Extremo **alto** (rival más duro) |
| Ofensa (Atk/SpA/Spe) abierta | Extremo **alto** (más amenaza) |
| Sin belief | Stats del estado Battle / defaults |

---

## 5. Integración experimental

| Uso | Contrato |
|-----|----------|
| Curriculum | Tras `random`, antes de `greedy_joint` |
| Suite offline | Opcional; Joint/Expectimax son obligatorios en §7 `rl_env` |
| Robust E2E | `--player-a greedy_solo` / `--player-b greedy_solo` |
| Validación | Sync Battle + soundness OTS en `ValidatedGreedyPlayer` |

---

## 6. Mapa de código

| Pieza | Ruta |
|-------|------|
| Bot IAP | `main.py` |
| Núcleo greedy | `greedy.py` (`best_double_action`, `CalcPoke`, `KO_BONUS`) |
| Calc batch | `calc_async.py` |
| Preview | `team_preview.py` |
| Tipado | `type_chart.py`, `species_names.py` |
| Poke-env | `pokeenv_player.py` (`GreedyDamagePlayer`) |
| Mecánicas compartidas | `showdown/battle_bots/move_mechanics/` |

---

## 7. Fortalezas y límites (tesis)

| Fortaleza | Límite |
|-----------|--------|
| Interpretable y barato | Ignora planes Protect+ataque / redirect conjuntos |
| Fuerte vs Random | Puede sobre-apilar daño en el mismo target |
| Calc realista | Sin horizonte \(T>1\); sin switches voluntarios tácticos |

---

## 8. Non-goals

- Producto \(A_1\times A_2\) (eso es Joint).
- Expectimax / mundos bayesianos.
- Optimizar RewardSpec Cand. 1/2.
- Sustituir MCTS (§6.c/§6.d).
- Terastallization (Reg M-C: inexistente).

---

## Changelog

| Versión | Fecha | Notas |
|---------|-------|-------|
| **1.2** | 2026-10-11 | **Reg M-C:** Terastallization **off** (non-goal); Mega sigue default ON. Kill-switch `format_rules.allow_tera`. |
| **1.1** | 2026-10-11 | **Team Preview §3** afinado: constantes, fórmula off/def/BST, indiferencia de lead en el mismo bring, contraste Random/Expectimax; tests `test_greedy_team_preview.py`. Combate sin cambio semántico. |
| **1.0** | 2026-10-10 | Contrato FROZEN: factored max-daño, `KO_BONUS=1000`, Tera-solo-si-KO (superseded 1.2), Mega default, preview \(\alpha=\beta=1,\gamma=0.01\). |

---

*Fin del contrato Greedy-Solo v1.2.*
