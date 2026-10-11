# Greedy-Solo Agent Spec — Max-daño factored \(T=1\) (VGC Gen 9)

| Campo | Valor |
|-------|--------|
| **Documento** | **Contrato exclusivo de implementación — Greedy-Solo** |
| **Versión** | **1.0** (alineado con `docs/rl_env_spec.md` **v1.9** §6.h.2) |
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
> **Inmutabilidad.** Bump **1.1+** aquí + `rl_env_spec` §6.h en el mismo cambio.

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

## 3. Team Preview

Módulo: `team_preview.py`.

| Ítem | Contrato |
|------|----------|
| Entrada | 6 mons propios + foes visibles (especies/tipos) |
| Búsqueda | Combinaciones \(\binom{6}{4}\); para cada combo, permutaciones de lead (2) |
| Score | \(\alpha\cdot\mathrm{off} + \beta\cdot\mathrm{def} + \gamma\cdot\mathrm{BST}\) |
| Defaults congelados | \(\alpha=1\), \(\beta=1\), \(\gamma=0.01\) |
| Empates | Mayor BST; luego estabilidad de índices |
| Roles de campo | **No** (Tailwind / TR / FO los añade Expectimax) |
| Salida | Orden 1-based `(lead1, lead2, bench1, bench2)` → string tipo `'1234'` |

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
| Tera | Solo si **convierte** un hit en KO garantizado que sin Tera no lo era |
| Mega | **Por defecto ON** si `canMega` (no solo en KO) |
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

---

## Changelog

| Versión | Fecha | Notas |
|---------|-------|-------|
| **1.0** | 2026-10-10 | Contrato FROZEN: factored max-daño, `KO_BONUS=1000`, Tera-solo-si-KO, Mega default, preview \(\alpha=\beta=1,\gamma=0.01\). |

---

*Fin del contrato Greedy-Solo v1.0.*
