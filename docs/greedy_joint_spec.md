# Greedy-Joint Agent Spec — Búsqueda 1-ply \(A_1\times A_2\) (VGC Gen 9)

| Campo | Valor |
|-------|--------|
| **Documento** | **Contrato exclusivo de implementación — Greedy-Joint** |
| **Versión** | **1.2** (alineado con `docs/rl_env_spec.md` **v1.14** §6.h.3) |
| **Estado** | **FROZEN — implementado en `showdown/battle_bots/greedy_joint/`** |
| **Algoritmo** | Greedy joint: \(\arg\max_{(a_1,a_2)} S(a_1,a_2)\) en horizonte \(T=1\) |
| **Rol** | Rival heurístico **principal** (curriculum avanzado, suite WR, ancla Expectimax / MCTS rollout) |
| **Code root** | `showdown/battle_bots/greedy_joint/` |
| **Id CLI / mosaic** | `greedy_joint` |
| **Player poke-env** | `GreedyJointPlayer` |
| **Validated** | `ValidatedGreedyJointPlayer` |
| **Deps** | Sin `torch`. Reutiliza calc/tipos de `greedy_solo/` |

> **Alcance.** Fuente de verdad para **Greedy-Joint**. Escalera:
> [`heuristic_agents_spec.md`](heuristic_agents_spec.md). Solo:
> [`greedy_solo_spec.md`](greedy_solo_spec.md). Expectimax (anclado aquí):
> [`expectimax_bayesian_spec.md`](expectimax_bayesian_spec.md).
>
> **Contraste con Solo.** Evalúa **pares** para sinergia de turno (redirect+ataque,
> doble foco, Fake Out+daño, HH, speed control).
>
> **Contraste con MCTS.** Horizonte \(T=1\) heurístico; **no** simula cadenas ni
> usa PUCT. El rollout MCTS usa `greedy_joint_snapshot` (hermano nativo en
> snapshot; ver `mcts_puct_spec.md`) — no el player poke-env completo.
>
> **Inmutabilidad.** Bump **1.1+** aquí + `rl_env_spec` §6.h juntos.

---

## 1. Por qué existe (lectura no técnica)

En VGC la jugada buena suele ser **de equipo**: uno protege o atrae golpes
mientras el otro ataca; ambos concentran fuego para un KO; Fake Out abre el
turno. Greedy-Joint mira todas las combinaciones razonables de “qué hace el
izquierdo × qué hace el derecho” **en este turno** y elige la de mayor puntaje.

**Analogía:** no elige la mejor jugada de cada pieza por separado; elige el
mejor *acorde* de dos notas para el compás actual.

---

## 2. Delimitación de decisión

| Ítem | Contrato |
|------|----------|
| Horizonte | \(T=1\) (1-ply) |
| Espacio | Producto de átomos legales por lado (moves; ver abajo) |
| Switches voluntarios en joint | **Prohibidos** en `best_joint_double_action` |
| Force switch | Fuera del producto; manejado en `main` / poke-env como Solo |
| Aprendizaje | **Ninguno** |
| Preview | **Igual que Solo** (tipado+BST; sin roles Expectimax) |

---

## 3. Team Preview

**Idéntico** a [`greedy_solo_spec.md`](greedy_solo_spec.md) **v1.1** §3
(módulo compartido `greedy_solo/team_preview.py`).  
Joint **no** tiene preview propio: tipado+BST sin roles; combate sí divergen.

---

## 4. Política de combate

### 4.1 Pipeline

`best_joint_double_action` en `joint.py`:

1. Construir **átomos** por lado (`JointAtom`: move, target, mega/tera, daños por foe, flags de utilidad).
2. Calc batch compartido (misma infraestructura que Solo).
3. Producto cartesiano de átomos válidos.
4. `_score_pair(left, right)` → score escalar; argmax.
5. Como máximo **una** mega por par (consistencia de flags).

### 4.2 Score de par (contrato)

Componentes (orden conceptual):

1. **Daño combinado** por rival, capado al % HP restante.
2. **`KO_BONUS = 1000`** si el roll máximo combinado garantiza KO (y no hay Substitute que lo invalide según reglas del scorer).
3. **Utilidad 1-ply** según kind del move (Protect, Fake Out, HH, speed, redirect, status, charge).
4. **Redirect:** bonus base; mucho mayor si el compañero hace daño alto o KO.
5. **Friendly-fire** y costos Power Herb / recharge: penalización.
6. **Focus split:** pequeño bonus si se reparte presión entre rivales.

### 4.3 Constantes de utilidad congeladas

| Constante | Valor |
|-----------|-------|
| `_PROTECT_BONUS` | 18.0 |
| `_FAKE_OUT_BONUS` | 25.0 |
| `_HELPING_HAND_BONUS` | 20.0 |
| `_SPEED_CONTROL_BONUS` | 12.0 |
| `_STATUS_BASE` | 4.0 |
| `_FOCUS_SPLIT_BONUS` | 8.0 |
| `_CHARGE_BASE` | 2.0 |
| `_REDIRECT_BASE` | 8.0 |
| `_REDIRECT_WITH_ATTACK` | 40.0 |
| `_REDIRECT_WITH_KO` | 70.0 |
| `_HIGH_DAMAGE_PCT` (umbral “alto”) | 45.0 |
| `KO_BONUS` (importado de Solo) | 1000.0 |

Clasificación de utility (`_utility_kind`): Protect-family, Follow Me / Rage Powder / Spotlight, Fake Out, Helping Hand, Tailwind / Trick Room / Icy Wind / Electroweb / Bulldoze / Glaciate, charge (turno que no dispara), status residual, else damage.

### 4.4 Mega / Tera (Reg M-C)

| Mecánica | Contrato |
|----------|----------|
| Mega | Por defecto ON si `canMega` (átomos / `apply_pending_mega`) |
| Tera | **Off** — Reg M-C no tiene Terastallization. `can_tera` forzado false vía `format_rules.allow_tera` |

### 4.5 Belief

Misma `apply_belief_stats` que Solo cuando el harness provee creencias.

---

## 5. Integración experimental

| Uso | Contrato |
|-----|----------|
| Curriculum RL | Rival tras Solo; checkpoint “best” a menudo vs Joint |
| Suite offline | **Obligatorio**; 200 BO1 |
| Mosaic / QD fitness | Oponente fijo típico |
| BC | Fuente de replays internos (con Expectimax) |
| Robust E2E | `--player-a greedy_joint` / `--player-b greedy_joint` |
| MCTS rollouts | `greedy_joint_snapshot` (spec MCTS) — hermano semántico |

---

## 6. Mapa de código

| Pieza | Ruta |
|-------|------|
| Bot IAP | `greedy_joint/main.py` |
| Núcleo joint | `greedy_joint/joint.py` |
| Calc / tipos / preview | `greedy_solo/{greedy,calc_async,team_preview,type_chart}.py` |
| Poke-env | `greedy_joint/pokeenv_player.py` |
| Validated | `inference_sim/validated_greedy_player.py` |

---

## 7. Fortalezas y límites (tesis)

| Fortaleza | Límite |
|-----------|--------|
| Captura sinergia VGC de un turno | No planifica turno 2+ |
| Rival principal interpretable | Costo \(\propto \|A_1\|\,\|A_2\|\) |
| Ancla estable para Expectimax/MCTS | Sin modelo explícito de \(\pi_{\mathrm{opp}}\) ni esperanza de rolls |

---

## 8. Non-goals

- Búsqueda multi-ply / MCTS / Nash.
- Phase2 bayesiana (Expectimax).
- Switches voluntarios dentro del producto joint.
- Usar \(S(a_1,a_2)\) como potencial Ng en \(R_t\).
- Terastallization (Reg M-C: inexistente; kill-switch `format_rules`).

---

## Changelog

| Versión | Fecha | Notas |
|---------|-------|-------|
| **1.2** | 2026-10-11 | **Reg M-C:** Tera **off**; Mega ON. Alineado `rl_env` v1.14. |
| **1.1** | 2026-10-11 | Preview: referencia a Solo **v1.1** §3 (sin código Joint propio). Combate sin cambio. |
| **1.0** | 2026-10-10 | Contrato FROZEN: 1-ply \(A_1\times A_2\), utilidades redirect/FO/HH/speed, `KO_BONUS=1000`, preview=Solo. |

---

*Fin del contrato Greedy-Joint v1.2.*
