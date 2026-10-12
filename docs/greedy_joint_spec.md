# Greedy-Joint Agent Spec — Búsqueda 1-ply \(A_1\times A_2\) (VGC Gen 9)

| Campo | Valor |
|-------|--------|
| **Documento** | **Contrato exclusivo de implementación — Greedy-Joint** |
| **Versión** | **1.4** (alineado con `docs/rl_env_spec.md` **v1.15** §6.h.3) |
| **Estado** | **FROZEN — implementado en `showdown/battle_bots/greedy_joint/`** |
| **Algoritmo** | Greedy joint: \(\arg\max_{(a_1,a_2)} S(a_1,a_2)\) en horizonte \(T=1\) |
| **Rol** | Rival heurístico **principal** (curriculum avanzado, suite WR, ancla Expectimax / MCTS rollout) |
| **Code root** | `showdown/battle_bots/greedy_joint/` |
| **CLI** | id `greedy_joint`; `--player-a greedy_joint` |
| **Training** | **No** |
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
> **Inmutabilidad.** Bump **1.5+** aquí + `rl_env_spec` §6.h juntos.
>
> **Capítulo Team Preview.** Preview = **igual que Solo** (§4 abajo); el valor
> diferencial de Joint está en el **combate** coordinado.

---

## 1. Por qué existe (lectura no técnica)

En VGC la jugada buena suele ser **de equipo**: uno protege o atrae golpes
mientras el otro ataca; ambos concentran fuego para un KO; Fake Out abre el
turno. Greedy-Joint mira combinaciones “qué hace el izquierdo × qué hace el
derecho” **en este turno** y elige la de mayor puntaje.

En **Team Preview** no inventa otra lógica: reutiliza el preview tipado de Solo
(el capítulo distingue baselines Random / Greedy / Expectimax; Joint y Solo
comparten el mismo “Greedy preview”).

**Analogía (combate):** no la mejor nota de cada instrumento, sino el mejor
*acorde* del compás.
**Analogía (preview):** mismo “once tipado” que Solo; la coordinación llega al
pitar el árbitro.

| Término | Significado aquí |
|---------|------------------|
| Joint / producto \(A_1\times A_2\) | Evaluar pares de acciones, no cada lado solo |
| 1-ply | Solo el turno actual (sin simular el siguiente) |
| Redirect | Follow Me / Rage Powder: atraer golpes |
| \(\pi_c\) en RL preview | Joint suele ser el piloto de combate congelado |
| Preview compartido | Módulo `greedy_solo/team_preview.py` |

---

## 2. Glosario y delimitación

| Ítem | Contrato |
|------|----------|
| Horizonte | \(T=1\) (1-ply) |
| Espacio | Producto de átomos legales por lado (moves; ver abajo) |
| Switches voluntarios en joint | **Prohibidos** en `best_joint_double_action` |
| Force switch | Fuera del producto; manejado en `main` / poke-env como Solo |
| Aprendizaje | **Ninguno** |
| Preview | **Igual que Solo** (tipado+BST; sin roles Expectimax) |

---

## 3. Algoritmo paso a paso

1. Team Preview delega en el selector Greedy-Solo tipado+BST.
2. En combate, construir `JointAtom` legales por activo: move, target,
   Mega/Tera, daño y flags de utilidad.
3. Calcular daños en batch y formar \(A_1\times A_2\).
4. Rechazar pares incompatibles, incluida más de una Mega.
5. Puntuar cada par con daño capado, KO, utilidad, redirect y foco.
6. Elegir el argmax y serializar. No simula el turno siguiente.

**Lectura intuitiva:** Solo elige dos notas buenas; Joint escucha cada acorde.
**Lectura de ingeniería:** sigue siendo 1-ply, pero el costo crece como
\(\lvert A_1\rvert\lvert A_2\rvert\).

---

## 4. Team Preview

**Idéntico** a [`greedy_solo_spec.md`](greedy_solo_spec.md) **v1.4** §4
(módulo compartido `greedy_solo/team_preview.py`).  
Joint **no** tiene preview propio: tipado+BST sin roles; combate sí divergen.

---

## 5. Combate / turnos

### 5.1 Pipeline

`best_joint_double_action` en `joint.py`:

1. Construir **átomos** por lado (`JointAtom`: move, target, mega/tera, daños por foe, flags de utilidad).
2. Calc batch compartido (misma infraestructura que Solo).
3. Producto cartesiano de átomos válidos.
4. `_score_pair(left, right)` → score escalar; argmax.
5. Como máximo **una** mega por par (consistencia de flags).

### 5.2 Score de par (contrato)

Componentes (orden conceptual):

1. **Daño combinado** por rival, capado al % HP restante.
2. **`KO_BONUS = 1000`** si el roll máximo combinado garantiza KO (y no hay Substitute que lo invalide según reglas del scorer).
3. **Utilidad 1-ply** según kind del move (Protect, Fake Out, HH, speed, redirect, status, charge).
4. **Redirect:** bonus base; mucho mayor si el compañero hace daño alto o KO.
5. **Friendly-fire** y costos Power Herb / recharge: penalización.
6. **Focus split:** pequeño bonus si se reparte presión entre rivales.

### 5.3 Constantes de utilidad congeladas

| Parámetro | Valor | Propósito | Cómo influye |
|---|---:|---|---|
| `_PROTECT_BONUS` | 18.0 | Valorar defensa inmediata | Hace competir Protect con daño moderado |
| `_FAKE_OUT_BONUS` | 25.0 | Valorar denegación de acción | Favorece aperturas con flinch legal |
| `_HELPING_HAND_BONUS` | 20.0 | Reconocer apoyo ofensivo | Premia potenciar el ataque compañero |
| `_SPEED_CONTROL_BONUS` | 12.0 | Reconocer control de orden | Añade valor a TW/TR/Icy Wind y familia |
| `_STATUS_BASE` | 4.0 | Dar piso a status residual | Evita score cero sin equipararlo a daño alto |
| `_FOCUS_SPLIT_BONUS` | 8.0 | Diversificar presión | Favorece targets distintos ligeramente |
| `_CHARGE_BASE` | 2.0 | Representar preparación | Da valor pequeño a un turno de carga |
| `_REDIRECT_BASE` | 8.0 | Reconocer redirect aislado | Evita descartarlo si el aliado no capitaliza |
| `_REDIRECT_WITH_ATTACK` | 40.0 | Modelar sinergia protectora | Sube redirect cuando acompaña daño alto |
| `_REDIRECT_WITH_KO` | 70.0 | Modelar máxima sinergia | Sube más si protege un KO |
| `_HIGH_DAMAGE_PCT` | 45.0 | Definir «daño alto» | Activa el bonus redirect+ataque |
| `KO_BONUS` | 1000.0 | Priorizar KO garantizado | Domina utilidades ordinarias |

Clasificación de utility (`_utility_kind`): Protect-family, Follow Me / Rage Powder / Spotlight, Fake Out, Helping Hand, Tailwind / Trick Room / Icy Wind / Electroweb / Bulldoze / Glaciate, charge (turno que no dispara), status residual, else damage.

### 5.4 Mega / Tera (Reg M-C)

| Mecánica | Contrato |
|----------|----------|
| Mega | Por defecto ON si `canMega` (átomos / `apply_pending_mega`) |
| Tera | **Off** — Reg M-C no tiene Terastallization. `can_tera` forzado false vía `format_rules.allow_tera` |

### 5.5 Belief

Misma `apply_belief_stats` que Solo cuando el harness provee creencias.

---

## 6. Entrenamiento

**No entrena — política fija.** Sus constantes no se actualizan con reward,
gradientes o replay. En la tesis es el rival heurístico principal y un piloto
congelado: «fuente de BC» describe recolección de ejemplos, no entrenamiento de
Joint.

---

## 7. Pipeline de Conclusión

Combina **preview Greedy-Solo + combate Greedy-Joint**. Id canónico:
`greedy_joint`, `combat='greedy_joint'`, `preview='greedy_solo'`, grupo
`heuristic`. Esta asimetría es intencional porque Joint no implementa selector
propio. Véase [`conclusion_eval_protocol.md`](conclusion_eval_protocol.md).

---

## 8. CLI / código / tests

| Uso | Contrato |
|-----|----------|
| Curriculum RL | Rival tras Solo; checkpoint “best” a menudo vs Joint |
| Suite offline | **Obligatorio**; 200 BO1 |
| Mosaic / QD fitness | Oponente fijo típico |
| BC | Fuente de replays internos (con Expectimax) |
| Robust E2E | `--player-a greedy_joint` / `--player-b greedy_joint` |
| MCTS rollouts | `greedy_joint_snapshot` (spec MCTS) — hermano semántico |

---

| Pieza | Ruta / comando |
|-------|------|
| Bot IAP | `greedy_joint/main.py` |
| Núcleo joint | `greedy_joint/joint.py` |
| Calc / tipos / preview | `greedy_solo/{greedy,calc_async,team_preview,type_chart}.py` |
| Poke-env | `greedy_joint/pokeenv_player.py` |
| Validated | `inference_sim/validated_greedy_player.py` |
| Robust smoke | `python -m showdown.inference_sim.run_robust_e2e --teams 4 --player-a greedy_joint --player-b random` |
| Tests | `tests/test_greedy_joint.py`; `tests/test_greedy_team_preview.py` |

---

## 9. Limitaciones y lectura cruzada

| Fortaleza | Límite |
|-----------|--------|
| Captura sinergia VGC de un turno | No planifica turno 2+ |
| Rival principal interpretable | Costo \(\propto \|A_1\|\,\|A_2\|\) |
| Ancla estable para Expectimax/MCTS | Sin modelo explícito de \(\pi_{\mathrm{opp}}\) ni esperanza de rolls |

---

### 9.1 Non-goals

- Búsqueda multi-ply / MCTS / Nash.
- Phase2 bayesiana (Expectimax).
- Switches voluntarios dentro del producto joint.
- Usar \(S(a_1,a_2)\) como potencial Ng en \(R_t\).
- Terastallization (Reg M-C: inexistente; kill-switch `format_rules`).

---

### 9.2 Lectura cruzada y referencias

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction*
  (2nd ed.). MIT Press.
- [`greedy_solo_spec.md`](greedy_solo_spec.md) (preview + calc compartidos);
  [`expectimax_bayesian_spec.md`](expectimax_bayesian_spec.md);
  [`mcts_puct_spec.md`](mcts_puct_spec.md) (rollout `greedy_joint_snapshot`).

---

## Changelog

| Versión | Fecha | Notas |
|---------|-------|-------|
| **1.4** | 2026-10-11 | Estructura dual-audience; utilidades explicadas; entrenamiento y Conclusión. Semántica intacta. |
| **1.3** | 2026-10-11 | Redacción dual (glosario, refs, preview vs combate). Semántica intacta. |
| **1.2** | 2026-10-11 | **Reg M-C:** Tera **off**; Mega ON. Alineado `rl_env` v1.14. |
| **1.1** | 2026-10-11 | Preview: referencia a Solo §3 (sin código Joint propio). Combate sin cambio. |
| **1.0** | 2026-10-10 | Contrato FROZEN: 1-ply \(A_1\times A_2\), utilidades redirect/FO/HH/speed, `KO_BONUS=1000`, preview=Solo. |

---

*Fin del contrato Greedy-Joint v1.4.*
