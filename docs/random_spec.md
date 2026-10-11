# Random Agent Spec — Piso estocástico legal (VGC Gen 9)

| Campo | Valor |
|-------|--------|
| **Documento** | **Contrato exclusivo de implementación — Random** |
| **Versión** | **1.0** (alineado con `docs/rl_env_spec.md` **v1.9** §6.h.1) |
| **Estado** | **FROZEN — implementado en `showdown/battle_bots/random/`** |
| **Algoritmo** | Muestreo estocástico sobre acciones **legales** del request Showdown |
| **Rol** | Piso de la escalera heurística; primer oponente de curriculum RL |
| **Code root** | `showdown/battle_bots/random/` |
| **Id CLI / mosaic** | `random` |
| **Deps** | Sin `torch`. Sin `@smogon/calc`. Sin belief / OTS en la decisión |

> **Alcance.** Fuente de verdad para implementar y citar el bot **Random**.
> La escalera completa y el glosario compartido viven en
> [`heuristic_agents_spec.md`](heuristic_agents_spec.md). Obs / Reward / `VGCEnv`
> viven en [`rl_env_spec.md`](rl_env_spec.md) — Random **no** optimiza \(R_t\).
>
> **Contraste.** A diferencia de Greedy-Solo / Joint / Expectimax, no usa tipado
> ni calculadora de daño: solo legales + RNG.
>
> **Inmutabilidad.** Cambiar un número o semántica congelada exige bump a **1.1+**
> de este doc **y** entrada en `rl_env_spec.md` §6.h en el mismo cambio.

---

## 1. Por qué existe (lectura no técnica)

En un experimento de aprendizaje o planificación hace falta un **piso**: un rival
que juegue “al azar pero sin romper las reglas”. Si el pipeline (servidor, teams,
sincronía) funciona, Random pierde casi siempre ante cualquier heurística sensata.
Si no, el fallo está en la infraestructura, no en el algoritmo aprendido.

**Analogía:** un dado que solo cae en caras permitidas por el árbitro (Showdown).

---

## 2. Delimitación de decisión

| Ítem | Contrato |
|------|----------|
| Formato | Gen 9 VGC dobles (Regulation M-C) |
| Episodio (eval) | 1 partida BO1 (o BO3 si el batch lo pide) |
| Unidad | Request Showdown: Team Preview **o** turno de combate |
| Acciones ilegales | **Prohibidas** (PP, `disabled`, trapped, forceSwitch) |
| Aprendizaje | **Ninguno** |
| Horizontes | \(T=0\) (sample); no mira el futuro |
| Calc / belief | **No** |

---

## 3. Team Preview

| Paso | Contrato |
|------|----------|
| Selección | `random.sample(range(1, 7), 4)` — 4 índices distintos uniformes |
| Orden | El orden de la muestra es el orden enviado (leads = primeros dos en la tupla de retorno del bot IAP) |
| Tipado / BST / roles | **Ignorados** |

Código: `BattleBot.action_team_preview` en `main.py`.

---

## 4. Turno de combate

### 4.1 Force switch

| Caso | Política |
|------|----------|
| Solo un slot forzado | Ese slot → `switch`; el otro → `pass` |
| Slot de banco | Si solo uno vivo → ese; si ambos vivos → uniforme en \(\{3,4\}\) |
| Doble force, un banco vivo | Switch al vivo; el otro `pass` |
| Doble force, ambos vivos | Activo izq → slot 3, activo der → slot 4 |

### 4.2 Turno normal (por activo)

Sea un activo no debilitado:

| Decisión | Contrato |
|----------|----------|
| Move vs switch | Si atrapado o sin reserva: **move**. Si no: `random.choices(['move','switch'], weights=(80,20))` |
| Move slot | Uniforme entre índices con PP \(>0\) y no `disabled` |
| Mega | Si `canMegaEvo` y el move **no** es carga/recarga sin `target`: uniforme \(\{0,1\}\). Como máximo **una** mega por turno (el activo derecho solo si el izquierdo no eligió mega) |
| Tera | **No** muestreado en v1.0 (flag ausente en este bot) |
| Target | Ver §4.3 |
| Sin moves legales | Cascada a `switch` en banco vivo |
| Activo debilitado | `pass` |

### 4.3 Targets (congelado)

| `target` del move | Política |
|-------------------|----------|
| Sin parámetros | `0` |
| Single-target | 90% uniforme en rivales \(\{1,2\}\); **10%** friendly-fire al aliado si vive (`-2` izq / `-1` der) |
| Adjacent foe | Uniforme \(\{1,2\}\) |
| Adjacent ally | Aliado fijo |
| Ally or self | 50% aliado / 50% self si el aliado vive; else self |
| Carga/recarga sin `target` | `0` |
| Desconocido | **Error** (`ValueError`) — no silenciar |

### 4.4 Conflictos de banco

Si el activo izquierdo ya eligió `switch`, el derecho **no** puede elegir el mismo slot: se fuerza `move` o el otro banco.

---

## 5. Números congelados

| Símbolo / knobs | Valor |
|-----------------|-------|
| \(P(\mathrm{move})\) | \(0.80\) |
| \(P(\mathrm{switch})\) | \(0.20\) |
| \(P(\mathrm{mega}\mid \mathrm{can})\) | \(0.50\) |
| \(P(\mathrm{friendly\,fire}\mid \mathrm{single})\) | \(0.10\) |

---

## 6. Integración experimental

| Uso | Contrato |
|-----|----------|
| Curriculum DDQN / PPO | Fase 1 / oponente en \(\Omega\) — ver specs RL |
| Suite offline WR | Incluido; **200 BO1** cuando el reporte RL lo exige |
| Robust E2E | `--player-b random` (default típico) |
| Validated player A | No es el player A “inteligente” por defecto; puede ser B |

**Nota sobre ActionSpec RL:** Random no muestra uniforme sobre el joint 576 del env; muestrea el lenguaje de órdenes Showdown por slot. Para el banco de pruebas basta como oponente estocástico débil.

---

## 7. Mapa de código

| Pieza | Ruta |
|-------|------|
| Bot IAP | `showdown/battle_bots/random/main.py` (`BattleBot`) |
| Helpers UI | `showdown/battle_bots/helpers.py` |

---

## 8. Non-goals

- Tipado, calc, belief, OTS, roles de preview.
- \(\varepsilon\)-greedy de Q-learning / exploración dirigida.
- Terastallize aleatorio (fuera de v1.0).
- Sustituir a heuristics o agentes aprendidos.

---

## Changelog

| Versión | Fecha | Notas |
|---------|-------|-------|
| **1.0** | 2026-10-10 | Contrato FROZEN: preview uniforme 4/6; 80/20 move/switch; mega 50%; friendly-fire 10%. |

---

*Fin del contrato Random v1.0.*
