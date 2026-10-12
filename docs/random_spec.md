# Random Agent Spec — Piso estocástico legal (VGC Gen 9)

| Campo | Valor |
|-------|--------|
| **Documento** | **Contrato exclusivo de implementación — Random** |
| **Versión** | **1.4** (alineado con `docs/rl_env_spec.md` **v1.15** §6.h.1) |
| **Estado** | **FROZEN — implementado en `showdown/battle_bots/random/`** |
| **Algoritmo** | Muestreo estocástico sobre acciones **legales** del request Showdown |
| **Rol** | Piso de la escalera heurística; primer oponente de curriculum RL |
| **Code root** | `showdown/battle_bots/random/` |
| **CLI** | id `random`; `--player-b random` en robust E2E |
| **Training** | **No** |
| **Deps** | Sin `torch`. Sin `@smogon/calc`. Sin belief / OTS en la decisión |

> **Alcance.** Fuente de verdad para implementar y citar el bot **Random**.
> La escalera completa y el glosario compartido viven en
> [`heuristic_agents_spec.md`](heuristic_agents_spec.md). Obs / Reward / `VGCEnv`
> viven en [`rl_env_spec.md`](rl_env_spec.md) — Random **no** optimiza \(R_t\).
>
> **Contraste.** A diferencia de Greedy-Solo / Joint / Expectimax, no usa tipado
> ni calculadora de daño: solo legales + RNG.
>
> **Inmutabilidad.** Cambiar un número o semántica congelada exige bump a **1.5+**
> de este doc **y** entrada en `rl_env_spec.md` §6.h en el mismo cambio.
>
> **Capítulo Team Preview.** Este bot también define el **piso** de la escalera de
> preview (§4): elige 4 de 6 al azar legal, sin tipado ni roles.

---

## 1. Por qué existe (lectura no técnica)

En un experimento de aprendizaje o planificación hace falta un **piso**: un rival
que juegue “al azar pero sin romper las reglas”. Si el pipeline (servidor, teams,
sincronización) funciona, Random pierde casi siempre ante cualquier heurística
sensata. Si no, el fallo está en la infraestructura, no en el algoritmo aprendido.

En **Team Preview** el mismo espíritu: no intenta “sacar bien”; solo demuestra que
el harness de `/team` y el BO1 funcionan antes de medir agentes más listos.

**Analogía:** un dado que solo cae en caras permitidas por el árbitro (Showdown).

| Término | Significado aquí |
|---------|------------------|
| Acción legal | Lo que el request de Showdown permite (PP, disabled, trapped…) |
| Estocástico | Elige al azar entre opciones legales |
| Piso / floor | Rival más débil de la escalera; control negativo |
| Team Preview | Elegir 4 de 6 y el orden de salida |
| Mega | Mecánica especial activa en Reg M-C (Tera **no** existe) |

---

## 2. Glosario y delimitación de decisión

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

## 3. Algoritmo paso a paso

1. Leer el request y distinguir preview, reemplazo forzado o turno normal.
2. Construir únicamente opciones legales según PP, `disabled`, `trapped` y reservas.
3. Muestrear la selección sin estimar daño ni valor futuro.
4. Resolver recursos compartidos: una Mega y ningún doble switch al mismo slot.
5. Serializar las órdenes para Showdown.

Es una política \(T=0\): para el aprendiz, no planifica; para ingeniería, prueba
el lenguaje de acciones y sus bordes.

---

## 4. Team Preview

Módulo: `showdown/battle_bots/random/team_preview.py`.  
Wiring IAP: `main.py` → `sample_team_preview_order()`.

### 4.1 Intuición

Elige 4 de 6 **sin mirar** tipado, BST ni roles. Es el piso estocástico de la
escalera de preview (antes de Greedy tipado y Expectimax con roles).

### 4.2 Contrato

| Parámetro | Valor | Propósito | Cómo influye |
|---|---:|---|---|
| API | `sample_team_preview_order(rng=None)` | Aislar el selector | Permite inyectar RNG en tests |
| `PREVIEW_SLOT_LO/HI` | 1 / 6 | Delimitar el roster | Impide índices fuera del equipo |
| `PREVIEW_PICK_COUNT` | 4 distintos | Cumplir el bring VGC | Produce dos leads y dos reservas |
| Distribución | Uniforme sobre \(P(6,4)=360\) | Evitar preferencias ocultas | Cada orden legal pesa igual |
| Leads | posiciones 0 y 1 | Fijar aperturas | Esos slots salen al campo |
| Banca | posiciones 2 y 3 | Fijar reservas | Esos slots quedan para cambios |
| Tipado/BST/roles/OTS | ignorados | Mantener el control puro | El matchup no sesga la muestra |
| `rng` | opcional | Reproducibilidad | Una seed fija repite el orden |
| Validación | `is_valid_preview_order` | Detectar salida mal formada | Rechaza rango, duplicados o cardinalidad inválidos |

### 4.3 Non-goals (preview)

- Sesgo hacia leads “buenos” o cobertura tipada.  
- Compartir lógica con Greedy / Expectimax.  
- Preview fijo `/team 1234` (eso es el env RL de combate, no este bot).

---


## 5. Combate / turnos

### 5.1 Force switch

| Parámetro | Valor | Propósito | Cómo influye |
|---|---|---|---|
| Un slot forzado | `switch`; otro `pass` | Respetar request parcial | Solo reemplaza el slot pedido |
| Una reserva viva | esa reserva | Evitar debilitados | Elimina el azar sin alternativa |
| Dos reservas vivas | uniforme en 3/4 | Mantener neutralidad | Cada reserva pesa 0.5 |
| Doble force, una reserva | switch + `pass` | Resolver escasez | Solo un activo entra |
| Doble force, dos reservas | izquierda→3, derecha→4 | Evitar colisión | Produce asignación legal |

### 5.2 Turno normal (por activo)

Sea un activo no debilitado:

| Parámetro | Valor | Propósito | Cómo influye |
|---|---:|---|---|
| \(P(move)\) | 0.80 | Hacer progresar la partida | Favorece moves si puede cambiar |
| \(P(switch)\) | 0.20 | Ejercitar cambios | Introduce switches legales |
| Move slot | uniforme entre PP \(>0\), no `disabled` | No sesgar por daño | Todos los moves legales pesan igual |
| \(P(Mega\mid canMega)\) | 0.50 | Ejercitar Mega | La activa en mitad de órdenes elegibles |
| Mega compartida | máximo una | Cumplir la mecánica | Derecha depende de izquierda |
| Tera | **OFF** | Cumplir Reg M-C | Nunca emite `terastallize` |
| Sin moves legales | switch vivo | Mantener legalidad | Cae a reserva |
| Activo debilitado | `pass` | No ordenar a un slot inactivo | Evita rechazo |

### 5.3 Targets (congelado)

| Parámetro | Valor | Propósito | Cómo influye |
|---|---:|---|---|
| Sin parámetros | target `0` | Emitir sintaxis válida | No adjunta target |
| Single-target rival | 0.90; uniforme en 1/2 | Orientar normalmente el ataque | Reparte presión entre foes |
| Friendly fire single-target | 0.10 | Conservar ruido histórico | Apunta al aliado vivo (`-2`/`-1`) |
| Adjacent foe | uniforme en 1/2 | Respetar la clase | Elige un rival |
| Adjacent ally | aliado fijo | Respetar la clase | Nunca elige foe |
| Ally or self | 0.50 / 0.50 | No privilegiar receptor | Sin aliado vivo, usa self |
| Carga/recarga sin `target` | `0` | Tratar orden especial | No añade target |
| Desconocido | `ValueError` | Detectar deriva | Falla de forma visible |

### 5.4 Conflictos de banco

Si el activo izquierdo ya eligió `switch`, el derecho **no** puede elegir el mismo slot: se fuerza `move` o el otro banco.

---

## 6. Entrenamiento

**No entrena — política fija.** No tiene pesos, gradientes, replay, reward ni
checkpoint. La seed controla azar, no aprendizaje. En la tesis es una referencia
estable cuyo comportamiento no mejora durante la evaluación.

---

## 7. Parámetros congelados

| Parámetro | Valor | Propósito | Cómo influye |
|---|---:|---|---|
| \(P(\mathrm{move})\) | 0.80 | Sesgo operativo principal | Más ataques que cambios |
| \(P(\mathrm{switch})\) | 0.20 | Explorar rama de cambio | Menos cambios que ataques |
| \(P(\mathrm{mega}\mid\mathrm{can})\) | 0.50 | Muestrear la mecánica | Mega no es sistemática |
| \(P(\mathrm{friendly\,fire}\mid\mathrm{single})\) | 0.10 | Conservar ruido histórico | Uno de cada diez single puede ir al aliado vivo |

---

## 8. Pipeline de Conclusión e integración experimental

Combina **preview Random + combate Random**. Su id en `conclusion_eval` es
`random`, con `combat='random'`, `preview='random'`, grupo `heuristic`.
Véase [`conclusion_eval_protocol.md`](conclusion_eval_protocol.md).

| Uso | Contrato |
|-----|----------|
| Curriculum DDQN / PPO | Fase 1 / oponente en \(\Omega\) — ver specs RL |
| Suite offline WR | Incluido; **200 BO1** cuando el reporte RL lo exige |
| Robust E2E | `--player-b random` (default típico) |
| Validated player A | No es el player A “inteligente” por defecto; puede ser B |

**Nota sobre ActionSpec RL:** Random no muestra uniforme sobre el joint 576 del env; muestrea el lenguaje de órdenes Showdown por slot. Para el banco de pruebas basta como oponente estocástico débil.

---

## 9. CLI / código / tests

| Pieza | Ruta |
|-------|------|
| Bot IAP | `showdown/battle_bots/random/main.py` (`BattleBot`) |
| Preview | `showdown/battle_bots/random/team_preview.py` |
| Helpers UI | `showdown/battle_bots/helpers.py` |
| Robust smoke | `python -m showdown.inference_sim.run_robust_e2e --teams 4 --player-a greedy_solo --player-b random` |
| Conclusión smoke | `python -m showdown.inference_sim.run_conclusion_e2e --smoke --seed 42 --start-ps` |
| Tests | `tests/test_random_team_preview.py`; `tests/test_conclusion_eval.py` |

---

## 10. Limitaciones y lectura cruzada

| Fortaleza | Límite |
|-----------|--------|
| Control negativo claro | No enseña tipado ni roles |
| Barato y reproducible con seed | Puede “ganar” por ruido en N pequeño |
| Válido en preview y combate | No es techo de la escalera |

---

### 10.1 Non-goals

- Tipado, calc, belief, OTS, roles de preview.
- \(\varepsilon\)-greedy de Q-learning / exploración dirigida.
- Terastallize (Reg M-C inexistente; non-goal permanente del formato tesis).
- Sustituir a heuristics o agentes aprendidos.

---

### 10.2 Lectura cruzada

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction*
  (2nd ed.). MIT Press. (línea base aleatoria / exploración).
- [`heuristic_agents_spec.md`](heuristic_agents_spec.md); [`rl_env_spec.md`](rl_env_spec.md) §6.h.
- Capítulo preview aprendido: [`team_preview_masked_ppo_spec.md`](team_preview_masked_ppo_spec.md).

---

## Changelog

| Versión | Fecha | Notas |
|---------|-------|-------|
| **1.4** | 2026-10-11 | Estructura dual-audience; parámetros explicados; entrenamiento y Conclusión explícitos. Semántica intacta. |
| **1.3** | 2026-10-11 | Redacción dual (glosario, fortalezas/límites, refs); foco capítulo preview. Semántica intacta. |
| **1.2** | 2026-10-11 | **Reg M-C:** Tera explicitamente **off** (non-goal de formato). |
| **1.1** | 2026-10-11 | **Team Preview §3** afinado: módulo `team_preview.py`, constantes, RNG inyectable; tests `test_random_team_preview.py`. Combate sin cambio semántico. |
| **1.0** | 2026-10-10 | Contrato FROZEN: preview uniforme 4/6; 80/20 move/switch; mega 50%; friendly-fire 10%. |

---

*Fin del contrato Random v1.4.*
