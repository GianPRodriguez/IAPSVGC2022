# Agente *Greedy-Joint* (v_stable): diseño, formalización y método de decisión

**Documento metodológico** para uso académico (tesis / trabajo de grado).  
**Alcance:** agente `greedy_joint` del repositorio IAPSVGC2022, configuración **congelada** `v_stable`.  
**Audiencia:** lectores con formación en informática o ciencias, **sin** conocimiento previo de *Pokémon* competitivo ni de *VGC*.  
**Implementación de referencia:** `showdown/battle_bots/greedy_joint/`.  
**Documento hermano:** `docs/tesis_agente_greedy_solo.md` (baseline factored).

---

## Resumen

*Greedy-Joint* es un agente de decisión para combates de *Pokémon* en formato **dobles**. Comparte con *Greedy-Solo* el mismo entorno (Showdown + poke-env), el mismo calculador de daño (`@smogon/calc`) y la misma proyección de estado (`CalcPoke`). La diferencia central es la **política de decisión**:

- *Solo* maximiza una puntuación **por activo** (política *factored*).
- *Joint* maximiza una puntuación sobre el **producto cartesiano** de acciones de ambos activos \(A_L \times A_R\) en horizonte **1 turno** (*1-ply*).

Así puede capturar sinergias típicas de dobles (*Protect* + ataque del compañero, redirección + KO, *Helping Hand*, foco combinado, evitar fuego amigo), a costa de un espacio de búsqueda mayor y de una función de utilidad con parámetros heurísticos.

La versión documentada aquí es **`v_stable`**: constantes de utilidad ajustadas mediante un ciclo A/B controlado frente a `greedy_solo` (pruebas P1–P3) y **congeladas** tras observar paridad empírica en el *harness* robusto (`--teams 200`, seed 10), sin veredicto estadístico de superioridad estricta.

---

## 1. Motivación y objetivos

### 1.1. Por qué un agente “conjunto” después del *solo*

En dobles, la acción óptima de un Pokémon **depende** de lo que haga su compañero. Una política factored puede:

- duplicar el mismo objetivo sin necesidad;
- atacar cuando un *Protect* del compañero + presión sería mejor;
- ignorar redirección (*Follow Me* / *Rage Powder*) que salva un KO.

*Greedy-Joint* responde a esa limitación con la búsqueda conjunta más simple posible: **enumerar pares legales y quedarse con el de mayor score**, sin árbol multi-turno ni modelo del rival.

### 1.2. Objetivos de diseño

| Objetivo | Criterio |
|---|---|
| Expresividad de dobles | Argmax sobre \(A_L \times A_R\) |
| Misma base mecánica que el baseline | Reutiliza calc, Team Preview y `CalcPoke` de *solo* |
| Tiempo interactivo | Horizonte \(T=1\); un batch de daño; sin MCTS |
| Interpretabilidad | Score aditivo (daño combinado + KO + utilities + FF) |
| Estabilidad científica | Constantes `v_stable` versionadas y no retocadas ad hoc |

### 1.3. Lo que este agente **no** pretende

- No planifica a \(d>1\) turnos.
- No modela la política del oponente (*expectimax* / Bayes).
- No es RL ni aprende pesos online.
- **No** es un teorema de que joint \(\ge\) solo: la comparación es empírica y depende de la función \(U\) y del rival.

---

## 2. Antecedentes (dominio)

Se asumen los mismos conceptos que en el documento de *Greedy-Solo* (combate por turnos, tipos, VGC dobles, Team Preview 4 de 6, Showdown, poke-env). Resumen mínimo:

- Cada turno el jugador elige **dos** órdenes.
- El espacio natural de decisión es el **par** \((a_L, a_R)\).
- Sinergias entre compañeros son el valor añadido teórico de joint frente a solo.

---

## 3. Formulación del problema

### 3.1. Política conjunta 1-ply

Sea \(s_t\) el estado observable. *Greedy-Joint* define

\[
(a_L^\star, a_R^\star)
= \arg\max_{(a_L,a_R) \in \mathcal{A}_L(s_t)\times\mathcal{A}_R(s_t)}
\; U(s_t, a_L, a_R),
\]

donde \(U\) es la utilidad heurística de la Sección 6 (daño combinado, KO, utilities de soporte, penalización de fuego amigo, costes de timing).

### 3.2. Relación con la política factored

Si \(U(s, a_L, a_R) = S_L(s,a_L) + S_R(s,a_R)\) con \(S\) idéntica a *solo* y sin términos cruzados, el argmax conjunto se factoriza. En la práctica \(U\) **incluye términos cruzados** (redirect+partner, Protect+daño del compañero, KO por focus-fire, friendly fire del par), por lo que joint **no** es un superset trivial de solo: puede elegir pares peores en daño puro si la utilidad de soporte está mal calibrada.

### 3.3. Complejidad

Con \(n_L = |\mathcal{A}_L|\) y \(n_R = |\mathcal{A}_R|\) átomos (candidatos por lado tras generar movimientos × objetivos × variantes), se evalúan \(O(n_L\, n_R)\) pares. El calc de daño sigue siendo el coste dominante y se comparte en *batch* (misma infraestructura que *solo*).

---

## 4. Arquitectura del sistema

```
┌─────────────────────────────────────────────────────────────┐
│  Pokémon Showdown (servidor local)                          │
└───────────────────────────┬─────────────────────────────────┘
                            │ protocolo PS
┌───────────────────────────▼─────────────────────────────────┐
│  poke-env  →  GreedyJointPlayer (pokeenv_player.py)         │
│    · Team Preview (módulo compartido con solo)              │
│    · force_switch; sin switch voluntario                    │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  joint.py  →  best_joint_double_action()                    │
│    · átomos por lado (JointAtom)                            │
│    · score de cada par U(left, right)                       │
│    · 1 Mega por par; Tera si convierte en KO                │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  calc_async / @smogon/calc  (compartido con greedy_solo)    │
└─────────────────────────────────────────────────────────────┘
```

| Módulo | Rol |
|---|---|
| `pokeenv_player.py` | Adaptador poke-env → `best_joint_double_action` |
| `joint.py` | Núcleo: átomos, \(U\), argmax del par |
| `greedy_solo/*` | Calc, tipos, Team Preview, `CalcPoke`, creencia de stats |
| `move_mechanics/` | Precisión, carga, FF, efectos |

---

## 5. Representación: el átomo `JointAtom`

Cada candidato de un lado se compacta en un átomo:

| Campo | Uso |
|---|---|
| `move_slot`, `target`, `move_name` | Orden Showdown |
| `mega` | 0 / Mega / Tera |
| `damage_pct`, `max_roll` | Daño por objetivo (tras precisión) |
| `ko_tids` | KO “duro” (acc ≈ 100% y \(d_{\min}\) suficiente) |
| `utility` | `damage` \| `protect` \| `fakeout` \| `redirect` \| `helpinghand` \| `speed` \| `charge` \| `status` |
| `effect_bonus` | Efectos + soft-KO (acc &lt; 99%) vía `accuracy_scale_ko_bonus` |
| `timing_penalty` | Power Herb + lock de recarga |
| Friendly fire | `hits_ally`, `ally_damage_pct`, `ally_ko` |

Los átomos de utilidad (no daño) se generan sin rango de calc; los de daño reutilizan el pipeline de *requests* de *solo* (incl. Tera solo si convierte en KO).

---

## 6. Función de utilidad del par \(U\) (método central)

### 6.1. Daño combinado por rival

Para cada rival \(f\) con PS actuales \(h\) y máximo \(H\):

\[
\mathrm{pct}_f = \min\!\Big(
  \mathrm{pct}_L(f)+\mathrm{pct}_R(f),\;
  100\cdot h/H
\Big).
\]

Se suma \(\mathrm{pct}_f\) al score (no se cuenta daño “sobre-kill” más allá de los PS restantes).

### 6.2. Término de KO (v_stable / P3)

Sea \(r_f\) la suma de `max_roll` de ambos átomos sobre \(f\).

- Si \(r_f \ge h\) y no hay *Substitute*:
  - **KO duro** (algún átomo tiene \(f\) en `ko_tids`): \(+\,\mathrm{KO\_BONUS}\) con \(\mathrm{KO\_BONUS}=1000\).
  - **Soft individual** (un átomo solo ya alcanzaba KO con acc &lt; 99%): el \(E[\mathrm{KO}]\) ya está en `effect_bonus` del átomo → **no** se re-suma en el par.
  - **Focus-fire nuevo** (ninguno KO solo, sí juntos): \(+\,\mathrm{KO\_BONUS}\cdot \min p_{\mathrm{acc}}\) de los movimientos que pegan a \(f\).

Objetivo: alinear el tratamiento de precisión con *solo* y evitar doble conteo.

### 6.3. Utilities de soporte (constantes `v_stable`)

| Constante | Valor `v_stable` | Rol |
|---|---|---|
| `_PROTECT_BONUS` | 8 | *Protect* si el par hace daño; si no, cae a `_STATUS_BASE` |
| `_FAKE_OUT_BONUS` | 12 | Fake Out (si hay objetivo afectable) |
| `_HELPING_HAND_BONUS` | 10 | HH si hay daño del compañero |
| `_SPEED_CONTROL_BONUS` | 6 | Tailwind / TR / Icy Wind / etc. |
| `_STATUS_BASE` | 2 | Fallback de estado / support sin daño |
| `_FOCUS_SPLIT_BONUS` | 0 | (desactivado) premio por pegar a ≥2 foes |
| `_CHARGE_BASE` | 2 | Base de movimientos de carga |
| `_REDIRECT_BASE` | 4 | Follow Me / Rage Powder sin partner fuerte |
| `_REDIRECT_WITH_ATTACK` | 15 | Partner con daño ≥ `_HIGH_DAMAGE_PCT` |
| `_REDIRECT_WITH_KO` | 25 | Partner con KO (duro o por roll) |
| `_HIGH_DAMAGE_PCT` | 55 | Umbral “partner pega fuerte” |
| Protect + daño partner | +3 | Sinergia explícita |
| HH + daño partner | +4 | Sinergia explícita |

**Redirect (sinergia):** si el compañero garantiza KO → `_REDIRECT_WITH_KO`; si su daño total ≥ 55% → `_REDIRECT_WITH_ATTACK`; si hace algo de daño → base + \(\min(0.25\cdot\mathrm{dmg}, 20)\); si no → base.

### 6.4. Friendly fire

Se resta `friendly_fire_pair_penalty` (alineado a `ALLY_KO_BONUS = 1000`) cuando un movimiento de área puede dañar/KO al compañero y el otro átomo no lo evita (p. ej. no está en *Protect*).

### 6.5. Forma compacta

\[
\begin{aligned}
U(a_L,a_R)
&= \sum_f \mathrm{pct}_f
+ \mathrm{KO\_pair}(a_L,a_R)
+ U_{\mathrm{utility}}(a_L,a_R)
+ e_L + e_R
- \tau_L - \tau_R
- \mathrm{FF}(a_L,a_R),
\end{aligned}
\]

con \(e\) = `effect_bonus`, \(\tau\) = `timing_penalty`.

### 6.6. Pseudocódigo

```
function GREEDY_JOINT_TURN(state):
    if force_switch:
        return FORCED_SWITCH_POLICY(state)   # igual espíritu tipológico que solo

    atoms_L ← BUILD_ATOMS(left, foes, field)
    atoms_R ← BUILD_ATOMS(right, foes, field)
    best ← -∞; pair ← None
    for aL in atoms_L:
        for aR in atoms_R:
            (aL', aR') ← ENFORCE_ONE_MEGA(aL, aR)
            u ← SCORE_PAIR(aL', aR', foes)
            if u > best: best ← u; pair ← (aL', aR')
    return DoubleBattleOrder(pair)
```

### 6.7. Switches

Solo en `force_switch`. **No** hay switch voluntario (comentario explícito en el player): el horizonte 1-ply no asigna valor de cambio de posición.

---

## 7. Team Preview y calc de daño

Idénticos en espíritu a *Greedy-Solo*:

- Preview: cobertura ofensiva/defensiva + \(\gamma\cdot\mathrm{BST}\) (`team_preview.py` compartido).
- Daño: batch `@smogon/calc`; Mega pendiente aplicada en el atacante si `can_mega`.
- Creencia de stats rival opcional vía `apply_belief_stats`.

---

## 8. Configuración `v_stable`: origen experimental

### 8.1. Motivación del ajuste

Con constantes iniciales más agresivas de support/redirect, un E2E `--teams 200 --seed 10` mostró a **solo** superior de forma clara (veredicto `NO` a favor de solo, WR crudo ~42% para joint). Hipótesis: utilities 1-ply sobredimensionadas restaban tempo de daño frente a un rival puramente ofensivo.

### 8.2. Ciclo de pruebas (resumen)

| Prueba | Cambio | Efecto a N=100 seed 10 |
|---|---|---|
| P1 | Bajar Protect / Fake Out / HH (+ sinergias) | ~nulo |
| P2 | Bajar redirect + umbral 55% | leve mejora |
| P3 | Split→0, speed/status↓, KO sin doble conteo | mejor punto del ciclo (aún UNCLEAR) |

### 8.3. Validación a N=200 (seed 10) con P1+P2+P3

| Métrica | Pre-tune (aprox.) | **v_stable** |
|---|---|---|
| Series WR (joint) | ~42% | **~51%** |
| Pares | solo dominante (`NO`) | 13–11, **UNCLEAR** |
| Infra | — | `NEED_FIX=NO`, sync/inf OK |

**Conclusión operativa:** `v_stable` restaura **paridad** con el baseline; **no** demuestra superioridad estadística (muchos *draws* de pares). Se congela para no seguir retocando knobs.

### 8.4. Política de mantenimiento

- No modificar las constantes de la Sección 6.3 sin un protocolo A/B documentado (mismo `--teams`, `--seed`, matchup).
- **Propuesta A (activa, `_DAMAGE_ISO=True`):** \(U \approx S_L+S_R\) + KO conjunto solo si nadie solo remata; overkill al mismo foe se anula.
- **Tune A2 (activo sobre A):** clip de % si ambos golpean el mismo foe; `+_JOINT_KO_SYNERGY` en KO conjunto real; Protect reactivo si HP%≤40 y el partner ataca.
- **Tune A3 (activo, `_SYNERGY_LITE=True`):** Fake Out / Helping Hand / speed / redirect **solo si el partner aporta daño o KO** (bonos modestos; redirect ×0.55).
- **Tune A4 (activo):** `_OPP_THREAT` — Protect/redirect por amenaza de tipos 1-ply; `_MECH_SYNERGY` — HH ×1.5 al daño del partner; Fake Out = daño calc + deny por amenaza del foe. Sin switch voluntario (eso es expectimax).
- **Tune A5 (activo, `_RESIDUAL_SECOND=True`):** tras KO de ≥1 foe, bonus acotado por daño/KO al otro activo vivo (preferir KO+chip vs overkill).
- **B-lite (activo, `_FOCUS_SPEED=True`):** Spe/prio **solo** al apilar dos hits al mismo foe (crédito KO/clip secuencial). `_SPEED_AWARE` global sigue off.
- **Campo Spe:** Trick Room / Tailwind **activos** entran al orden 1-ply (`FieldContext.trick_room` / `tailwind` / `opp_tailwind`).
- **Tune A9 (activo, sinergia residual):** `_FIELD_SETUP` — TW/TR solo con Spe desfavorable y campo ausente; `_PRIO_WIN_GATE` — FO deny / status prio / tempo ES si ganan la carrera; `_SPREAD_SYNERGY` — merge de daño spread + bonus si ambos foes vivos y focus no remata.
- **Spe carrera (activo):** `apply_belief_stats` con Spe abierta usa point-estimate (`preferred_sp`/mediana); Atk/SpA/bulk abiertos siguen en extremo alto. Flag `greedy_solo._SPE_OPEN_POINT_ESTIMATE`.
- **Tune A6/A7/A8 (off):** focus-waste, Spe-gate Protect, FO+partner pressure — off tras N=100.
- **Propuesta B (off, `_SPEED_AWARE=False`):** Spe-order global; off tras N=100.
- Rollback: `_SPE_OPEN_POINT_ESTIMATE=False` → Spe abierta a max; `_FIELD_SETUP`/`_PRIO_WIN_GATE`/`_SPREAD_SYNERGY=False` → A5+B-lite+campo; `_FOCUS_SPEED=False` → A5; `_RESIDUAL_SECOND=False` → A4; `_DAMAGE_ISO=False` → v_stable.

---

## 9. Evaluación recomendada (CMD)

Matchup canónico (joint = A validated, solo = B):

```bat
python -m showdown.inference_sim.run_robust_e2e --teams 200 --seed 10 --player-a greedy_joint --player-b greedy_solo
```

Comprobación de solidez con **otra semilla** (recomendada tras congelar):

```bat
python -m showdown.inference_sim.run_robust_e2e --teams 200 --seed 11 --player-a greedy_joint --player-b greedy_solo
```

Interpretar `verdict` / `pair_*` / `NEED_FIX` en `logs/robust_e2e_report.txt` (y fixplan). No analizar turnos a mano salvo fallo de sync/inferencia.

---

## 10. Limitaciones

1. **Miopía \(T=1\)** y utilities fijas pueden premiar soporte que pierde la carrera de daño.
2. **Sin modelo de oponente:** las sinergias asumen valor intrínseco del support.
3. **Espacio \(A_L\times A_R\):** crece rápido con muchos movimientos/objetivos; mitigado porque VGC limita opciones legales por turno.
4. **Paridad ≠ superioridad:** IC anchos con muchos *draws*.
5. Comparte limitaciones de estado/calc con *solo* (sincronización, creencia de stats).

---

## 11. Relación con otros agentes

| Agente | Relación |
|---|---|
| `greedy_solo` | Baseline factored; oponente / control en A/B |
| `expectimax_bayesian` | Añade incertidumbre / política rival; joint es peldaño intermedio |
| `random` | Control negativo |

---

## 12. Mapa de código

| Concepto | Ubicación |
|---|---|
| Constantes `v_stable` | `joint.py` (bloque STABLE) |
| Átomos y \(U\) | `joint.py` → `_atoms_from_side`, `_score_pair`, `_utility_score` |
| Argmax del par | `best_joint_double_action` |
| Player | `pokeenv_player.py` → `GreedyJointPlayer` |
| Calc / preview | `greedy_solo/calc_async.py`, `team_preview.py` |
| E2E | `showdown/inference_sim/run_robust_e2e.py` |

---

## 13. Conclusión

*Greedy-Joint* (`v_stable`) operacionaliza la idea de que, en dobles, la decisión natural es el **par** de acciones. Conserva la infraestructura mecánica de *Greedy-Solo* y añade una utilidad conjunta 1-ply con utilities de VGC calibradas empíricamente hasta alcanzar **paridad estable** frente al baseline. Su valor científico es ser el eslabón interpretable entre una heurística factored y agentes con modelo de incertidumbre o aprendizaje, con parámetros explícitos y un protocolo de evaluación reproducible.

---

## Apéndice A — Glosario breve

| Término | Significado |
|---|---|
| 1-ply | Optimización mirando solo el turno actual |
| Átomo | Candidato de acción de un lado (movimiento × objetivo × flags) |
| Focus-fire | Ambos activos atacan al mismo rival |
| Friendly fire | Daño al compañero por movimientos de área |
| Redirect | *Follow Me* / *Rage Powder* / *Spotlight* |
| v_stable | Configuración congelada P1+P2+P3 |

## Apéndice B — Cómo citar

> Agente *Greedy-Joint* (`v_stable`) del proyecto IAPSVGC2022: política conjunta 1-ply sobre \(A_L\times A_R\) con utilidad de daño combinado y soporte VGC, documentada en `docs/tesis_agente_greedy_joint.md`.

Ante divergencias código ↔ documento, prevalece `showdown/battle_bots/greedy_joint/joint.py`.
