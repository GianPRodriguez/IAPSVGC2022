# Agente *Greedy-Solo*: diseño, formalización y método de decisión

**Documento metodológico** para uso académico (tesis / trabajo de grado).  
**Alcance:** agente `greedy_solo` del repositorio IAPSVGC2022.  
**Audiencia:** lectores con formación en informática o ciencias, **sin** conocimiento previo de *Pokémon* competitivo ni de *VGC*.  
**Implementación de referencia:** `showdown/battle_bots/greedy_solo/`.

---

## Resumen

*Greedy-Solo* es un agente de decisión para combates de *Pokémon* en formato **dobles** (dos Pokémon activos por bando). En cada turno, transforma el estado observable de la batalla en un conjunto de acciones legales, estima el daño esperado de cada ataque mediante un calculador oficial de daño (`@smogon/calc`), asigna una **puntuación escalar** a cada acción y elige, de forma **independiente para cada Pokémon activo**, la acción de mayor puntuación (*política factored / greedy*).

No es un agente de aprendizaje por refuerzo ni un planificador de horizonte largo: es una **heurística miope (1 turno)** diseñada como *baseline* reproducible, interpretable y computacionalmente barata, frente a la cual se pueden comparar políticas más elaboradas (*greedy-joint*, *expectimax*, RL, etc.).

---

## 1. Motivación y objetivos

### 1.1. Por qué hace falta un agente “simple”

En investigación sobre agentes de juego, un *baseline* fuerte y bien documentado permite:

1. **Aislar** mejoras atribuibles a búsqueda, modelo del oponente o aprendizaje, frente a meras mejoras en el cálculo de daño.
2. **Reproducir** experimentos con semilla y protocolo fijos.
3. **Explicar** decisiones: la política greedy admite auditoría (“elige el máximo de esta función de score”).

### 1.2. Objetivos del diseño *Greedy-Solo*

| Objetivo | Criterio de diseño |
|---|---|
| Correctez mecánica básica | Usar el mismo modelo de daño que la comunidad competitiva (`@smogon/calc`) |
| Tiempo real | Un *batch* de cálculos por turno; sin árbol de juego profundo |
| Interpretabilidad | Score aditivo explícito (daño %, KO, efectos, penalizaciones) |
| Dobles sin explosión combinatoria | Decisión **factored**: cada activo maximiza por su cuenta |
| Uso como baseline | Integración con *poke-env* / Showdown local y *harness* E2E del proyecto |

### 1.3. Lo que este agente **no** pretende

- No modela la política del rival (no es *expectimax* / Bayes).
- No planifica a varios turnos (no es MCTS ni minimax de profundidad \(d>1\)).
- No optimiza el **par** de acciones de forma conjunta (eso es *Greedy-Joint*).
- No aprende parámetros online (no es RL).

---

## 2. Antecedentes: el dominio de decisión

### 2.1. ¿Qué es un combate de *Pokémon*?

*Pokémon* es un juego por turnos. Cada combatiente controla criaturas (*Pokémon*) con:

- **Tipos** (p. ej. Fuego, Agua): definen multiplicadores de efectividad (super eficaz ×2, poco eficaz ×0.5, inmune ×0).
- **Estadísticas**: PS (puntos de salud), Ataque, Defensa, Ataque Especial, Defensa Especial, Velocidad.
- **Movimientos**: acciones con potencia, precisión, categoría (físico / especial / estado) y efectos secundarios.
- **Objetos, habilidades, clima, terreno**: modificadores del daño y de la legalidad de ciertos movimientos.

El combate termina cuando un bando no puede continuar (p. ej. todos sus Pokémon debilitados).

### 2.2. ¿Qué es VGC y por qué son “dobles”?

**VGC** (*Video Game Championships*) es el circuito oficial de *Pokémon* competitivo en videojuegos. El formato estándar relevante aquí es de **combates dobles**:

- Cada jugador lleva un **equipo de 6** Pokémon.
- Tras el **Team Preview**, elige **4** para la batalla.
- En cada momento hay **2 activos** por bando en el campo (hasta 4 Pokémon interactuando a la vez).
- Cada turno, el jugador emite **dos órdenes** (una por activo): atacar (con objetivo), protegerse, cambiar, etc.

Esto cambia radicalmente el problema respecto a *singles* (un solo activo): hay **interacción entre compañeros** (fuego amigo, redirección, *Helping Hand*, etc.) y el espacio de acciones es el **producto cartesiano** \(A_L \times A_R\).

### 2.3. Showdown como entorno de simulación

[Pokémon Showdown](https://pokemonshowdown.com/) es un simulador de combates usado por la comunidad. Este proyecto conecta agentes Python (vía **poke-env**) a un servidor Showdown local, con formatos de campeonato (p. ej. *Champions* / VGC BO3). El agente no “ve” píxeles: recibe el **protocolo de batalla** (estado textual estructurado) y responde con órdenes `/choose`.

---

## 3. Formulación del problema

### 3.1. Vista como proceso de decisión secuencial

Sea \(s_t\) el estado observable al inicio del turno \(t\). El agente elige una acción conjunta

\[
a_t = (a_t^{(L)}, a_t^{(R)}) \in \mathcal{A}_L(s_t) \times \mathcal{A}_R(s_t),
\]

donde \(L\) y \(R\) son los dos activos propios. El entorno (Showdown + rival) produce \(s_{t+1}\) y, al final del episodio, un resultado \(z \in \{-1,0,+1\}\) (derrota / empate / victoria).

Una política óptima maximizaría \(\mathbb{E}[z]\) (u otra recompensa). *Greedy-Solo* **aproxima** esa política con una heurística de un solo paso.

### 3.2. Política factored (definición)

En lugar de

\[
a_t^\star = \arg\max_{(a_L,a_R)} \; U(s_t, a_L, a_R),
\]

*Greedy-Solo* resuelve **dos** problemas independientes:

\[
a_t^{(L)} = \arg\max_{a \in \mathcal{A}_L(s_t)} \; S(s_t, a;\; \text{activo}=L),
\qquad
a_t^{(R)} = \arg\max_{a \in \mathcal{A}_R(s_t)} \; S(s_t, a;\; \text{activo}=R),
\]

con la misma función de puntuación \(S\) (Sección 5). Tras elegir ambos máximos, aplica **reglas de desambiguación** locales (un solo *Mega* por bando; posible *split* de foco si ambos eligen el mismo objetivo).

**Implicación teórica:** la política factored **no** garantiza maximizar una utilidad conjunta \(U\). Puede fallar en sinergias (p. ej. *Protect* + ataque del compañero). A cambio, el coste computacional crece como \(O(|\mathcal{A}_L| + |\mathcal{A}_R|)\) en el número de candidatos a scorear tras el cálculo de daño, no como \(O(|\mathcal{A}_L|\,|\mathcal{A}_R|)\).

---

## 4. Arquitectura del sistema

```
┌─────────────────────────────────────────────────────────────┐
│  Pokémon Showdown (servidor local)                          │
└───────────────────────────┬─────────────────────────────────┘
                            │ protocolo PS
┌───────────────────────────▼─────────────────────────────────┐
│  poke-env  →  GreedyDamagePlayer (pokeenv_player.py)        │
│    · Team Preview                                           │
│    · force_switch / choose_move                             │
└───────────────────────────┬─────────────────────────────────┘
                            │ CalcPoke, MoveOption, FieldContext
┌───────────────────────────▼─────────────────────────────────┐
│  greedy.py  →  best_double_action()                         │
│    · batch de requests de daño                              │
│    · scoring → ScoredAction                                 │
│    · argmax por lado + post-reglas                          │
└───────────────────────────┬─────────────────────────────────┘
                            │ JSON requests
┌───────────────────────────▼─────────────────────────────────┐
│  calc_async.py  →  damage_calc_bridge  →  @smogon/calc      │
│    rangos [daño_min, daño_max]                              │
└─────────────────────────────────────────────────────────────┘
```

| Módulo | Rol |
|---|---|
| `pokeenv_player.py` | Adaptador al entorno: lee batalla poke-env, construye `CalcPoke`, emite `DoubleBattleOrder` |
| `greedy.py` | Núcleo: scoring, `best_double_action`, switches forzados |
| `calc_async.py` | Cliente asíncrono al microservicio de daño |
| `team_preview.py` | Selección de 4/6 por cobertura de tipos + BST |
| `type_chart.py` | Matriz de tipos (preview / matchups) |
| `species_names.py` | Normalización de nombres de especies |
| `move_mechanics/` (paquete compartido) | Precisión, carga, recarga, prioridad, efectos esperados |

---

## 5. Representación de estado y acciones

### 5.1. Estructura `CalcPoke`

Cada Pokémon relevante se proyecta a un *dataclass* con campos usados por el calc y el scoring:

- Identidad: `species`, `types`, `tera_type`, `is_terastallized`
- *Build*: `nature`, `item`, `ability`, `stats`
- Dinámicos: `cur_hp`, `boosts`, `status`
- Volátiles / flags: `is_protected`, `has_substitute`, `is_helping_hand`, `is_redirecting`, `just_switched`, `is_two_turn_charging`, `must_recharge`, `allies_fainted`

### 5.2. Contexto de campo `FieldContext`

Clima, terreno, salas (*Wonder/Magic Room*), *Gravity*, pantallas (*Reflect*, *Light Screen*, *Aurora Veil*) y flags de habilidades de ruina. Estos modificadores se serializan en los *requests* al calc.

### 5.3. Espacio de acciones por activo

Para un activo con movimientos disponibles:

- Cada **movimiento ofensivo** × cada **objetivo legal** (típicamente rivales `1` y `2` en notación Showdown).
- Variantes **Mega** (si `can_mega`) y **Terastallize** (si `can_tera` y el Tera convierte un no-KO en KO garantizado; ver 5.5).
- Movimientos de **estado / utilidad** (score por efectos esperados, sin pasar por el calc de daño).
- **Switch forzado** cuando Showdown lo exige (`force_switch`); el switch voluntario casi no se usa si hay daño positivo.

### 5.4. Salida `ScoredAction`

\[
\texttt{ScoredAction} = (\texttt{score},\; \texttt{move\_slot},\; \texttt{target},\; \texttt{mega},\; \texttt{damage\_pct},\; \texttt{ko})
\]

donde `mega ∈ {0,1,2}` codifica: sin transformación / Mega / Tera.

### 5.5. Reglas especiales de transformación

| Mecánica | Política *Greedy-Solo* |
|---|---|
| **Mega Evolution** | Si está disponible, se asume Mega en el cálculo y se marca `mega=1` por defecto en la acción elegida. Si **ambos** lados querrían Mega, se conserva solo en el de mayor `score`. |
| **Terastallization** | Se evalúa una variante `mega=2` **solo** si el golpe con Tera es KO garantizado y **sin** Tera no lo era. Evita gastar el recurso Tera en daño marginal. |

---

## 6. Función de puntuación \(S\) (método central)

### 6.1. Daño esperado desde un rango

El calc devuelve un rango entero \((d_{\min}, d_{\max})\) (roll de daño). Se usa:

\[
\hat{d} = \frac{d_{\min} + d_{\max}}{2},
\qquad
\mathrm{pct} = 100 \cdot \frac{\hat{d}}{\mathrm{HP}_{\max}}.
\]

**KO garantizado** (*guaranteed KO*):

\[
\mathrm{raw\_ko} \iff d_{\min} \ge \mathrm{HP}_{\mathrm{actual}}
\]

(y no hay *Substitute* que bloquee el KO al mon detrás).

### 6.2. Escalado por precisión

Sea \(p_{\mathrm{acc}} \in (0,1]\) la precisión del movimiento. Entonces:

\[
\mathrm{pct}' = \mathrm{pct} \cdot p_{\mathrm{acc}},
\qquad
\mathrm{KO\_term} = \mathbf{1}[\mathrm{raw\_ko}] \cdot \mathrm{KO\_BONUS} \cdot p_{\mathrm{acc}},
\]

con constante de diseño

\[
\mathrm{KO\_BONUS} = 1000.
\]

La magnitud 1000 hace que **cualquier KO esperado** domine el porcentaje de daño típico (\(\mathrm{pct}' \le 100\)), alineando la heurística con el objetivo de debilitar rivales.

Además se suma un *tie-break* pequeño \(\propto p_{\mathrm{acc}}\) para preferir movimientos más precisos ante empates casi exactos.

### 6.3. Score de un golpe ofensivo (forma aditiva)

Para un candidato ataque \(a\) sobre un defensor concreto:

\[
\begin{aligned}
S_{\mathrm{hit}}(a)
&= \mathrm{pct}'
+ \mathrm{KO\_term}
+ S_{\mathrm{efectos}}(a)
+ S_{\mathrm{carga}}(a)
- C_{\mathrm{PowerHerb}}(a)
- C_{\mathrm{recarga}}(a)
+ \varepsilon_{\mathrm{acc}}(a).
\end{aligned}
\]

Donde:

- \(S_{\mathrm{efectos}}\): esperanza de efectos secundarios / estado (módulo `move_mechanics`), ya sensibilizada a precisión cuando aplica.
- \(S_{\mathrm{carga}}\): bonus si un movimiento de carga dispara este turno (clima / *Power Herb*).
- \(C_{\mathrm{PowerHerb}}\): coste de consumir *Power Herb* (constante compartida del proyecto, p. ej. 18).
- \(C_{\mathrm{recarga}}\): penalización por quedar bloqueado en recarga (*Hyper Beam*, etc.; p. ej. 28, reducida si el golpe es KO).

Movimientos que **no disparan** este turno (primera fase de carga, o mon en *must recharge*) reciben score bajo / nulo según el caso.

### 6.4. Movimientos no ofensivos

No se envían al calc. Su score proviene de \(S_{\mathrm{efectos}}\) sobre el mejor objetivo rival, menos costes de timing. En la práctica, suelen perder frente a un ataque con \(\mathrm{pct}'>0\), salvo efectos muy valorados.

### 6.5. Filtros de legalidad / fracaso mecánico

Antes de scorear (o descartando candidatos):

- *Fake Out* contra tipos/habilidades inmunes a Normal (salvo excepciones como *Scrappy*).
- Movimientos de prioridad que fallan por terreno (*Psychic Terrain*) o habilidades (*Armor Tail*, etc.).
- *Gate* de rangos de daño en fases de carga / recarga (`gate_damage_range`).

### 6.6. Decisión por lado y post-proceso

1. \(\hat{a}_L = \arg\max S\) sobre candidatos del activo izquierdo; análogo \(\hat{a}_R\).
2. Si ambos tienen `mega=1`, se anula Mega en el de menor score.
3. Si ambos eligen el **mismo** objetivo y hay ≥2 rivales, se considera un *retarget* del derecho a otro foe si:
   - el daño alternativo es \(\ge 1\%\), y
   - la suma de scores no cae por debajo del **85%** del foco original.

Esto es una corrección heurística al problema clásico de “doble foco innecesario” en políticas factored.

### 6.7. Pseudocódigo

```
function GREEDY_SOLO_TURN(state):
    if force_switch:
        return FORCED_SWITCH_POLICY(state)

    L, R ← activos propios; foes ← rivales activos
    requests ← BUILD_DAMAGE_REQUESTS(L, R, foes, field)  # incluye variantes Tera
    ranges ← SMÓGON_CALC_BATCH(requests)

    actions_L ← SCORE_CANDIDATES(L, ranges_L)
    actions_R ← SCORE_CANDIDATES(R, ranges_R)

    aL ← argmax score(actions_L)
    aR ← argmax score(actions_R)
    (aL, aR) ← RESOLVE_MEGA_CONFLICT(aL, aR)
    (aL, aR) ← MAYBE_SPLIT_FOCUS(aL, aR, actions_R)

    return DoubleBattleOrder(aL, aR)
```

---

## 7. Cálculo de daño (`@smogon/calc`)

### 7.1. Por qué externalizar el daño

La fórmula de daño de *Pokémon* depende de decenas de modificadores. Reimplementarla a mano introduce sesgos. El proyecto delega en **`@smogon/calc`**, biblioteca de referencia de la comunidad Smogon, expuesta mediante un puente (`damage_calc_bridge`) llamado en *batch* desde `calc_async.py`.

### 7.2. Semántica del rango

El rango \([d_{\min}, d_{\max}]\) captura la **varianza del roll** de daño (típicamente 85%–100% del daño base en generaciones modernas), no la incertidumbre de *build* del rival. La incertidumbre de estadísticas rivales se aborda aparte vía creencias / OTS cuando el *pipeline* de inferencia está activo (`apply_belief_stats`).

### 7.3. Inferencia de stats (opcional, acoplable)

`apply_belief_stats` puede sobrescribir stats del rival:

- Si el conjunto de *sets* posibles (SP) está estrecho → estimación puntual.
- Si el *bulk* (HP/Def/SpD) sigue abierto → extremo **alto** (peor caso para KO).
- Si Atk/SpA siguen abiertos → extremo **alto** (amenaza de daño).
- Si **Spe** sigue abierta → point-estimate (`preferred_sp` / mediana) para carreras 1-ply (TW/TR, B-lite, prio); no max SP (sesgaba setup/orden). Flag `_SPE_OPEN_POINT_ESTIMATE`.

Así el greedy evita “KOs falsos” por subestimar la defensa rival, sin inflar Spe rival en las carreras.

---

## 8. Team Preview (antes del combate)

Antes del primer turno, Showdown pide el orden de 4 Pokémon de 6.

*Greedy-Solo* enumera combinaciones \(C(6,4)\) (y permutaciones de *leads*) y maximiza

\[
J = \alpha\, O + \beta\, D + \gamma\, \mathrm{BST},
\]

con \(\alpha=\beta=1\), \(\gamma=0.01\) por defecto:

- \(O\): cobertura ofensiva de tipos vs tipos rivales (`offensive_coverage_score`).
- \(D\): resistencia defensiva agregada (`defensive_resistance_score`).
- \(\mathrm{BST}\): suma de bases estadística (*Base Stat Total*) como desempate suave.

Los dos primeros del orden salen de *lead*; los otros dos a banca.

**Nota metodológica:** es una heurística de *matchup* de tipos, no un predictor de *win rate* empírico.

---

## 9. Switches forzados

Cuando un activo es debilitado (u otras causas de `force_switch`), el agente elige el reservista que maximiza un score de **matchup tipológico** frente a los tipos rivales activos (`best_forced_switch_slot` + `matchup_vs_actives`).

Los **switches voluntarios** en medio de turno se desincentivan: si hay daño positivo disponible, se ataca; la política privilegiada es ofensiva miope.

---

## 10. Integración experimental en el proyecto

### 10.1. Rol en el *harness* robusto

En `run_robust_e2e`, `greedy_solo` puede actuar como:

- **Player A** *validated* (espejo de estado Battle + soundness de inferencia OTS), o
- **Player B** oponente.

El protocolo A/B con pools disjuntos y BO3 cruzados permite estimar si otra política (p. ej. *joint* o *expectimax*) supera a este baseline con intervalos empíricos sobre *pares* de series.

### 10.2. Métricas típicas (nivel tesis)

| Métrica | Significado |
|---|---|
| WR de series / batallas | Frecuencia de victorias (informativa) |
| WR de pares cruzados | Resultado primario del diseño A/B |
| IC 95% bootstrap | Si incluye 50%, el veredicto es *UNCLEAR* |
| Reward Spec (C1/C2) | Telemetría densa/sparse; no sustituye el veredicto de pares |

Para un trabajo de grado, conviene reportar: \(N\) equipos, semilla, matchup, WR crudo, pares, IC y limitaciones de potencia estadística.

---

## 11. Complejidad y coste

Sea \(M\) el número de movimientos por activo, \(T\) objetivos típicos (≈2), y \(V\) variantes (base + Tera selectivo). El número de *requests* al calc por turno es \(O(M\cdot T\cdot V)\) **por lado**, agregados en un solo *batch*.

No hay expansión de árbol: el coste dominante es I/O al proceso Node del calc, mitigado por *warmup* TCP y *chunks*.

---

## 12. Limitaciones (discusión científica)

1. **Miopía:** ignora amenaza del rival el turno siguiente (no valora “matar antes de ser matado” más allá del KO inmediato propio).
2. **Factored:** no elige el mejor *par* de acciones; puede subutilizar soporte (*Protect*, redirección, *Helping Hand*).
3. **Utilidad de estado débil:** muchos movimientos de apoyo quedan con score bajo frente a daño.
4. **Sin modelo de oponente:** no distingue un rival aleatorio de uno greedy.
5. **Dependencia del calc y del estado observado:** errores de sincronización Battle/protocolo o de creencia de stats sesgan \(S\).
6. **Team Preview tipológico:** puede elegir *leads* subóptimos ante *builds* concretas (ítems, *speed control*, etc.).

Estas limitaciones motivan agentes posteriores (*Greedy-Joint*, *Expectimax-Bayesian*, RL) **sin invalidar** el uso de *Greedy-Solo* como baseline.

---

## 13. Relación con otros agentes del mismo repositorio

| Agente | Diferencia clave respecto a *Greedy-Solo* |
|---|---|
| `greedy_joint` | Maximiza score sobre \(A_L \times A_R\) (sinergias 1-ply; utilities de soporte) |
| `expectimax_bayesian` | Horizonte / incertidumbre / política rival aproximada |
| `random` | Control negativo |

Una afirmación del tipo “*joint* ≥ *solo*” **no** es un teorema bajo funciones \(S\) distintas: solo es comprobable empíricamente bajo un protocolo fijo (mismo pool, semilla, formato).

---

## 14. Mapa de código (trazabilidad tesis ↔ repo)

| Concepto del documento | Archivo / símbolo |
|---|---|
| Política factored + score | `greedy.py` → `best_double_action`, `_actions_from_ranges` |
| `KO_BONUS = 1000` | `greedy.py` |
| Batch `@smogon/calc` | `calc_async.py` → `calc_damage_batch` |
| Player poke-env | `pokeenv_player.py` → `GreedyDamagePlayer` |
| Team Preview | `team_preview.py` → `best_team_preview_order` |
| Tipos | `type_chart.py` |
| Precisión / KO×acc | `move_mechanics/accuracy.py` |
| Carga / Power Herb | `move_mechanics/charge.py` |
| Recarga | `move_mechanics/recharge.py` |
| E2E robusto | `showdown/inference_sim/run_robust_e2e.py` |

---

## 15. Conclusión

*Greedy-Solo* operacionaliza una idea clásica de IA de juegos —**elegir en cada estado la acción de mayor valor heurístico inmediato**— adaptada a *Pokémon* VGC dobles mediante:

1. proyección fiel del estado a un calc de daño de referencia;
2. una función de score aditiva centrada en daño esperado y KO;
3. factorización por activo para mantener tiempo real;
4. reglas explícitas y auditables para Mega, Tera, precisión y post-foco.

Como artefacto científico, su valor principal es la **transparencia metodológica** y su papel de **línea base** en la evaluación de agentes más expresivos dentro del mismo entorno Showdown.

---

## Apéndice A — Glosario breve

| Término | Significado |
|---|---|
| Activo | Pokémon en el campo este turno |
| Banca / reservista | Pokémon del equipo de 4 no activo |
| BO3 | *Best of 3*: serie al mejor de tres juegos |
| BST | Suma de estadísticas base de la especie |
| KO | *Knock Out*: PS del rival llegan a 0 |
| Lead | Pareja que sale al inicio |
| OTS | *Open Team Sheet*: revelación de equipos (según formato) |
| PS / HP | Puntos de salud |
| Showdown | Simulador de combates |
| Tera | *Terastallization*: cambio temporal de tipo |
| VGC | Circuito competitivo oficial en dobles |

## Apéndice B — Cómo citar este documento en la tesis

> Agente *Greedy-Solo* del proyecto IAPSVGC2022: política factored de maximización de daño esperado con calculador `@smogon/calc`, documentada en `docs/tesis_agente_greedy_solo.md` del repositorio.

Fecha de generación del documento: alineada al estado del código en la rama de trabajo del autor. Ante divergencias, prevalece la implementación en `showdown/battle_bots/greedy_solo/`.
