# MCTS + PUCT Agent Spec — planificación open-loop (VGC Gen 9)

| Campo | Valor |
|---|---|
| **Documento** | Contrato exclusivo — MCTS+PUCT open-loop (solo turnos) |
| **Versión** | **5.2** por defecto (combate 5.0 confirmado + preview por búsqueda + SP del rival por prior meta, sin inferencia); 6.0 probada sin efecto neto (alineada con [`rl_env_spec.md`](rl_env_spec.md) v1.20) |
| **Estado** | v5.0 confirmada y por defecto (protocolo §5.4: > joint 65,5 %, expectimax 63,6 %, solo 66,8 %, y +4,5 pts sobre v4); v4.0 (3b) con `MCTS_PUCT_VERSION=4`, v3.1 (3a) con `=3`. Preview por búsqueda (`mean`, §4.6) por defecto desde el 22-11-2026 (protocolo §5.5: +4,1 pts contra joint y contra expectimax; 69,0 % / 67,7 %); v2 (S2) con `MCTS_PUCT_VERSION=2`, v1 con `=1`. Diagnóstico: [`mcts_puct_diagnostico.md`](mcts_puct_diagnostico.md) |
| **Training** | **No**: sin gradientes ni `.pt` de combate |
| **Ajuste** | Hiperparámetros de búsqueda (sobre todo \(N_{\rm sim}\)); barrido automatizado |
| **CLI smoke** | `python -m showdown.rl.mcts_puct --seed 0 --sims 16 --smoke` |
| **CLI barrido** | `python -m showdown.inference_sim.run_mcts_sims_sweep --agent mcts_puct ...` |
| **Env presupuesto** | `MCTS_SIMS` (default config `200`); `MCTS_PUCT_VERSION` (`5` default = S3 fase 3c; `6` = 3d; `4` = 3b; `3` = S3 fase 3a; `2` = S2; `1` = búsqueda original); `MCTS_S3_CALC=0` = S3 sin calculadora (tests) |
| **Dependencias** | NumPy; v3: calculadora real (`damage_calc_bridge`, Node + `damage-calc`); no requiere Torch |

> **Para quien aprende IA.** Este agente no “estudia” partidas previas. En cada
> turno imagina muchas continuaciones (simulaciones), anota qué acciones
> parecieron buenas y juega la más visitada. Es planificación con presupuesto,
> no aprendizaje por gradiente.

---

## 1. Por qué existe

MCTS cambia costo de entrenamiento por costo de decisión. Desde el estado
actual simula futuros con un modelo local S2, equilibra explorar acciones nuevas
y explotar las prometedoras mediante PUCT, y juega la raíz más visitada. Es un
baseline interpretable frente a DDQN/PPO/QD, que comprimen experiencia en pesos.

**Analogía:** un jugador que, con reloj, juega mentalmente muchas variantes
antes de mover; cuando se acaba el tiempo, elige la línea que más veces “sobrevivió”
en su análisis — no una red entrenada ayer.

Hermano: [`mcts_puct_nash_spec.md`](mcts_puct_nash_spec.md) (matriz simultánea).
Preview en Conclusión: greedy de tipado + BST desde v3.1 (antes [`team_preview_nash_spec.md`](team_preview_nash_spec.md), nunca entrenado).

---

## 2. Glosario

| Término | Significado |
|---|---|
| Simulación | Una trayectoria imaginada dentro de S2 |
| Open-loop | El rival se samplea; no tiene nodo estratégico propio |
| Arista / acción | Un joint legal \(a=24a_1+a_2\) desde el nodo |
| \(N\) | Visitas: cuántas sims pasaron por esa arista |
| \(W\) | Suma de retornos Candidate 1 acumulados en la arista |
| \(Q=W/N\) | Valor medio estimado de la arista |
| \(P\) | Prior (aquí uniforme sobre legales) |
| PUCT | Score \(Q\) + bono de exploración con \(N\) y \(P\) |
| Rollout | Política `greedy_joint_snapshot` desde la hoja |
| Backup | Propagar el retorno \(z\) hacia la raíz |
| \(N_{\rm sim}\) | Presupuesto: sims por **decisión de turno**, no épocas de train |
| Meseta | Zona donde subir \(N_{\rm sim}\) ya casi no mejora WR |

---

## 3. Qué son \(Q\) y \(N\) (y qué no son)

### 3.1 No son parámetros de entrenamiento

\(Q\) y \(N\) **no** se “aprenden” entre partidas ni se guardan en un checkpoint.
Son **estadísticas temporales del árbol** de la decisión actual:

1. Empiezan en cero al crear el árbol del turno.
2. Cada simulación que elige una arista hace \(N \leftarrow N+1\) y \(W \leftarrow W+z\).
3. Se recalcula \(Q=W/N\).
4. Al jugar la orden real, **el árbol se descarta**. El siguiente turno vuelve a cero.

Lo que sí fijas a mano (o con el barrido) es \(N_{\rm sim}\), \(c_{\rm puct}\), etc.

### 3.2 Intuición numérica

| Símbolo | Pregunta que responde | Fiabilidad |
|---|---|---|
| \(Q\) | “¿Qué tan bien me fue *en promedio* cuando elegí esto?” | Baja si \(N\) es pequeño |
| \(N\) | “¿Cuántas veces miré esto?” | Es el contador de evidencia |

Escala de retorno aquí: Candidate 1, \(z\in\{-1,0,+1\}\) (derrota / empate-cap / victoria).
Así \(Q\in[-1,1]\) aproximadamente.

### 3.3 Durante la búsqueda vs al final

**Durante las sims** PUCT elige la arista con mayor

\[
\text{score}(a)=Q(s,a)+c_{\rm puct}\,P(s,a)\frac{\sqrt{N(s)}}{1+N(s,a)}.
\]

El segundo término (bono) es grande cuando \(N(s,a)\) es pequeño → obliga a
probar acciones poco vistas.

**Al cerrar el turno** se juega \(\arg\max_a N(s,a)\) (desempate \(Q\)), no el
máximo \(Q\) aislado. Motivo: un \(Q\) alto con \(N=2\) puede ser suerte; un
\(N\) grande resume el proceso acumulado de selección.

### 3.4 Ejemplo trabajado

Con \(Q_A=0.30,\,N_A=20\) y \(Q_B=0.10,\,N_B=2\):

1. **PUCT (mitad de la búsqueda):** B puede ganar el score temporalmente porque
   su bono de exploración es alto (\(N_B=2\)).
2. **Tras más visitas a B:** el bono cae; si B sigue mediocre, A recupera el score.
3. **Decisión final:** gana A (\(N_A=20>2\)), aunque en algún momento B haya sido
   “la más interesante” para explorar.

Para un aprendiz: \(Q\) es la nota media; \(N\) es cuántos exámenes; PUCT manda
hacer exámenes a los poco evaluados; al entregar la respuesta final, confías en
quien tiene más exámenes, no en una nota 10 con una sola prueba.

---

## 4. Algoritmo paso a paso (v2.0)

1. Convertir la batalla en `BattleSnapshot` y **preparar la raíz** (`prepare_root`):
   movimientos propios en el orden de `available_moves` (= índice de la acción real);
   banca rival = vistos no excluidos + no vistos con mayor `p_brought` hasta 4 traídos.
2. **Aristas de la raíz** = joints legales reales (`pe_adapter.legal_joint_mask`, que
   filtra por `valid_orders`); sin poke-env, joints podados del snapshot (§5.1). Si hay
   una sola arista se juega sin buscar. Mega preferida (`prefer_mega`, §5.1).
3. **Prior** \(P=(1-\varepsilon)\,\mathrm{softmax}(\text{daño}/\tau)+\varepsilon/|A|\)
   con el daño estimado S2 de cada joint; Protect y cambios conservan masa por la parte
   uniforme.
4. Repetir \(N_{\rm sim}\) veces, **siempre desde la raíz** (open-loop real):
   - Bajar con PUCT; las aristas sin visitar usan **FPU = media del nodo padre**.
   - En cada ply: **remuestrear** la acción rival (softmax del daño rival con
     \(\tau_{\rm opp}\); uniforme sobre sus joints podados con prob. \(\varepsilon_{\rm opp}\))
     y el azar de S2; `step_sim` (sin chip de estado) + **reemplazo tras KO**.
   - Si el ply alcanza un nodo nuevo: se expande (joints podados, Top-K por daño) y
     se valora con `rollout_turns` turnos (greedy propio, rival muestreado) +
     **evaluación de material** \(v=(HP_{\rm propio}-HP_{\rm rival})/400\in[-1,1]\);
     terminal = ±1.
   - Backup: \(N{+}{=}1,\,W{+}{=}z\) en el camino.
5. Jugar \(\arg\max N\) en la raíz (desempate \(Q\), luego id menor).
6. Descartar el árbol.

En cambio forzado en la raíz el rival no actúa en ese ply. Los nodos open-loop
guardan solo estadísticas de **nuestras** secuencias: en cada visita se descartan
aristas que ya no son jugables en el estado muestreado (`joint_still_valid`).

### Ciclo de vida del presupuesto

Cada turno real: raíz nueva → sims con RNG `seed|turn|k` → decisión → telemetría
(`sims`, `nodes`, `avg_depth`, `failures`, `root_actions`, `root_visited`,
`root_max_n`, `elapsed_sec`) → árbol muerto. Una excepción de simulación aporta
\(z=0\) y se registra. Coste medido en el laptop: ~2 ms/sim (v1: ~130 ms/sim).

### 4.1 Qué hacía v1 (y por qué se cambió)

v1 (`search.py`, intacto porque MCTS–Nash lo importa) expandía la máscara del
snapshot (~660 joints: 4 objetivos por movimiento, `pass`, mega duplicada), probaba
primero las aristas no visitadas **en orden de id** y guardaba el hijo de cada acción
con **una sola** muestra rival y de azar. Con \(N_{\rm sim}\le 660\) cada arista
recibía como mucho una visita y se jugaba el id más bajo con \(Q=+1\). Detalle y
medidas en [`mcts_puct_diagnostico.md`](mcts_puct_diagnostico.md).

---

### 4.2 Modelo S3 (v3.0, fase 3a: `s3.py`)

S3 sustituye a S2 como modelo de transición **y** de puntuación (prior, rival del
árbol y rollout greedy). S2 sigue intacto para MCTS–Nash y para `MCTS_PUCT_VERSION=2`.

| Pieza | Cómo |
|---|---|
| Tabla de daño | Al inicio de cada decisión, **un lote** a la calculadora real (`@smogon/calc` Champions, mismos pedidos que P5: `obs_calc.build_request`): cada atacante × defensor × movimiento dañino para los 4 Pokémon de cada lado (activos y banca), variante individual y de área (`isSpread`), y fuego amigo de `allAdjacent`. ~100–130 pedidos; ~19 ms sin caché, ~6 ms con caché; arranque de Node ~10 s una vez por proceso. La tirada se sortea uniforme en el rango. Pedido sin respuesta → heurística S2 para ese par |
| Stats | Propios exactos (`|request|`). Rival: SP preferido por el prior meta leave-one-out de sus creencias (`iap_battle.stat_beliefs`, `preferred_sp`), como greedy/expectimax; sin creencias, punto medio del rango de la hoja |
| Dex | El de Champions (`team_preview_obs`), no `API/JSONs/moves.json`: a este le faltan 21 de los 212 movimientos del meta (Make It Rain, Kowtow Cleave, Glaive Rush…), que con S2 no hacían nada. La poda de objetivos de v3 usa también este dex |
| Orden | Cambios primero; luego movimientos por prioridad (`move_priority`: Prankster, Grassy Glide…) y velocidad efectiva (`effective_speed`: boosts, Scarf, parálisis; Tailwind ×2 por lado; Trick Room invierte); desempate al azar. Quien cae antes de actuar no actúa |
| Objetivos | Individual: si el rival elegido cayó, se redirige al otro. `allAdjacentFoes`: rivales vivos; `allAdjacent`: también el aliado; ×0,75 si golpea a más de uno. Precisión por movimiento |
| Puntuación | Valor de un hueco = Σ objetivos de precisión × (daño medio acotado al HP + 40·P(KO)); golpear al aliado resta. Prior, rival (softmax) y rollout greedy la usan |

Fuera de 3a (fases 3b/3c): Protect, Fake Out, redirección, Helping Hand, Tailwind /
Trick Room como acciones, Intimidate, estados, clima/terreno cambiantes, boosts dentro
del árbol, críticos. Los movimientos de estado no hacen nada. La tabla se calcula con
el estado de la raíz (HP, boosts, campo) y no se recalcula dentro del árbol.

---

### 4.3 Fase 3b (v4.0, confirmada; por defecto del 21-11-2026 hasta v5)

S3 con `phase='3b'`. Mecánicas añadidas (todas desde el dex de Champions):

| Mecánica | Modelo |
|---|---|
| Protect y variantes (`stallingMove` + volátil protect, Spiky Shield, Baneful Bunker, King's Shield…) | Prioridad +4; éxito con prob. \((1/3)^n\), \(n\) = usos seguidos (`protect_counter`); bloquea los movimientos con flag `protect`; Feint (`breaksProtect`) la rompe. Endure no protege |
| Wide Guard | Bloquea los golpes de área sobre su lado ese turno |
| Fake Out / First Impression | Solo en el primer turno en campo (`turns_active = 0`); si no, fallan. Fake Out hace retroceder al objetivo (no actúa) salvo Inner Focus, Shield Dust o Covert Cloak |
| Follow Me / Rage Powder | Los golpes individuales del rival hacia ese lado van al redirector; Rage Powder no afecta a tipo Planta, Overcoat ni Safety Goggles |
| Helping Hand | El daño del aliado ese turno ×1,5 |

Contadores: `turns_active` sube al final del turno para los activos; un cambio o un
reemplazo lo pone a 0 (y `protect_counter` también).

Puntuación (prior, rival del árbol, rollout): Protect = 0,5 · \((1/3)^n\) · amenaza entrante;
Fake Out = daño + 0,5 · daño que el objetivo haría (0 fuera del primer turno);
Follow Me / Rage Powder = 0,4 · amenaza individual al aliado; Helping Hand = 0,5 · mejor
daño del aliado; Wide Guard = 0,5 · daño en área sobre el lado. Amenaza = Σ rivales del
mejor daño esperado (tabla de S3) acotado al HP.

---

### 4.4 Fase 3c (v5.0, por defecto desde el 21-11-2026: `MCTS_PUCT_VERSION=5`)

S3 con `phase='3c'` (incluye todo 3b). Mecánicas añadidas:

| Mecánica | Modelo |
|---|---|
| Velocidad dinámica | Tras cada acción se elige la siguiente con la velocidad **actual** (Gen 9): Tailwind, Trick Room, Icy Wind, parálisis o un cambio de etapas reordenan el resto del turno |
| Tailwind | Lado ×2 de velocidad, 4 turnos (el actual incluido) |
| Trick Room | Prioridad −7; invierte el orden dentro de cada prioridad, 5 turnos; repetirlo lo cancela |
| Estados | Sueño 1–3 turnos (no actúa; al despertar actúa), parálisis (×0,5 velocidad, 25 % de no actuar), quemadura (×0,5 daño físico salvo Guts; 1/16 por turno), veneno (1/8). Inmunidades: ya con estado, Eléctrico/par, Tierra/Thunder Wave, Fuego/quemadura, Veneno y Acero/veneno, polvo (Planta, Overcoat, Safety Goggles), Insomnia / Vital Spirit / Sweet Veil |
| Boosts en el árbol | Movimientos de estado (`boosts`, p. ej. Swords Dance, Nasty Plot), secundarios (`secondaries`, p. ej. Icy Wind −1 Vel, Snarl −1 AtEsp), `selfBoosts` (Close Combat, Draco Meteor, Make It Rain), Parting Shot (−1 Atk/−1 AtEsp, falta en el dex). Clear Body / White Smoke / Full Metal Body bloquean bajadas del rival; Defiant / Competitive (+2) |
| Daño | Tabla de la raíz × razón de etapas (atk/def o spa/spd) respecto a la raíz × quemadura × clima actual / clima de la raíz (Fuego/Agua ×1,5 / ×0,5) |
| Al entrar | Intimidate (−1 Atk a los rivales; inmunes Clear Body, Inner Focus, Oblivious, Own Tempo, Scrappy, Hyper Cutter…; Guard Dog +1) y clima por habilidad (Drought, Drizzle, Sand Stream, Snow Warning, Orichalcum Pulse): al cambiar, al reemplazar tras KO y al pivotar |
| Clima | Por movimiento (Sunny Day, Rain Dance…) o habilidad, 5 turnos; Swift Swim / Chlorophyll / Sand Rush / Slush Rush ×2 velocidad; daño de arena 1/16 |
| Pivotes | U-turn, Volt Switch, Flip Turn (si golpean) y Parting Shot: entra el de banca con más HP |

Puntuación añadida: Tailwind / Trick Room = 25 × (fracción de duelos de velocidad ganados
con el efecto − sin él); sueño 0,8 · daño del objetivo, parálisis 0,35, quemadura 0,5 si su
mejor ataque es físico (0,1 si no), veneno 8; boosts propios 0,15 · propio mejor daño por
etapa de ataque; bajadas al rival 0,2 · su daño; Icy Wind / Electroweb +5 por rival.

Fuera de 3c: terreno (cubierto en la fase 3d, §4.5), críticos, Quick Guard, Spotlight,
Yawn, objetos consumibles, habilidades raras.

---

### 4.5 Fase 3d: terreno (v6.0, candidata: `MCTS_PUCT_VERSION=6`)

S3 con `phase='3d'` (incluye 3c). Motivo: en la v5 el terreno de la raíz ya entraba en la
tabla de daño, pero el que aparece dentro del árbol no; en el meta, Grassy Surge (Rillaboom)
está en 195 de 1800 sets (~65 % de los equipos) con Grassy Glide en 191, y Psychic Surge
(Indeedee-F) en 34 con Expanding Force en 30; ningún set usa movimientos de terreno.

| Mecánica | Modelo |
|---|---|
| Terreno al entrar | Grassy / Psychic / Electric / Misty Surge, Hadron Engine (al cambiar, reemplazar o pivotar); 5 turnos (sin dato de la raíz: ~3 restantes) |
| Daño | × factor de terreno actual / factor del terreno de la raíz: Grass / Electric / Psychic ×1,3 si el atacante está en el suelo; Dragon ×0,5 contra defensor en el suelo con Misty; Earthquake / Bulldoze / Magnitude ×0,5 contra defensor en el suelo con Grassy |
| Grassy Glide | Prioridad +1 si hay Grassy Terrain y el usuario está en el suelo (se evalúa al ordenar el turno) |
| Psychic Terrain | Bloquea los movimientos con prioridad > 0 dirigidos a rivales en el suelo (daño y estado: Fake Out, Sucker Punch, Aqua Jet, Extreme Speed, Prankster) |
| Expanding Force | En Psychic Terrain y con el usuario en el suelo: golpea a los dos rivales (×0,75) con poder ×1,5 (más el ×1,3 del terreno) |
| Grassy Terrain | Cura 1/16 al final del turno a quien esté en el suelo |
| Misty / Electric | Misty bloquea estados no volátiles y Electric el sueño a quien esté en el suelo |

"En el suelo" = no Volador, sin Levitate ni Air Balloon.

---

### 4.6 Preview por búsqueda (por defecto desde el 22-11-2026; `MCTS_PREVIEW=greedy` para el anterior)

Decisión del usuario (21-11-2026): como método de árbol, el preview también debe mirar los
primeros turnos. Hallazgo previo: el preview greedy (tipado + BST) elige bien los 4 Pokémon
pero **no elige los leads** (su puntuación ignora el orden; salen en el orden de la hoja).

`showdown/rl/mcts_puct/search_preview.py`:

| Paso | Detalle |
|---|---|
| Candidatas propias | 3 mejores combinaciones de 4 según la regla greedy × 6 parejas de leads = 18 formaciones |
| Candidatas del rival | Sus 2 mejores combinaciones (hoja abierta; SP por prior meta) × 6 leads = 12 |
| Modelo | Una tabla S3 para los 6 × 6 Pokémon (~150 pedidos) y la fase de S3 del combate (3c en v5, 3d en v6) |
| Simulación | 8 simulaciones × 3 turnos por par (softmax de la puntuación S3 en ambos lados, efectos al entrar de los leads); valor = material |
| Decisión | `mean` (mejor promedio contra las 12 formaciones del rival, por defecto) o `maximin` (`MCTS_PREVIEW_DECISION`); desempate: el orden greedy |
| Seguridad | `deadline_sec = 90` (`MCTS_PREVIEW_DEADLINE`; 25 hasta el 22-11-2026); si se agota o falla, orden greedy y anomalía `preview_search_fallback` |

Coste medido en el laptop: ~12–15 s por preview (1728 simulaciones), ~24 s con el arranque de
Node. Memo por proceso (21-11-2026, commit `118e998`): la búsqueda es determinista dado (equipos,
config), así que las partidas 2 y 3 del BO3 reutilizan la formación de la 1; `MCTS_PREVIEW_DEADLINE`
sube el límite bajo carga. La búsqueda corre en un hilo (`asyncio.to_thread`, commit `3269711`)
para no bloquear el bucle de poke-env. **Confirmado el 22-11-2026** (protocolo §5.5, C9+C10, en
pareado frente al greedy): +4,11 pts contra joint (p = 3·10⁻¹²) y +4,06 contra expectimax
(p = 1·10⁻¹¹); 0 fallbacks con 90 s. En el servidor tarda ~7 s por preview.

---

## 5. Observación, acción y máscaras

Contrato live: [`rl_env_spec.md`](rl_env_spec.md). La búsqueda usa snapshots, no
el vector de ObsSpec v2. Aristas de la raíz = joints legales de poke-env
(`valid_orders`, ActionSpec v2: mega por movimiento); Tera id 37 siempre off (Reg M-C).
Terminal = Candidate 1; hoja no terminal = material (§4).

### 5.1 Poda de acciones (`actions.py`)

| Regla | Detalle |
|---|---|
| Objetivo según `moves.json` | `normal`/`adjacentFoe`/`any`/`randomNormal` → rival a o b (vivos); `adjacentAlly` → aliado (si vive); resto (área, self, campo) → objetivo 3 |
| `pass` | Solo si el hueco no tiene otra acción |
| Cambios | Banca viva; los dos huecos no cambian al mismo Pokémon |
| Mega | Dentro del árbol no hay variantes mega (S2 no las distingue). En la raíz, si existe el mismo joint con mega en un hueco más, se quita el joint sin mega |
| Top-K | Dentro del árbol, `inner_top_k` joints de mayor daño; la raíz no se recorta |

**No hay Team Preview en este MDP de combate** (MCTS no elige `/team` como
acción de búsqueda). Desde v3.1 el preview es el **greedy** de tipado + BST
(`GreedyDamagePlayer.teampreview`), el mismo de solo/joint/expectimax; desde el 22-11-2026 el
default es el **preview por búsqueda** (§4.6), que usa el modelo S3 para elegir 4 y leads
(`MCTS_PREVIEW=greedy` vuelve al greedy).
`MCTS_PREVIEW=nash` usa [`team_preview_nash`](team_preview_nash_spec.md) vía
`frozen_preview` (fallback `/team 1234` si falta checkpoint; hoy no hay uno entrenado).

---

## 6. “Mejorar” el agente sin entrenamiento

No hay `final.pt`. Mejorar = ajustar presupuesto/reglas y medir:

| Palanca | Efecto típico |
|---|---|
| Subir \(N_{\rm sim}\) | Menos ruido, más latencia; luego **meseta** |
| \(c_{\rm puct}\) | Más/menos exploración durante el árbol |
| Rollout / rival en árbol | Cambian el sesgo del valor de hoja (cambio fuerte de agente) |

La palanca principal y limpia para tesis es \(N_{\rm sim}\).

### 6.1 Meseta de \(N_{\rm sim}\)

Subir sims no mejora el WR de forma lineal. Se busca el **menor** \(N_{\rm sim}\)
tal que el siguiente escalón ya no paga:

- mejora de WR &lt; umbral (pp), **y**
- el tiempo crece al menos un factor dado.

Defaults de config: **200** sims (`16` en smoke). Env: `MCTS_SIMS`.

### 6.2 Barrido automatizado

```bat
python -m showdown.inference_sim.run_mcts_sims_sweep --agent mcts_puct ^
    --player-b greedy_joint --seed 10 --teams 40 ^
    --delta-wr-pp 1.5 --time-ratio 1.8 --metric wr_raw --start-ps
```

| Parámetro CLI | Default | Propósito | Cómo influye |
|---|---:|---|---|
| `--agent` | `both` / `mcts_puct` | Qué árbol barrer | Open-loop y Nash tienen grillas distintas |
| `--player-b` | `greedy_joint` | Rival del robust e2e | Cambia la dificultad del WR; debe fijarse al citar meseta |
| `--seed` | `10` | Repro batch + MCTS_SEED | Otra seed → otra curva (misma metodología) |
| `--teams` | `40` | Tamaño del batch robusto | Más → WR más estable, más horas |
| `--sims` | `50,100,200,400` | Rejilla de presupuestos | Define los puntos de la curva |
| `--delta-wr-pp` | `1.5` | Criterio de meseta (pp) | Más estricto → exige más ganancia para subir N |
| `--time-ratio` | `1.8` | Factor de tiempo | Más bajo → declara meseta antes |
| `--metric` | `wr_raw` | WR crudo o de pares | `pair_wr` alinea con métrica BO3 del robusto |
| `--dry-run` | off | Solo lógica/tabla fake | Valida CLI sin PS |

Por cada punto: `MCTS_SIMS=<N>` + `run_robust_e2e --player-a mcts_puct`.

#### Tracking del barrido (cómo cambia \(N_{\rm sim}\))

El hiperparámetro que **varía** es solo `n_sims` (vía `MCTS_SIMS`). El resto
(`c_puct`, prior, rollout, …) queda **congelado** y se registra en cada evento
para auditar que no se movió a escondidas.

| Artefacto | Contenido |
|-----------|-----------|
| `logs/mcts_sims_sweep_mcts_puct_history.jsonl` | Un evento por paso: `sweep_start`, `hyperparam_step` (prev→nuevo N, ΔWR, time_ratio), `plateau_decision` |
| `logs/mcts_sims_sweep_mcts_puct_progress.json` | Estado vivo (`running`/`done`, N actual, recomendación interim) |
| `logs/mcts_sims_sweep_mcts_puct_tracking.txt` | Log humano paso a paso hasta la meseta |
| `logs/mcts_sims_sweep_mcts_puct.{txt,json}` | Tabla final + recomendación + copia del tracking |
| `logs/figures/mcts_puct_sims_sweep_latest.svg` | WR vs paso del barrido (real) |
| `logs/figures/hp_tune_ledger.jsonl` | Append histórico de recomendaciones |

En consola cada paso imprime `TRACK step i/n N_sim prev->cur ...`. La
recomendación **interim** se recalcula con los puntos ya vistos (útil si el
job se corta).

Lógica pura (testeable): `showdown/inference_sim/mcts_sims_sweep.py`.  
Tests: `tests/test_mcts_sims_sweep.py`.

El valor recomendado se **congela** antes de Conclusión (igual que un checkpoint).

```bat
python -m showdown.rl.mcts_puct --seed 0 --sims 16 --smoke
```

---

## 7. Hiperparámetros de búsqueda

| Parámetro | Valor | Propósito | Cómo influye |
|---|---:|---|---|
| \(N_{\rm sim}\) | `200` (`16` smoke) | Presupuesto por decisión | Más reduce varianza y aumenta latencia; barrer hasta meseta |
| \(c_{\rm puct}\) | `1.5` | Exploración PUCT | Mayor visita acciones poco probadas |
| Prior \(P\) | softmax(daño/`20`) + `0.25` uniforme (v1: uniforme) | Iniciar aristas | Guía sin red; ε protege Protect/cambios |
| FPU | media del padre (v1: 0) | Valor de aristas sin visitar | Evita explorar en orden de id |
| `inner_top_k` | `12` | Acotar ramificación interna | Más = más cobertura, menos profundidad |
| Rival en árbol | softmax(daño/`10`) + `0.10` uniforme podado (v1: uniforme sobre ~660) | Transición open-loop | Rival que ataca, con algo de variedad |
| Muestreo rival/azar | en cada visita (v1: una vez por arista) | Open-loop real | \(Q\) promedia respuestas rivales |
| Hoja | `rollout_turns=2` + material (v1: greedy hasta turno 40) | Valorar hojas | Menos ruido y mucho más rápido |
| `max_depth` | `6` plies | Acotar árbol | Más allá, se valora con rollout corto |
| Reemplazo tras KO | automático, más HP (v1: no) | Estado coherente | Evita lados con un solo activo |
| Chip de estado S2 | off (v1: 8–18 % de daño) | Protect no daña | Corrige un artefacto de S2 |
| `prefer_mega` | on | Mega con el mismo movimiento | S2/S3 no distinguen mega (S3 usa la forma Mega si lleva la piedra) |
| Modelo | `s3` (v3; `s2` en v2) | Transición y puntuación | §4.2 |
| Max turns | `40` | Corte del combate simulado | Cap → 0 |
| Descuento | `1` | Backup | — |
| RNG | seed+turn+sim | Repro | Azar S2 / accuracy |
| Acción final | máx. \(N\), desempate \(Q\), id | Decidir raíz | Prefiere evidencia acumulada |

---

## 8. Ilegales, máscaras y fallos

- Joints legales desde el snapshot; Tera 22 ausente.
- Mega solo si el estado simulado la permite.
- Sin aristas → `(pass, pass)`.
- L1: fallback seguro del player live.
- L2: desync → decisión inválida en harness validated.
- Fallo interno S2 → \(z=0\) + `failures`.
- Más sims **no** corrigen sesgo sistemático de S2.
- L3/crash según `rl_env_spec`.

---

## 9. Training frente a inferencia

No hay training. Toda la cómputo útil es inferencia. Cambiar `MCTS_SIMS` cambia
calidad/latencia al instante; no produce modelo reutilizable.

Desarrollo: Candidate 1, fidelidad S2↔S1, latencia, profundidad, fallos, barrido
\(N_{\rm sim}\).  
**Conclusión:** congela presupuesto, S2, rollout y seed policy; no adapta el
árbol entre partidas. Protocolo:
[`conclusion_eval_protocol.md`](conclusion_eval_protocol.md).

---

## 10. Team Preview y Conclusión

Pipeline G: **preview por búsqueda (S3, `mean`) → MCTS+PUCT combate** desde el 22-11-2026;
preview greedy (tipado + BST) de v3.1 hasta entonces (antes `team_preview_nash`, que nunca se
entrenó). No comparten árbol ni parámetros. El \(N_{\rm sim}\) congelado del barrido es
parte del artefacto de evaluación.

---

## 11. Mapa de código

| Archivo | Responsabilidad |
|---|---|
| `showdown/rl/mcts_puct/agent.py` | `MCTSPuctConfig` / API; `version` 1/2 |
| `showdown/rl/mcts_puct/openloop.py` | v2: open-loop, FPU, hoja corta, `prepare_root`, reemplazo tras KO |
| `showdown/rl/mcts_puct/actions.py` | v2: poda de joints, prior, rival del árbol, `prefer_mega` (v3: puntuador y clase de objetivo de S3) |
| `showdown/rl/mcts_puct/s3.py` | v3: modelo S3 (tabla de daño real, orden del turno, área, precisión, puntuación) |
| `showdown/rl/mcts_puct/search.py` | v1 (y `make_sim_rng`/`rollout` usados por MCTS–Nash) |
| `showdown/rl/mcts_puct/tree.py` | nodos, aristas, PUCT, \(Q,N,W\) |
| `showdown/rl/mcts_puct/simulator.py` | S2 |
| `showdown/rl/mcts_puct/greedy_snapshot.py` | rollout |
| `showdown/rl/mcts_puct/player.py` | poke-env + `MCTS_SIMS`; raíz desde `valid_orders`; contadores `mcts_stats` |
| `showdown/inference_sim/mcts_sims_sweep.py` | meseta (lógica) |
| `showdown/inference_sim/run_mcts_sims_sweep.py` | CLI barrido |
| `showdown/battle_bots/mcts_puct/` | integración live |

---

## 12. Limitaciones

- **S3 (v3) cubre solo la fase 3a**: daño real y orden del turno. Protect, Fake Out,
  redirección, Helping Hand y los efectos de varios turnos (Tailwind/Trick Room como
  acciones, Intimidate, estados) siguen sin modelar: valen "no hacer nada".
- La tabla de daño es la de la raíz: boosts, clima o HP que cambian dentro del árbol no
  la recalculan (salvo el HP restante, que acota el daño en la puntuación).
- v2 (S2): sin velocidad, sin stats reales, dex sin 21 movimientos del meta.
- Rival del árbol = softmax de daño: no anticipa Protect, redirección ni cambios.
- La evaluación de material ignora estados, velocidad y campo.
- Sin poke-env (bot IAP) la raíz usa la poda del snapshot y no ofrece mega.
- Tera off por formato.

---

## 13. Lectura de telemetría

`sims` debe coincidir con el presupuesto. `nodes` / `avg_depth` describen la
forma del árbol. `failures>0` investiga S2. La acción más visitada puede tener
\(Q\) menor que otra poco probada: no es contradicción (ver §3). Conservar seed
y turno para reproducir RNG.

---

## Changelog

| Versión | Fecha | Notas |
|---|---|---|
| **5.2** (por defecto desde 2026-11-22) | 2026-11-22 | **Sin inferencia de SP por defecto** (`ValidatedMCTSPuctPlayer.default_sp_info_mode = 'meta'`; combate y preview iguales a 5.1). Diagnóstico §9.1: la inferencia estrechó 0 de 14 016 stats en partidas de G; meta frente a inference +0,02 pts (p = 1), con 14 % menos de tiempo; oracle_spe +0,08 (p = 0,85). `--sp-info-a inference` la reactiva. |
| **5.1** (por defecto desde 2026-11-22) | 2026-11-22 | **Preview por búsqueda por defecto** (§4.6; combate idéntico a 5.0). Confirmado (protocolo §5.5, C9+C10): +4,11 pts sobre el greedy contra joint y +4,06 contra expectimax, Holm m = 2; G v5 + búsqueda 69,0 % contra joint y 67,7 % contra expectimax. `MCTS_PREVIEW` por defecto `search` (`greedy` / `nash` opcionales); `MCTS_PREVIEW_DEADLINE` por defecto 90 s. |
| **6.0** (probada, sin efecto neto) | 2026-11-21 | Fase 3d (§4.5): terreno al entrar por habilidad y su duración, ajuste de daño por terreno respecto a la raíz, prioridad de Grassy Glide según el terreno actual, bloqueo de prioridad en Psychic Terrain, Expanding Force en área ×1,5, curación de Grassy, Misty / Electric contra estados. `MCTS_PUCT_VERSION=6`. Exploración (diagnóstico §6.6): 64,5 % frente a 64,2 % de v5, McNemar p = 0,73; sin efecto también en el subgrupo con terreno (+0,26 pts, p = 0,70). No se adopta; se confirma v5. |
| **5.0** (por defecto desde 2026-11-21) | 2026-11-21 | Fase 3c (§4.4): velocidad dinámica, Tailwind, Trick Room, estados con inmunidades y residuales, boosts en el árbol (estado, secundarios, `selfBoosts`, Parting Shot) con la tabla ajustada por etapas / quemadura / clima, Intimidate y clima al entrar, clima por movimiento, pivotes. Uso en el meta: Tailwind 213, Parting Shot 129, Trick Room 96, Icy Wind 17, Snarl 20. `MCTS_PUCT_VERSION=5`. Exploración: 64,2 % contra joint, +4,0 pts sobre v4 (diagnóstico §6.5). **Confirmada el 21-11-2026** (protocolo §5.4, C7+C8, 34/34 semillas): > greedy_joint 65,52 % [64,68; 66,34], > expectimax 63,60 % [62,72; 64,44], > greedy_solo 66,80 % [65,96; 67,57], y frente a v4 McNemar +4,48 pts (1535 frente a 1080, p = 6·10⁻¹⁹); por la regla del pre-registro pasa a ser el valor por defecto. |
| **4.0** (confirmada; por defecto hasta v5) | 2026-11-20 | Fase 3b (§4.3): Protect y variantes con \((1/3)^n\), Feint, Wide Guard, Fake Out / First Impression con primer turno y retroceso, Follow Me / Rage Powder, Helping Hand; puntuación de jugadas de apoyo por amenaza evitada. Uso en el meta (1800 sets): Protect 1182, Fake Out 459, Helping Hand 73, Wide Guard 57, Follow Me 35, Rage Powder 29. `MCTS_PUCT_VERSION=4`. **Confirmada el 21-11-2026** (protocolo §5.3, C5+C6, 34/34 semillas): > greedy_joint 62,61 % [61,80; 63,41], > expectimax 61,18 % [60,34; 62,11], > greedy_solo 64,30 % [63,47; 65,08], y frente a v3.1 McNemar +5,09 pts (1582 frente a 1066, p = 10⁻²³); por la regla del pre-registro pasa a ser el valor por defecto. |
| **3.1.1** | 2026-11-20 | Sin cambios de decisión: S3 pide a la calculadora con `persist=False` (solo caché en memoria). La caché en disco `calc_cache.sqlite` no tiene límite y la clave lleva el HP actual: con S3 llegó a 386 GB en un día y llenó el disco del servidor. Confirmación de v3.1 (protocolo §5.2): superioridad práctica sobre joint (56,3 %), expectimax (56,3 %) y solo (59,6 %). |
| **3.1** | 2026-11-20 | **Preview greedy por defecto** (tipado + BST, el de solo/joint/expectimax), decisión del usuario: `team_preview_nash` nunca se entrenó (solo un checkpoint de smoke con etiquetas mock) y G caía al orden fijo `/team 1234` (−4 pts medidos en v2). `MCTS_PREVIEW=nash` vuelve al preview de familia. No se usa el preview PPO (sin componentes RL en G). Combate idéntico a v3.0. |
| **3.0** | 2026-11-20 | Fase 3a del plan: modelo **S3** (`s3.py`) en lugar de S2 para G. Tabla de daño real por decisión (un lote a la calculadora, mismos pedidos que P5), stats del rival por prior meta LOO, orden por prioridad y velocidad efectiva (Tailwind, Trick Room), redirección de golpes individuales, área ×0,75 con fuego amigo, precisión por movimiento; prior, rival del árbol y rollout puntúan con la tabla (+ bono de KO). Dex de Champions (el antiguo no tenía 21 movimientos del meta). `MCTS_PUCT_VERSION=2` reproduce v2. Motivo: v2 con meseta en 37–38 % contra joint aunque se suban las sims (diagnóstico §5.2). |
| **2.0** | 2026-11-19 | Diagnóstico de v1 ([`mcts_puct_diagnostico.md`](mcts_puct_diagnostico.md)): raíz de ~660 joints explorada en orden de id, hijo fijo por arista (no era open-loop), rival uniforme, chip de 8–18 % en movimientos de estado, ~26 s/decisión. v2: raíz desde `valid_orders`, poda por objetivo, prior softmax(daño)+ε, FPU, open-loop real con remuestreo, rival softmax(daño), hoja = 2 turnos + material, reemplazo tras KO, banca rival creíble, movimientos alineados con `available_moves`, contadores de fallback/excepción. ~2 ms/sim. `MCTS_PUCT_VERSION=1` reproduce v1. S2 sin cambios por defecto (MCTS–Nash intacto). El barrido de `logs/mcts_sims_sweep_mcts_puct.*` era `dry_run` (datos falsos) y no cuenta. |
| **1.7** | 2026-11-15 | Objetivo aliado del slot 0 corregido en el adaptador de órdenes (−2; antes −1 = sí mismo, rechazado por Showdown). Sin cambios de obs. |
| **1.6** | 2026-11-14 | ActionSpec v2 (40 ids por slot, mega como flag de cada movimiento; joint 1600) y máscaras desde `valid_orders`. |
| **1.5** | 2026-10-11 | Inferencia bajo Preview Nash congelado (`frozen_preview`). |
| **1.4** | 2026-10-11 | §Q/N pedagógico; meseta \(N_{\rm sim}\); CLI `run_mcts_sims_sweep`; rival/seed/criterio; **tracking** history/progress de cambios de N_sim. |
| **1.3** | 2026-10-11 | PUCT, lifecycle, fallos, evaluación, code map. |
| **1.2** | 2026-10-11 | Dual-audience; pairing Nash-preview. |
| **1.1** | 2026-10-10 | Lectura pedagógica. |
| **1.0** | 2026-10-10 | Contrato open-loop PUCT. |
