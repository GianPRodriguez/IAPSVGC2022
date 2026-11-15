# Representación de entrada y salida de los agentes RL: cambios y fundamento teórico

**Documento metodológico** para uso académico (tesis / trabajo de grado).
**Alcance:** observación, espacio de acciones, máscaras, arquitectura de entrada y aumentación de
datos de los agentes con aprendizaje (E–K): Team Preview (Masked PPO, Payoff Nash, QD-PGA) y combate
(DDQN, PPO, QD-PPO, QD-PGA, QD-PPGA). MCTS (G, H) comparte el espacio de acciones.
**Audiencia:** lectores con formación en informática o aprendizaje automático, sin conocimiento previo
de *Pokémon* competitivo.
**Commits:** `791e5b2` (preview v2, evaluación pareada, logs), `a000576` (combate v2, acciones v2,
máscaras), `dbb188b` (codificador estructurado, simetrías, objetivo aliado).
**Contratos normativos:** [`rl_env_spec.md`](rl_env_spec.md) v1.20 (§2, §3, §3.c, §3.d) y los specs de
cada agente. Este documento explica **por qué**; los specs dicen **qué**.

---

## Resumen

La versión anterior de la entrada de las redes descartaba información disponible (identidades
codificadas con hash, segmento de creencias en cero, sin estadísticas propias ni hoja rival) y su
espacio de acciones no podía expresar todas las jugadas legales (la Mega Evolución solo con el primer
movimiento; el slot 1 no podía apuntar a su aliado). Además, la máscara de entrenamiento era más
permisiva que la de inferencia, lo que contaminaba el *bootstrap* de DDQN.

Los cambios siguen cuatro principios, cada uno con un fundamento distinto:

| Principio | Cambio | Fundamento |
|---|---|---|
| **Toda la información disponible** | Obs OTS completa (preview 5676, combate 6332), historial, creencias explícitas | En un POMDP la política óptima depende del estado de creencia; omitir información produce *aliasing* de estados |
| **Política completa y consistente** | 40 acciones por slot (mega × movimiento × objetivo), máscara = órdenes válidas de Showdown | Una parametrización que excluye acciones legales restringe la clase de políticas; máscaras inconsistentes sesgan el *target* de Q-learning |
| **Sesgo inductivo correcto** | Codificador con pesos compartidos por bloque | Compartir pesos entre entidades del mismo tipo reduce parámetros y multiplica la señal por muestra (como la convolución o *Deep Sets*) |
| **Explotar simetrías del juego** | Aumentación con el grupo \(\mathbb{Z}_2^3\) | Si el MDP es invariante a una transformación, también lo son \(Q^*\) y \(\pi^*\) (homomorfismos de MDP) |

---

## 1. Contexto del problema

### 1.1. El juego como POMDP

El combate de dobles del formato Reg M-C es un juego estocástico de dos jugadores con información
imperfecta. Cada jugador lleva 6 Pokémon, ve la hoja pública del rival (*Open Team Sheet*, OTS:
especie, objeto, habilidad y movimientos) y elige 4. Lo que **no** se ve:

- los *Stat Points* (SP, 0–32 por estadística) del rival, que fijan sus estadísticas reales;
- cuáles 4 de los 6 trajo el rival, hasta que aparecen en el campo;
- la acción simultánea del rival en cada turno.

Desde el punto de vista de un agente, el problema es un POMDP \((\mathcal{S}, \mathcal{A}, P, R, \Omega, O)\).
La política óptima no es función de la observación instantánea \(o_t\) sino del **estado de creencia**
\(b_t(s) = \Pr(s_t = s \mid o_{1:t}, a_{1:t-1})\) (Kaelbling, Littman y Cassandra, 1998). Una red que
recibe \(\phi(o_t)\) solo puede aproximar \(\pi^*(b_t)\) en la medida en que \(\phi(o_t)\) sea (casi)
suficiente para \(b_t\).

### 1.2. Estado anterior

| Aspecto | v1 | Problema |
|---|---|---|
| Obs de combate | 1024D: especie, tipos e item como *hash* md5; segmento *belief* de 256 en cero para DDQN/PPO | El hash no preserva similitud (dos especies parecidas quedan lejos) y colisiona; un segmento constante no aporta nada |
| Estadísticas | Ausentes (ni las propias, que se conocen exactas) | La red no puede saber quién es más rápido ni cuánto daño hace |
| Hoja rival | Ausente | Se desperdicia la información OTS, que es pública |
| Obs de preview | 512D, solo especies | No distingue un mismo Pokémon con distinto objeto o set |
| Acciones | 24 por slot; id 21 = Mega + movimiento 0 | La Mega solo se podía usar con el primer movimiento |
| Máscara de entrenamiento | Permitía pasar, objetivos y cambios inválidos | Showdown rechazaba la orden, se ejecutaba una aleatoria y DDQN hacía *bootstrap* sobre Q nunca entrenados |
| Material del rival | Los no vistos contaban como debilitados | Saltos espurios en la recompensa densa |

---

## 2. Principio 1: dar a la red toda la información disponible

### 2.1. Por qué importa (aliasing y suficiencia)

Si dos estados de creencia \(b \neq b'\) con acciones óptimas distintas producen la misma entrada
\(\phi(o)=\phi(o')\), ninguna red puede actuar bien en ambos: es el *perceptual aliasing* (Whitehead y
Ballard, 1991). La pérdida es irreducible por más datos o capacidad que se agreguen. Por eso el criterio
de diseño fue: **toda variable que un jugador humano conoce al decidir debe estar en la entrada, en una
forma aprendible**:

- identidades como *one-hot* sobre un vocabulario congelado (no hash);
- magnitudes normalizadas a \([0,1]\) o \([-1,1]\);
- sin segmentos constantes.

### 2.2. Bloque OTS por Pokémon (compartido por preview y combate)

Cada Pokémon se codifica con 464 valores, derivados del *dex* de Showdown congelado
(`API/JSONs/champions_dex_snapshot.json`):

| Parte | Dim | Contenido |
|---|---|---|
| Semántica | 193 | Estadísticas base de la forma efectiva (incluida la Mega si el objeto es su piedra), rango de estadísticas a nivel 50, tipos, habilidad, objeto, movimientos por tipo y potencia, utilidades (Protect, Fake Out, Tailwind, redirección, etc.), roles |
| Identidad | 105 + 76 + 90 | *One-hot* de especie, objeto y habilidad |

**Estadísticas: exactas o como intervalo.** Las propias se conocen exactas (SP y naturaleza del
*teambuilder*). Las del rival dependen de SP ocultos; con \(\mathrm{SP}\in[0,32]\), cada estadística
\(x\) queda en un intervalo \([x_{\min}, x_{\max}]\) que se codifica por sus dos extremos. Así la red
recibe la **incertidumbre** explícitamente, no un valor puntual inventado. Si la inferencia de SP
estrecha el rango (`OBS_STAT_INFERENCE=1`), se usan sus cotas. Hoy está apagada por defecto, porque en
partidas reales no poda nada y su prior comparte equipos con la evaluación (ver
[`resultados_informacion_sp.md`](resultados_informacion_sp.md)).

### 2.3. Observación de Team Preview (5676)

\[
D_{\text{preview}} = \underbrace{6\cdot 464}_{\text{propios}} + \underbrace{6\cdot 464}_{\text{rival}} + \underbrace{3\cdot 36}_{\text{matrices}} = 5676
\]

Las tres matrices \(6\times 6\) contienen el daño esperado propio→rival, el daño rival→propio y la
relación de velocidad.

**Features de dominio precalculadas.** El daño y la velocidad son funciones fijas y conocidas de
estadísticas, tipos, objetos y habilidades. Aprenderlas desde cero exigiría muchos datos; calcularlas e
incluirlas es un caso de *feature engineering* que reduce la complejidad muestral sin restringir la
clase de funciones, porque la red sigue recibiendo las variables crudas.

**Orden canónico del rival.** La hoja rival es un conjunto: su orden no tiene significado. Si la entrada
dependiera del orden de llegada, la red tendría que aprender la misma respuesta para las \(6!\)
permutaciones. Ordenar por especie y objeto (*canonicalización*) elimina esa variación irrelevante. El
equipo propio, en cambio, mantiene el orden del *teambuilder*, porque la acción (qué 4 y en qué orden)
se indexa por esa posición.

### 2.4. Observación de combate (6332)

\[
D_{\text{combate}} = \underbrace{10\cdot(464+58)}_{\text{Pokémon}} + \underbrace{8\cdot 81}_{\text{mov. propios}} + \underbrace{8\cdot 43}_{\text{mov. rivales}} + \underbrace{72}_{\text{matrices}} + \underbrace{48}_{\text{campo}} = 6332
\]

Cada uno de los 10 bloques (4 propios: activos a, b y 2 reservas; 6 del rival) es el bloque OTS más 58
variables dinámicas: HP, estado y su contador, *boosts*, 13 volátiles, *perish*, contador de Protect,
si ya megaevolucionó, si conserva el objeto, si puede megaevolucionar, incertidumbre de SP por
estadística, historial y pertenencia al equipo traído.

#### Creencia sobre el equipo traído: \(P(\text{traído}) = \frac{4-k}{6-k}\)

El rival trae 4 de sus 6. Sin información sobre su elección, la hipótesis de máxima entropía es que
todos los subconjuntos de 4 son igualmente probables. Si ya se vieron \(k\) Pokémon distintos (todos
traídos con certeza), quedan \(4-k\) plazas repartidas entre los \(6-k\) no vistos. Por simetría entre
los no vistos, cada uno ocupa una plaza con probabilidad

\[
P(\text{traído} \mid k) = \frac{4-k}{6-k}, \qquad k \in \{0,\dots,4\}.
\]

Con \(k=4\) la probabilidad es 0: los 2 restantes **no pueden aparecer**. Se marcan como descartados, su
bloque queda en cero y salen de las matrices. Antes contaminaban los cálculos de amenaza con Pokémon que
no estaban en el combate.

#### Historial: aproximar la propiedad de Markov

La observación instantánea no es markoviana en este juego. Varias reglas dependen del pasado:

- *Choice lock*: con ciertos objetos, se repite el último movimiento;
- *Encore*: se repite el movimiento forzado;
- Protect consecutivo: la probabilidad de éxito cae;
- *Fake Out*: solo funciona en el primer turno en el campo.

Una red sin memoria no puede distinguir "acaba de entrar" de "lleva 3 turnos". Hay dos soluciones
clásicas: una red recurrente (LSTM) que aprende su propio resumen, o un **resumen manual
suficiente**. Se eligió lo segundo, más barato y más fácil de verificar. El historial se extrae del
protocolo de Showdown (`showdown/rl/battle_history.py`) e incluye:

- por Pokémon: turnos activo, recencia y tipo del último movimiento, movimiento bloqueado, HP perdido el
  último turno, veces que entró y movimientos desde que entró;
- por bando: un resumen del último turno (Protect, cambio, debilitado, Fake Out, ataque en área).

#### Movimientos alineados con la acción

Los 4 movimientos de cada activo propio se colocan **en el mismo orden que el índice de movimiento de
la acción**. Así, la feature del movimiento \(i\) y la salida de la acción \(i\) quedan en
correspondencia fija. La red aprende "valor de usar el movimiento \(i\)" a partir de las propiedades de
ese movimiento, no de una lista desordenada. Cada movimiento incluye:

- daño esperado contra cada rival y contra el aliado;
- **KO posible** (con la tirada de daño máxima) y **KO garantizado** (con la mínima);
- si la prioridad queda anulada (*Psychic Terrain*, *Armor Tail*, *Prankster* contra Siniestro) y si
  el movimiento sería redirigido (*Storm Drain*, *Lightning Rod*);
- si fue el último usado y si está bloqueado.

Los KO posible y garantizado separan el riesgo del valor esperado, una distinción que importa con
información incompleta.

### 2.5. Recompensa: material del rival sin saltos espurios

La recompensa densa (Candidate 2) depende de la diferencia de material (HP y bajas). Si los Pokémon no
vistos del rival contaban como debilitados, el potencial \(\Phi(s)\) cambiaba **sin que el agente
hiciera nada** cuando aparecía uno nuevo. Ese salto es ruido en la recompensa. Ahora los no vistos
cuentan como sanos, y el equipo propio se filtra a los 4 traídos. El potencial depende solo de
información observada de forma estable.

---

## 3. Principio 2: política completa y máscaras consistentes

### 3.1. Espacio de acciones v2 (40 por slot, 1600 conjuntas)

| Ids | Acción |
|---|---|
| 0–15 | movimiento \(i\in\{0..3\}\) × objetivo \(t\in\{\)rival a, rival b, aliado, sin objetivo/sí mismo\(\}\) |
| 16–31 | **Mega Evolución + el mismo** movimiento × objetivo |
| 32–35 | cambio a la reserva \(k\) |
| 36 | pasar |
| 37 | Tera (siempre enmascarada: Tera está desactivada en Reg M-C) |
| 38–39 | reservadas |

En Showdown, megaevolucionar es un *flag* de la orden de movimiento: se megaevoluciona **y** se usa un
movimiento. Nunca va con un cambio, y solo una vez por bando. En v1 existía un único id "Mega +
movimiento 0", así que la clase de políticas representables \(\Pi_{v1}\) era un subconjunto estricto de
las políticas legales. Si la acción óptima era "megaevolucionar y usar el movimiento 2", ninguna red
podía expresarla. Ahora el agente aprende **cuándo** megaevolucionar y **con qué** movimiento.

Las acciones muertas (Tera, reservadas, objetivos imposibles) se enmascaran con logit \(-10^{9}\). Su
probabilidad es exactamente 0 y su gradiente también, así que el único costo es tener unas pocas salidas
sin uso. A cambio, el *layout* se mantiene estable para todos los agentes.

### 3.2. Objetivos relativos al slot (corrección)

En Showdown, las posiciones propias son −1 (slot 1) y −2 (slot 2). El adaptador enviaba "aliado" como −1
para ambos slots, así que para el slot 1 eso significaba **apuntarse a sí mismo**. Showdown rechazaba la
orden, el filtro la quitaba de la máscara y, en la práctica, el slot 1 nunca podía apuntar a su aliado
(*Pollen Puff* para curar, *Instruct*, *Skill Swap*, etc.), mientras que el slot 2 sí podía. Ahora
"aliado" es −2 para el slot 1 y −1 para el slot 2. Además de devolver acciones legales, esto hace que el
espacio de acciones sea simétrico entre los dos slots, condición necesaria para la aumentación de la §5.

### 3.3. Máscara de entrenamiento = máscara de inferencia

La máscara se construye ahora desde `battle.valid_orders`, las órdenes que Showdown declara legales. Hay
dos efectos teóricos.

**1. Transiciones corruptas.** Antes, una acción ilegal se reemplazaba por otra aleatoria
(*fallback*). La transición guardada \((s, a, r, s')\) registraba \(a\), pero el entorno había
ejecutado otra acción \(\tilde a\). Para un método *off-policy* como DDQN, esa tupla es falsa: atribuye a
\(a\) la consecuencia de \(\tilde a\).

**2. Sesgo de maximización sobre acciones nunca entrenadas.** El *target* de Double DQN es

\[
y = r + \gamma\, Q_{\bar\theta}\!\big(s', \arg\max_{a' \in M(s')} Q_\theta(s', a')\big).
\]

Si la máscara \(M(s')\) incluye acciones que en la práctica nunca se ejecutan, sus valores \(Q_\theta\)
nunca se corrigen. Quedan en su valor de inicialización más el ruido de la generalización, y el
\(\arg\max\) tiende a elegir justamente esos errores positivos. Es el sesgo de sobreestimación por
maximización (Thrun y Schwartz, 1993; van Hasselt, 2010), agravado porque el error **no** disminuye con
más datos. Usar la misma máscara exacta en entrenamiento e inferencia elimina ambos efectos.

---

## 4. Principio 3: sesgo inductivo con pesos compartidos

### 4.1. El problema del MLP plano

Un MLP cuya primera capa recibe los 6332 valores concatenados asigna un vector de pesos distinto a cada
posición. "Incineroar en el bloque 1" e "Incineroar en el bloque 3" activan pesos disjuntos, y lo que la
red aprende sobre un Pokémon en una posición no se transfiere a otra. Con \(n\) bloques del mismo tipo,
la misma función se aprende \(n\) veces, cada una con \(1/n\) de los datos relevantes.

### 4.2. Codificador estructurado

`showdown/rl/encoders.py` aplica **un mismo** sub-MLP \(f_\psi\) a cada bloque de un grupo y concatena
los resultados **en orden**:

\[
\mathrm{enc}(o) = \big[f_\psi(m_1), \dots, f_\psi(m_{10}),\ g_\eta(u_1), \dots, g_\eta(u_8),\ h_\zeta(v_1), \dots, h_\zeta(v_8),\ \text{matrices},\ \text{campo}\big]
\]

| Obs | Grupos (MLP compartido por grupo) | Crudo | Salida |
|---|---|---|---|
| Combate (6332) | 10 Pokémon `522→128→64` · 8 movimientos propios `81→64→32` · 8 rivales `43→64→32` | matrices + campo (120) | 1272 |
| Preview (5676) | 12 Pokémon `464→128→64` | matrices (108) | 876 |

Después del codificador, el torso y las cabezas no cambian (`→512→512` en combate, `→256→256` en
preview).

**Fundamento.** Es el mismo sesgo inductivo que hace eficientes a las redes convolucionales (pesos
compartidos entre posiciones espaciales) y a *Deep Sets* (Zaheer et al., 2017), que aplican \(\phi\) a cada
elemento de un conjunto. Hay dos consecuencias:

1. **Más señal por muestra.** Cada transición actualiza \(f_\psi\) con 10 ejemplos de Pokémon, no con
   uno. El gradiente de \(\psi\) es la suma de las contribuciones de todas las posiciones.
2. **Menos parámetros**, lo que reduce la varianza del estimador y el sobreajuste:

| Red | MLP plano | Codificador estructurado |
|---|---|---|
| DDQN | 4,33 M | 1,82 M |
| PPO (y actores QD) | 3,55 M | 1,05 M |
| Preview PPO | 1,61 M | 0,45 M |
| Preview Nash | 1,58 M | 0,42 M |

**Por qué concatenar y no agregar (*pooling*).** *Deep Sets* suma los \(\phi(x_i)\) para obtener
invariancia a permutaciones. Aquí eso sería incorrecto: las acciones son **posicionales** ("el slot 1
ataca al rival b", "cambiar a la reserva 2"), así que la red debe saber quién está en cada lugar.
Concatenar en orden conserva esa información. El costo es que la red, por sí sola, **no** es equivariante
a intercambiar posiciones; eso se le enseña con la aumentación de la §5.

**Para QD.** En QD-PGA/PPGA el codificador forma parte del genotipo del actor: la mutación gaussiana
\(\sigma=0{,}02\) también lo perturba. Como un mismo peso se aplica a 10 bloques, una perturbación en
\(\psi\) tiene un efecto coherente sobre todos los Pokémon, en vez de alterar posiciones de forma
independiente.

**Ablación:** `RL_OBS_ENCODER=flat` restaura el MLP plano. Los *checkpoints* de una arquitectura no
cargan en la otra.

---

## 5. Principio 4: explotar las simetrías del juego

### 5.1. Simetrías de un MDP

Una transformación \(g\) (sobre estados y acciones) es una **simetría** del MDP si conserva dinámica y
recompensa:

\[
P(g s' \mid g s, g a) = P(s' \mid s, a), \qquad R(g s, g a) = R(s, a).
\]

En ese caso \(Q^*(gs, ga) = Q^*(s,a)\) y \(\pi^*(ga \mid gs) = \pi^*(a \mid s)\). Es el marco de los
homomorfismos de MDP (Ravindran y Barto, 2001; van der Pol et al., 2020). Conocer las simetrías permite
generar, a partir de cada experiencia, otras igual de válidas sin jugar más partidas.

### 5.2. Las simetrías de un combate de dobles

En dobles todos los Pokémon en campo son adyacentes, así que la posición izquierda/derecha no cambia la
mecánica. El orden de las 2 reservas propias solo fija el índice de la acción de cambio. Se obtiene el
grupo \(\mathbb{Z}_2^3\), 8 elementos, todos involuciones que conmutan:

| Generador | Sobre la obs | Sobre la acción |
|---|---|---|
| \(\sigma_o\): activos propios a↔b | bloques, movimientos, filas de las matrices, cambio forzado por slot, daño rival sobre "nuestro activo 0/1" | intercambia slot 1 y slot 2 (aliado y "sí mismo" son relativos al slot, por eso no cambian) |
| \(\sigma_f\): activos rivales a↔b | bloques rivales, movimientos rivales, columnas de las matrices, features "contra rival 0/1" de cada movimiento propio | objetivo rival a ↔ rival b, en movimientos y en Mega |
| \(\sigma_b\): reservas 1↔2 | bloques y filas de las matrices | cambio 32 ↔ 33 |

Para cada \(g\), la obs se transforma con una permutación de índices precalculada
(\(o' = o[P_g]\)) y las acciones y máscaras con otra (`showdown/rl/symmetry.py`).

**Restricción de \(\sigma_b\).** La codificación canónica pone primero la reserva viva. Si una reserva
está debilitada, el estado permutado (debilitada primero) **nunca ocurre** en el juego, y entrenar con
él gastaría capacidad en estados imposibles. Por eso \(\sigma_b\) solo se aplica cuando ambas reservas
están presentes y vivas.

**Verificación.** Las permutaciones no se derivaron a mano y luego se supusieron correctas. Un test
construye una batalla, construye la batalla **espejada de verdad** (intercambiando los objetos en el
snapshot) y comprueba que su observación sea exactamente la observación original permutada, para los 8
elementos del grupo (`tests/test_rl_symmetry.py`). También verifica que cada elemento sea una
involución, que el grupo conmute y que las acciones legales sigan siéndolo tras la transformación.

### 5.3. Cómo se usa en cada algoritmo

**DDQN (*off-policy*): aumentación exacta.** A cada transición muestreada del *replay* se le aplica un
elemento aleatorio \(g\) a \((s, a)\) y otro independiente \(h\) a \((s', M(s'))\). Como \(Q^*\) es
invariante, el *target* sigue siendo correcto. Es la idea de DrQ y RAD (Kostrikov et al., 2020; Laskin
et al., 2020), con simetrías exactas en lugar de aumentaciones aproximadas de imagen. La máscara conjunta
es simétrica bajo \(\sigma_o\), porque las únicas restricciones entre slots (una sola Mega, no cambiar
ambos al mismo Pokémon) son simétricas.

**Crítico compartido de QD-PGA/PPGA: aumentación exacta.** Igual que DDQN, sobre \(s\) y \(s'\) en la
actualización TD(0), porque \(V^*\) también es invariante.

**PPO y actores QD (*on-policy*): consistencia, no datos nuevos.** El objetivo recortado de PPO usa el
cociente \(\pi_\theta(a\mid s)/\pi_{\text{old}}(a\mid s)\), que supone que \(a\) se muestreó de
\(\pi_{\text{old}}(\cdot\mid s)\). Una muestra permutada \((gs, ga)\) **no** se muestreó de
\(\pi_{\text{old}}(\cdot\mid gs)\), así que usarla como dato *on-policy* sesga el gradiente. Siguiendo
DrAC (Raileanu et al., 2021), la simetría entra como regularización:

\[
\mathcal{L} = \mathcal{L}_{\text{PPO}} + c_s\,\mathbb{E}\Big[\big(\log\pi_\theta(ga \mid gs) - \mathrm{sg}[\log\pi_\theta(a\mid s)]\big)^2\Big] + c_v\,\mathbb{E}\big[(V_\theta(gs) - \hat R)^2\big],
\]

donde \(\mathrm{sg}\) detiene el gradiente, \(c_s = 0{,}1\) (`sym_coef`) y \(c_v\) es el coeficiente de
valor de PPO. El primer término empuja a la política a ser equivariante. El segundo es exacto, porque el
retorno \(\hat R\) es el mismo para estados simétricos.

**Detalle de la política autorregresiva.** PPO decide primero el slot 1 y después el slot 2,
condicionado. \(\sigma_o\) invierte ese orden: el nuevo slot 1 necesita la máscara **incondicional** del
slot 2, que el *buffer* no guarda. La condición solo quita la Mega (si el slot 1 megaevolucionó) y el
cambio ya tomado, así que cuando \(a_1\) no es Mega ni cambio, la máscara guardada coincide exactamente
con la incondicional. \(\sigma_o\) se aplica solo a esas filas, que son la mayoría. Así el término de
consistencia nunca usa una máscara aproximada.

### 5.4. Relación con el codificador

El codificador comparte pesos **dentro** de cada bloque, pero su concatenación posicional no es
equivariante. La aumentación enseña la equivariancia que falta. Juntos se aproximan a una red
equivariante por construcción (van der Pol et al., 2020), sin rediseñar las cabezas de salida de 6 redes.
Esa alternativa queda como trabajo futuro.

**Ablación:** `RL_SYM_AUG=0` apaga la aumentación. Se registran `sym_frac_our/foe/bench` (proporción de
cada generador aplicada), `sym_pi_loss` y `sym_v_loss`.

---

## 6. Evaluación pareada del Team Preview

### 6.1. Problema

El *checkpoint* y el ajuste de hiperparámetros del preview PPO se elegían por \(-\mathcal{L}_{\text{PG}}\).
La pérdida de política no mide desempeño: depende de ventajas normalizadas por lote y del recorte de PPO,
y no es monótona respecto del retorno esperado.

### 6.2. Solución: números aleatorios comunes

Ahora la puntuación es `MEAN_WR`, la tasa de victoria sobre un conjunto **fijo** de contextos (equipos y
semillas), comparada con una línea base evaluada **en los mismos contextos**
(`showdown/rl/preview_eval.py`). Para la diferencia \(\Delta = X - Y\) entre política y línea base,

\[
\mathrm{Var}(\Delta) = \mathrm{Var}(X) + \mathrm{Var}(Y) - 2\,\mathrm{Cov}(X, Y).
\]

Evaluar ambas en los mismos contextos induce \(\mathrm{Cov}(X,Y) > 0\) (los emparejamientos difíciles lo
son para ambas) y reduce la varianza de \(\Delta\). Es la técnica de números aleatorios comunes (*common
random numbers*). Se reportan `MEAN_WR`, `BASE_WR` y `DELTA_WR ± SE`. El generador del entorno se guarda
y se restaura, así que evaluar no altera el entrenamiento.

### 6.3. Registro para la tesis

Todos los entrenamientos escriben `train_metrics.jsonl` (`showdown/rl/train_metrics.py`) con:
- episodios, ventanas de tasa de victoria por rival, KOs y mezcla de acciones;
- en preview: diagnósticos de PPO (KL aproximada, fracción recortada, varianza explicada, acciones
  efectivas) y estrategia por rol (tasa de *lead*, tasa de victoria con y sin cada rol).

Con esto se puede graficar no solo si el agente mejora, sino **qué aprende** y qué lo frena.

---

## 7. Validación y estado

| Verificación | Estado |
|---|---|
| Permutaciones de simetría contra batallas espejadas (8 elementos) | Pasa (laptop, tests acotados) |
| Formas, número de parámetros y ablación `flat` de las 6 clases de red | Pasa |
| Actualizaciones de DDQN y PPO con simetría activa | Pasa |
| Máscara desde `valid_orders`, historial, bloques no vacíos | Pasa con datos sintéticos |
| Suite completa en el servidor | **Pendiente** |
| Corrida corta en vivo contra Showdown (tasa de *fallback* ≈ 0, historial correcto, alineación acción–movimiento) | **Pendiente**: es lo más importante antes de entrenar, porque un desalineamiento afectaría a 8 de los 11 agentes |
| `run_preview_value_probe` (n = 3000) | **Pendiente** |

---

## 8. Limitaciones y trabajo futuro

- **Modelo de daño aproximado.** Las features de daño usan la fórmula de nivel 50 con estadísticas,
  STAB, tipos, inmunidades por habilidad, objetos, *boosts*, clima, terreno, pantallas y quemadura. No
  modelan críticos ni habilidades poco comunes. La red recibe también las variables crudas, así que puede
  corregir errores sistemáticos.
- **Historial resumido.** Ventana corta y contadores acotados. Una red recurrente podría capturar
  patrones más largos del rival (por ejemplo, su frecuencia de Protect), a mayor costo.
- **Inferencia de SP.** El canal existe (`OBS_STAT_INFERENCE`), pero hoy no estrecha rangos en partidas
  reales, y su prior se construye con los mismos equipos de la evaluación. Debe mantenerse apagado
  mientras eso no se corrija, para no filtrar información.
- **Simetrías del preview no explotadas.** El espacio de 360 acciones (\(P(6,4)\)) contiene redundancia:
  el orden de los dos *leads* y el de las dos reservas no cambia la mecánica, así que hay
  \(\binom{6}{2}\binom{4}{2} = 90\) formaciones distintas, cada una con 4 codificaciones. También se podría
  aumentar permutando el equipo propio. Ambas son extensiones naturales de la §5.
- **Equivariancia aprendida, no garantizada.** La aumentación la induce, pero la arquitectura no la
  impone. Las redes equivariantes por construcción son la alternativa rigurosa.
- **Consistencia en PPO.** El término de la §5.3 iguala la probabilidad de la acción muestreada en su
  órbita, no la distribución completa. Es un estimador por muestras de la regularización de DrAC.

---

## 9. Variables de ablación (para reportar en la tesis)

| Variable | Valores | Efecto |
|---|---|---|
| `RL_OBS_ENCODER` | `shared` (defecto) / `flat` | Codificador estructurado vs. MLP plano |
| `RL_SYM_AUG` | `1` (defecto) / `0` | Aumentación y consistencia por simetría |
| `OBS_STAT_INFERENCE` | `0` (defecto) / `1` | Rangos de SP inferidos en la obs (entrenamiento **e** inferencia juntos) |
| `PS_RL_FOE_VALIDATED` | `1` (defecto) / `0` | Rivales heurísticos validados vs. crudos (rl_env_spec v1.19) |

Un diseño factorial mínimo con `RL_OBS_ENCODER` × `RL_SYM_AUG`, a igual presupuesto de pasos, permite
atribuir la mejora de cada componente.

---

## Referencias

- Kaelbling, L. P., Littman, M. L., y Cassandra, A. R. (1998). Planning and acting in partially
  observable stochastic domains. *Artificial Intelligence*, 101(1–2), 99–134.
- Kostrikov, I., Yarats, D., y Fergus, R. (2020). Image augmentation is all you need: Regularizing deep
  reinforcement learning from pixels. *arXiv:2004.13649* (DrQ).
- Laskin, M., Lee, K., Stooke, A., Pinto, L., Abbeel, P., y Srinivas, A. (2020). Reinforcement learning
  with augmented data. *NeurIPS* (RAD).
- Raileanu, R., Goldstein, M., Yarats, D., Kostrikov, I., y Fergus, R. (2021). Automatic data
  augmentation for generalization in reinforcement learning. *NeurIPS* (DrAC).
- Ravindran, B., y Barto, A. G. (2001). Symmetries and model minimization in Markov decision processes.
  *Technical Report 01-43*, University of Massachusetts.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., y Klimov, O. (2017). Proximal policy optimization
  algorithms. *arXiv:1707.06347*.
- Thrun, S., y Schwartz, A. (1993). Issues in using function approximation for reinforcement learning.
  *Proceedings of the Connectionist Models Summer School*.
- van der Pol, E., Worrall, D., van Hoof, H., Oliehoek, F., y Welling, M. (2020). MDP homomorphic
  networks: Group symmetries in reinforcement learning. *NeurIPS*.
- van Hasselt, H. (2010). Double Q-learning. *NeurIPS*.
- van Hasselt, H., Guez, A., y Silver, D. (2016). Deep reinforcement learning with double Q-learning.
  *AAAI*.
- Whitehead, S. D., y Ballard, D. H. (1991). Learning to perceive and act by trial and error. *Machine
  Learning*, 7(1), 45–83.
- Zaheer, M., Kottur, S., Ravanbakhsh, S., Poczos, B., Salakhutdinov, R., y Smola, A. (2017). Deep sets.
  *NeurIPS*.
