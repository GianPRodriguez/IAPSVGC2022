# Payoff Nash — Team Preview Exclusive Spec (VGC Gen 9 Regulation M-C)

| Campo | Valor |
|-------|-------|
| **Documento** | Contrato exclusivo — Payoff Nash solo Team Preview |
| **Versión** | **1.2** (alineado con `docs/rl_env_spec.md` v1.15 §6.j) |
| **Estado** | **FROZEN** — implementado en `showdown/rl/team_preview_nash/` |
| **Algoritmo** | Value Net de bajo rango + equilibrio de Nash por LP |
| **Formato** | Gen 9 VGC **Regulation M-C**, OTS, Mega on, **Tera off** |
| **Código** | `showdown/rl/team_preview_nash/` y `team_preview_{env,obs,action}.py` |
| **CLI** | `python -m showdown.rl.team_preview_nash.train --seed N` |

Este agente no aprende directamente una policy con PPO. Aprende a estimar el
resultado de cada par de formaciones y resuelve ese juego para elegir.

## 1. Por qué existe

Una política que siempre escoge la formación de mayor media puede ser
explotable: el rival aprende su patrón y responde. Payoff Nash modela preview
como una decisión simultánea. Filas son nuestras 360 formaciones; columnas, las
360 rivales; cada celda estima qué tan favorable es ese cruce.

Una red aproxima la tabla completa y un solver busca una mezcla que maximice el
resultado garantizado frente a la mejor respuesta rival. Para quien aprende IA:
la red aporta predicción; la programación lineal aporta estrategia. Ninguna de
las dos, por separado, implementa el agente completo.

## 2. Glosario

| Término | Explicación |
|---------|-------------|
| Payoff | Resultado esperado de un par concreto de acciones |
| Value Net \(V_\theta\) | Red supervisada que estima la matriz de win-rate; no es una policy |
| \(\hat M_{ij}\) | Win-rate estimado de nuestra acción \(i\) contra la rival \(j\) |
| Juego suma cero | Transformación \(A=2\hat M-1\), con pagos en \([-1,1]\) |
| Estrategia mixta | Distribución de probabilidad sobre varias formaciones |
| Equilibrio de Nash | Par de mezclas donde ningún jugador mejora cambiando unilateralmente |
| \(p^\star,q^\star\) | Mezclas de equilibrio para fila y columna |
| \(v^\star\) | Valor garantizado estimado del juego |
| LP | Programa lineal que calcula la mezcla minimax |
| Low-rank | Factorización latente que construye 129 600 celdas con un matmul |
| Fictitious Play | Solver iterativo de respaldo si falla el LP |
| Cand. 1 | \(+1/-1/0\); para labels se mapea a \(1/0/0.5\) |

## 3. Algoritmo: Nash LP + Value Net low-rank

### 3.1 Aprender la matriz

1. `env.reset()` produce una observación OTS \(s\in\mathbb{R}^{512}\).
2. Se muestrean independientemente \(i,j\sim U\{0,\ldots,359\}\).
3. Ambas selecciones se decodifican a `/team`; el piloto congelado
   `greedy_joint` juega el BO1 y produce Cand. 1.
4. La etiqueta se expresa como win-rate:
   \[
   r=(R+1)/2\in\{0,0.5,1\}.
   \]
5. Un tronco `512→256→256` genera contexto \(h\). Cada acción tiene embedding
   de dimensión 32; dos cabezas producen vectores fila/columna de dimensión 64:
   \[
   \hat M_{ij}=\sigma\left(
   \langle\Phi_{\rm row}(h,e_i),\Phi_{\rm col}(h,e_j)\rangle\right).
   \]
6. BCE suave es la pérdida por defecto; MSE está soportada. Adam actualiza solo
   \(V_\theta\), nunca \(\pi_c\).

La factorización de bajo rango es importante: para inferencia se calculan 360
vectores fila y 360 columna, y un solo producto matricial forma la tabla
\(360\times360\). No se hacen 129 600 forwards independientes.

### 3.2 Resolver el juego

1. Transformar probabilidades a pagos: \(A=2\hat M-1\).
2. Resolver
   \[
   \max_p\min_q p^\top A q,
   \qquad v^\star=p^{\star\top}Aq^\star.
   \]
3. Usar primero `scipy.optimize.linprog`/HiGHS, con timeout `0.2 s`.
4. Si falla, usar Fictitious Play; como último recurso, softmax de la media de
   filas.
5. En batalla muestrear \(a\sim p^\star\). En evaluación determinista usar
   \(\arg\max p^\star\).

La mezcla no es ruido añadido: evita anunciar siempre la misma acción cuando
varias opciones se cubren mutuamente. Probabilidades menores que \(10^{-6}\)
se consideran fuera del soporte, aunque no necesariamente sean cero exacto.

## 4. MDP, acción y observación

### 4.1 Juego de un paso

Las OTS se conocen, pero la selección concreta rival no se observa al decidir.
Es un juego simultáneo de horizonte uno. Producción ejecuta un BO1 de máximo 40
turnos por label con `greedy_joint` congelado; self-play preview está off.

La matriz no debe llamarse antisimétrica. En un matchup general,
\(\hat M_{ij}\neq1-\hat M_{ji}\): intercambiar formaciones no equivale
necesariamente a intercambiar equipos, lado y representación.

### 4.2 Acción y máscara

Se reutiliza exactamente Flat-360:
\[
|A|=P(6,4)=360,\quad a\in\{0,\ldots,359\}.
\]
La biyección Lehmer y `/team` están en `team_preview_action.py`. La interfaz
`act(..., mask=...)` renormaliza \(p^\star\) tras eliminar acciones ilegales;
con pools completos la máscara es toda uno.

### 4.3 Observación compartida 512D

| Bloque | Dimensión | Contenido |
|--------|-----------|-----------|
| Equipo propio | 210 | Seis Pokémon × 35 features |
| Equipo rival | 210 | Mismo encoder OTS |
| Cobertura \(P_1\to P_2\) | 36 | Una sola matriz, calculada en `reset()` |
| Contexto | 56 | Clima/terreno cero y padding |

Compartir encoder con PPO y QD evita comparar algoritmos sobre información
distinta. No incluye Stellar/Tera: **Tera está desactivada en Regulation M-C**.

## 5. Entrenamiento detallado

### 5.1 Loop, datos y CLI

Por update se generan 2048 pares uniformes (64 en smoke). El dataset de cada
batch es `(obs[512], i, j, r)`. Se baraja y optimiza cuatro épocas en minibatches
de 128, con BCE suave por defecto.

El muestreo uniforme cubre la matriz sin depender del solver actual. Su coste es
gastar datos en pares irrelevantes; Nash-on-policy queda fuera de esta versión.
Congelar `greedy_joint` mantiene estable la función objetivo: si el piloto
cambiara mientras se ajusta la red, una misma celda adquiriría significados
distintos a lo largo del entrenamiento.

El CLI actual usa `TeamPreviewEnv(backend='mock')` y `pair_label`, por tanto
genera labels del proxy de cobertura. Producción live requiere
`TeamPreviewEnv(combat_fn=...)` con `greedy_joint` congelado. Debe registrarse el
origen de los labels.

```bat
python -m showdown.rl.team_preview_nash.train --seed 42 --updates 10
python -m showdown.rl.team_preview_nash.train --seed 42 --smoke
```

`--seed` es obligatorio; `--device` vale `cpu`.

### 5.2 Qué significa un checkpoint

El checkpoint de inferencia es el `state_dict` de `LowRankPayoffNet`: tronco,
embeddings fila/columna y ambas cabezas. También debe guardar versión del spec,
seed, loss, updates, pares vistos y origen `mock/live`. El solver no se aprende,
pero su método, timeout y fallback son parte del artefacto desplegable.

Para reanudar hacen falta además optimizador y RNG. El CLI devuelve el agente,
pero no serializa automáticamente; `DONE team_preview_nash` no significa que
exista un archivo de pesos.

## 6. Hiperparámetros

| Parámetro | Valor | Propósito | Cómo influye |
|-----------|-------|-----------|--------------|
| Tronco | `512→256→256`, ReLU | Codificar contexto OTS | Más capacidad modela matchups, con más riesgo de sobreajuste |
| Embedding acción \(d_a\) | `32` | Representar cada formación | Mayor dimensión separa acciones, pero aumenta parámetros |
| Latente \(d_{\rm lat}\) | `64` | Rango de la factorización | Mayor rango expresa matrices más complejas y cuesta más |
| Loss | `bce` por defecto; `mse` soportada | Ajustar \(\hat M\) a labels | BCE enfatiza probabilidad; MSE cambia geometría del error |
| Learning rate | `3e-4` | Paso de Adam | Alto acelera y puede desestabilizar |
| Gradient clip | `0.5` | Limitar gradientes | Reduce saltos por labels ruidosos |
| Pares/update | `2048`; smoke `64` | Datos por ajuste | Más pares cubren mejor la matriz |
| Minibatch | `128` | Tamaño de SGD | Equilibra ruido y memoria |
| Épocas | `4` | Reutilizar pares | Más ajuste por dato, con riesgo de sobreajuste |
| Timeout LP | `0.2 s` | Presupuesto del solver | Bajo reduce latencia y aumenta uso de fallback |
| Umbral soporte | `1e-6` | Métrica de acciones activas | Mayor umbral reporta mezclas más pequeñas |
| Seed | Obligatoria | Reproducibilidad | Fija inicialización, pares y muestreo de \(p^\star\) |

## 7. Inferencia y despliegue en batalla

1. Codificar ambas OTS en 512D.
2. Cargar el Value Net y construir \(\hat M\) por producto low-rank.
3. Calcular \(A=2\hat M-1\) y resolver \(p^\star,q^\star,v^\star\).
4. Aplicar máscara y renormalizar.
5. Muestrear \(p^\star\) en batalla; usar argmax para tablas deterministas.
6. Convertir a `/team ####` y transferir control al combatiente.

Se registran método del solver, \(v^\star\), entropía, soporte, tiempo y si hubo
fallback. Una estrategia obtenida tras fallback no debe reportarse como LP.

## 8. Pipeline de Conclusión

Nash-preview se usa con las dos familias MCTS:

- **MCTS + Nash-preview**
- **MCTS–Nash + Nash-preview**

El Value Net y el checkpoint del combatiente se congelan antes del round-robin.
El protocolo común está en
[`conclusion_eval_protocol.md`](conclusion_eval_protocol.md).

## 9. Limitaciones

- El equilibrio solo es tan bueno como la calibración de \(\hat M\).
- El muestreo uniforme puede dedicar muchos BO1 a celdas poco relevantes.
- Resolver un LP \(360\times360\) añade CPU y puede activar fallbacks.
- El modelo low-rank impone una restricción expresiva.
- El CLI directo usa proxy mock y no guarda checkpoints.
- No hay Nash-on-policy ni self-play preview.
- Tera está desactivada; otro formato requiere revalidar encoder y datos.

## 10. Referencias

- von Neumann & Morgenstern (1944), *Theory of Games and Economic Behavior*.
- McMahan, Gordon & Blum (2003), planificación con costes adversariales.
- Lanctot et al. (2017), enfoque game-theoretic para MARL.
- [`rl_env_spec.md`](rl_env_spec.md) §4 y §6.j.
- [`team_preview_masked_ppo_spec.md`](team_preview_masked_ppo_spec.md).

## Changelog

| Versión | Fecha | Notas |
|---------|-------|-------|
| **1.2** | 2026-10-11 | Value Net, low-rank, LP, entrenamiento, despliegue y Conclusión; alineación actualizada a rl_env v1.15. |
| **1.1** | 2026-10-11 | Nombre Payoff Nash, glosario y límites. |
| **1.0** | 2026-10-11 | Low-rank, HiGHS, pares uniformes, \(A=2M-1\), Flat-360/512D. |

*Fin del contrato Team Preview Payoff Nash v1.2.*
