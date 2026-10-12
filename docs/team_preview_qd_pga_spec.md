# QD-PGA — Team Preview Exclusive Spec (VGC Gen 9 Regulation M-C)

| Campo | Valor |
|-------|-------|
| **Documento** | Contrato exclusivo — QD-PGA MAP-Elites solo Team Preview |
| **Versión** | **1.2** (alineado con `docs/rl_env_spec.md` v1.15 §6.k) |
| **Estado** | **FROZEN** — implementado en `showdown/rl/team_preview_qd_pga/` |
| **Algoritmo** | Grid MAP-Elites + mutación Gaussiana o mejora PPO-clip (PGA) |
| **Formato** | Gen 9 VGC **Regulation M-C**, OTS, Mega on, **Tera off** |
| **Código** | `showdown/rl/team_preview_qd_pga/` y módulos `team_preview_*` |
| **CLI** | `python -m showdown.rl.team_preview_qd_pga.train --seed N` |

El resultado no es una sola policy, sino un archivo de políticas de preview:
la mejor conocida para cada región de comportamiento.

## 1. Por qué existe

Optimizar únicamente win-rate suele producir un campeón que repite un estilo.
Puede ser fuerte, pero frágil ante cambios de matchup. Quality-Diversity (QD)
pregunta algo más amplio: «¿qué políticas distintas pueden ser buenas?».

MAP-Elites funciona como un museo de 100 salas. Dos descriptores de
comportamiento (BD) indican en qué sala entra una policy; dentro queda solo la
de mayor fitness. La diversidad no se suma a la recompensa: calidad decide
quién gana una sala, mientras el BD decide qué políticas compiten entre sí.

PGA añade gradiente de política a MAP-Elites. Unos descendientes se exploran
con mutación aleatoria y otros se mejoran con PPO. Así combina exploración
paramétrica barata con una mejora dirigida por Cand. 1.

## 2. Glosario

| Término | Explicación |
|---------|-------------|
| QD | Quality-Diversity: buscar calidad y cobertura de comportamientos |
| MAP-Elites | Archivo discretizado que conserva una elite por celda |
| Elite | Mejor actor observado en una celda según fitness |
| BD | Behavior Descriptor; coordenadas del estilo, no objetivo de recompensa |
| Fitness \(f\) | Media de Cand. 1 en partidas de evaluación |
| Coverage | Fracción de las 100 celdas ocupadas |
| QD-score | Suma de fitness de las celdas según implementación del archivo |
| Genotipo | Pesos de la red actor |
| Fenotipo | Comportamiento observado, resumido por \((b_1,b_2)\) |
| PGA | Policy Gradient Assisted: mejora algunos descendientes con PPO |
| Crítico \(V_\phi\) | Predice retorno y reduce varianza; es compartido, no se archiva |
| Actor-only | Cada celda guarda pesos del actor, no un crítico propio |
| Burn rollout | Episodios de una mutación sin aprendizaje ni replay |

## 3. Algoritmo: archivo QD-PGA, paso a paso

### 3.1 Bootstrap

1. Inicializar actor, crítico compartido y archivo vacío \(10\times10\).
2. Recolectar \(B_{\rm boot}=4096\) episodios (64 smoke) y mejorar el actor con
   PPO-clip durante \(K=2\) épocas.
3. Evaluarlo durante \(G=20\) BO1 (4 smoke), calcular fitness y BD medios.
4. Si hay al menos \(g_{\min}=10\) resultados válidos (2 smoke), insertar el
   actor en su celda.

### 3.2 Emisión de descendientes

Durante \(N_{\rm outer}=256\) iteraciones (2 smoke), se emiten cuatro
descendientes por iteración (2 smoke):

1. Elegir uniformemente una celda ocupada y cargar su actor. Si el archivo está
   vacío, usar el actor actual.
2. Con probabilidad \(p_{\rm mut}=0.5\), sumar ruido Gaussiano
   \(\mathcal N(0,0.02^2)\) a sus pesos y ejecutar 128 episodios burn
   (16 smoke), sin aprendizaje ni replay.
3. En caso contrario, ejecutar PPO-clip sobre 128 episodios, usando el crítico
   compartido \(V_\phi\). Compartirlo transfiere una referencia de valor entre
   estilos y evita guardar un crítico por celda.
4. Evaluar el descendiente con acción greedy masked, obtener \(f,b_1,b_2\).
5. Discretizar el BD en la rejilla y reemplazar solo si
   \(f_{\rm new}>f_{\rm old}\). Los empates no reemplazan.
6. Reportar coverage, QD-score y mejor fitness.

Este genotipo no es una máscara «6 choose 4» ni un espacio 90/180: son los
pesos de una policy Flat-360. La acción elegida es el fenotipo observado.

### 3.3 Descriptores de comportamiento

`bd.py` calcula ambos valores en \([0,1]\) a partir de `/team` y especies OTS:

- \(b_1\), control de velocidad: \(0.5\) si el bring contiene una especie de
  Trick Room y \(0.5\) si contiene Tailwind u otro speed control.
- \(b_2\), agresión: proporción ponderada de leads y banca agresivos. Fake Out
  cuenta como agresivo; redirect, Tailwind y Trick Room cuentan como soporte;
  la banca pesa la mitad que los leads.

Los BD son heurísticos. **BD no pertenece a \(R\) ni a \(f\)**; alterar esta
separación convertiría el algoritmo en optimización escalar, no MAP-Elites.

## 4. MDP, acción y observación

| Elemento | Contrato |
|----------|----------|
| Horizonte | \(T=1\), una selección por episodio |
| Acción | Flat-360 con IAM; cuatro índices distintos |
| Observación | \(512=210+210+36+56\), compartida con PPO/Nash |
| Recompensa/fitness | Cand. 1; sin Cand. 2 ni \(\Delta HP\) |
| Piloto | `greedy_joint` congelado |
| Self-play preview | Off |

La policy produce 360 logits. IAM se aplica tanto al muestrear rollouts como en
la evaluación greedy, de modo que una policy nunca recibe mérito por acciones
ilegales. La observación contiene dos equipos OTS, una matriz de cobertura
\(P_1\to P_2\) calculada en `reset()` y contexto reservado.

La recompensa es:
\[
R=\begin{cases}+1&\text{victoria}\\-1&\text{derrota}\\0&\text{empate}\end{cases},
\qquad
f=\frac{1}{g}\sum_{e=1}^{g}R^{(e)}.
\]

Regulation M-C usa Mega on y **Terastallization off**; Tera/Stellar no se
codifica ni se activa.

## 5. Entrenamiento detallado

### 5.1 Loop, rivales, datos y CLI

Los rollouts PPO guardan `obs[512]`, acción, `old_logp`, Cand. 1 y máscara. La
ventaja es \(R-V_\phi(s)\), normalizada por batch. El actor usa PPO-clip y
entropía; el crítico compartido ajusta MSE contra Cand. 1. Los burn rollouts
solo ejecutan la policy mutada y descartan transiciones.

El contrato de evaluación del archivo usa un mosaico de rivales de preview:
**25% random, 25% greedy y 50% expectimax**. Fitness primario nunca es un score
tipado estático. `greedy_joint` permanece congelado durante combate para que
las diferencias entre elites provengan del preview, no de coadaptación del
piloto.

El CLI implementado actualmente crea `TeamPreviewEnv(backend='mock')`; no
conecta por sí solo el mosaico ni Showdown live. Producción debe proporcionar
el runner/callback live que preserve esas proporciones y ejecute
`greedy_joint`. Los reportes deben distinguir `mock` de `live`.

```bat
python -m showdown.rl.team_preview_qd_pga.train --seed 42
python -m showdown.rl.team_preview_qd_pga.train --seed 42 --smoke
```

`--seed` es obligatorio; `--device` vale `cpu`.

### 5.2 Qué significa un checkpoint

El artefacto principal es el **archivo**, no solo la mejor red: para cada celda
ocupada contiene coordenadas BD, fitness, `state_dict` actor-only y metadatos
de origen. Debe añadir versión, seed, budgets, procedencia `mock/live` y tablas
heurísticas usadas por los BD.

Para reanudar exactamente también se guardan crítico compartido,
optimizadores, RNG e iteración outer. Para desplegar una sola policy basta el
actor de la elite elegida y el contrato obs/acción. El CLI actual devuelve
`Archive`, pero no lo serializa; la línea `DONE` solo informa coverage/celdas.

## 6. Hiperparámetros

| Parámetro | Valor | Propósito | Cómo influye |
|-----------|-------|-----------|--------------|
| Grid | `10×10` | Discretizar dos BD | Más celdas dan detalle y exigen más evaluaciones |
| \(B_{\rm boot}\) | `4096`; smoke `64` | Crear policy semilla | Más episodios mejoran el arranque |
| \(T_{\rm emit}\) | `128`; smoke `16` | Datos por descendiente | Más reduce ruido y aumenta coste |
| \(n_{\rm emit}\) | `4`; smoke `2` | Descendientes por outer | Más explora el archivo más rápido |
| \(N_{\rm outer}\) | `256`; smoke `2` | Iteraciones totales | Define gran parte del budget |
| \(p_{\rm mut}\) | `0.5` | Alternar mutación/PPO | Alto explora más y explota menos gradiente |
| \(\sigma\) | `0.02` | Escala de mutación | Alto produce estilos más lejanos y más frágiles |
| PPO \(K\) | `2` | Épocas de mejora | Más explota datos y puede alejarse del padre |
| PPO \(\epsilon\) | `0.2` | Limitar cambio del actor | Menor hace mejora más conservadora |
| Entropía \(c_e\) | `0.02` | Sostener exploración | Alto evita colapso temprano |
| Learning rate | `3e-4` actor/crítico | Paso de Adam | Alto acelera, con riesgo de inestabilidad |
| Gradient clip | `0.5` | Acotar gradientes PPO | Reduce updates extremos |
| Minibatch | `128` | Tamaño de SGD | Equilibra memoria y ruido |
| \(\gamma\) | `1.0` | Retorno del bandit | Conserva Cand. 1 completa |
| \(G,g_{\min}\) | `20,10`; smoke `4,2` | Fiabilidad/elegibilidad | Más partidas estabilizan fitness/BD |
| Reemplazo | Estricto `>` | Proteger elite | Evita rotación por empates |
| Selección padre | Uniforme en celdas llenas | Dar oportunidad a cada nicho | No favorece automáticamente al campeón |

## 7. Inferencia y despliegue en batalla

Hay dos modos legítimos:

1. **Campeón global:** seleccionar \(\pi^\star=\arg\max f\), cargar su actor,
   construir obs 512D, aplicar IAM, tomar argmax y enviar `/team ####`.
2. **Elite por estilo:** elegir una celda antes de la serie según el estilo
   deseado y desplegar su actor. La regla de elección debe fijarse antes de ver
   resultados para evitar selección retrospectiva.

La evaluación principal usa la elite de mayor fitness y 200 BO1 con acción
argmax masked. Se reportan también coverage, QD-score, celda y BD.

## 8. Pipeline de Conclusión

QD-PGA-preview alimenta las tres familias QD de combate:

- **QD-PPO + QD-PGA-preview**
- **QD-PGA + QD-PGA-preview**
- **QD-PPGA + QD-PGA-preview**

Archivo, regla de selección de elite y checkpoint de combate se congelan antes
del round-robin. Véase
[`conclusion_eval_protocol.md`](conclusion_eval_protocol.md).

## 9. Limitaciones

- Los BD dependen de listas heurísticas de especies y pueden simplificar sets.
- Coverage alto no implica automáticamente mayor win-rate.
- Fitness con \(G=20\) sigue siendo ruidoso; una elite puede entrar por azar.
- La mutación en pesos es ciega y puede destruir comportamiento útil.
- Un crítico compartido puede favorecer estilos frecuentes.
- El CLI directo usa proxy mock y no implementa el mosaico live completo.
- El archivo no se serializa automáticamente.
- Tera está desactivada; activarla invalida parte del dominio.

## 10. Referencias

- Nilsson & Cully (2021), *Policy Gradient Assisted MAP-Elites*.
- Mouret & Clune (2015), *Illuminating Search Spaces by Mapping Elites*.
- Schulman et al. (2017), PPO.
- Huang & Ontañón (2022), invalid action masking.
- [`qd_pga_map_elites_spec.md`](qd_pga_map_elites_spec.md), variante de combate.
- [`team_preview_masked_ppo_spec.md`](team_preview_masked_ppo_spec.md).

## Changelog

| Versión | Fecha | Notas |
|---------|-------|-------|
| **1.2** | 2026-10-11 | MAP-Elites, BD, loop, despliegue y familias QD; alineación documental actualizada a rl_env v1.15. |
| **1.1** | 2026-10-11 | Redacción dual, glosario y límites. |
| **1.0** | 2026-10-11 | Archivo actor-only, BD, Cand. 1, budget y variación mutación/PGA. |

*Fin del contrato Team Preview QD-PGA v1.2.*
