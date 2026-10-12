# Masked PPO — Team Preview Exclusive Spec (VGC Gen 9 Regulation M-C)

| Campo | Valor |
|-------|-------|
| **Documento** | Contrato exclusivo — Masked PPO solo Team Preview |
| **Versión** | **1.2** (alineado con `docs/rl_env_spec.md` v1.15 §6.i) |
| **Estado** | **FROZEN** — implementado en `showdown/rl/team_preview_ppo/` |
| **Algoritmo** | PPO + Invalid Action Masking (IAM), acción Flat-360 |
| **Formato** | Gen 9 VGC **Regulation M-C**, OTS, Mega on, **Tera off** |
| **Código** | `showdown/rl/team_preview_ppo/` y `team_preview_{env,obs,action}.py` |
| **CLI** | `python -m showdown.rl.team_preview_ppo.train --seed N` |

Este agente solo elige cuatro Pokémon, con los dos leads primero y los dos de
banca después. No decide movimientos durante el combate.

## 1. Por qué existe

Team Preview condiciona toda la partida. Si se aprende junto con los turnos, no
se puede atribuir una mejora a uno u otro componente. Este módulo lo convierte
en una sola decisión: observa ambas plantillas OTS, elige una de 360 formaciones
y recibe el resultado final del BO1.

La analogía es un director técnico que elige la alineación y entrega el partido
a un entrenador fijo. Ese entrenador es \(\pi_c\). Congelarlo hace comparables
los episodios: cambia el selector de preview, no el piloto de combate.

## 2. Glosario

| Término | Explicación |
|---------|-------------|
| Política \(\pi(a\mid s)\) | Probabilidad con que el actor elige cada formación dado el estado |
| Valor \(V(s)\) | Retorno esperado antes de elegir; referencia para juzgar la acción |
| Ventaja | En este bandit, señal derivada de \(R-V(s)\) que guía al actor |
| PPO | Mejora la política limitando cuánto puede cambiar en una actualización |
| IAM | Enmascarado que da probabilidad cero a acciones ilegales antes del muestreo |
| OTS | Open Team Sheets: información pública disponible en preview |
| Bandit contextual | MDP de horizonte \(T=1\): contexto, una acción y un resultado |
| Flat-360 | Las \(P(6,4)=360\) selecciones ordenadas de cuatro miembros distintos |
| \(\pi_c\) | Piloto `greedy_joint`, congelado al entrenar preview |
| Cand. 1 | \(+1\) victoria, \(-1\) derrota, \(0\) empate |
| Checkpoint | Pesos aprendidos y metadatos necesarios para reproducir inferencia |

## 3. Algoritmo: PPO con IAM, paso a paso

La red comparte un tronco MLP. La **policy head** produce 360 logits
(preferencias sin normalizar); la **value head** produce \(V(s)\), un escalar.

1. `env.reset()` construye \(s\in\mathbb{R}^{512}\) y máscara legal \(M\).
2. La red calcula logits \(z\) y valor \(V(s)\).
3. IAM aplica
   \[
   \tilde z_i=z_i+(1-M_i)(-10^9),\qquad
   \pi(a\mid s)=\operatorname{Softmax}(\tilde z).
   \]
   Así no se desperdician episodios ni gradiente en acciones inexecutable.
4. Durante entrenamiento se muestrea \(a\); en evaluación se usa el argmax
   masked. La acción se decodifica a `/team t1t2t3t4`.
5. \(\pi_c\) juega el BO1 sin actualizar parámetros y devuelve Cand. 1.
6. El rollout guarda observación, acción, recompensa, log-probabilidad antigua,
   valor y máscara.
7. PPO calcula \(\rho=\exp(\log\pi_{\rm new}-\log\pi_{\rm old})\) y minimiza
   \[
   -\min(\rho A,\operatorname{clip}(\rho,1-\epsilon,1+\epsilon)A).
   \]
   El clip evita saltos destructivos; la pérdida de valor enseña \(V\), y la
   entropía conserva exploración.
8. La misma máscara almacenada se reutiliza en todas las épocas PPO.

## 4. MDP, acción y observación

### 4.1 MDP y recompensa

| Elemento | Contrato |
|----------|----------|
| Horizonte | \(T=1\), una acción de preview por episodio |
| Episodio producción | Preview + un BO1, máximo 40 turnos |
| Política de combate | `greedy_joint` congelada; ablación opcional `expectimax_bayesian` |
| Recompensa | Cand. 1, \(R\in\{-1,0,+1\}\), sin \(\Delta HP\) |
| Self-play preview / BC | Off / fuera de alcance |

### 4.2 Acción Flat-360

\[
|A|=P(6,4)=360,\qquad a\in\{0,\ldots,359\}.
\]

Gym y logits son 0-based; Showdown recibe índices 1-based. La codificación
Lehmer usa factores \((60,12,3,1)\), contando en cada posición el índice dentro
de los miembros aún disponibles. `decode_preview_action` invierte el proceso y
`action_to_team_string` suma uno. Ejemplo:
\((0,1,2,3)\leftrightarrow a=0\leftrightarrow\)`/team 1234`.

Con seis miembros todas las acciones son legales. `legal_mask(pool_size)`
mantiene el contrato para pools reducidos.

### 4.3 Observación

\[
512=210_{P1}+210_{P2}+36_{\rm cobertura}+56_{\rm contexto}.
\]

| Bloque | Dimensión | Contenido |
|--------|-----------|-----------|
| Equipo propio | \(6\times35\) | Stats normalizados, 18 tipos, seis flags item/ability y cinco roles |
| Equipo rival | \(6\times35\) | La misma representación desde OTS |
| Cobertura | 36 | Matriz \(P_1\to P_2\), aplanada |
| Contexto | 56 | Clima/terreno en cero y padding reservado |

La cobertura es \(M_{ij}=\log(d_{\max}/HP_j+\varepsilon)\) y se calcula en
`reset()`, no en los turnos. Stellar/Tera no forma parte del encoder:
Regulation M-C se ejecuta con **Terastallization desactivada**.

## 5. Entrenamiento detallado

### 5.1 Loop, datos y CLI

Cada update recolecta 2048 episodios (64 en smoke). Cada fila contiene
`obs[512]`, acción, Cand. 1, `old_logp`, valor y `action_mask[360]`. Luego se
baraja y recorre cuatro veces en minibatches de 128.

En producción, `greedy_joint` juega todos los combates congelado. No recibe
gradientes, replay ni cambios de parámetros desde preview. Esto evita una diana
móvil y permite atribuir la señal al selector bajo una competencia fija.

El CLI implementado crea hoy `TeamPreviewEnv(backend='mock')`, por lo que el
comando directo usa el proxy de cobertura. El camino live es
`TeamPreviewEnv(combat_fn=...)`; ese callback debe ejecutar el BO1 con
`greedy_joint` y devolver Cand. 1. Un run mock no debe reportarse como live.

```bat
python -m showdown.rl.team_preview_ppo.train --seed 42 --updates 10
python -m showdown.rl.team_preview_ppo.train --seed 42 --smoke
```

`--seed` es obligatorio; `--device` vale `cpu` por defecto.

### 5.2 Qué significa un checkpoint

Un checkpoint de inferencia contiene los pesos completos de
`PreviewActorCritic` (tronco, policy head y value head) y debe registrar versión
del spec, seed, hiperparámetros, episodios/updates y procedencia `mock` o `live`.
Para reanudar exactamente también requiere optimizador y estados RNG.

El CLI actual devuelve el agente en Python, pero no escribe un archivo. La
persistencia corresponde al runner que lo invoque; `DONE team_preview_ppo` no
implica que se hayan guardado pesos.

## 6. Hiperparámetros

| Parámetro | Valor | Propósito | Cómo influye |
|-----------|-------|-----------|--------------|
| Red | `512→256→256→(360,1)`, ReLU | Representación compartida de policy/value | Más capacidad modela interacciones OTS, con riesgo de sobreajuste |
| \(\gamma\) | `1.0` | Descuento | En \(T=1\), conserva toda la recompensa |
| \(\lambda\) GAE | `1.0` | Construir ventaja | En este bandit usa el retorno terminal |
| \(\epsilon_{\rm clip}\) | `0.2` | Limitar cambios PPO | Menor valor hace updates más conservadores |
| \(c_v\) | `0.5` | Peso de pérdida de valor | Alto prioriza calibrar \(V\) |
| \(c_e\) | `0.02` | Bono de entropía | Alto mantiene exploración |
| Learning rate | `3e-4` | Paso de Adam | Alto aprende rápido, pero puede inestabilizar |
| Gradient clip | `0.5` | Acotar gradiente | Reduce saltos por batches ruidosos |
| Episodios/update | `2048`; smoke `64` | Tamaño del rollout | Más reduce ruido a mayor coste |
| Minibatch \(B\) | `128` | Tamaño de SGD | Controla ruido y memoria |
| Épocas \(K\) | `4` | Reutilizar rollout | Más señal, pero mayor deriva de la policy antigua |
| Seed | Obligatoria | Reproducibilidad | Fija inicialización y muestreo |

## 7. Inferencia y despliegue en batalla

1. Construir exactamente la observación 512D desde ambas OTS.
2. Cargar una red compatible y calcular logits.
3. Aplicar la máscara legal antes de seleccionar.
4. Usar argmax masked para evaluación o muestrear la categórica masked para
   juego estocástico.
5. Decodificar y enviar `/team ####`.
6. Transferir el control al combatiente; preview no vuelve a actuar.

La suite propia usa 200 BO1 por baseline y siempre declara checkpoint, modo de
selección y \(\pi_c\).

## 8. Pipeline de Conclusión

Este preview alimenta dos familias:

- **DDQN + PPO-preview**
- **PPO + PPO-preview**

Selector y combatiente son checkpoints separados, fijados antes del torneo y
sin reentrenamiento durante él. Véase el protocolo de snapshot, pares, swap y
métricas en [`conclusion_eval_protocol.md`](conclusion_eval_protocol.md).

## 9. Limitaciones

- El aprendizaje hereda competencia y sesgos de `greedy_joint`.
- El CLI directo usa proxy mock, no Showdown live.
- Cand. 1 es una señal escasa y no explica una derrota.
- Flat-360 distingue orden de banca aunque varios baselines no lo hagan.
- No incluye self-play, BC ni serialización automática.
- Tera está desactivada; activarla cambia observación y dinámica.

## 10. Referencias

- Schulman et al. (2017), *Proximal Policy Optimization Algorithms*.
- Huang & Ontañón (2022), *A Closer Look at Invalid Action Masking*.
- [`rl_env_spec.md`](rl_env_spec.md) §4 y §6.i.
- [`ppo_spec.md`](ppo_spec.md), combatiente PPO separado.
- [`team_preview_nash_spec.md`](team_preview_nash_spec.md) y
  [`team_preview_qd_pga_spec.md`](team_preview_qd_pga_spec.md).

## Changelog

| Versión | Fecha | Notas |
|---------|-------|-------|
| **1.2** | 2026-10-11 | Algoritmo, loop, checkpoint, despliegue y Conclusión; alineación documental actualizada a rl_env v1.15. |
| **1.1** | 2026-10-11 | Redacción dual inicial, glosario y límites. |
| **1.0** | 2026-10-11 | Flat-360 IAM, obs 512D, Cand. 1 y \(\pi_c\)=`greedy_joint`. |

*Fin del contrato Team Preview Masked PPO v1.2.*
