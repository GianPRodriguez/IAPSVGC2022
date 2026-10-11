# MCTS–PUCT/Nash Agent Spec — SM-MCTS B1 (VGC Gen 9)

| Campo | Valor |
|-------|--------|
| **Documento** | **Contrato exclusivo — MCTS–PUCT/Nash (SM-MCTS B1)** |
| **Versión** | **1.1** (alineado con `docs/rl_env_spec.md` **v1.10** §6.d) |
| **Estado** | **FROZEN — implementado en `showdown/rl/mcts_puct_nash/`** |
| **Algoritmo** | MCTS de turnos simultáneos: PUCT sobre **celdas** de matriz + backup Nash (Fictitious Play) |
| **Rol** | Planificación profunda **simultánea** bajo Cand. 1 (sibling de open-loop) |
| **Code root** | `showdown/rl/mcts_puct_nash/` |
| **CLI** | `python -m showdown.rl.mcts_puct_nash` |
| **Deps** | `numpy` (sin torch). Reusa S2 de `mcts_puct`. |

> **Contraste.** Open-loop ([`mcts_puct_spec.md`](mcts_puct_spec.md)) samplea
> \(a_{\mathrm{opp}}\sim U\). Este agente elige **celdas** \((a_i,b_j)\) con PUCT y
> propaga valores Nash \(v^\star=\sigma^{\star\top}\hat A\tau^\star\).
>
> **Inmutabilidad.** Bump **1.2+** + `rl_env_spec` §6.d juntos.

---

## 0. Por qué existe (lectura no técnica)

En VGC ambos bandos eligen **a la vez**. El MCTS open-loop finge que el rival es
un dado uniforme. Aquí se construye una **matriz de payoffs** estimada entre tus
mejores jugadas (Top-K) y las del rival (Top-K en el snapshot volteado), se elige
qué casilla explorar con PUCT, y al subir el resultado se resuelve un **equilibrio
aproximado** (Fictitious Play) para saber “cuánto vale esta posición si ambos
juegan bien en promedio”.

**Analogía:** no solo preguntas “¿qué pasa si yo hago X y el rival algo al azar?”,
sino “¿qué casilla de la tabla X-mío × Y-suyo merece más visitas?” y al final
juegas según la mezcla Nash de la raíz.

| Término | Significado aquí |
|---------|------------------|
| Celda \((i,j)\) | Par (acción propia Top-K, acción rival Top-K) |
| \(\hat A\) | Matriz empírica de returns |
| Nash \(v^\star\) | Valor de equilibrio de \(\hat A\) |
| Fictitious Play | Algoritmo iterativo para mezclas \(\sigma^\star,\tau^\star\) |
| Top-K | Recorte a \(K=16\) joints por lado |

---

## 1. Entorno heredado

| Contrato | Uso |
|----------|-----|
| Cand. 1 | \(z\in\{+1,-1,0\}\) |
| S2 + `greedy_joint_snapshot` | Desde `mcts_puct/` |
| Máscaras legales | IAM en cada expand |
| Belief plano | Paridad agentes v1 |
| Taxonomía | Stochastic **simultaneous-move** (no Poker de info oculta clásico) |

Riesgo S2↔S1: igual que el sibling; documentar en ablaciones.

---

## 2. Topología (Modelo B1)

- Nodo = estado \(s\).
- Arista = celda \((a_i,b_j)\), \(a_i\in I(s)\), \(b_j\in J(s)\).
- Transición: un `step_sim(s,a_i,b_j)` → hijo.
- PUCT sobre celdas:

\[
U(s,(i,j))=Q_{ij}+c\,P(i,j\mid s)\frac{\sqrt{N(s)}}{1+n_{ij}}
\]

con \(Q_{ij}=\hat A_{ij}\) (0 si \(n=0\)), \(P=1/(|I||J|)\) uniforme en celdas Top-K.

---

## 3. Top‑K

- \(K=16\).
- Score = `score_joint` (sibling).
- \(I(s)\): top-K joints legales propios; \(J(s)\): top-K en snapshot **flipped**.
- Si \(|L|<K\), usar todos. Empates → joint id menor.

---

## 4. Bucle de simulación (backup)

```
each sim_id:
  path ← []; node ← root; rng ← hash(seed, turn, sim_id)
  while True:
    if not expanded: expand Top-K + init n,W
    if finished: z ← Cand1; break
    pick cell (i,j) by PUCT (prefer n=0)
    path.append; child ← step_sim
    if child finished / first-visit leaf:
      z ← Cand1 or rollout(greedy_joint both); update cell; break
    node ← child
  # Backup:
  deepest: n+=1; W+=z
  then repeatedly: z ← NashValue(Â); ancestors n+=1; W+=z
play: σ* at root (T_fp=500) → a = argmax_i σ*_i
```

**Contrato preciso:**

1. Arista más profunda: tras hoja \(z\), \(n{+}{=}1\), \(W{+}{=}z\).
2. \(z\leftarrow v^\star(\hat A)\) vía Fictitious Play.
3. Ancestros en reverse: \(n{+}{=}1\), \(W{+}{=}z\) (ya \(v^\star\)), luego de nuevo \(z\leftarrow v^\star\).

---

## 5. Fictitious Play

| Nodo | \(T_{\mathrm{fp}}\) |
|------|---------------------|
| Interno | `200` |
| Raíz (play) | `500` |

Init uniforme; salida = promedio de últimos \(W=50\) vectores de frecuencia;
múltiples óptimos → max-entropía entre \(\varepsilon\)-óptimos (\(\varepsilon=10^{-6}\)).

---

## 6. Hiperparámetros congelados

| Parámetro | Valor |
|-----------|--------|
| \(N_{\mathrm{sim}}\) | `50` (smoke `8`) |
| \(c\) | `1.5` |
| \(K\) | `16` |
| Samples / visita celda | `1` (media online) |
| Prior | Uniforme en celdas Top-K |
| Play | \(\arg\max_i\sigma^\star_{\mathrm{root}}\) |
| Max turns | `40` |
| Seed | Obligatorio |

**Ablación de paridad:** igualar conteos `step_sim` / rollouts vs open-loop; reportar también defaults nativos (50 vs 200).

---

## 7. Módulos

`topk.py`, `nash.py`, `tree.py`, `search.py`, `agent.py` / `player.py`;
reuso `mcts_puct.simulator`, `greedy_snapshot`.

---

## 8. Tests de aceptación

Celdas solo Top-K legales; matching-pennies → FP \(\approx(0.5,0.5)\), \(v\approx0\);
backup padre = \(v^\star\); RNG reproducible; smoke `--sims 8`.

---

## 9. Fortalezas y límites (tesis)

| Fortaleza | Límite |
|-----------|--------|
| Modela simultaneidad con matriz + Nash | Más caro / menos sims que open-loop (50 vs 200) |
| Top-K acota explosión combinatoria | Top-K puede omitir joyas fuera del ranking greedy |
| Contraste limpio vs open-loop | Sigue dependiendo de S2 + rollout Joint |

---

## 10. Non-goals

Nash-solo-en-raíz; open-loop como modelo primario; prior neural; tracking clásico de info imperfecta tipo Poker.

---

## 11. Referencias

Rosin (2011); Silver et al. (2017); Brown (1951) fictitious play; Huang & Ontañón (2022);
[`mcts_puct_spec.md`](mcts_puct_spec.md); [`rl_env_spec.md`](rl_env_spec.md).

---

## Changelog

| Versión | Fecha | Notas |
|---------|-------|-------|
| **1.1** | 2026-10-10 | Redacción amena (ES); semántica intacta. |
| **1.0** | 2026-10-10 | SM-MCTS B1: cell PUCT, Nash backup, \(N_{\mathrm{sim}}=50\), \(K=16\). |

---

*Fin del contrato MCTS–PUCT/Nash v1.1.*
