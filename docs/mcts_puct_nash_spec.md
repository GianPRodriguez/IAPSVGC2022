# MCTS–PUCT/Nash Agent Spec — SM-MCTS (VGC Gen 9)

| Field | Value |
|-------|--------|
| **Document** | **MCTS–PUCT/Nash (SM-MCTS B1) — exclusive implementation contract** |
| **Version** | **1.0** (aligned with `docs/rl_env_spec.md` **v1.5** §6.d) |
| **Status** | **FROZEN — implemented in `showdown/rl/mcts_puct_nash/`** |
| **Algorithm** | Simultaneous-move MCTS: PUCT over matrix **cells** + Nash (Fictitious Play) backup |
| **Role** | Deep simultaneous planning under Cand. 1 (sibling of open-loop [`mcts_puct_spec.md`](mcts_puct_spec.md)) |
| **Code root** | `showdown/rl/mcts_puct_nash/` |
| **CLI** | `python -m showdown.rl.mcts_puct_nash` |
| **Deps** | `numpy` only (no torch). Reuses S2 from `mcts_puct`. |

> **Contrast:** Open-loop MCTS+PUCT samples \(a_{\mathrm{opp}}\sim U\). This agent selects **cells** \((a_i,b_j)\) with PUCT and backs up **Nash values** \(v^\star=\sigma^{\star\top}\hat A\tau^\star\).

---

## 1. Inherited environment

| Contract | Usage |
|----------|--------|
| Cand. 1 | \(z\in\{+1,-1,0\}\) |
| S2 simulator + `greedy_joint_snapshot` | From `showdown/rl/mcts_puct/` |
| Legal masks | IAM on every expand |
| Flat belief | Parity with v1 agents |
| Taxonomy | Stochastic **simultaneous-move** (concurrent-action uncertainty; not classic hidden-info Poker) |

**S2↔S1 divergence risk:** same as sibling — document in thesis ablations.

---

## 2. Topology (Model B1)

- Node = state \(s\).
- Edge = matrix cell \((a_i,b_j)\) with \(a_i\in I(s)\), \(b_j\in J(s)\).
- Transition: one S2 `step_sim(s, a_i, b_j)` → child.
- PUCT over cells:

\[
U(s,(i,j))=Q_{ij}+c\,P(i,j\mid s)\frac{\sqrt{N(s)}}{1+n_{ij}}
\]

with \(Q_{ij}=\hat A_{ij}\) (0 if \(n_{ij}=0\)), \(P=1/(|I||J|)\) uniform on Top‑K cells.

---

## 3. Top‑K

- \(K=16\).
- Score = `score_joint` (sibling).
- \(I(s)\): top‑K legal self joints; \(J(s)\): top‑K on **flipped** snapshot (opp).
- If \(|L|<K\), use all. Ties → smaller joint id.

---

## 4. Simulation loop (single backup rule)

```
each sim_id:
  path ← []
  node ← root; rng ← hash(seed, turn, sim_id)
  while True:
    if not node.expanded: expand Top-K + init n,W
    if node.state.finished: z ← Cand1; break
    pick cell (i,j) by PUCT (prefer n=0)
    path.append(node, i, j)
    child ← step_sim / children[(i,j)]
    if child.finished:
      z ← Cand1(child); update cell (i,j) with z; break
    if child first visit (leaf):
      z ← rollout(greedy_joint_snapshot both); update cell; break
    node ← child
  # Backup up path:
  for (node, i, j) in reversed(path):
    # cell already updated with z at creation time for deepest;
    # for upper nodes: ensure cell (i,j) got this trajectory's z once
    if cell not yet updated this sim: n_ij+=1; W_ij+=z
    z ← NashValue(Â)   # FP; parent receives v*
play: σ* at root with T_fp=500 → a = argmax_i σ*_i
```

**Precise backup (implementation contract):**

1. Deepest edge: after leaf \(z\), set \(n_{ij}\mathrel{+=}1\), \(W_{ij}\mathrel{+=}z\).
2. Then \(z \leftarrow v^\star(\hat A)\) via Fictitious Play.
3. For each ancestor edge in reverse: \(n\mathrel{+=}1\), \(W\mathrel{+=}z\) (here \(z\) is already \(v^\star\) from below), then \(z \leftarrow v^\star(\hat A)\) again.

Leaf terminal: same with \(z\in\{+1,-1,0\}\).

---

## 5. Fictitious Play

| Node | \(T_{\mathrm{fp}}\) |
|------|---------------------|
| Internal | `200` |
| Root (play) | `500` |

- Init: uniform.
- Output: average of last \(W=50\) frequency vectors.
- Multiple optima: max-entropy among \(\varepsilon\)-optimal mixes (\(\varepsilon=10^{-6}\)).

---

## 6. Frozen hyperparameters

| Parameter | Value |
|-----------|--------|
| \(N_{\mathrm{sim}}\) | `50` (smoke `8`) |
| \(c\) | `1.5` |
| \(K\) | `16` |
| Samples per cell visit | `1` (online mean) |
| Prior | Uniform on Top‑K cells |
| Play | \(\arg\max_i\sigma^\star_{\mathrm{root}}\) |
| Max turns | `40` |
| Seed | Required |

**Ablation parity:** equalize S2 `step_sim` / terminal-rollout counts vs open-loop MCTS+PUCT; also report native defaults (50 vs 200).

---

## 7. Modules

| Module | Role |
|--------|------|
| `topk.py` | Top‑K \(I,J\) |
| `nash.py` | Fictitious Play + \(v^\star\) |
| `tree.py` | Nodes / cells / PUCT |
| `search.py` | SM-MCTS loop |
| `agent.py` / `player.py` | Act + Validated |
| Reuse | `mcts_puct.simulator`, `greedy_snapshot` |

---

## 8. Acceptance tests

1. Cells only from legal Top‑K.
2. Toy matching-pennies matrix → FP \(\approx(0.5,0.5)\), \(v\approx0\).
3. Backup: parent value is \(v^\star\) after child update.
4. RNG reproducible for fixed `(seed,turn,sim_id)`.
5. Smoke: mock game `--sims 8`.

---

## 9. Non-goals

- Nash-at-root-only (rejected for this agent)
- Open-loop opp sampling as primary model
- Neural prior
- Classic imperfect-info belief tracking

---

## 10. References

- Rosin (2011); Silver et al. (2017) — PUCT form.
- Brown (1951) — fictitious play.
- Huang & Ontañón (2022) — invalid action masking.
- Sibling [`mcts_puct_spec.md`](mcts_puct_spec.md); [`rl_env_spec.md`](rl_env_spec.md) Cand. 1.

---

## Changelog

| Version | Date | Notes |
|---------|------|-------|
| **1.0** | 2026-10-10 | SM-MCTS B1: cell PUCT, online Â, Nash backup, \(N_{\mathrm{sim}}=50\), \(K=16\). |

---

*End of MCTS–PUCT/Nash spec 1.0.*
