# MCTS + PUCT Agent Spec — Standard (VGC Gen 9)

| Field | Value |
|-------|--------|
| **Document** | **MCTS + PUCT (standard) — exclusive implementation contract** |
| **Version** | **1.0** (aligned with `docs/rl_env_spec.md` **v1.4** §6.c) |
| **Status** | **FROZEN — implemented in `showdown/rl/mcts_puct/` + `battle_bots/mcts_puct`** |
| **Algorithm** | Monte Carlo Tree Search + PUCT (Rosin 2011; Silver et al. 2017 formula) |
| **Role** | Main **pure-planning** agent under sparse Cand. 1 (not PPO/DDQN) |
| **Code root** | `showdown/rl/mcts_puct/` |
| **CLI** | `python -m showdown.rl.mcts_puct` |
| **Deps** | **No torch** for v1.0 baseline (`numpy` only). Heuristic bots unchanged. |

> **Scope:** Single source of truth for **implementing** MCTS+PUCT.
> Obs / Action / Reward Cand. 1 / `VGCEnv` are imported from [`rl_env_spec.md`](rl_env_spec.md) v1.4.
>
> **Sibling:** MCTS + Nash (matrix row 4) is a **separate** exclusive spec — out of scope here.
>
> **Immutability:** Bump this doc to 1.1+ **and** `rl_env_spec.md` §6.c together.

---

## 1. Inherited environment

| Contract | Source | MCTS usage |
|----------|--------|------------|
| `VGCEnv` | §1 | Real-game I/O; search uses S2 cloneable sim |
| Reward **Candidate 1** | §4 | Terminal \(z\in\{+1,-1,0\}\) (tie \(=0\), never Cand. 2 \(-0.25\)) |
| Joint legal mask | §2 | Expand / select only \(a\in L(s)\) |
| No `@smogon/calc` | §1 | Also forbidden inside S2 search steps |

**Belief:** flat / unused in v1.0 search (parity with DDQN/PPO flat-belief agents).

---

## 2. MDP delimitation

| Item | Contract |
|------|----------|
| Episode | BO1; max turns **40**; truncate ≠ loss |
| Decision | One combat turn → joint \(a\in\{0,\ldots,575\}\) |
| Team Preview | Out of MDP: `/team 1234` |
| Training | **None** (pure search). Neural prior → **v1.1** |
| Player | `ValidatedMCTSPuctPlayer` |

---

## 3. Action space

**Joint legal edges** (DDQN-compatible):

\[
a \in L(s)\subseteq\{0,\ldots,575\},\quad L(s)=\{a:M_{\mathrm{joint}}(s,a)=1\}
\]

No auto-regressive intermediate nodes (would break MDP parity with DDQN/PPO).

---

## 4. Search (single-agent, open-loop opponent)

### 4.1 Structure

- Decision nodes belong **only** to the searching agent.
- Edges = legal \(a_{\mathrm{self}}\).
- Transition: sample \(a_{\mathrm{opp}}\sim U(L_{\mathrm{opp}}(s))\), then S2 step with True MC stochasticity.
- **Not** a 2P alternating / minimax tree (simultaneous turns).

### 4.2 PUCT selection

\[
a^\star=\arg\max_{a\in L(s)}\Big[
Q(s,a)+c_{\mathrm{puct}}\,P(s,a)\,\frac{\sqrt{N(s)}}{1+N(s,a)}
\Big]
\]

with \(Q=W/N\) (0 if \(N=0\)). Dirichlet noise: **off** in eval.

### 4.3 Expansion

On first visit to a non-terminal node: create edges for all \(a\in L(s)\) with prior \(P(a)=1/|L(s)|\).

### 4.4 Leaf / rollout

If non-terminal after expanding a new child state: **rollout to terminal** (turn cap 40) with **`greedy_joint_snapshot` for both players**. Return Cand. 1 \(z\).

### 4.5 Backup

Tree is single-agent open-loop. Terminal return \(z\in\{+1,-1,0\}\) propagates **unchanged** (no \(-v\)).

For each edge on the path:

\[
N(s,a)\mathrel{+=}1,\quad W(s,a)\mathrel{+=}z\quad(\gamma=1,\ \text{no discount})
\]

### 4.6 Real-game action

\[
a_{\mathrm{play}}=\arg\max_{a\in L(s_0)} N(s_0,a)
\]

(Eval: visit-greedy. **Not** the rollout policy.)

---

## 5. Simulator S2

**Default:** in-process `BattleSnapshot` + local damage heuristic (no Node, no poke-env).

**S1** (live poke-env/PS): offline **fidelity ablation only**.

**`greedy_joint_snapshot`:** damage-heuristic joint greedy **native on `BattleSnapshot`**, no network / poke-env calls. Used **only** in rollouts (both sides).

**Divergence risk (thesis):** If S2 damage/speed heuristics diverge substantially from S1, MCTS converges to decisions optimal for a hallucinated reality. Quantify via S1 fidelity ablations.

**Stochasticity:** True MC sample (damage rolls, accuracy). No Expectimax chance nodes.

---

## 6. Three policies (do not conflate)

| Role | Policy | Where used |
|------|--------|------------|
| Prior \(P(s,a)\) | Uniform legal | PUCT formula |
| Opponent model (tree) | Uniform legal | \(a_{\mathrm{opp}}\) when transitioning in the tree |
| Rollout (leaf) | `greedy_joint_snapshot` | Both players until terminal / turn 40 |
| Played action | \(\arg\max N\) | Sent to Showdown / `VGCEnv` |

Neural prior (PPO/DDQN): **v1.1 only**.

---

## 7. RNG reproducibility

Each simulation uses a **local** RNG seeded by a stable mix of
`(global_seed, root_turn, sim_id)` (e.g. hash → `numpy` SeedSequence)
so sim \(k\) at turn \(t\) always yields the same opp samples and damage rolls.

---

## 8. Frozen hyperparameters (v1.0 baseline)

| Parameter | Value | Notes |
|-----------|-------|-------|
| \(N_{\mathrm{sim}}\) | `200` | `16` smoke. Thesis curve: `{50, 200, 800}` |
| \(c_{\mathrm{puct}}\) | `1.5` | AlphaZero-family constant |
| Tree | Single-agent, open-loop opp | Expand \(a_{\mathrm{self}}\); sample \(a_{\mathrm{opp}}\) |
| Leaf | Rollout to terminal (cap 40) | `greedy_joint_snapshot` both sides; Cand. 1 |
| Simulator | S2 approximate | S1 = fidelity ablation |
| Prior \(P(s,a)\) | Uniform legal | \(1/\|L(s)\|\) |
| Opp model (tree) | Uniform legal | \(a_{\mathrm{opp}}\sim U(L_{\mathrm{opp}})\) |
| Rollout policy | `greedy_joint_snapshot` | Not used as prior / not as played action |
| Action space | Joint 576 legal-only | Parity with DDQN |
| Dirichlet | Off | Eval reproducibility |
| Stochasticity | True MC sample | No chance nodes |
| Seed | Required CLI | Drives per-sim RNG |

---

## 9. Modules & CLI

| Module | Responsibility |
|--------|----------------|
| `simulator.py` | S2 clone / step / terminal Cand. 1 |
| `greedy_snapshot.py` | `greedy_joint_snapshot` |
| `tree.py` | Nodes / edges / PUCT |
| `search.py` | Simulate loop + backup |
| `agent.py` | `act(snapshot) → joint` |
| `player.py` | poke-env + Validated |
| `bench.py` / `__main__.py` | Smoke / eval CLI |
| Tests | `tests/test_rl_mcts_puct.py` |

```bat
python -m showdown.rl.mcts_puct --seed 0 --sims 16 --smoke
```

---

## 10. Evaluation suite

| Opponent | Games | Preview | Sims |
|----------|-------|---------|------|
| `random` | 200 | `/team 1234` | frozen default |
| `greedy_joint` | 200 | `/team 1234` | frozen |
| `expectimax_bayesian` | 200 | `/team 1234` | frozen |

---

## 11. Telemetry (not in reward)

`sims_per_decision`, `tree_nodes`, `avg_depth`, `sim_failures`, `eval_wr_<opp>`, wall time.

---

## 12. Acceptance tests

1. Expanded / selected edges always legal.
2. PUCT argmax matches formula on a mock node.
3. Backup: \(W\mathrel{+=}z\) with \(z\in\{+1,-1,0\}\), no sign flip.
4. Cand. 1 terminals (tie \(=0\)).
5. **RNG:** fixed `(seed, turn, sim_id)` → identical opp action + damage sequence.
6. Smoke: mock game with `--sims 16`.

---

## 13. Non-goals

- **MCTS + Nash equilibrium matrix** (explicitly a **separate sibling spec**)
- Neural prior / value in v1.0
- QD / BD in search
- `@smogon/calc` / Node in S2
- 2P alternating minimax tree
- Team Preview in MDP
- AR Slot1→Slot2 tree nodes

---

## 14. References

- Rosin, C. D. (2011). Multi-armed bandits with episode context. *AMAI* (PUCT theory).
- Silver, D., et al. (2017). Mastering the game of Go without human knowledge. *Nature* (implemented PUCT form \(P\sqrt{N}/(1+N)\)).
- Kocsis & Szepesvári (2006). Bandit based Monte-Carlo planning. *ECML*.
- [`rl_env_spec.md`](rl_env_spec.md) v1.4; siblings [`ddqn_spec.md`](ddqn_spec.md), [`ppo_spec.md`](ppo_spec.md).

---

## Changelog

| Version | Date | Notes |
|---------|------|-------|
| **1.0** | 2026-10-10 | Frozen: joint 576, S2, open-loop, Cand. 1 rollout via `greedy_joint_snapshot`, True MC, uniform prior/opp, \(N_{\mathrm{sim}}=200\), \(c_{\mathrm{puct}}=1.5\). |

---

*End of MCTS+PUCT agent spec 1.0.*
