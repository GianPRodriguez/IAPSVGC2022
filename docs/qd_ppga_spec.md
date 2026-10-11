# QD-PPGA Agent Spec — Proximal Policy Gradient Arborescence / MAP-Elites (VGC Gen 9)

| Field | Value |
|-------|--------|
| **Document** | **QD-PPGA — exclusive implementation contract** |
| **Version** | **1.0** (aligned with `docs/rl_env_spec.md` **v1.8** §6.g) |
| **Status** | **FROZEN — implemented in `showdown/rl/qd_ppga/`** |
| **Algorithm** | Grid MAP-Elites + **proximal-only** Masked PPO emitters + shared \(V_\phi\) + **arborescence** parent selection |
| **Role** | Quality–Diversity **policy** path (PPGA); illuminate BD; characterize elites |
| **Code root** | `showdown/rl/qd_ppga/` |
| **CLI** | `python -m showdown.rl.qd_ppga.train` |
| **Deps** | `requirements-rl.txt` — `torch >= 2.0.0` (reuses `ppo/`, `qd_ppo/`, `qd_pga/` critic/replay/PG) |

> **Scope:** Single source of truth for **implementing QD-PPGA**.
> Shared contracts live in [`rl_env_spec.md`](rl_env_spec.md). Masked PPO math in
> [`ppo_spec.md`](ppo_spec.md). Critic/replay/PG collect reuse [`qd_pga_map_elites_spec.md`](qd_pga_map_elites_spec.md).
>
> **Contrast with QD-PPO:** per-elite full AC + uniform parents vs **actor-only** + shared \(V_\phi\) + arborescence.  
> **Contrast with QD-PGA:** hybrid \(p_{\mathrm{mut}}=0.5\) mut/PG vs **\(p_{\mathrm{mut}}=0\)** proximal-only + arborescence.  
> **Non-goal:** continuous DQD / measure gradients / xNES (Batra arborescence adapted to discrete grid).
>
> **Immutability:** bump this doc to 1.1+ **and** `rl_env_spec.md` §6.g together.

---

## 0. OPEN → FROZEN (v1.0)

| ID | Decision |
|----|----------|
| **R1** | Grid \(10\times 10\); **actor-only** elites; shared \(V_\phi\) |
| **R2** | \(p_{\mathrm{mut}}=0\) — **no** genetic mutation |
| **R3** | Variation = Masked PPO-clip only; Cand. 2; mosaic; IAM mandatory |
| **R4** | Critic: off-policy **1-step TD(0)** Huber; replay 200k; \(\gamma=0.95\); LR \(3\times10^{-4}\) |
| **R5** | Actor: **GAE-\(\lambda\)** vs \(V_\phi\); \(\lambda=\gamma=0.95\); **normalize per emit** |
| **R6** | Arborescence parents: \(\alpha=0.5\) uniform(filled) else frontier \(c^\star\) |
| **R7** | \(c^\star\) = empty cell, else filled with **min** \(f\); parent = min \(L_1\) on BD to \(c^\star\) |
| **R8** | \(T_{\mathrm{emit}}=2048\), \(K_{\mathrm{pg}}=4\), \(n_{\mathrm{emit}}=4\) |
| **R9** | \(N_{\mathrm{outer}}=\lfloor 900000/8192\rfloor=109\) |
| **R10** | \(f\) Cand. 1, \(G=20\), \(g_{\min}=10\); mosaic 25/25/50 |
| **R11** | Bootstrap \(B_{\mathrm{boot}}=100\,000\); serial emitter; deploy \(\pi^\star=\arg\max f\) |
| **R12** | \(T_{\mathrm{total}}=10^6\) **env steps only** |

**Defaults (parity Masked PPO / PGA):** \(c_e=0.01\), \(\varepsilon=0.2\), \(c_v=0.5\); critic \(G_c=8\times256\); suite 200 BO1 vs `random` / `greedy_joint` / `expectimax_bayesian`; ≤4 aux PG emits from Stage-1; empty archive → seed actor.

---

## 1. Inherited environment

Same as QD-PGA: `VGCEnv`, Obs 1024D, AR+IAM, Cand. 2 improve / Cand. 1 fitness, BDSpec, `/team 1234`.

---

## 2. Archive (actor-only)

Identical payload/binning/insert/Coverage/QD-score to QD-PGA.

---

## 3. Shared critic \(V_\phi\) + replay

Identical to QD-PGA §§3–4 (TD(0) Huber, 200k, PG writes, L2/crash drop).

---

## 4. Proximal variation (only)

```
parent ← sample_arborescence(A; α=0.5)   # or seed if empty
π' ← deepcopy(actor(parent))
π' ← PPO_improve(π'; T_emit=2048, K=4, R=Cand2, mosaic, V_φ, GAE)
(f, b, g) ← evaluate Cand1 (G=20)
if g ≥ 10: try_insert(A, π', f, b)
```

No mutation branch; no mut burn-rollout.

---

## 5. Arborescence parent selection

| Case | Action |
|------|--------|
| Archive empty | Return bootstrap seed actor |
| \(U\sim\mathrm{Unif}<\alpha\) (\(\alpha=0.5\)) | Uniform over filled cells |
| Else | Pick \(c^\star\): uniform among **empty** cells if any; else cell with **min** \(f\). Parent = filled elite minimizing \(L_1\) distance between BD vectors (or bin centers) and \(c^\star\) |

---

## 6. Mosaic / fitness / suite

Same as QD-PPO / QD-PGA.

---

## 7. Outer loop & budget

```
bootstrap ≤ B_boot Cand2 (replay + critic + actor PPO)
seed archive with ≤ 4 aux PG emits from Stage-1 checkpoint
N_outer ← floor((T - steps_used) / (n_emit * T_emit))   # = 109 nominal
for t in 1..N_outer:
    for e in 1..n_emit:
        parent ← arborescence(A)
        PG emit T_emit → critic → actor PPO-K=4 → eval → insert?
deploy π* = argmax f
```

| Parameter | Value |
|-----------|--------|
| \(T_{\mathrm{total}}\) | `1_000_000` env steps |
| \(B_{\mathrm{boot}}\) | `100_000` |
| \(T_{\mathrm{emit}}\) | `2048` |
| \(K_{\mathrm{pg}}\) | `4` |
| \(n_{\mathrm{emit}}\) | `4` |
| \(N_{\mathrm{outer}}\) | `109` |
| \(\alpha\) | `0.5` |
| \(p_{\mathrm{mut}}\) | `0` |
| Workers | `1` (serial) |

---

## 8. Modules & CLI

| Module | Role |
|--------|------|
| `arborescence.py` | Parent sampling \(\alpha=0.5\) |
| `improve.py` | Re-export `qd_pga.improve.pg_improve` |
| `critic.py` / `replay.py` / `archive.py` | Re-export QD-PGA |
| `mosaic.py` / `evaluate.py` | Re-export QD-PPO helpers |
| `train.py` | Budgeted outer loop |
| `agent.py` / `player.py` | \(\pi^\star\); `ValidatedQDPPGAPlayer` |
| Tests | `tests/test_rl_qd_ppga.py` |

```bat
python -m showdown.rl.qd_ppga.train --seed 0 --smoke
python -m showdown.rl.qd_ppga.train --seed 0 --device cpu
```

---

## 9. Acceptance tests

1. \(p_{\mathrm{mut}}=0\) — no mutate path in train.  
2. \(N_{\mathrm{outer}}=\lfloor 900000/8192\rfloor=109\).  
3. Arborescence: with \(\alpha=0\) always picks frontier; empty → seed.  
4. Actor-only elites; IAM; \(g<10\) reject.  
5. Smoke ≥1 outer iter on mock.

---

## 10. Non-goals (v1.0)

- Genetic mutation (QD-PGA)  
- Continuous DQD / measure Jacobians / xNES coefficient opt  
- Per-elite critic; CVT; ME-ES primary  
- Expand-\(\Omega\) / self-play as primary diversity  
- BD inside \(R_t\)

---

## 11. References

- Batra, S., Tjanaka, B., Fontaine, M., Petrenko, A., Nikolaidis, S., & Sukhatme, G. (2024). Proximal Policy Gradient Arborescence for Quality Diversity Reinforcement Learning. *ICLR*.  
- Nilsson, O., & Cully, A. (2021). Policy gradient assisted MAP-Elites. *GECCO*.  
- Schulman et al. (2017). PPO.  
- Huang & Ontañón (2022). Invalid action masking.  
- Sutton & Barto (2018); Yannakakis & Togelius (2018).  
- [`ppo_spec.md`](ppo_spec.md), [`qd_pga_map_elites_spec.md`](qd_pga_map_elites_spec.md), [`qd_ppo_spec.md`](qd_ppo_spec.md).

---

## Changelog

| Version | Date | Notes |
|---------|------|-------|
| **1.0** | 2026-10-10 | Frozen: actor-only + shared \(V_\phi\); \(p_{\mathrm{mut}}=0\); arborescence \(\alpha=0.5\); \(T_{\mathrm{emit}}=2048\), \(K=4\), \(N_{\mathrm{outer}}=109\); mosaic/fitness/suite parity. |

---

*End of QD-PPGA agent spec 1.0.*
