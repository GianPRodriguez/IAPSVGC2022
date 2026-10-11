# QD-PGA-MAP-Elites Agent Spec — Policy Gradient Assisted MAP-Elites (VGC Gen 9)

| Field | Value |
|-------|--------|
| **Document** | **QD-PGA-MAP-Elites — exclusive implementation contract** |
| **Version** | **1.0** (aligned with `docs/rl_env_spec.md` **v1.7** §6.f) |
| **Status** | **FROZEN — implemented in `showdown/rl/qd_pga/`** |
| **Algorithm** | Grid MAP-Elites + hybrid variation: Gaussian mutation **or** short Masked PPO-clip (PGA) with **shared** critic \(V_\phi\) |
| **Role** | Quality–Diversity **policy** path (PGA variant); illuminate BD; characterize elites |
| **Code root** | `showdown/rl/qd_pga/` |
| **CLI** | `python -m showdown.rl.qd_pga.train` |
| **Deps** | `requirements-rl.txt` — `torch >= 2.0.0` (reuses `showdown/rl/ppo/`, mosaic/eval helpers from `qd_ppo/`) |

> **Scope:** Single source of truth for **implementing QD-PGA-MAP-Elites**.
> Shared contracts (Obs 1024D, ActionSpec AR + IAM, Reward Cand. 2/1, BDSpec,
> `VGCEnv`) live in [`rl_env_spec.md`](rl_env_spec.md). Masked PPO math (clip, IAM)
> lives in [`ppo_spec.md`](ppo_spec.md) v1.0 — **reuse; do not fork**.
>
> **Contrast with QD-PPO:** per-elite full AC + PPO-only variation vs **actor-only**
> elites + **shared** \(V_\phi\) + hybrid mutation / PG emitters.
>
> **Contrast with QD-PPGA:** [`qd_ppga_spec.md`](qd_ppga_spec.md) — proximal-only + arborescence (\(p_{\mathrm{mut}}=0\)).
>
> **Immutability:** bump this doc to 1.1+ **and** `rl_env_spec.md` §6.f together.

---

## 0. OPEN → FROZEN (v1.0)

| ID | Decision |
|----|----------|
| **P1** | Regular 2D Grid MAP-Elites \(10\times 10\) (same BD bins as QD-PPO) |
| **P2** | **Actor-only** elite payload (torso + policy heads); **shared** critic \(V_\phi\) |
| **P3** | Hybrid variation: \(p_{\mathrm{mut}}=0.5\) Gaussian vs PG emitter |
| **P4** | Mutation \(\sigma=0.02\) isotropic Gaussian on actor float params |
| **P5** | Mutation **does not** write the shared replay |
| **P6** | PG emit: \(T_{\mathrm{emit}}=1024\), PPO-clip \(K_{\mathrm{pg}}=2\), Cand. 2, mosaic |
| **P7** | Shared replay capacity \(200\,000\); PG collect **does** write (L2/crash drop) |
| **P8** | Critic: off-policy **1-step TD(0)**, Huber, \(\gamma=0.95\) |
| **P9** | Actor advantage: on-policy **GAE-\(\lambda\)** vs \(V_\phi\), \(\lambda=\gamma=0.95\); **normalize per emit** |
| **P10** | \(f=\) mean Cand. 1 over \(G=20\); insert iff \(g\ge g_{\min}=10\) |
| **P11** | Mosaic foes 25/25/50; serial emitter; deploy \(\pi^\star=\arg\max f\) |
| **P12** | Bootstrap \(B_{\mathrm{boot}}=100\,000\) Cand. 2 on mosaic (counts in \(T\)) |
| **P13** | \(T_{\mathrm{total}}=10^6\) **env steps only** (ignore gradient updates) |

**Operational freeze:**

| Item | Contract |
|------|----------|
| \(n_{\mathrm{emit}}\) | `4` |
| \(N_{\mathrm{outer}}\) | \(\lfloor(T-B_{\mathrm{boot}})/(n_{\mathrm{emit}}T_{\mathrm{emit}})\rfloor=\lfloor 900000/4096\rfloor=219\) |
| Every emit | Costs \(T_{\mathrm{emit}}\) env steps (PG: collect+update; mut: burn-rollout, no replay) |
| Fitness / suite | Eval overhead; **not** in \(T_{\mathrm{total}}\) |
| Replace | Strict \(f_{\mathrm{new}}>f_{\mathrm{old}}\) |
| IAM | Mandatory at sample + PPO recompute |

---

## 1. Inherited environment

| Contract | Source | PGA usage |
|----------|--------|-----------|
| `VGCEnv` | `rl_env_spec` §1 | Rollouts; BD in `info` |
| Obs 1024D | §3 | Actor + \(V_\phi\) torso |
| AR + **IAM (mandatory)** | §2 / `ppo_spec` §3 | Sample + PPO recompute |
| Cand. 2 | §4 | PG / bootstrap / mut burn-rollout |
| Cand. 1 | §4 | Archive fitness \(U=f\) only |
| BDSpec | §5 | Archive coordinates |
| Preview | `/team 1234` | Out of MDP |

### 1.1 Taxonomy \(R\) vs \(U\)

| Signal | Symbol | When |
|--------|--------|------|
| Cand. 2 | \(R_t^{(2)}\) | PG collect, bootstrap, mut burn |
| Cand. 1 | \(U(\pi)=f(\pi)\) | Archive insert + suite |
| BD | \(b\in[0,1]^2\) | Cell index; not in \(R_t\) or \(U\) |

---

## 2. Archive (Grid MAP-Elites, actor-only)

| Item | Contract |
|------|----------|
| Grid | \(10\times 10\) on \([0,1]^2\); same binning as QD-PPO |
| Payload | Actor `state_dict` keys (exclude `value.*`) + \(f\) + BD + meta |
| Insert | Empty → insert; else replace iff \(f'>f\) |
| Parent sample | Uniform over filled cells |

\[
\mathrm{Coverage}=\frac{|\mathrm{filled}|}{100},\quad
\mathrm{QD\text{-}score}=\sum_{c\in\mathrm{filled}}\max(0,f(e_c))
\]

---

## 3. Shared critic \(V_\phi\)

| Item | Contract |
|------|----------|
| Net | MLP \(1024\to 512\to 512\to 1\) (same width as PPO torso) |
| Target | \(y_t=r_t+\gamma V_\phi(s_{t+1})\) with stop-grad on \(y\); \(\gamma=0.95\) |
| Loss | Huber(\(V_\phi(s_t), y_t\)); optional scale \(c_v=0.5\) |
| Data | Minibatches from replay \(200\,000\) |
| LR | \(3\times 10^{-4}\) |
| Counter | Critic gradient steps **do not** increment \(T\) |

---

## 4. Replay

| Item | Contract |
|------|----------|
| Capacity | \(200\,000\) |
| Tuple | \((s,r,s',\mathrm{terminated})\) (value learning) |
| Write | **PG branch + bootstrap only** |
| Drop | `skip_buffer` / `trunc_reason=='crash'` |
| Turn-cap | Store with `terminated=False`; bootstrap via \(V_\phi(s')\) in TD target |

---

## 5. Hybrid variation

With probability \(p_{\mathrm{mut}}=0.5\):

1. \(\theta'\leftarrow\theta+\mathcal{N}(0,\sigma^2 I)\), \(\sigma=0.02\) (actor floats).
2. Burn-rollout \(T_{\mathrm{emit}}\) Cand. 2 steps (mosaic); **no** replay write; **no** PG.
3. Evaluate Cand. 1; try insert.

Otherwise (PG):

1. Clone parent actor; collect \(T_{\mathrm{emit}}\) Cand. 2 steps; **write** replay.
2. Values / GAE from shared \(V_\phi\); normalize advantages **per emit**.
3. PPO-clip \(K_{\mathrm{pg}}=2\), \(\varepsilon=0.2\), \(c_e=0.01\); update **actor only**.
4. Critic TD updates from replay (\(G_c\) minibatches; default `8×256`).
5. Evaluate Cand. 1; try insert.

---

## 6. Mosaic foes

\[
P(\texttt{random})=0.25,\ 
P(\texttt{greedy\_solo})=0.25,\ 
P(\texttt{greedy\_joint})=0.50
\]

i.i.d. per episode. Same as QD-PPO.

---

## 7. Outer loop & budget

```
bootstrap ≤ B_boot Cand2 (replay + critic + actor PG)
seed archive with ≤ 4 aux emits from Stage-1 checkpoint (p_mut coin)
N_outer ← floor((T - steps_used) / (n_emit * T_emit))   # = 219 nominal
for t in 1..N_outer:
    for e in 1..n_emit:
        parent ← Uniform(filled) or seed actor
        if U(0,1) < p_mut: mutate → burn T_emit → eval → insert?
        else: PG collect T_emit → critic → actor PPO-K=2 → eval → insert?
deploy π* = argmax f
```

| Parameter | Value |
|-----------|--------|
| \(T_{\mathrm{total}}\) | `1_000_000` env steps |
| \(B_{\mathrm{boot}}\) | `100_000` |
| \(T_{\mathrm{emit}}\) | `1024` |
| \(n_{\mathrm{emit}}\) | `4` |
| \(N_{\mathrm{outer}}\) | `219` (derived) |
| \(p_{\mathrm{mut}}\) | `0.5` |
| \(\sigma\) | `0.02` |
| \(K_{\mathrm{pg}}\) | `2` |
| Workers | `1` (serial) |

---

## 8. Fitness & BD

Identical to QD-PPO: \(G=20\), \(g_{\min}=10\), \(f\) over all \(G\), BD over eligible \(g\), mosaic foes, greedy masked.

### Offline suite (\(\pi^\star\))

200 BO1 each vs `random`, `greedy_joint`, `expectimax_bayesian`.

---

## 9. Modules & CLI

| Module | Role |
|--------|------|
| `archive.py` | Actor-only grid helpers (bin/insert via QD-PPO `Archive`) |
| `critic.py` | Shared \(V_\phi\) + TD(0) Huber |
| `replay.py` | Capacity 200k value transitions |
| `mutate.py` | Gaussian \(\sigma=0.02\) |
| `improve.py` | PG emit + mut burn-rollout |
| `evaluate.py` | Re-export QD-PPO Cand. 1 eval |
| `mosaic.py` | Re-export 25/25/50 |
| `train.py` | Budgeted outer loop |
| `agent.py` / `player.py` | \(\pi^\star\); `ValidatedQDPGAPlayer` |
| Tests | `tests/test_rl_qd_pga.py` |

```bat
python -m showdown.rl.qd_pga.train --seed 0 --smoke
python -m showdown.rl.qd_pga.train --seed 0 --device cpu
```

---

## 10. Acceptance tests

1. IAM legal samples.  
2. Mutation does not increase `len(replay)`.  
3. PG collect increases replay (when not all skipped).  
4. Actor-only elite keys (no `value.*`).  
5. \(N_{\mathrm{outer}}=\lfloor 900000/4096\rfloor=219\).  
6. \(g<10\) never inserts.  
7. Smoke ≥1 outer iter on mock.

---

## 11. Non-goals (v1.0)

- QD-PPGA; ME-ES as primary; CVT  
- Per-elite critic; mutation → replay  
- Expand-\(\Omega\) / self-play as primary diversity  
- BD inside \(R_t\); confuse \(R^{(2)}\) with \(U\)  
- Multi-emitter parallel; Team Preview in MDP  

---

## 12. References

- Nilsson, O., & Cully, A. (2021). Policy gradient assisted MAP-Elites. *GECCO*.  
- Mouret & Clune (2015). Illuminating search spaces by mapping elites. *arXiv:1504.04909*.  
- Schulman et al. (2017). PPO. *arXiv:1707.06347*.  
- Sutton & Barto (2018). *Reinforcement learning: An introduction* (2nd ed.).  
- Huang & Ontañón (2022). Invalid action masking.  
- [`ppo_spec.md`](ppo_spec.md), [`qd_ppo_spec.md`](qd_ppo_spec.md), [`rl_env_spec.md`](rl_env_spec.md).

---

## Changelog

| Version | Date | Notes |
|---------|------|-------|
| **1.0** | 2026-10-10 | Frozen: shared \(V_\phi\) TD(0)+Huber; GAE-\(\lambda\) actor; \(p_{\mathrm{mut}}=0.5\), \(\sigma=0.02\); no mut→replay; \(T_{\mathrm{emit}}=1024\), \(K=2\), \(N_{\mathrm{outer}}=219\); actor-only elites; mosaic; \(T=10^6\) env steps. |

---

*End of QD-PGA-MAP-Elites agent spec 1.0.*
