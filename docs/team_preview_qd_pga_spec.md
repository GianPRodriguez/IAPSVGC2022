# QD-PGA — Team Preview Exclusive Spec (VGC Gen 9 Regulation M-C)

| Campo | Valor |
|-------|--------|
| **Documento** | **Contrato exclusivo — QD-PGA-MAP-Elites solo Team Preview** |
| **Versión** | **1.0** (alineado con `docs/rl_env_spec.md` **v1.13** §6.k) |
| **Estado** | **FROZEN — implementado en `showdown/rl/team_preview_qd_pga/`** |
| **Algoritmo** | Grid MAP-Elites + mutación Gaussiana **o** PPO-clip (PGA); \(V_\phi\) compartido |
| **Rol** | Archivo de **políticas** Flat-360 de Team Preview (estilos de bring/leads) |
| **Formato** | Gen 9 VGC **Regulation M-C**, Open Team Sheets |
| **Code root** | `showdown/rl/team_preview_qd_pga/` |
| **Env / Obs / Action** | Reuso: `team_preview_env.py`, `team_preview_obs.py`, `team_preview_action.py` |
| **CLI** | `python -m showdown.rl.team_preview_qd_pga.train --seed N` |
| **Deps** | `torch >= 2.0`; \(\pi_c=\)`greedy_joint` frozen (mock en CI) |

> **Hermanos:** [`team_preview_masked_ppo_spec.md`](team_preview_masked_ppo_spec.md),
> [`team_preview_nash_spec.md`](team_preview_nash_spec.md);
> combate [`qd_pga_map_elites_spec.md`](qd_pga_map_elites_spec.md).
>
> **Inmutabilidad.** Bump **1.1+** aquí + `rl_env_spec` §6.k juntos.

---

## 0. Por qué existe

Una política PPO preview = un estilo. QD-PGA preview = museo \(10\times10\) de
elites actor-only indexados por BD de bring (speed-control vs offense/support).

---

## 1. MDP (bandit \(T=1\))

| Ítem | Contrato |
|------|----------|
| Acción | Flat-360 Lehmer **un golpe** (no AR de 4 pasos) |
| Obs | \(512=210+210+36+56\) |
| IAM | Softmax masked |
| \(R\) / \(f\) | Cand. **1** \(\{+1,-1,0\}\); **sin** Cand. 2 |
| \(\pi_c\) | `greedy_joint` frozen |
| Self-play preview | Off |

---

## 2. Archivo

Grid \(10\times10\); payload **actor-only** (sin `value.*`); replace estricto;
padres uniforme sobre filled; deploy \(\pi^\star=\arg\max f\).

---

## 3. BD preview \(b\in[0,1]^2\)

Flags OTS/especies (tablas heuristics). BD ∉ \(R\), BD ∉ \(f\).

\[
b_1=\mathrm{clip}_{[0,1]}\big(
0.5\cdot\mathbf{1}[\mathrm{TR\ en\ bring}]
+0.5\cdot\mathbf{1}[\mathrm{TW\ o\ speed\text{-}control}]
\big)
\]

\[
b_2=\mathrm{clip}_{[0,1]}\Big(
\tfrac{N_{\mathrm{leads\ aggro}}+0.5\,N_{\mathrm{bench\ aggro}}}{3}
\Big)
\]

Aggro ≈ Fake Out / no-redirect damage lead (flags `FAKE_OUT` / no `REDIRECT`).

---

## 4. Fitness

\[
f=\tfrac{1}{g}\sum R^{(e)},\quad G=20,\ g_{\min}=10
\]

Mosaic train+eval: **25% random / 25% greedy / 50% expectimax** preview
(mock env usa proxy; live hook opcional).

---

## 5. \(V_\phi\) compartido + PGA

| Ítem | Contrato |
|------|----------|
| Critic | MLP \(512\to256\to256\to1\); \(\gamma=1\); no por celda |
| \(p_{\mathrm{mut}}\) | `0.5` |
| \(\sigma\) | `0.02` |
| Mut | Ruido actor; burn \(T_{\mathrm{emit}}\); **sin** replay |
| PG | Collect + PPO-clip \(K=2\), \(\varepsilon=0.2\), \(c_e=0.02\); IAM; write replay |

---

## 6. Budget (env step = 1 episodio preview)

| Parámetro | Prod | Smoke |
|-----------|------|-------|
| \(B_{\mathrm{boot}}\) | **4096** | 64 |
| \(T_{\mathrm{emit}}\) | **128** | 16 |
| \(n_{\mathrm{emit}}\) | **4** | 2 |
| \(N_{\mathrm{outer}}\) | **256** | 2 |
| \(G\) / \(g_{\min}\) | 20 / 10 | 4 / 2 |

Suite reportada: **200 BO1** vs baselines preview; eval \(\arg\max\) masked.

---

## 7. Non-goals

Genotipo \(\{0,1\}^6\) / 180 / 90; fitness tipado; PGA desde red combate;
BD combate §5; PPGA/arborescencia; crítico por celda; self-play preview.

---

## 8. Referencias

Nilsson & Cully (2021); Mouret & Clune (2015); Schulman (2017); Huang & Ontañón (2022);
[`qd_pga_map_elites_spec.md`](qd_pga_map_elites_spec.md),
[`team_preview_masked_ppo_spec.md`](team_preview_masked_ppo_spec.md).

---

## Changelog

| Versión | Fecha | Notas |
|---------|-------|-------|
| **1.0** | 2026-10-11 | FROZEN: Flat-360 archivo actor-only; BD \(b_1,b_2\); Cand.1; \(N_{\mathrm{outer}}=256\); mut/PGA. |

---

*Fin del contrato Team Preview QD-PGA v1.0.*
