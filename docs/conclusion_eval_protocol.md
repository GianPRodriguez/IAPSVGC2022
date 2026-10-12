# Protocolo de evaluación final (Conclusión)

| Campo | Valor |
|-------|--------|
| **Documento** | Contrato de la eval final de la tesis (preview + turnos) |
| **Versión** | **1.2** |
| **Estado** | **FROZEN — diseño**; harness `showdown.inference_sim.run_conclusion_e2e` |
| **Formato** | Champions **Reg M-C**, Mega **on**, Tera **off** |
| **Code** | `showdown/inference_sim/conclusion_eval/`, `run_conclusion_e2e.py` |
| **Índice de specs** | [`SPECS_INDEX.md`](SPECS_INDEX.md) |

> **Para quien aprende IA.** Hasta aquí cada agente se midió en su capítulo
> (solo preview, o solo turnos, o solo vs random). La Conclusión responde:
> *si armo un bot completo (formación + combate) y lo enfrento a los demás en
> el mismo meta congelado, ¿quién gana?* No se reentrena: se **evalúa**.

---

## 1. Por qué existe

En investigación es fácil confundir “mejoró el preview” con “mejoró el bot”.
Este protocolo fija:

1. Los **mismos** 250 equipos (corte final).
2. Los **mismos** 100 emparejamientos de equipos para todos los cruces.
3. Un **swap** de lados para que un equipo naturalmente fuerte no favorezca
   siempre al mismo agente.
4. Un ranking interpretable (puntos por cruce) con empates controlados (\(\varepsilon\)).

**Analogía:** un torneo de club con fixture y canchas fijas: nadie cambia de
reglamento a mitad, y todos juegan las mismas “canchas” (pares de equipos).

---

## 2. Glosario

| Término | Significado |
|---------|-------------|
| Pipeline | Bot completo = política de **Team Preview** + política de **turnos** |
| Familia | Emparejar preview y combate del mismo paradigma (PPO/Nash/QD) |
| Snapshot | Lista congelada de 250 equipos + split A/B + 100 pares |
| Cruce | Enfrentamiento pipeline \(i\) vs \(j\) (200 BO1) |
| BO1 | Una partida; no hay series BO3 en este protocolo |
| Swap | Repetir los mismos pares con equipos intercambiados |
| \(\varepsilon\) | Banda de empate del cruce (no es exploración RL) |
| Wilson CI | Intervalo de confianza del win-rate binomial |
| DQ | Descalificación si fallos sistemáticos (> 2%) |
| Seed permanente | Un entero que fija snapshot, pares y orden de cruces |

---

## 3. Pipelines (11)

| Grupo | Pipelines | Preview | Turnos |
|-------|-----------|---------|--------|
| Familia RL | `ddqn`, `ppo` | `team_preview_ppo` | DDQN / PPO |
| Familia planning | `mcts_puct`, `mcts_puct_nash` | `team_preview_nash` | MCTS / MCTS–Nash |
| Familia QD | `qd_ppo`, `qd_pga`, `qd_ppga` | `team_preview_qd_pga` | QD-* |
| Heurísticas | `random`, `greedy_solo`, `greedy_joint`, `expectimax_bayesian` | propio §3 | propio combate |

La **lista canónica** (orden \(i < j\) → asientos p1/p2) se congela en
`logs/conclusion_eval/pipelines.json` **justo antes** del run. Default de
desarrollo: `conclusion_eval/pipelines.py`.

| Parámetro | Valor | Propósito | Cómo influye |
|-----------|------:|-----------|--------------|
| Nº pipelines | 11 | Cubrir familias + baselines | Más pipelines → más cruces \(C(n,2)\) |
| Checkpoints | última versión lista al inicio | Congelar calidad | Reentrenar a mitad invalida el ranking |
| Preview wiring | `PipelineSpec.preview` + `preview_checkpoint` | Familia coherente | Players de familia usan `frozen_preview` (PPO/Nash/QD-PGA); fallback `/team 1234` |

Contratos de cada pieza: ver [`SPECS_INDEX.md`](SPECS_INDEX.md).

---

## 4. ¿Hay entrenamiento aquí?

**No.** La Conclusión es solo evaluación.

| Etapa previa | Qué se entrenó / fijó | Spec |
|--------------|----------------------|------|
| Preview PPO / Nash / QD-PGA | Pesos o archive de formación | `team_preview_*_spec` |
| DDQN / PPO / QD combate | Pesos de turnos | `ddqn` / `ppo` / `qd_*` |
| MCTS* | Presupuesto de sims (no gradientes) | `mcts_*_spec` |
| Heurísticas | Nada (reglas fijas) | `*_spec` heurísticos |

Al arrancar el harness se cargan esos artefactos y **no** se actualizan.

---

## 5. Snapshot de equipos

| Parámetro | Valor | Propósito | Cómo influye |
|-----------|------:|-----------|--------------|
| `snapshot_n` | 250 | Meta representativo al corte | Menos equipos → menos diversidad; más → más costo de curación |
| Pool A / B | 125 / 125 | Disjuntos | Evita que ambos usen el mismo paste |
| Dedup | por species+items | Quitar clones | Reduce sesgo a stacks repetidos |
| Seed | permanente (ej. 42) | Reproducibilidad | Cambiarla rehace A/B y pares |
| Crecimiento repo | **off** durante Conclusión | Estabilidad | Añadir equipos a mitad rompería el freeze |

Pasos: corte final → legales Reg M-C → dedup → sample 250 con seed → split A/B →
persistir `logs/conclusion_eval/snapshot.json`.

---

## 6. Pares de equipos (los mismos 100)

| Parámetro | Valor | Propósito | Cómo influye |
|-----------|------:|-----------|--------------|
| `n_pairs` | 100 | Tamaño muestral por lado | Sube precisión del WR y el costo ×2 (por el swap) |
| Muestreo | sin reemplazo | Pares únicos | Con reemplazo repetiría matchups y empeora interpretación |
| Alcance | **todos** los 55 cruces | Comparabilidad | Si cada cruce tuviera pares distintos, el ranking mezcla fuerza de agente y de fixture |
| BO1 por cruce | 200 = 100 AB + 100 BA | Equilibrar equipos | Sin swap, un pool fuerte favorece siempre al mismo asiento de equipo |

Total: \(55 \times 200 = 11\,000\) BO1.

---

## 7. Asientos p1 / p2

Para el cruce \((i,j)\) con \(i < j\) en la lista canónica:

- Agente \(i\) = **p1**, agente \(j\) = **p2** (fijo).
- Mitad AB: p1 usa \(t_A\), p2 usa \(t_B\).
- Mitad BA: p1 usa \(t_B\), p2 usa \(t_A\).

| Parámetro | Valor | Propósito | Cómo influye |
|-----------|------:|-----------|--------------|
| Asiento agente | fijo por cruce | Reproducibilidad protocolo | Evita mezclar sesgo p1/p2 con sesgo de equipo |
| Swap | solo equipos | Control de fuerza de team | Si también se intercambiaran asientos, se confunden dos factores |

---

## 8. Fallos de batalla

Showdown no empata. Abortos raros (WS, timeout harness, etc.):

1. **Reintento 1 vez** el mismo BO1 (mismos equipos y lados).
2. Si falla otra vez → `failures[]`; **no** sustituir por otro par.
3. WR sobre válidos; \(k\approx 0\) se menciona en limitaciones.
4. Si un pipeline falla en **> 2%** de sus partidas → **DQ**.

| Parámetro | Valor | Propósito | Cómo influye |
|-----------|------:|-----------|--------------|
| Reintentos | 1 | Absorber fallos raros | Más reintentos ocultan inestabilidad del agente |
| Sustitución de pares | prohibida | Mantener los mismos 100 | Remuestrear cambiaría el fixture |
| Umbral DQ | 2% | Separar ruido de bot roto | Muy bajo → DQ por rarezas; muy alto → ranking sucio |

---

## 9. Métricas (qué va a Conclusión)

| Métrica | Propósito | Cómo influye / se lee |
|---------|-----------|------------------------|
| WR agregado + **IC Wilson 95%** | Fuerza bruta del cruce | IC ancho → no sobreinterpretar 52–48 |
| Puntos con \(\varepsilon=0.025\) | Ranking 1–11 estable | Empata cruces “casi 50–50”; evita pelear por 101–99 |
| WR lado AB vs BA | Control de sesgo de pool | Si divergen mucho, el split A/B está desbalanceado |
| WR vs bloque heurístico | Claim vs baselines | Responde “¿superamos lo clásico?” |
| Tiempo / timeouts | Solo si hay timeouts sistemáticos | Con presupuesto holgado, no es claim principal |

### Puntos (\(\varepsilon = 0.025\), \(N=200\))

- Empate de cruce si \(|2w-N| \le 2\varepsilon N\) → victorias **95…105** → **0.5 / 0.5**.
- Si no → **1** al de mayor WR.

Ejemplos: 104–96 y 105–95 empatan; **106–94** da el punto.

### Desempate de puestos

1. Puntos de cruce  
2. WR global  
3. H2H  
4. WR vs `{random, greedy_solo, greedy_joint, expectimax_bayesian}`  
5. Id canónico  

---

## 10. Orden de cruces

Los 55 pares \(\{i,j\}\) se barajan con la seed permanente
(`deterministic_cross_order`). Sirve para reanudar el harness y reproducir el
orden en logs.

---

## 11. CLI y artefactos

```bat
python -m showdown.inference_sim.run_conclusion_e2e --write-pipelines
python -m showdown.inference_sim.run_conclusion_e2e --freeze-snapshot --seed 42
python -m showdown.inference_sim.run_conclusion_e2e --seed 42 --start-ps --run
python -m showdown.inference_sim.run_conclusion_e2e --smoke --seed 42 --start-ps
```

PS local: `node pokemon-showdown start --no-security` (vía `--start-ps`).

| Artefacto | Rol |
|-----------|-----|
| `pipelines.json` | Lista canónica de los 11 |
| `snapshot.json` | 250 equipos, A/B, 100 pares |
| `progress.json` | Avance crash-safe |
| `results.json` | Cruces + ranking machine-readable |
| `report.txt` | Tabla de Conclusión |

Tests de protocolo (sin PS): `tests/test_conclusion_eval.py`.

---

## 12. Limitaciones

- El WR está condicionado al meta del snapshot y a \(\pi\) de preview/combate
  elegidas; no es “fuerza absoluta en todo VGC”.
- Familia coherente ≠ victoria automática: un RL flojo puede quedar bajo Expectimax.
- Hasta cablear checkpoints `team_preview_*` en players de combate, el preview
  efectivo puede ser el del bot actual (documentar al congelar `pipelines.json`).

## Changelog

| Versión | Notas |
|---------|--------|
| **1.2** | Preview familiar cableado en players (`frozen_preview` + `preview_checkpoint`); orden preview→combat |
| **1.1** | Redacción dual-audience; tablas propósito/influencia; entrenamiento = no; enlace SPECS_INDEX |
| **1.0** | Diseño inicial + harness |
