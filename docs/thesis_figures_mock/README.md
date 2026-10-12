# Mock figuras tesis (navegador, lenguaje claro)

Vista previa de gráficas para Cap. resultados con **datos sintéticos** y textos pensados para un lector no técnico, más una **galería** que intenta cargar SVG reales de `logs/figures/`.

> **Figuras reales** (desde `*_history.jsonl` / `reporte.json` al cerrar tune/train):
> `logs/figures/*_latest.svg` y ledger append-only `logs/figures/hp_tune_ledger.jsonl`.
> Regenerar: `python -m showdown.inference_sim.run_plot_tune_history --all`.
>
> La pestaña **Figuras reales** las carga vía HTTP relativo (`../../logs/figures/`).
> Con doble clic (`file://`) el navegador suele bloquear el fetch: usá el servidor abajo
> o la página Streamlit **Figuras y tune**.

## Abrir

```bat
:: recomendado (galería real funciona)
cd ruta\del\repo
py -3 -m http.server 8765
:: http://localhost:8765/docs/thesis_figures_mock/

:: o solo mock:
start docs\thesis_figures_mock\index.html
```

También: `streamlit run showdown/ui/app.py` → **Figuras y tune**.

## Pestañas

1. **Guía rápida** — glosario (WR, pasos, preview congelado, tune, ledger, familias)  
2. **¿Aprenden con el tiempo?** — panorámica de todos  
3. **Familia aprendizaje** — DDQN **junto a** PPO  
4. **Familia planificación** — MCTS+PUCT **junto a** Nash  
5. **Familia diversidad** — QD-PPO · PGA · PPGA  
6. **Pipelines (preview → turnos)** — pairing por familia  
7. **Tune HP e histórico** — mock de trials + prev→nuevo + ledger  
8. **Figuras reales** — SVG desde `logs/figures/` (+ upload local)  
9. **Todos frente a frente** — barras globales + diversidad vs calidad + tabla  

Cada tarjeta mock incluye: *qué significa*, *qué buscar* (caja beige) y etiqueta de *quién se compara*.
