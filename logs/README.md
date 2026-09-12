# Logs de batalla

| Archivo | Uso |
|---------|-----|
| **CMD** | Fuente principal: cada `|request|` imprime un bloque `TURNO N [KIND]` con YO / RIVAL / ACCION. |
| `latest_state.txt` | Copia del último bloque (se sobrescribe). |
| `reporte.txt` | Historial append de esos bloques + banners BO3. |
| `logs_showdown.txt` | Mensajes raw del websocket. |

Formato del bloque:
- Título: `TURNO N [FORCE|DECISION|WAIT|ERR]` (incluye **WAIT** para paridad de rqid con Showdown)
- Nombres duales:
  - Ditto: `ditto -> dragonite`
  - Illusion deducida (PS aún muestra disfraz): `indeedee-f (ZORO/Illusion)`
  - Tras `|replace|` de PS: `zoroark (Illusion was indeedee-f)`
- `stats: Atk=… Def=… SpA=… SpD=… Spe=…` (y sufijo `SPLIT` si Guard Split está activo)
- `ACCION -> ...` dentro del bloque (`(wait)` si no hay decisión)
- OTS (plantilla showteam) solo una vez por partida individual
- Separadores BO3: `BO3 GAME N START` / `BO3 GAME N END | win: ...`

**Limpieza:** al **iniciar** un nuevo BO3 se vacían los tres archivos. Al **terminar** un BO3 **no** se limpian (quedan para revisión).

### Stat Points + damage-calc (multi-agente)

Inferencia de Atk/Def/SpA/SpD/Spe del rival. El calc Champions se comparte así:

1. **Servidor TCP** `127.0.0.1:8765` (un Node para todos los agentes; arranque lazy).
2. Fallback: worker stdio local por proceso (`DAMAGE_CALC_FORCE_LOCAL=1`).
3. Caché SQLite compartida: `logs/calc_cache.sqlite`.

Variables útiles: `DAMAGE_CALC_ROOT`, `DAMAGE_CALC_HOST`, `DAMAGE_CALC_PORT`, `DAMAGE_CALC_CACHE`.

### Harness e2e de inferencia (poke-env + pokemon-showdown Champions)

Valida soundness de Stat Points contra equipos reales Reg M-C ([MetaVGC](https://metavgc.com/teams/featured/regulation-mc)).

1. Arranca el PS Champions local **con `--no-security`** (default del repo; sin rate limits):
   ```bat
   cd ..\pokemon-showdown
   node pokemon-showdown start --no-security
   ```
   Opt-out solo si hace falta: `set PS_NO_SECURITY=0` antes de `--start-ps`.
2. Corre batch heurístico vs random (default 100 BO1, rápido, reporte autónomo).
   Si el PS ya estaba sin `--no-security`, reinícialo:
   ```bat
   set DAMAGE_CALC_FORCE_LOCAL=1
   python -m showdown.inference_sim.run_heuristic_e2e --battles 100 --restart-ps
   ```
   Salida mínima: progreso cada 10 + línea `DONE ... report=logs\heuristic_e2e_report.txt`.
   Progreso crash-safe: `logs\heuristic_e2e_progress.json`.
3. Soundness mirror (opcional): `python -m showdown.inference_sim.runner --battles 10`
4. **Validación Battle vs PS local** (módulo aparte; heuristic vs random + contraste de estado):
   ```bat
   python -m showdown.inference_sim.run_battle_state_e2e --battles 20 --restart-ps
   ```
   Salida: `DONE ... state_mismatch=YES|NO ... report=logs\battle_state_e2e_report.txt`
   Progreso: `logs\battle_state_e2e_progress.json`. No revisar turno a turno.
5. **Robusto A/B + BO3 + sync + inferencia OTS** (recomendado tras cambios de reglas):
   ```bat
   python -m showdown.inference_sim.run_robust_e2e --teams 100
   ```
   N par → N/2 BO3 (pool A heuristic / pool B random, sin solape). Reporte:
   `logs\robust_e2e_report.txt` + `logs\robust_e2e_fixplan.txt`
   (`NEED_FIX=YES|NO`, `NEED_FLAGS_ADD=YES|NO`).
   - `NEED_FIX=NO` → no abras Cursor por sync/SP.
   - `NEED_FLAGS_ADD=YES` → gaps de flags Battle/calc (informativo; no gasta créditos
     solo); pedir «agrega flags de inferencia» si quieres implementarlos.
   Consulta: `python -m showdown.inference_sim.robust_fixplan`

Para refrescar equipos desde la web: `python showdown/inference_sim/scrape_metavgc.py`.

Para empezar limpio a mano:

```bat
del logs\reporte.txt
del logs\latest_state.txt
del logs\logs_showdown.txt
```
