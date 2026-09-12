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

1. Arranca el PS Champions local (`noguestsecurity = true` en `config/config.js`):
   ```bat
   cd ..\pokemon-showdown
   node pokemon-showdown start
   ```
2. Corre batallas mirror:
   ```bat
   set DAMAGE_CALC_FORCE_LOCAL=1
   python -m showdown.inference_sim.runner --battles 3
   ```
3. Reporte: `logs/inference_sim_report.txt`

Equipos embebidos: `showdown/inference_sim/team_factory.py` (SableyeVGC, Robin_VGC, Roid Blancaflor).

Para empezar limpio a mano:

```bat
del logs\reporte.txt
del logs\latest_state.txt
del logs\logs_showdown.txt
```
