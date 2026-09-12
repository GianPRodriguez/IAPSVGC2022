# IAPSVGC — Agente VGC (Pokémon Showdown)

Bot / interfaz de batallas VGC sobre Pokémon Showdown.

## Arranque

```bash
python -m pip install -r requirements.txt
python start.py
```

Credenciales y formato en `.env` (`PS_USERNAME`, `PS_PASSWORD`, `POKEMON_MODE`, `TEAM_DIR`, `BATTLE_BOT`).

Dependencias opcionales:

```bash
pip install -r requirements-dev.txt      # pytest
pip install -r requirements-legacy.txt   # IAs antiguas (NEAT / TensorFlow)
```

## Estructura

| Ruta | Rol |
|------|-----|
| `start.py` / `run.py` | UI Tk y orquestación WebSocket |
| `showdown/` | Motor de batalla y parsers |
| `showdown/battle_bots/random` | Bot activo de pruebas |
| `API/` | Datos de movimientos / Pokémon |
| `teams/` | Equipos exportados |
| `tests/` | Tests de regresión |
| `assets/images/` | Imágenes de la UI |
| `docs/` | PDFs de ayuda + notas de motor |
| `logs/` | Reportes de batallas |
| `old/` | **IAs y scripts legacy** (temporal) |

## Bots (`BATTLE_BOT`)

- `random` — activo (recomendado ahora)
- `safest` — entrada manual por CMD
- `old_smart` / `old_drl` — IAs antiguas (ver `old/README.md`)
