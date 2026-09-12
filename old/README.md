# Legacy (`old/`)

Código de **IAs antiguas** (neuroevolución NEAT y deep reinforcement learning) y scripts de entrenamiento/prueba asociados.

Se conserva aquí de forma temporal. **No usar en el flujo activo** (`BATTLE_BOT=random` + Módulos 1/2).

Dependencias:

```bash
pip install -r requirements-legacy.txt
```

Cuando se integren las nuevas IAs, este directorio se podrá eliminar.

| Ruta | Contenido |
|------|-----------|
| `neuroevolution/` | Entrenamiento NEAT / genomas |
| `deep_reinforcement_learning/` | Entrenamiento DDQN |
| `scripts/` | Scripts sueltos (`pruebita*`, `testAgent*`, `visualize`, etc.) |

Bots de batalla legacy (siguen importables vía `.env`):

- `showdown/battle_bots/old_smart/` → `BATTLE_BOT=old_smart`
- `showdown/battle_bots/old_drl/` → `BATTLE_BOT=old_drl`
