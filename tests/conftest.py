"""Aislamiento de variables de entorno que el código fija a propósito en procesos RL.

``showdown.rl.pokeenv_backend.rl_team_split`` hace ``os.environ.setdefault('PS_TEAM_SPLIT',
'train')`` para que, dentro de un entrenamiento, el prior de SP y los mundos de expectimax
salgan del mismo grupo de equipos que el entorno. En un proceso de entrenamiento eso es lo
deseado; en una sesión de pytest, la variable quedaba puesta para los tests siguientes y
cambiaba qué equipos devuelve ``active_teams()``.
"""
import os

import pytest


@pytest.fixture(autouse=True)
def _isolate_team_split():
    before = os.environ.get('PS_TEAM_SPLIT')
    try:
        yield
    finally:
        if before is None:
            os.environ.pop('PS_TEAM_SPLIT', None)
        else:
            os.environ['PS_TEAM_SPLIT'] = before
