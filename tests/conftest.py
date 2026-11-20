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


@pytest.fixture(autouse=True)
def _isolate_obs_calc(monkeypatch):
    """P5: los trainers exportan RL_OBS_ABILITIES (se aislaría entre tests) y, con la corrección activa,
    la observación llama a la calculadora real (Node). Por defecto los tests no la usan
    (RL_OBS_CALC=0, respaldo en Python); los tests de integración la activan explícitamente."""
    before = os.environ.get('RL_OBS_ABILITIES')
    monkeypatch.setenv('RL_OBS_CALC', os.environ.get('RL_OBS_CALC_TESTS', '0'))
    # MCTS+PUCT v3 (S3) tampoco llama a Node en los tests salvo que se pida (respaldo S2 por pedido).
    monkeypatch.setenv('MCTS_S3_CALC', os.environ.get('MCTS_S3_CALC_TESTS', '0'))
    try:
        yield
    finally:
        if before is None:
            os.environ.pop('RL_OBS_ABILITIES', None)
        else:
            os.environ['RL_OBS_ABILITIES'] = before
