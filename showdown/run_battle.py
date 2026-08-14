import multiprocessing
import os
import io
import ast
import sys
import math
import json
import time
import queue
import config
import importlib
import constants
import traceback
import concurrent.futures

from os import path
from copy import deepcopy
from showdown.battle_modifier import update_battle_environment
from showdown.battle_modifier import updateStatsOpponents
from showdown.websocket_client import PSWebsocketClient
from showdown.battle import Battle
from showdown.battle_bots.helpers import format_decision

def NeuralNetworkInput(battle):
    listNeuralNetworkInput = []
    # 1 to 4
    if battle.force_switch:
        izquierdaSwitch = (battle.request_json)["forceSwitch"][0]
        derechaSwitch = (battle.request_json)["forceSwitch"][1]
        if izquierdaSwitch and derechaSwitch:
            listNeuralNetworkInput.append(1.0)
            listNeuralNetworkInput.append(0.0)
            listNeuralNetworkInput.append(0.0)
            listNeuralNetworkInput.append(0.0)
        elif izquierdaSwitch:
            listNeuralNetworkInput.append(0.0)
            listNeuralNetworkInput.append(0.0)
            listNeuralNetworkInput.append(1.0)
            listNeuralNetworkInput.append(0.0)
        elif derechaSwitch:
            listNeuralNetworkInput.append(0.0)
            listNeuralNetworkInput.append(1.0)
            listNeuralNetworkInput.append(0.0)
            listNeuralNetworkInput.append(0.0)
        else:
            print("ERROR 1 - NeuralNetworkInput")
    else:
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
    # 5 to 10
    if battle.user.active1.name == 'zacian-crowned':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
    elif battle.user.active1.name == 'calyrex-shadow':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.user.active1.name == 'incineroar':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.user.active1.name == 'gastrodon':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.user.active1.name == 'rillaboom':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.user.active1.name == 'thundurus-incarnate':
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    else:
        print("ERROR 2 - NeuralNetworkInput")
    # 11 to 14
    listNeuralNetworkInput.append(battle.user.active1.HPB*1.0)
    listNeuralNetworkInput.append(battle.user.active1.ATB*1.0)
    listNeuralNetworkInput.append(battle.user.active1.DEB*1.0)
    listNeuralNetworkInput.append(battle.user.active1.VEB*1.0)
    # 15 to 17
    listNeuralNetworkInput.append(battle.user.active1.ASL*1.0)
    listNeuralNetworkInput.append(battle.user.active1.PAR*1.0)
    listNeuralNetworkInput.append(battle.user.active1.DRO*1.0)
    # 18 to 27
    if battle.user.active1.isDynamax:
        listNeuralNetworkInput.append(1.0)
    else:
        listNeuralNetworkInput.append(0.0)
    if battle.user.active1.can_dynamax:
        listNeuralNetworkInput.append(1.0)
    else:
        listNeuralNetworkInput.append(0.0)
    if not battle.force_switch:
        if battle.user.active1.movePP[0] > 0:
            listNeuralNetworkInput.append(1.0)
        else:
            listNeuralNetworkInput.append(0.0)
        try:
            if (battle.request_json)['active'][0]['moves'][0]['target'] == 'allAdjacentFoes' or (battle.request_json)['active'][0]['moves'][0]['target'] == 'allySide' or (battle.request_json)['active'][0]['moves'][0]['target'] == 'self' or (battle.request_json)['active'][0]['moves'][0]['target'] == 'all':
                listNeuralNetworkInput.append(0.0)
            else:
                listNeuralNetworkInput.append(1.0)
        except:
            listNeuralNetworkInput.append(0.0)
        if battle.user.active1.movePP[1] > 0:
            listNeuralNetworkInput.append(1.0)
        else:
            listNeuralNetworkInput.append(0.0)
        try:
            if (battle.request_json)['active'][0]['moves'][1]['target'] == 'allAdjacentFoes' or (battle.request_json)['active'][0]['moves'][1]['target'] == 'allySide' or (battle.request_json)['active'][0]['moves'][1]['target'] == 'self' or (battle.request_json)['active'][0]['moves'][1]['target'] == 'all':
                listNeuralNetworkInput.append(0.0)
            else:
                listNeuralNetworkInput.append(1.0)
        except:
            listNeuralNetworkInput.append(0.0)
        if battle.user.active1.movePP[2] > 0:
            listNeuralNetworkInput.append(1.0)
        else:
            listNeuralNetworkInput.append(0.0)
        try:
            if (battle.request_json)['active'][0]['moves'][2]['target'] == 'allAdjacentFoes' or (battle.request_json)['active'][0]['moves'][2]['target'] == 'allySide' or (battle.request_json)['active'][0]['moves'][2]['target'] == 'self' or (battle.request_json)['active'][0]['moves'][2]['target'] == 'all':
                listNeuralNetworkInput.append(0.0)
            else:
                listNeuralNetworkInput.append(1.0)
        except:
            listNeuralNetworkInput.append(0.0)
        if battle.user.active1.movePP[3] > 0:
            listNeuralNetworkInput.append(1.0)
        else:
            listNeuralNetworkInput.append(0.0)
        try:
            if (battle.request_json)['active'][0]['moves'][3]['target'] == 'allAdjacentFoes' or (battle.request_json)['active'][0]['moves'][3]['target'] == 'allySide' or (battle.request_json)['active'][0]['moves'][3]['target'] == 'self' or (battle.request_json)['active'][0]['moves'][3]['target'] == 'all':
                listNeuralNetworkInput.append(0.0)
            else:
                listNeuralNetworkInput.append(1.0)
        except:
            listNeuralNetworkInput.append(0.0)
    else:
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    # 28 to 33
    if battle.user.active2.name == 'zacian-crowned':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
    elif battle.user.active2.name == 'calyrex-shadow':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.user.active2.name == 'incineroar':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.user.active2.name == 'gastrodon':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.user.active2.name == 'rillaboom':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.user.active2.name == 'thundurus-incarnate':
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    else:
        print("ERROR 3 - NeuralNetworkInput")
    # 34 to 37
    listNeuralNetworkInput.append(battle.user.active2.HPB*1.0)
    listNeuralNetworkInput.append(battle.user.active2.ATB*1.0)
    listNeuralNetworkInput.append(battle.user.active2.DEB*1.0)
    listNeuralNetworkInput.append(battle.user.active2.VEB*1.0)
    # 38 to 40
    listNeuralNetworkInput.append(battle.user.active2.ASL*1.0)
    listNeuralNetworkInput.append(battle.user.active2.PAR*1.0)
    listNeuralNetworkInput.append(battle.user.active2.DRO*1.0)
    # 41 to 50
    if battle.user.active2.isDynamax:
        listNeuralNetworkInput.append(1.0)
    else:
        listNeuralNetworkInput.append(0.0)
    if battle.user.active2.can_dynamax:
        listNeuralNetworkInput.append(1.0)
    else:
        listNeuralNetworkInput.append(0.0)
    if not battle.force_switch:
        if battle.user.active2.movePP[0] > 0:
            listNeuralNetworkInput.append(1.0)
        else:
            listNeuralNetworkInput.append(0.0)
        try:
            if (battle.request_json)['active'][1]['moves'][0]['target'] == 'allAdjacentFoes' or (battle.request_json)['active'][1]['moves'][0]['target'] == 'allySide' or (battle.request_json)['active'][1]['moves'][0]['target'] == 'self' or (battle.request_json)['active'][1]['moves'][0]['target'] == 'all':
                listNeuralNetworkInput.append(0.0)
            else:
                listNeuralNetworkInput.append(1.0)
        except:
            listNeuralNetworkInput.append(0.0)
        if battle.user.active2.movePP[1] > 0:
            listNeuralNetworkInput.append(1.0)
        else:
            listNeuralNetworkInput.append(0.0)
        try:
            if (battle.request_json)['active'][1]['moves'][1]['target'] == 'allAdjacentFoes' or (battle.request_json)['active'][1]['moves'][1]['target'] == 'allySide' or (battle.request_json)['active'][1]['moves'][1]['target'] == 'self' or (battle.request_json)['active'][1]['moves'][1]['target'] == 'all':
                listNeuralNetworkInput.append(0.0)
            else:
                listNeuralNetworkInput.append(1.0)
        except:
            listNeuralNetworkInput.append(0.0)
        if battle.user.active2.movePP[2] > 0:
            listNeuralNetworkInput.append(1.0)
        else:
            listNeuralNetworkInput.append(0.0)
        try:
            if (battle.request_json)['active'][1]['moves'][2]['target'] == 'allAdjacentFoes' or (battle.request_json)['active'][1]['moves'][2]['target'] == 'allySide' or (battle.request_json)['active'][1]['moves'][2]['target'] == 'self' or (battle.request_json)['active'][1]['moves'][2]['target'] == 'all':
                listNeuralNetworkInput.append(0.0)
            else:
                listNeuralNetworkInput.append(1.0)
        except:
            listNeuralNetworkInput.append(0.0)
        if battle.user.active2.movePP[3] > 0:
            listNeuralNetworkInput.append(1.0)
        else:
            listNeuralNetworkInput.append(0.0)
        try:
            if (battle.request_json)['active'][1]['moves'][3]['target'] == 'allAdjacentFoes' or (battle.request_json)['active'][1]['moves'][3]['target'] == 'allySide' or (battle.request_json)['active'][1]['moves'][3]['target'] == 'self' or (battle.request_json)['active'][1]['moves'][3]['target'] == 'all':
                listNeuralNetworkInput.append(0.0)
            else:
                listNeuralNetworkInput.append(1.0)
        except:
            listNeuralNetworkInput.append(0.0)
    else:
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    # 51 to 56
    if battle.user.reserved[0].name == 'zacian-crowned':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
    elif battle.user.reserved[0].name == 'calyrex-shadow':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.user.reserved[0].name == 'incineroar':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.user.reserved[0].name == 'gastrodon':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.user.reserved[0].name == 'rillaboom':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.user.reserved[0].name == 'thundurus-incarnate':
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    else:
        print("ERROR 4 - NeuralNetworkInput")
    # 57 to 60
    listNeuralNetworkInput.append(battle.user.reserved[0].HPB*1.0)
    listNeuralNetworkInput.append(battle.user.reserved[0].ATB*1.0)
    listNeuralNetworkInput.append(battle.user.reserved[0].DEB*1.0)
    listNeuralNetworkInput.append(battle.user.reserved[0].VEB*1.0)
    # 61 to 62
    listNeuralNetworkInput.append(battle.user.reserved[0].ASL*1.0)
    listNeuralNetworkInput.append(battle.user.reserved[0].PAR*1.0)
    # 63 to 68
    if battle.user.reserved[1].name == 'zacian-crowned':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
    elif battle.user.reserved[1].name == 'calyrex-shadow':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.user.reserved[1].name == 'incineroar':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.user.reserved[1].name == 'gastrodon':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.user.reserved[1].name == 'rillaboom':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.user.reserved[1].name == 'thundurus-incarnate':
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    else:
        print("ERROR 5 - NeuralNetworkInput")
    # 69 to 72
    listNeuralNetworkInput.append(battle.user.reserved[1].HPB*1.0)
    listNeuralNetworkInput.append(battle.user.reserved[1].ATB*1.0)
    listNeuralNetworkInput.append(battle.user.reserved[1].DEB*1.0)
    listNeuralNetworkInput.append(battle.user.reserved[1].VEB*1.0)
    # 73 to 74
    listNeuralNetworkInput.append(battle.user.reserved[1].ASL*1.0)
    listNeuralNetworkInput.append(battle.user.reserved[1].PAR*1.0)
    # 75
    if battle.user.side_conditions['tailwind'] > 0:
        listNeuralNetworkInput.append(1.0)
    else:
        listNeuralNetworkInput.append(0.0)
    # 76 to 77
    if battle.user.side_conditions['auroraveil'] > 0:
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(1.0)
    else:
        if battle.user.side_conditions['lightscreen'] > 0:
            listNeuralNetworkInput.append(1.0)
        else:
            listNeuralNetworkInput.append(0.0)
        if battle.user.side_conditions['reflect'] > 0:
            listNeuralNetworkInput.append(1.0)
        else:
            listNeuralNetworkInput.append(0.0)
    # 78 to 96
    if battle.opponent.active1.type1 == 'Null':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
    elif battle.opponent.active1.type1 == 'normal':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active1.type1 == 'fire':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active1.type1 == 'water':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active1.type1 == 'grass':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active1.type1 == 'electric':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active1.type1 == 'ice':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active1.type1 == 'fighting':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active1.type1 == 'poison':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active1.type1 == 'ground':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active1.type1 == 'flying':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active1.type1 == 'psychic':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active1.type1 == 'bug':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active1.type1 == 'rock':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active1.type1 == 'ghost':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active1.type1 == 'dragon':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active1.type1 == 'dark':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active1.type1 == 'steel':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active1.type1 == 'fairy':
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    else:
        print("ERROR 6 - NeuralNetworkInput")
    # 97 to 115
    if battle.opponent.active1.type2 == 'Null':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
    elif battle.opponent.active1.type2 == 'normal':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active1.type2 == 'fire':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active1.type2 == 'water':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active1.type2 == 'grass':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active1.type2 == 'electric':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active1.type2 == 'ice':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active1.type2 == 'fighting':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active1.type2 == 'poison':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active1.type2 == 'ground':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active1.type2 == 'flying':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active1.type2 == 'psychic':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active1.type2 == 'bug':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active1.type2 == 'rock':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active1.type2 == 'ghost':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active1.type2 == 'dragon':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active1.type2 == 'dark':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active1.type2 == 'steel':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active1.type2 == 'fairy':
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    else:
        print("ERROR 7 - NeuralNetworkInput")
    # 116 to 119
    listNeuralNetworkInput.append(battle.opponent.active1.HPB*1.0)
    listNeuralNetworkInput.append(battle.opponent.active1.ATB*1.0)
    listNeuralNetworkInput.append(battle.opponent.active1.DEB*1.0)
    listNeuralNetworkInput.append(battle.opponent.active1.VEB*1.0)
    # 120 to 123
    if battle.opponent.active1.isDynamax:
        listNeuralNetworkInput.append(1.0)
    else:
        listNeuralNetworkInput.append(0.0)
    listNeuralNetworkInput.append(battle.opponent.active1.ASL*1.0)
    listNeuralNetworkInput.append(battle.opponent.active1.PAR*1.0)
    listNeuralNetworkInput.append(battle.opponent.active1.DRO*1.0)
    # 124 to 141
    listNeuralNetworkInput.append(battle.opponent.active1.moveType.get('normal',0.0)*1.0)
    listNeuralNetworkInput.append(battle.opponent.active1.moveType.get('fire',0.0)*1.0)
    listNeuralNetworkInput.append(battle.opponent.active1.moveType.get('water',0.0)*1.0)
    listNeuralNetworkInput.append(battle.opponent.active1.moveType.get('grass',0.0)*1.0)
    listNeuralNetworkInput.append(battle.opponent.active1.moveType.get('electric',0.0)*1.0)
    listNeuralNetworkInput.append(battle.opponent.active1.moveType.get('ice',0.0)*1.0)
    listNeuralNetworkInput.append(battle.opponent.active1.moveType.get('fighting',0.0)*1.0)
    listNeuralNetworkInput.append(battle.opponent.active1.moveType.get('poison',0.0)*1.0)
    listNeuralNetworkInput.append(battle.opponent.active1.moveType.get('ground',0.0)*1.0)
    listNeuralNetworkInput.append(battle.opponent.active1.moveType.get('flying',0.0)*1.0)
    listNeuralNetworkInput.append(battle.opponent.active1.moveType.get('psychic',0.0)*1.0)
    listNeuralNetworkInput.append(battle.opponent.active1.moveType.get('bug',0.0)*1.0)
    listNeuralNetworkInput.append(battle.opponent.active1.moveType.get('rock',0.0)*1.0)
    listNeuralNetworkInput.append(battle.opponent.active1.moveType.get('ghost',0.0)*1.0)
    listNeuralNetworkInput.append(battle.opponent.active1.moveType.get('dragon',0.0)*1.0)
    listNeuralNetworkInput.append(battle.opponent.active1.moveType.get('dark',0.0)*1.0)
    listNeuralNetworkInput.append(battle.opponent.active1.moveType.get('steel',0.0)*1.0)
    listNeuralNetworkInput.append(battle.opponent.active1.moveType.get('fairy',0.0)*1.0)
    # 142 to 160
    if battle.opponent.active2.type1 == 'Null':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
    elif battle.opponent.active2.type1 == 'normal':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active2.type1 == 'fire':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active2.type1 == 'water':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active2.type1 == 'grass':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active2.type1 == 'electric':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active2.type1 == 'ice':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active2.type1 == 'fighting':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active2.type1 == 'poison':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active2.type1 == 'ground':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active2.type1 == 'flying':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active2.type1 == 'psychic':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active2.type1 == 'bug':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active2.type1 == 'rock':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active2.type1 == 'ghost':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active2.type1 == 'dragon':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active2.type1 == 'dark':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active2.type1 == 'steel':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active2.type1 == 'fairy':
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    else:
        print("ERROR 8 - NeuralNetworkInput")
    # 161 to 179
    if battle.opponent.active2.type2 == 'Null':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
    elif battle.opponent.active2.type2 == 'normal':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active2.type2 == 'fire':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active2.type2 == 'water':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active2.type2 == 'grass':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active2.type2 == 'electric':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active2.type2 == 'ice':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active2.type2 == 'fighting':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active2.type2 == 'poison':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active2.type2 == 'ground':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active2.type2 == 'flying':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active2.type2 == 'psychic':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active2.type2 == 'bug':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active2.type2 == 'rock':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active2.type2 == 'ghost':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active2.type2 == 'dragon':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active2.type2 == 'dark':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active2.type2 == 'steel':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.opponent.active2.type2 == 'fairy':
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    else:
        print("ERROR 9 - NeuralNetworkInput")
    # 180 to 183
    listNeuralNetworkInput.append(battle.opponent.active2.HPB*1.0)
    listNeuralNetworkInput.append(battle.opponent.active2.ATB*1.0)
    listNeuralNetworkInput.append(battle.opponent.active2.DEB*1.0)
    listNeuralNetworkInput.append(battle.opponent.active2.VEB*1.0)
    # 184 to 187
    if battle.opponent.active2.isDynamax:
        listNeuralNetworkInput.append(1.0)
    else:
        listNeuralNetworkInput.append(0.0)
    listNeuralNetworkInput.append(battle.opponent.active2.ASL*1.0)
    listNeuralNetworkInput.append(battle.opponent.active2.PAR*1.0)
    listNeuralNetworkInput.append(battle.opponent.active2.DRO*1.0)
    # 188 to 205
    listNeuralNetworkInput.append(battle.opponent.active2.moveType.get('normal',0.0)*1.0)
    listNeuralNetworkInput.append(battle.opponent.active2.moveType.get('fire',0.0)*1.0)
    listNeuralNetworkInput.append(battle.opponent.active2.moveType.get('water',0.0)*1.0)
    listNeuralNetworkInput.append(battle.opponent.active2.moveType.get('grass',0.0)*1.0)
    listNeuralNetworkInput.append(battle.opponent.active2.moveType.get('electric',0.0)*1.0)
    listNeuralNetworkInput.append(battle.opponent.active2.moveType.get('ice',0.0)*1.0)
    listNeuralNetworkInput.append(battle.opponent.active2.moveType.get('fighting',0.0)*1.0)
    listNeuralNetworkInput.append(battle.opponent.active2.moveType.get('poison',0.0)*1.0)
    listNeuralNetworkInput.append(battle.opponent.active2.moveType.get('ground',0.0)*1.0)
    listNeuralNetworkInput.append(battle.opponent.active2.moveType.get('flying',0.0)*1.0)
    listNeuralNetworkInput.append(battle.opponent.active2.moveType.get('psychic',0.0)*1.0)
    listNeuralNetworkInput.append(battle.opponent.active2.moveType.get('bug',0.0)*1.0)
    listNeuralNetworkInput.append(battle.opponent.active2.moveType.get('rock',0.0)*1.0)
    listNeuralNetworkInput.append(battle.opponent.active2.moveType.get('ghost',0.0)*1.0)
    listNeuralNetworkInput.append(battle.opponent.active2.moveType.get('dragon',0.0)*1.0)
    listNeuralNetworkInput.append(battle.opponent.active2.moveType.get('dark',0.0)*1.0)
    listNeuralNetworkInput.append(battle.opponent.active2.moveType.get('steel',0.0)*1.0)
    listNeuralNetworkInput.append(battle.opponent.active2.moveType.get('fairy',0.0)*1.0)
    # 206
    if battle.opponent.side_conditions['tailwind'] > 0:
        listNeuralNetworkInput.append(1.0)
    else:
        listNeuralNetworkInput.append(0.0)
    # 207 to 208
    if battle.opponent.side_conditions['auroraveil'] > 0:
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(1.0)
    else:
        if battle.opponent.side_conditions['lightscreen'] > 0:
            listNeuralNetworkInput.append(1.0)
        else:
            listNeuralNetworkInput.append(0.0)
        if battle.opponent.side_conditions['reflect'] > 0:
            listNeuralNetworkInput.append(1.0)
        else:
            listNeuralNetworkInput.append(0.0)
    # 209
    if battle.trick_room > 0:
        listNeuralNetworkInput.append(1.0)
    else:
        listNeuralNetworkInput.append(0.0)
    # 210 to 214
    if battle.weather == 'none' or battle.weather == None:
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
    elif battle.weather == 'raindance':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.weather == 'sunnyday':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.weather == 'sandstorm':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.weather == 'hail':
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    else:
        print("ERROR 10 - NeuralNetworkInput")
    # 215 to 219
    if battle.terrain == 'none' or battle.terrain == None:
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
    elif battle.terrain == 'electricterrain':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.terrain == 'grassyterrain':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.terrain == 'mistyterrain':
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    elif battle.terrain == 'psychicterrain':
        listNeuralNetworkInput.append(1.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
        listNeuralNetworkInput.append(0.0)
    else:
        print("ERROR 11 - NeuralNetworkInput")
    return listNeuralNetworkInput


####################################################################################################################################################
### OK - Retorna True si la batalla ha terminado, False en caso contrario, usa como referencia el battle_tag y el mensaje recibido del WebSocket ###
####################################################################################################################################################
def battle_is_finished(battle_tag: str, msg: str) -> bool:
    return (
        msg.startswith(">{}".format(battle_tag)) and
        (constants.WIN_STRING in msg) and constants.CHAT_STRING not in msg
    )

##############################################################################################################################################
# Retorna la acción en texto, en base al estado actual del entorno (battle) y el agente usado, que luego es enviado inmediatamente al WebSocket
##############################################################################################################################################
async def createBattleAction(battle, msg):
    # ¿Hay solicitud de JSON pendiente? 
    if msg:
        # Actualiza el estado de los pkmn del usuario en base al json recibido
        battle.user.updatePkmnUser(msg, first_turn=False)
        # Actualiza SOLO las estadísticas de los PKMN Oponentes Activos
        await updateStatsOpponents(battle)

    # listNeuralNetworkInput es el estado del entorno, una lista de 220 valores que se envía a la red neuronal para que decida la mejor acción
    listNeuralNetworkInput = NeuralNetworkInput(battle)
    # Se define la mejor acción en base al listNeuralNetworkInput: moveSwitch1, nMoveSwitch1, dynamax1, toMove1, moveSwitch2, nMoveSwitch2, dynamax2, toMove2
    moveSwitch1, nMoveSwitch1, dynamax1, toMove1, moveSwitch2, nMoveSwitch2, dynamax2, toMove2, battle.reward, battle.score = await battle.action_battle(battle, listNeuralNetworkInput)
    # actionToWebSocket es lo que envío directamente al WebSocket para ejecutar la acción, format_decision lo formatea
    actionToWebSocket = await format_decision(moveSwitch1, nMoveSwitch1, dynamax1, toMove1, moveSwitch2, nMoveSwitch2, dynamax2, toMove2)

    # Retorno el mensaje listo para enviarlo al WebSocket
    return battle, actionToWebSocket

###############################################################################################################################################
# OK - Dado el mensaje, se obtiene el ganador de la Batalla Individual y devuelvo el Nombre del Ganador, en caso contrario lanzo excepción
###############################################################################################################################################
async def handle_finished_battle(ps_websocket_client: PSWebsocketClient, battle, q, mensaje: str):

    if constants.WIN_STRING in mensaje:  # Si hay un ganador en la Batalla Individual...
        winner = mensaje.split(constants.WIN_STRING)[-1].split('\n')[0].strip(" ")  # Se almacena el nombre del ganador
    else:
        raise ValueError("¡La batalla terminó sin un ganador válido") # Sino, se lanza excepción

    q.put([-1, winner == battle.opponent.account_name, battle.user.account_name, battle.opponent.account_name])  # Envío el ganador a la Interfaz e indico que ha finalizado la Batalla Individual
    print("[INFO]  Batalla Individual: {} - Ganador: {}".format(battle.battleBO3_count, winner))

    await ps_websocket_client.send_message(battle.battle_tag, [config.battle_ending_message])  # Es opcional enviar un mensaje de despedida al oponente...
    await ps_websocket_client.leave_battle(battle.battle_tag, save_replay=config.save_replay)  # Es opcional guardar la batalla...


###################################################################################################################################################
### OK - Método que se encarga de manejar los mensajes de actualización de entorno y tipo de solicitud de acción, retorna la acción a realizar  ###
###################################################################################################################################################
async def handle_active_battle_turn(ps_websocket_client: PSWebsocketClient, battle: Battle, msg_t: str) -> None: 
    # Recibir el mensaje del WebSocket que contiene el |request|
    msg_request = await ps_websocket_client.receive_message()

    # Actualizar todas las variables del entorno (Pokémon activos, stats, estados, etc.)
    await update_battle_environment(battle, msg_t, msg_request)

    # 1. Si el servidor solicita esperar (p. ej. el oponente está realizando U-Turn / cambio)
    if battle.wait or "wait" in msg_request:
        return

    # 2. Si el turno requiere una acción activa o un cambio forzado (forceSwitch)
    if battle.force_switch or "active" in msg_request or "forceSwitch" in msg_request:
        battle.error = False
        battle, battle_turn_action = await createBattleAction(battle, msg_request)
        await ps_websocket_client.send_message(battle.battle_tag, battle_turn_action)
        return

    # 3. Si no es ni wait, ni active, ni forceSwitch, lanzamos una excepción específica
    raise ValueError(f"Se ha recibido un tipo de |request| inválido o desconocido: {msg_request}")

#######################################################################################################################
# OK - MENSAJES CON ERROR... NO USAR POR AHORA
#######################################################################################################################
async def handle_invalid_action(ps_websocket_client, battle, q):
    # Se genera la entrada de la red neuronal para registrar la acción inválida
    listNeuralNetworkInput = NeuralNetworkInput(battle)

    if battle.cantActions != 0:
        q.put([-10,listNeuralNetworkInput,battle.turn+1,0,battle.score,math.floor(battle.cantInvActions*100/battle.cantActions)])
    else:
        q.put([-10,listNeuralNetworkInput,battle.turn+1,0,battle.score,math.floor(-1)])
    return "None"


async def handle_error_message(ps_websocket_client, battle, q, split_msg):
    if len(split_msg) < 3 or split_msg[2] == '[Invalid choice] Sorry, too late to make a different move; the next turn has already started':
        return await handle_invalid_action(ps_websocket_client, battle, q)
    return None

#######################################################################################################################
# OK - Escoge la acción para la vista previa del equipo
#######################################################################################################################
async def send_action_team_preview(ps_websocket_client, battle):

    print("[INFO] Acción Vista Previa del Equipo") # Se imprime en CMD la info de la Fase

    # Se escoge los 4 Pokémon: action_team_preview = tupla (5, 2, 1, 4)
    action_team_preview = await battle.action_team_preview(battle)
    # Se formatea en String; action_team_preview_str = "5214"
    action_team_preview_str = "".join(map(str, action_team_preview))
    # Se prepara el String que se va a enviar al WebSocket: mensaje = "/team 314256|2"
    mensaje = ["/team {}|{}".format(action_team_preview_str, battle.rqid)]
    print("[SEND] Acción Vista Previa del Equipo: {}".format(mensaje))
    
    # Se envía la acción al WebSocket y regreso al método que llamó a esta función, que es start_team_preview_phase
    await ps_websocket_client.send_message(battle.battle_tag, mensaje)

#######################################################################################################################
# OK - Obtiene el equipo del oponente en base a la vista previa del equipo
#######################################################################################################################
async def get_opponent_team(preview_string_lines: str, opponent_id: str) -> list:
    equipo = []
    # El prefijo que estamos buscando en las líneas, por ejemplo: "|showteam|p2|"
    target_prefix = f"|showteam|{opponent_id}|"
    
    # Recorremos cada línea del string
    for linea in preview_string_lines.strip().split('\n'):
        if linea.startswith(target_prefix):
            # Removemos el prefijo para quedarnos solo con la cadena de los Pokémon
            team_data = linea[len(target_prefix):]
            
            # Cada Pokémon está separado por el carácter ']'
            pokemon_raw_list = team_data.split(']')
            
            for pkmn_str in pokemon_raw_list:
                if not pkmn_str.strip():
                    continue
                
                # Los atributos de cada Pokémon están separados por '|'
                atributos = pkmn_str.split('|')
                
                # Extraemos según el índice correspondiente en la cadena de Showdown
                nombre = atributos[0]
                objeto = atributos[2]
                habilidad = atributos[3]
                movimientos = atributos[4].split(',')
                naturaleza = atributos[5]
                
                # Formateamos el Pokémon y lo agregamos a la lista final
                pkmn_formateado = [nombre, objeto, habilidad, movimientos, naturaleza]
                equipo.append(pkmn_formateado)
            
            # Una vez que encontramos y procesamos la línea del equipo, podemos detener la búsqueda
            break
            
    return equipo

#######################################################################################################################
# OK - Obtiene el battle_tag de cada una de las Batallas Individuales
#######################################################################################################################
async def get_battle_tag(ps_websocket_client: PSWebsocketClient):
    while True:
        # Voy recibiendo mensajes del WebSocket hasta que aparezca el mensaje que contiene init
        msg = await ps_websocket_client.receive_message()
        # msg = ">battle-gen8vgc2022-1614956049-sefnavzupbfhf6qq7ge3ckmkpoqyilkpw|init|battle|title|PopplioPeru98 vs. Braindeadmeta|j|☆PopplioPeru98"
        print("[get_battle_tag_and_opponent] msg: {}".format(msg))

        if not isinstance(msg, str) or not msg.strip():
            raise ValueError("Se recibió un mensaje vacío desde el WebSocket")

        parts = msg.split('|')
        first_msg = parts[0]
        # first_msg = ">battle-gen8vgc2022-1614956049-sefnavzupbfhf6qq7ge3ckmkpoqyilkpw"

        if 'battle' in first_msg:
            battle_tag = first_msg.replace('>', '').strip()  # battle_tag = "battle-gen8vgc2022-1614956049-sefnavzupbfhf6qq7ge3ckmkpoqyilkpw"
            print("[get_battle_tag_and_opponent] Battle Tag: {}".format(battle_tag))
            return battle_tag

#######################################################################################################################
# OK - Inicia una Nueva Batalla Individual Pokémon
#######################################################################################################################
async def start_battle(ps_websocket_client, battle):
    # Obtengo el battle_tag de la Batalla Individual
    battle.battle_tag = await get_battle_tag(ps_websocket_client)

    team_preview_message = ""

    # Entra en el bucle hasta que reciba el mensaje que contiene la vista previa del equipo del oponente
    while constants.START_TEAM_PREVIEW not in team_preview_message:
        team_preview_message = await ps_websocket_client.receive_message()

        if not isinstance(team_preview_message, str) or not team_preview_message.strip():
            raise ValueError("Se recibió un mensaje vacío desde el WebSocket")

    # Ahora toca los datos de mi equipo para la fase de vista previa del equipo
    while True:
        # Recibo un mensaje del WebSocket
        request_message = await ps_websocket_client.receive_message()
        print("[start_battle] msg: {}".format(request_message))

        if not isinstance(request_message, str) or not request_message.strip():
            raise ValueError("Se recibió un mensaje vacío desde el WebSocket")

        split_msg = request_message.split('|')

        # Si la cadena de arriba...
        if len(split_msg) > 2 and split_msg[1].strip() == 'request' and split_msg[2].strip():
            # Se obtiene el JSON TEAM PREVIEW de la batalla
            request_payload = split_msg[2].strip().strip("'")
            user_json = json.loads(request_payload)
            print("[initialize_battle_with_tag] User JSON: {}".format(user_json))

            user_id = user_json[constants.SIDE][constants.ID]  # user_id = "p1"
            opponent_id = constants.ID_LOOKUP[user_id]  # opponent_id = "p2"

            # Actualización de variables de la Clase Batalla
            battle.opponent.id = opponent_id
            battle.request_json = user_json
            break

    opponent_team = await get_opponent_team(team_preview_message, battle.opponent.id) # ["Swampert","Swampertite","Torrent",["WaveCrash", "Surf","BulkUp","FlipTurn"],"Brave"] x6

    # Inicializa la Clase Batalla con los datos de los 6 PKMN usuario y oponente, se ejecuta 1 vez por Batalla Individual
    await battle.initialize_team_preview(user_json, opponent_team)

    # Envía el saludo para la Batalla Individual
    await ps_websocket_client.send_message(battle.battle_tag, [config.greeting_message])

    # Inicia el contador de tiempo para la Batalla Individual
    if (battle.botMode == constants.CHALLENGE_USER or battle.botMode == constants.ACCEPT_CHALLENGE):
        await ps_websocket_client.send_message(battle.battle_tag, ['/timer on'])

#######################################################################################################################################################
# OK - Inicia la Batalla BO3, deja todo listo para que inicie la Batalla Individual, retorno la Clase Batalla con el bo3_tag
#######################################################################################################################################################
async def start_battleBO3(ps_websocket_client, botMode):
    # Creo la Clase Batalla, que estará con vida durante el BO3, y que contiene toda la información de la Batalla, tanto del usuario como del oponente
    battle = importlib.import_module('showdown.battle_bots.{}.main'.format("random"))
    battle.botMode = botMode

    # Inicia la Batalla VGC
    while True:
        # Voy recibiendo mensajes del WebSocket hasta que aparezca el mensaje que contiene init
        message = await ps_websocket_client.receive_message()
        # message = ">battle-gen8vgc2022-1614956049-sefnavzupbfhf6qq7ge3ckmkpoqyilkpw|init|battle|title|PopplioPeru98 vs. Braindeadmeta|j|☆PopplioPeru98"
        print("[get_battle_tag_and_opponent] msg: {}".format(message))

        if not isinstance(message, str) or not message.strip():
            raise ValueError("Se recibió un mensaje vacío desde el WebSocket")

        parts = message.split('|')
        first_part = parts[0]
        # first_part = ">battle-gen8vgc2022-1614956049-sefnavzupbfhf6qq7ge3ckmkpoqyilkpw"

        # Rechazo de la solicitud de batalla
        if len(parts) >= 5 and parts[4].startswith("/nonotify"):
            raise Exception("¡El oponente ha rechazado la solicitud de batalla!")
        # Nombre de usuario no encontrado
        if message.startswith("|popup|The user"):
            raise Exception("¡El nombre de usuario no ha sido encontrado!")

        if 'battle' in first_part:
            battle_tag = first_part.replace('>', '').strip()  # battle_tag = "battle-gen8vgc2022-1614956049-sefnavzupbfhf6qq7ge3ckmkpoqyilkpw"
            print("[get_battle_tag_and_opponent] Battle Tag BO3: {}".format(battle_tag))
            # Asigno el battle_tag de la Batalla BO3
            battle.battleBO3_tag = battle_tag
            user_name = parts[-1].replace('☆', '').strip()  # user_name = "PopplioPeru98"
            opponent_name = parts[4].replace(user_name, '').replace('vs.', '').strip()  # opponent_name = "Braindeadmeta"
            print("[get_battle_tag_and_opponent] Opponent Name: {}".format(opponent_name))
            battle.opponent.account_name = opponent_name
            break

    # Retorna la Clase Batalla, que va a ser usada durante toda la Batalla BO3
    return battle

#######################################################################################################################################################
# OK - Flujo Principal Batalla BO3, devuelve el ganador o genera excepción si hubo un error
#######################################################################################################################################################
async def pokemon_battle(ps_websocket_client: PSWebsocketClient, q: multiprocessing.Queue, botmode: str) -> str:

    try:
        battle = await start_battleBO3(ps_websocket_client, botmode) # Crea la Clase Batalla BO3 | init (1)
    except Exception as e:
        print(f"[ERROR]  ¡Ha sucedido un error al momento de Comenzar la Batalla BO3!: {e}")
        q.put([-6])
        raise
    
    while True: # Para cada uno de las Batallas Individuales dentro del BO3
        try:
            await start_battle(ps_websocket_client, battle) # Inicialización de la Batalla Individual | init (x), clearpoke, |request|{"teamPreview"...
            await send_action_team_preview(ps_websocket_client, battle) # Envío la acción para el Team Preview
        except Exception as e:
            print(f"[ERROR]  ¡Ha sucedido un error al momento de Iniciar la Batalla Individual!: {e}")
            q.put([-6])
            raise

        while True: # Para cada uno de los turnos de la Batalla Individual
            msg = await ps_websocket_client.receive_message() # Recibo Mensajes por cada turno de la Batalla Individual

            if battle_is_finished(battle.battle_tag, msg): # ¿Ha finalizado la Batalla Individual? |win|
                try: # Si ha finalizado...
                    await handle_finished_battle(ps_websocket_client, battle, q, msg) # Entonces se obtiene el ganador de la Batalla Individual...
                except Exception as e:
                    print(f"[ERROR]  ¡Ha sucedido un error al momento de Finalizar la Batalla Individual!: {e}")
                    q.put([-6])
                    raise
                break # Y se sale del bucle
            else:
                if "|inactive|" in msg:
                    continue
                elif "|error|" in msg:
                    await handle_action_error(ps_websocket_client, battle)
                else:
                    await handle_active_battle_turn(ps_websocket_client, battle, msg) # Si no lo tiene, se maneja el nuevo turno confirmado
        
        if not (await battleBO3_is_finished(battle)): # ¿Ha finalizado la Batalla BO3?
            await confirm_next_battle(ps_websocket_client, battle.battle_tag) # Si no ha finalizado, entonces se espera la siguiente Batalla Individual | confirmReady
        else:
            winner = await handle_finished_battleBO3(ps_websocket_client, battle, q, msg) # Sí ha finalizado la Batalla BO3, entonces se obtiene el ganador y se sale del bucle
            print("[INFO] Ganador de la Batalla BO3: {}".format(winner))
            return winner