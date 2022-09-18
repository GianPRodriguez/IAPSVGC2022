import logging
import asyncio
import config
import constants
import requests
import math
from collections import defaultdict
from numpy import sqrt 

logger = logging.getLogger(__name__)


# async def action_team_preview(self, battle):
class Box():
    def __init__(self, rewardx1OpponentTeam: int, i: int, j: int):
        self.rewardx1OpponentTeam = rewardx1OpponentTeam
        self.i = i
        self.j = j
    
    def __lt__(self, other):
        return self.rewardx1OpponentTeam < other.rewardx1OpponentTeam
    
    def get(self):
        return self.i, self.j, self.rewardx1OpponentTeam


# async def action_team_preview(self, battle):
class pkmnSubReward:
    def __init__(self, HPB, ATB, DEB, VEB, type1, type2, dictMoves):
        self.HPB = HPB
        self.ATB = ATB
        self.DEB = DEB
        self.VEB = VEB
        self.type1 = type1
        self.type2 = type2
        self.ATBoost = 1
        self.DEBoost = 1
        self.dictMoves = dictMoves

    def addTypesMod(self, ATBoost, DEBoost):
        self.ATBoost = ATBoost
        self.DEBoost = DEBoost

# Formato de Acción
async def format_decision(battle, moveSwitch1, nMoveSwitch1, dynamax1, toMove1, moveSwitch2, nMoveSwitch2, dynamax2, toMove2):
    # Formats a decision for communication with Pokemon-Showdown
    if nMoveSwitch1 == 0:
        nMoveSwitch1 = ""
    if toMove1 == 0:
        toMove1 = ""
    if nMoveSwitch2 == 0:
        nMoveSwitch2 = ""
    if toMove2 == 0:
        toMove2 = ""
    if dynamax1 == 0:
        dynamax1 = ""
    else:
        dynamax1 = "dynamax"
    if dynamax2 == 0:
        dynamax2 = ""
    else:
        dynamax2 = "dynamax"
    message = "/choose {} {} {} {}, {} {} {} {}".format(moveSwitch1, nMoveSwitch1, dynamax1, toMove1, moveSwitch2, nMoveSwitch2, dynamax2, toMove2)
    return [message, str(battle.rqid)]


async def getMoveName(battle, position, nMoveSwitch, dynamax):
    if position == 0:
        if dynamax == 1 or battle.user.active1.isDynamax == True:
            maxMoveWS = battle.request_json.get('active','False')[position].get('maxMoves','False').get('maxMoves','False')[nMoveSwitch].get('move', 'False')
            if maxMoveWS in constants.maxMoves:
                return constants.maxMoves[maxMoveWS]
            else:
                logger.info("NO DEBE ENTRAR ACÁ - if dynamax == 1 or battle.user.active1.isDynamax == True:")
        else:
            return battle.request_json.get('active','False')[position].get('moves','False')[nMoveSwitch].get('move', 'False')
    elif position == 1:
        if dynamax == 1 or battle.user.active2.isDynamax == True:
            maxMoveWS = battle.request_json.get('active','False')[position].get('maxMoves','False').get('maxMoves','False')[nMoveSwitch].get('move', 'False')
            if maxMoveWS in constants.maxMoves:
                return constants.maxMoves[maxMoveWS]
            else:
                logger.info("NO DEBE ENTRAR ACÁ - elif position == 1:")
        else:
            return battle.request_json.get('active','False')[position].get('moves','False')[nMoveSwitch].get('move', 'False')
    else:
        logger.info("NO DEBE ENTRAR ACÁ - async def getMoveName(battle, position, nMoveSwitch, dynamax):")


# async def action_team_preview(self, battle):
def findRewardSingles(index, isOpponent, battle):
    dictIndex = battle.cadenaTP[index]
    subIndex0, subIndex1, subIndex2, subIndex3 = -1, -1, -1, -1
    if isOpponent == 1:
        # 11 - Pkmn1
        if dictIndex.get(str(index),'False').get('0','False') == 1 and dictIndex.get(str(index),'False').get('1','False') == 1:
            subIndex0 = 0
        elif dictIndex.get(str(index),'False').get('2','False') == 1 and dictIndex.get(str(index),'False').get('3','False') == 1:
             subIndex0 = 1
        elif dictIndex.get(str(index),'False').get('4','False') == 1 and dictIndex.get(str(index),'False').get('5','False') == 1:
             subIndex0 = 2
        elif dictIndex.get(str(index),'False').get('6','False') == 1 and dictIndex.get(str(index),'False').get('7','False') == 1:
             subIndex0 = 3
        elif dictIndex.get(str(index),'False').get('8','False') == 1 and dictIndex.get(str(index),'False').get('9','False') == 1:
             subIndex0 = 4
        elif dictIndex.get(str(index),'False').get('10','False') == 1 and dictIndex.get(str(index),'False').get('11','False') == 1:
             subIndex0 = 5
        # 10 - Pkmn2
        if dictIndex.get(str(index),'False').get('0','False') == 1 and dictIndex.get(str(index),'False').get('1','False') == 0:
            subIndex1 = 0
        elif dictIndex.get(str(index),'False').get('2','False') == 1 and dictIndex.get(str(index),'False').get('3','False') == 0:
             subIndex1 = 1
        elif dictIndex.get(str(index),'False').get('4','False') == 1 and dictIndex.get(str(index),'False').get('5','False') == 0:
             subIndex1 = 2
        elif dictIndex.get(str(index),'False').get('6','False') == 1 and dictIndex.get(str(index),'False').get('7','False') == 0:
             subIndex1 = 3
        elif dictIndex.get(str(index),'False').get('8','False') == 1 and dictIndex.get(str(index),'False').get('9','False') == 0:
             subIndex1 = 4
        elif dictIndex.get(str(index),'False').get('10','False') == 1 and dictIndex.get(str(index),'False').get('11','False') == 0:
             subIndex1 = 5
        # 01 - Pkmnr1
        if dictIndex.get(str(index),'False').get('0','False') == 0 and dictIndex.get(str(index),'False').get('1','False') == 1:
            subIndex2 = 0
        elif dictIndex.get(str(index),'False').get('2','False') == 0 and dictIndex.get(str(index),'False').get('3','False') == 1:
             subIndex2 = 1
        elif dictIndex.get(str(index),'False').get('4','False') == 0 and dictIndex.get(str(index),'False').get('5','False') == 1:
             subIndex2 = 2
        elif dictIndex.get(str(index),'False').get('6','False') == 0 and dictIndex.get(str(index),'False').get('7','False') == 1:
             subIndex2 = 3
        elif dictIndex.get(str(index),'False').get('8','False') == 0 and dictIndex.get(str(index),'False').get('9','False') == 1:
             subIndex2 = 4
        elif dictIndex.get(str(index),'False').get('10','False') == 0 and dictIndex.get(str(index),'False').get('11','False') == 1:
             subIndex2 = 5
        # 01 - Pkmnr2
        if dictIndex.get(str(index),'False').get('0','False') == 0 and dictIndex.get(str(index),'False').get('1','False') == 1 and subIndex2 != 0:
            subIndex3 = 0
        elif dictIndex.get(str(index),'False').get('2','False') == 0 and dictIndex.get(str(index),'False').get('3','False') == 1 and subIndex2 != 1:
             subIndex3 = 1
        elif dictIndex.get(str(index),'False').get('4','False') == 0 and dictIndex.get(str(index),'False').get('5','False') == 1 and subIndex2 != 2:
             subIndex3 = 2
        elif dictIndex.get(str(index),'False').get('6','False') == 0 and dictIndex.get(str(index),'False').get('7','False') == 1 and subIndex2 != 3:
             subIndex3 = 3
        elif dictIndex.get(str(index),'False').get('8','False') == 0 and dictIndex.get(str(index),'False').get('9','False') == 1 and subIndex2 != 4:
             subIndex3 = 4
        elif dictIndex.get(str(index),'False').get('10','False') == 0 and dictIndex.get(str(index),'False').get('11','False') == 1 and subIndex2 != 5:
             subIndex3 = 5
        # Asignacion
        pkmn1 = pkmnSubReward(battle.opponent.reserved[subIndex0].HPB, battle.opponent.reserved[subIndex0].ATB, battle.opponent.reserved[subIndex0].DEB, battle.opponent.reserved[subIndex0].VEB, battle.opponent.reserved[subIndex0].type1, battle.opponent.reserved[subIndex0].type2, battle.opponent.reserved[subIndex0].moveType)
        pkmn2 = pkmnSubReward(battle.opponent.reserved[subIndex1].HPB, battle.opponent.reserved[subIndex1].ATB, battle.opponent.reserved[subIndex1].DEB, battle.opponent.reserved[subIndex1].VEB, battle.opponent.reserved[subIndex1].type1, battle.opponent.reserved[subIndex1].type2, battle.opponent.reserved[subIndex1].moveType)
        pkmnr1 = pkmnSubReward(battle.opponent.reserved[subIndex2].HPB, battle.opponent.reserved[subIndex2].ATB, battle.opponent.reserved[subIndex2].DEB, battle.opponent.reserved[subIndex2].VEB, battle.opponent.reserved[subIndex2].type1, battle.opponent.reserved[subIndex2].type2, battle.opponent.reserved[subIndex2].moveType)
        pkmnr2 = pkmnSubReward(battle.opponent.reserved[subIndex3].HPB, battle.opponent.reserved[subIndex3].ATB, battle.opponent.reserved[subIndex3].DEB, battle.opponent.reserved[subIndex3].VEB, battle.opponent.reserved[subIndex3].type1, battle.opponent.reserved[subIndex3].type2, battle.opponent.reserved[subIndex3].moveType)
        return pkmn1, pkmn2, pkmnr1, pkmnr2
    else:
        # 11 - Pkmn1
        if dictIndex.get(str(index),'False').get('0','False') == 1 and dictIndex.get(str(index),'False').get('1','False') == 1:
            subIndex0 = 0
        elif dictIndex.get(str(index),'False').get('2','False') == 1 and dictIndex.get(str(index),'False').get('3','False') == 1:
             subIndex0 = 1
        elif dictIndex.get(str(index),'False').get('4','False') == 1 and dictIndex.get(str(index),'False').get('5','False') == 1:
             subIndex0 = 2
        elif dictIndex.get(str(index),'False').get('6','False') == 1 and dictIndex.get(str(index),'False').get('7','False') == 1:
             subIndex0 = 3
        elif dictIndex.get(str(index),'False').get('8','False') == 1 and dictIndex.get(str(index),'False').get('9','False') == 1:
             subIndex0 = 4
        elif dictIndex.get(str(index),'False').get('10','False') == 1 and dictIndex.get(str(index),'False').get('11','False') == 1:
             subIndex0 = 5
        # 10 - Pkmn2
        if dictIndex.get(str(index),'False').get('0','False') == 1 and dictIndex.get(str(index),'False').get('1','False') == 0:
            subIndex1 = 0
        elif dictIndex.get(str(index),'False').get('2','False') == 1 and dictIndex.get(str(index),'False').get('3','False') == 0:
             subIndex1 = 1
        elif dictIndex.get(str(index),'False').get('4','False') == 1 and dictIndex.get(str(index),'False').get('5','False') == 0:
             subIndex1 = 2
        elif dictIndex.get(str(index),'False').get('6','False') == 1 and dictIndex.get(str(index),'False').get('7','False') == 0:
             subIndex1 = 3
        elif dictIndex.get(str(index),'False').get('8','False') == 1 and dictIndex.get(str(index),'False').get('9','False') == 0:
             subIndex1 = 4
        elif dictIndex.get(str(index),'False').get('10','False') == 1 and dictIndex.get(str(index),'False').get('11','False') == 0:
             subIndex1 = 5
        # 01 - Pkmnr1
        if dictIndex.get(str(index),'False').get('0','False') == 0 and dictIndex.get(str(index),'False').get('1','False') == 1:
            subIndex2 = 0
        elif dictIndex.get(str(index),'False').get('2','False') == 0 and dictIndex.get(str(index),'False').get('3','False') == 1:
             subIndex2 = 1
        elif dictIndex.get(str(index),'False').get('4','False') == 0 and dictIndex.get(str(index),'False').get('5','False') == 1:
             subIndex2 = 2
        elif dictIndex.get(str(index),'False').get('6','False') == 0 and dictIndex.get(str(index),'False').get('7','False') == 1:
             subIndex2 = 3
        elif dictIndex.get(str(index),'False').get('8','False') == 0 and dictIndex.get(str(index),'False').get('9','False') == 1:
             subIndex2 = 4
        elif dictIndex.get(str(index),'False').get('10','False') == 0 and dictIndex.get(str(index),'False').get('11','False') == 1:
             subIndex2 = 5
        # 01 - Pkmnr1
        if dictIndex.get(str(index),'False').get('0','False') == 0 and dictIndex.get(str(index),'False').get('1','False') == 1 and subIndex2 != 0:
            subIndex3 = 0
        elif dictIndex.get(str(index),'False').get('2','False') == 0 and dictIndex.get(str(index),'False').get('3','False') == 1 and subIndex2 != 1:
             subIndex3 = 1
        elif dictIndex.get(str(index),'False').get('4','False') == 0 and dictIndex.get(str(index),'False').get('5','False') == 1 and subIndex2 != 2:
             subIndex3 = 2
        elif dictIndex.get(str(index),'False').get('6','False') == 0 and dictIndex.get(str(index),'False').get('7','False') == 1 and subIndex2 != 3:
             subIndex3 = 3
        elif dictIndex.get(str(index),'False').get('8','False') == 0 and dictIndex.get(str(index),'False').get('9','False') == 1 and subIndex2 != 4:
             subIndex3 = 4
        elif dictIndex.get(str(index),'False').get('10','False') == 0 and dictIndex.get(str(index),'False').get('11','False') == 1 and subIndex2 != 5:
             subIndex3 = 5
        # Asignacion
        pkmn1 = pkmnSubReward(battle.user.reserved[subIndex0].HPB, battle.user.reserved[subIndex0].ATB, battle.user.reserved[subIndex0].DEB, battle.user.reserved[subIndex0].VEB, battle.user.reserved[subIndex0].type1, battle.user.reserved[subIndex0].type2, battle.user.reserved[subIndex0].moveType)
        pkmn2 = pkmnSubReward(battle.user.reserved[subIndex1].HPB, battle.user.reserved[subIndex1].ATB, battle.user.reserved[subIndex1].DEB, battle.user.reserved[subIndex1].VEB, battle.user.reserved[subIndex1].type1, battle.user.reserved[subIndex1].type2, battle.user.reserved[subIndex1].moveType)
        pkmnr1 = pkmnSubReward(battle.user.reserved[subIndex2].HPB, battle.user.reserved[subIndex2].ATB, battle.user.reserved[subIndex2].DEB, battle.user.reserved[subIndex2].VEB, battle.user.reserved[subIndex2].type1, battle.user.reserved[subIndex2].type2, battle.user.reserved[subIndex2].moveType)
        pkmnr2 = pkmnSubReward(battle.user.reserved[subIndex3].HPB, battle.user.reserved[subIndex3].ATB, battle.user.reserved[subIndex3].DEB, battle.user.reserved[subIndex3].VEB, battle.user.reserved[subIndex3].type1, battle.user.reserved[subIndex3].type2, battle.user.reserved[subIndex3].moveType)
        return pkmn1, pkmn2, pkmnr1, pkmnr2


def formula(pkmn):
    #if pkmn.type1 == 'psychic':
    #    print("ATB: {} ATBoost: {} VEB: {} HPB: {} DEBoost: {} DEB: {}".format(pkmn.ATB, pkmn.ATBoost, pkmn.VEB, pkmn.HPB, pkmn.DEBoost, pkmn.DEB))
    return sqrt((pkmn.ATB*pkmn.ATBoost*pkmn.VEB)/10)*sqrt((pkmn.HPB*pkmn.DEBoost*pkmn.DEB)/10)


# async def action_team_preview(self, battle):
def findGroupReward(pkmno1, pkmno2, pkmnor1, pkmnor2, pkmnu1, pkmnu2, pkmnur1, pkmnur2, battle):
    pkmnu1.addTypesMod(getATBoost(pkmnu1.dictMoves, pkmno1.type1, pkmno1.type2, pkmno2.type1, pkmno2.type2, battle), getDEBoost(pkmno1.dictMoves, pkmno2.dictMoves, pkmnu1.type1, pkmnu1.type2, battle))
    #print("pkmnu1.dictMoves: {} pkmno1.dictMoves.dictMoves: {} pkmno2.dictMoves: {}".format(pkmnu1.dictMoves, pkmno1.dictMoves, pkmno2.dictMoves))
    #print("getATBoost: {} getDEBoost: {}".format(getATBoost(pkmnu1.dictMoves, pkmno1.type1, pkmno1.type2, pkmno2.type1, pkmno2.type2, battle), getDEBoost(pkmno1.dictMoves, pkmno2.dictMoves, pkmnu1.type1, pkmnu1.type2, battle)))
    pkmnu2.addTypesMod(getATBoost(pkmnu2.dictMoves, pkmno1.type1, pkmno1.type2, pkmno2.type1, pkmno2.type2, battle), getDEBoost(pkmno1.dictMoves, pkmno2.dictMoves, pkmnu2.type1, pkmnu2.type2, battle))
    pkmno1.addTypesMod(getATBoost(pkmno1.dictMoves, pkmnu1.type1, pkmnu1.type2, pkmnu2.type1, pkmnu2.type2, battle), getDEBoost(pkmnu1.dictMoves, pkmnu2.dictMoves, pkmno1.type1, pkmno1.type2, battle))
    pkmno2.addTypesMod(getATBoost(pkmno2.dictMoves, pkmnu1.type1, pkmnu1.type2, pkmnu2.type1, pkmnu2.type2, battle), getDEBoost(pkmnu1.dictMoves, pkmnu2.dictMoves, pkmno2.type1, pkmno2.type2, battle))
    #if pkmnu1.type1.lower() == 'psychic' and pkmnu2.type1.lower() == 'grass' and pkmnur1.type1.lower() == 'fairy' and pkmnur2.type1.lower() == 'electric' and pkmno1.type1.lower() == 'water' and pkmno1.type2.lower() == 'ground' and pkmno2.type1.lower() == 'fire' and pkmno2.type2.lower() == 'dark' and pkmnor1.type1.lower() == 'dark' and pkmnor1.type2.lower() == 'flying' and pkmnor2.type1.lower() == 'fairy':
    #    print("----------")
    #    print("formula(pkmnu1): ATB: {} ATBoost: {} VEB: {} HPB: {} DEBoost: {} DEB: {}".format(pkmnu1.ATB, pkmnu1.ATBoost, pkmnu1.VEB, pkmnu1.HPB, pkmnu1.DEBoost, pkmnu1.DEB))
    #    print("formula(pkmnu1)*1.1: {}".format(formula(pkmnu1)*1.1))
    #    print("formula(pkmnu2): ATB: {} ATBoost: {} VEB: {} HPB: {} DEBoost: {} DEB: {}".format(pkmnu2.ATB, pkmnu2.ATBoost, pkmnu2.VEB, pkmnu2.HPB, pkmnu2.DEBoost, pkmnu2.DEB))
    #    print("formula(pkmnu2): {}".format(formula(pkmnu2)))
    #    print("0.9*formula(pkmnur1): {}".format(0.9*formula(pkmnur1)))
    #    print("0.9*formula(pkmnur2): {}".format(0.9*formula(pkmnur2)))
    #    print("SUMA USUARIO: {}".format(formula(pkmnu1)*1.1 + formula(pkmnu2) + 0.9*(formula(pkmnur1)+formula(pkmnur2))))
    #    print("----------")
    #    print("formula(pkmno1)*1.1: {}".format(formula(pkmno1)*1.1))
    #    print("formula(pkmno2): {}".format(formula(pkmno2)))
    #    print("0.9*formula(pkmnor1): {}".format(0.9*formula(pkmnor1)))
    #    print("0.9*formula(pkmnor2): {}".format(0.9*formula(pkmnor2)))
    #    print("SUMA OPONENTE: {}".format(-(formula(pkmno1)*1.1 + formula(pkmno2) + 0.9*(formula(pkmnor1)+formula(pkmnor2)))))
    #    print("TOTAL SUMA: {}".format(formula(pkmnu1)*1.1 + formula(pkmnu2) + 0.9*(formula(pkmnur1)+formula(pkmnur2)) - (formula(pkmno1)*1.1 + formula(pkmno2) + 0.9*(formula(pkmnor1)+formula(pkmnor2)))))
    return formula(pkmnu1)*1.1 + formula(pkmnu2) + 0.9*(formula(pkmnur1)+formula(pkmnur2)) - (formula(pkmno1)*1.1 + formula(pkmno2) + 0.9*(formula(pkmnor1)+formula(pkmnor2)))


# async def action_team_preview(self, battle):
def findTeam(index ,battle):
    dictIndex = battle.cadenaTP[index]
    subIndex0, subIndex1, subIndex2, subIndex3 = -1, -1, -1, -1
    # 11 - Pkmn1
    if dictIndex.get(str(index),'False').get('0','False') == 1 and dictIndex.get(str(index),'False').get('1','False') == 1:
        subIndex0 = 0
    elif dictIndex.get(str(index),'False').get('2','False') == 1 and dictIndex.get(str(index),'False').get('3','False') == 1:
        subIndex0 = 1
    elif dictIndex.get(str(index),'False').get('4','False') == 1 and dictIndex.get(str(index),'False').get('5','False') == 1:
        subIndex0 = 2
    elif dictIndex.get(str(index),'False').get('6','False') == 1 and dictIndex.get(str(index),'False').get('7','False') == 1:
        subIndex0 = 3
    elif dictIndex.get(str(index),'False').get('8','False') == 1 and dictIndex.get(str(index),'False').get('9','False') == 1:
        subIndex0 = 4
    elif dictIndex.get(str(index),'False').get('10','False') == 1 and dictIndex.get(str(index),'False').get('11','False') == 1:
        subIndex0 = 5
    # 10 - Pkmn2
    if dictIndex.get(str(index),'False').get('0','False') == 1 and dictIndex.get(str(index),'False').get('1','False') == 0:
        subIndex1 = 0
    elif dictIndex.get(str(index),'False').get('2','False') == 1 and dictIndex.get(str(index),'False').get('3','False') == 0:
        subIndex1 = 1
    elif dictIndex.get(str(index),'False').get('4','False') == 1 and dictIndex.get(str(index),'False').get('5','False') == 0:
        subIndex1 = 2
    elif dictIndex.get(str(index),'False').get('6','False') == 1 and dictIndex.get(str(index),'False').get('7','False') == 0:
        subIndex1 = 3
    elif dictIndex.get(str(index),'False').get('8','False') == 1 and dictIndex.get(str(index),'False').get('9','False') == 0:
        subIndex1 = 4
    elif dictIndex.get(str(index),'False').get('10','False') == 1 and dictIndex.get(str(index),'False').get('11','False') == 0:
        subIndex1 = 5
    # 01 - Pkmnr1
    if dictIndex.get(str(index),'False').get('0','False') == 0 and dictIndex.get(str(index),'False').get('1','False') == 1:
        subIndex2 = 0
    elif dictIndex.get(str(index),'False').get('2','False') == 0 and dictIndex.get(str(index),'False').get('3','False') == 1:
        subIndex2 = 1
    elif dictIndex.get(str(index),'False').get('4','False') == 0 and dictIndex.get(str(index),'False').get('5','False') == 1:
        subIndex2 = 2
    elif dictIndex.get(str(index),'False').get('6','False') == 0 and dictIndex.get(str(index),'False').get('7','False') == 1:
        subIndex2 = 3
    elif dictIndex.get(str(index),'False').get('8','False') == 0 and dictIndex.get(str(index),'False').get('9','False') == 1:
        subIndex2 = 4
    elif dictIndex.get(str(index),'False').get('10','False') == 0 and dictIndex.get(str(index),'False').get('11','False') == 1:
        subIndex2 = 5
    # 01 - Pkmnr1
    if dictIndex.get(str(index),'False').get('0','False') == 0 and dictIndex.get(str(index),'False').get('1','False') == 1 and subIndex2 != 0:
        subIndex3 = 0
    elif dictIndex.get(str(index),'False').get('2','False') == 0 and dictIndex.get(str(index),'False').get('3','False') == 1 and subIndex2 != 1:
        subIndex3 = 1
    elif dictIndex.get(str(index),'False').get('4','False') == 0 and dictIndex.get(str(index),'False').get('5','False') == 1 and subIndex2 != 2:
        subIndex3 = 2
    elif dictIndex.get(str(index),'False').get('6','False') == 0 and dictIndex.get(str(index),'False').get('7','False') == 1 and subIndex2 != 3:
        subIndex3 = 3
    elif dictIndex.get(str(index),'False').get('8','False') == 0 and dictIndex.get(str(index),'False').get('9','False') == 1 and subIndex2 != 4:
        subIndex3 = 4
    elif dictIndex.get(str(index),'False').get('10','False') == 0 and dictIndex.get(str(index),'False').get('11','False') == 1 and subIndex2 != 5:
        subIndex3 = 5
    return subIndex0+1, subIndex1+1, subIndex2+1, subIndex3+1


# Team Preview & Battle
def getATBoost(dictMoves, o1type1, o1type2, o2type1, o2type2, battle):
    cadena = battle.cadenaTypes
    # aux
    aux1 = -1
    aux2 = -1
    # iterator
    for key in dictMoves:
        aux1 = max(aux1, float(cadena.get(key,'False').get(o1type1,1)) * float(cadena.get(key,'False').get(o1type2,1)) * dictMoves[key])
        aux2 = max(aux2, float(cadena.get(key,'False').get(o2type1,1)) * float(cadena.get(key,'False').get(o2type2,1)) * dictMoves[key])
    return (aux1 + aux2)/2


# Team Preview & Battle
def getDEBoost(dictMoves1, dictMoves2, utype1, utype2, battle):
    cadena = battle.cadenaTypes
    # aux
    aux1 = -1
    aux2 = -1
    # iterator
    for key in dictMoves1:
        aux1 = max(aux1, float(cadena.get(key,'False').get(utype1,1)) * float(cadena.get(key,'False').get(utype2,1)) * dictMoves1[key])
    for key in dictMoves2:
        aux2 = max(aux2, float(cadena.get(key,'False').get(utype1,1)) * float(cadena.get(key,'False').get(utype2,1)) * dictMoves2[key])
    return 2/(aux1 + aux2)