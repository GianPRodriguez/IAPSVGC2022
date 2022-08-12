import random
import json
import logging
import constants

from numpy import sqrt
from showdown.battle import Battle
from showdown.battle_bots.helpers import getMoveName
from showdown.battle_bots.helpers import getColorMove
from showdown.battle_bots.helpers import getToMovePokemon


logger = logging.getLogger(__name__)

class Box():
    def __init__(self, rewardx1OpponentTeam: int, i: int, j: int):
        self.rewardx1OpponentTeam = rewardx1OpponentTeam
        self.i = i
        self.j = j
    
    def __lt__(self, other):
        return self.rewardx1OpponentTeam < other.rewardx1OpponentTeam
    
    def get(self):
        return self.i, self.j, self.rewardx1OpponentTeam

class pkmnSubReward:
    def __init__(self, HPB, ATB, DEB, VEB, type1, type2):
        self.HPB = HPB
        self.ATB = ATB
        self.DEB = DEB
        self.VEB = VEB
        self.type1 = type1
        self.type2 = type2
        self.ATBoost = -1
        self.DEBoost = -1

    def addTypesMod(self, ATBoost, DEBoost):
        self.ATBoost = ATBoost
        self.DEBoost = DEBoost

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
        pkmn1 = pkmnSubReward(battle.opponent.reserved[subIndex0].HPB, battle.opponent.reserved[subIndex0].ATB, battle.opponent.reserved[subIndex0].DEB, battle.opponent.reserved[subIndex0].VEB, battle.opponent.reserved[subIndex0].type1, battle.opponent.reserved[subIndex0].type2)
        pkmn2 = pkmnSubReward(battle.opponent.reserved[subIndex1].HPB, battle.opponent.reserved[subIndex1].ATB, battle.opponent.reserved[subIndex1].DEB, battle.opponent.reserved[subIndex1].VEB, battle.opponent.reserved[subIndex1].type1, battle.opponent.reserved[subIndex1].type2)
        pkmnr1 = pkmnSubReward(battle.opponent.reserved[subIndex2].HPB, battle.opponent.reserved[subIndex2].ATB, battle.opponent.reserved[subIndex2].DEB, battle.opponent.reserved[subIndex2].VEB, battle.opponent.reserved[subIndex2].type1, battle.opponent.reserved[subIndex2].type2)
        pkmnr2 = pkmnSubReward(battle.opponent.reserved[subIndex3].HPB, battle.opponent.reserved[subIndex3].ATB, battle.opponent.reserved[subIndex3].DEB, battle.opponent.reserved[subIndex3].VEB, battle.opponent.reserved[subIndex3].type1, battle.opponent.reserved[subIndex3].type2)
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
        pkmn1 = pkmnSubReward(battle.user.reserved[subIndex0].HPB, battle.user.reserved[subIndex0].ATB, battle.user.reserved[subIndex0].DEB, battle.user.reserved[subIndex0].VEB, battle.user.reserved[subIndex0].type1, battle.user.reserved[subIndex0].type2)
        pkmn2 = pkmnSubReward(battle.user.reserved[subIndex1].HPB, battle.user.reserved[subIndex1].ATB, battle.user.reserved[subIndex1].DEB, battle.user.reserved[subIndex1].VEB, battle.user.reserved[subIndex1].type1, battle.user.reserved[subIndex1].type2)
        pkmnr1 = pkmnSubReward(battle.user.reserved[subIndex2].HPB, battle.user.reserved[subIndex2].ATB, battle.user.reserved[subIndex2].DEB, battle.user.reserved[subIndex2].VEB, battle.user.reserved[subIndex2].type1, battle.user.reserved[subIndex2].type2)
        pkmnr2 = pkmnSubReward(battle.user.reserved[subIndex3].HPB, battle.user.reserved[subIndex3].ATB, battle.user.reserved[subIndex3].DEB, battle.user.reserved[subIndex3].VEB, battle.user.reserved[subIndex3].type1, battle.user.reserved[subIndex3].type2)
        return pkmn1, pkmn2, pkmnr1, pkmnr2

def formula(pkmn):
    if pkmn.ATBoost != -1:
        return sqrt((pkmn.ATB*pkmn.ATBoost*pkmn.VEB)/10)*sqrt((pkmn.HPB*pkmn.DEBoost*pkmn.DEB)/10)
    else:
        return sqrt((pkmn.ATB*pkmn.VEB)/10)*sqrt((pkmn.HPB*pkmn.DEB)/10)

def getATBoost(utype1, utype2, o1type1, o1type2, o2type1, o2type2, battle):
    cadena = battle.cadenaTypes
    # All Boosts
    boost1 = float(cadena.get(utype1,'False').get(o1type1,1))
    boost2 = float(cadena.get(utype1,'False').get(o1type2,1))
    boost3 = float(cadena.get(utype1,'False').get(o2type1,1))
    boost4 = float(cadena.get(utype1,'False').get(o2type2,1))
    if cadena.get(utype2,'False') == 'False':
        return ((boost1 * boost2) + (boost3 * boost4))/2
    else:
        boost5 = float(cadena.get(utype2,'False').get(o1type1,1))
        boost6 = float(cadena.get(utype2,'False').get(o1type2,1))
        boost7 = float(cadena.get(utype2,'False').get(o2type1,1))
        boost8 = float(cadena.get(utype2,'False').get(o2type2,1))
        return (((boost1 * boost2) + (boost3 * boost4))/2 + ((boost5 * boost6) + (boost7 * boost8))/2)/2

def getDEBoost(utype1, utype2, o1type1, o1type2, o2type1, o2type2, battle):
    cadena = battle.cadenaTypes
    # All Boosts
    boost1 = float(cadena.get(o1type1,'False').get(utype1,1))
    boost3 = float(cadena.get(o2type1,'False').get(utype1,1))
    if cadena.get(o1type2,'False') == 'False':
        boost2 = 1
    else:
        boost2 = float(cadena.get(o1type2,'False').get(utype1,1))
    if cadena.get(o2type2,'False') == 'False':
        boost4 = 1
    else:
        boost4 = float(cadena.get(o2type2,'False').get(utype1,1))
    if cadena.get(utype2,'False') == 'False':
        return ((boost1 * boost2) + (boost3 * boost4))/2
    else:
        boost5 = float(cadena.get(o1type1,'False').get(utype2,1))
        boost7 = float(cadena.get(o2type1,'False').get(utype2,1))
        if cadena.get(o1type2,'False') == 'False':
            boost6 = 1
        else:
            boost6 = float(cadena.get(o1type2,'False').get(utype2,1))
        if cadena.get(o2type2,'False') == 'False':
            boost8 = 1
        else:
            boost8 = float(cadena.get(o2type2,'False').get(utype2,1))
        return (((boost1 * boost2) + (boost3 * boost4))/2 + ((boost5 * boost6) + (boost7 * boost8))/2)/2

def findGroupReward(pkmno1, pkmno2, pkmnor1, pkmnor2, pkmnu1, pkmnu2, pkmnur1, pkmnur2, battle):
    pkmnu1.addTypesMod(getATBoost(pkmnu1.type1, pkmnu1.type2, pkmno1.type1, pkmno1.type2, pkmno2.type1, pkmno1.type2, battle), getDEBoost(pkmnu1.type1, pkmnu1.type2, pkmno1.type1, pkmno1.type2, pkmno2.type1, pkmno1.type2, battle))
    pkmnu2.addTypesMod(getATBoost(pkmnu2.type1, pkmnu2.type2, pkmno1.type1, pkmno1.type2, pkmno2.type1, pkmno1.type2, battle), getDEBoost(pkmnu2.type1, pkmnu2.type2, pkmno1.type1, pkmno1.type2, pkmno2.type1, pkmno1.type2, battle))
    pkmno1.addTypesMod(getATBoost(pkmno1.type1, pkmno1.type2, pkmnu1.type1, pkmnu1.type2, pkmnu2.type1, pkmnu1.type2, battle), getDEBoost(pkmno1.type1, pkmno1.type2, pkmnu1.type1, pkmnu1.type2, pkmnu2.type1, pkmnu1.type2, battle))
    pkmno2.addTypesMod(getATBoost(pkmno2.type1, pkmno2.type2, pkmnu1.type1, pkmnu1.type2, pkmnu2.type1, pkmnu1.type2, battle), getDEBoost(pkmno2.type1, pkmno2.type2, pkmnu1.type1, pkmnu1.type2, pkmnu2.type1, pkmnu1.type2, battle))
    return formula(pkmnu1)*1.1 + formula(pkmnu2) + 0.9*(formula(pkmnur1)+formula(pkmnur2)) - (formula(pkmno1)*1.1 + formula(pkmno2) + 0.9*(formula(pkmnor1)+formula(pkmnor2)))

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

class BattleBot(Battle):
    def __init__(self, *args, **kwargs):
        super(BattleBot, self).__init__(*args, **kwargs)

    async def action_team_preview(self, battle):
        bestRewardIJ = []
        for i in range(180):
            rewardx1OpponentTeam = -1000000
            bestJ = -1
            o1, o2, or1, or2 = findRewardSingles(i, 1, battle)
            for j in range(180):
                u1, u2, ur1, ur2 = findRewardSingles(j, 0, battle)
                aux = findGroupReward(o1, o2, or1, or2, u1, u2, ur1, ur2, battle)
                if aux > rewardx1OpponentTeam:
                    rewardx1OpponentTeam = aux
                    bestJ = j
            bestRewardIJ.append(Box(rewardx1OpponentTeam, i, bestJ))
            #print(bestRewardIJ[i].get())
        bestRewardIJ.sort()
        print(bestRewardIJ[0].get())
        print(bestRewardIJ[1].get())
        print(bestRewardIJ[2].get())
        print(bestRewardIJ[3].get())
        print(bestRewardIJ[4].get())
        print(bestRewardIJ[5].get())
        print(bestRewardIJ[6].get())
        # bestJ -> team choosed
        eleccion0, eleccion1, eleccion2, eleccion3 = findTeam(bestRewardIJ[3].j, battle)
        # get names of Pkmn
        namePokemon0 = battle.request_json.get('side','False').get('pokemon', 'False')[eleccion0-1].get('details', 'False').partition(",")[0]
        namePokemon1 = battle.request_json.get('side','False').get('pokemon', 'False')[eleccion1-1].get('details', 'False').partition(",")[0]
        namePokemon2 = battle.request_json.get('side','False').get('pokemon', 'False')[eleccion2-1].get('details', 'False').partition(",")[0]
        namePokemon3 = battle.request_json.get('side','False').get('pokemon', 'False')[eleccion3-1].get('details', 'False').partition(",")[0]
        # Send to Queue
        battle.q.put([1,namePokemon0, namePokemon1, namePokemon2, namePokemon3, 0, 0])
        # Return 4 integers
        return eleccion0, eleccion1, eleccion2, eleccion3

    async def action_battle(self, battle):
        moveSwitch1 = None
        nMoveSwitch1 = 0
        dynamax1 = 0
        toMove1 = 0
        moveSwitch2 = None
        nMoveSwitch2 = 0
        dynamax2 = 0
        toMove2 = 0
        # Inicio del Árbol...
        if battle.force_switch:
            # Si hay al menos un cambio obligatorio...
            izquierdaSwitch = (battle.request_json)["forceSwitch"][0]
            derechaSwitch = (battle.request_json)["forceSwitch"][1]
            # Si alguno es falso... hay que hacer pass
            if izquierdaSwitch == True and derechaSwitch == False:
                # derecha
                moveSwitch2 = 'pass'
                nMoveSwitch2 = 0
                dynamax2 = 0
                toMove2 = 0
                # izquierda... ¿con quién cambio?
                moveSwitch1 = 'switch'
                dynamax1 = 0
                toMove1 = 0
                # Se toma la decisión...
                switch3F = battle.user.reserved[0].fainted
                switch4F = battle.user.reserved[1].fainted
                if switch3F == False and switch4F == True:
                    # Solo Pokémon 3 puede reemplazar 
                    nMoveSwitch1 = 3
                elif switch3F == True and switch4F == False:
                    # Solo Pokémon 4 puede reemplazar
                    nMoveSwitch1 = 4
                else:
                    # Pokémon 3 y 4 pueden reemplazar
                    nMoveSwitch1 = random.choice([3, 4])
            elif izquierdaSwitch == False and derechaSwitch == True:
                # izquierda
                moveSwitch1 = 'pass'
                nMoveSwitch1 = 0
                dynamax1 = 0
                toMove1 = 0
                # derecha... ¿con quien cambio?
                moveSwitch2 = 'switch'
                dynamax2 = 0
                toMove2 =  0
                # Se toma la decisión...
                switch3F = battle.user.reserved[0].fainted
                switch4F = battle.user.reserved[1].fainted
                if switch3F == False and switch4F == True:
                    # Solo Pokémon 3 puede reemplazar 
                    nMoveSwitch2 = 3
                elif switch3F == True and switch4F == False:
                    # Solo Pokémon 4 puede reemplazar
                    nMoveSwitch2 = 4
                else:
                    # Pokémon 3 y 4 pueden reemplazar
                    nMoveSwitch2 = random.choice([3, 4])
            else: # Los 2 Pokémon hacen cambio... los 2 suplentes deben estar disponibles... o solo hay 1 Pokémon disponible
                # izquierda... ¿con quién cambio?
                if battle.user.reserved[0].fainted or battle.user.reserved[1].fainted:
                    moveSwitch1 = 'switch'
                    dynamax1 = 0
                    toMove1 = 0
                    # Se toma la decisión...
                    if battle.user.reserved[0].fainted:
                        nMoveSwitch1 = 4
                    else:
                        nMoveSwitch1 = 3
                else:
                    moveSwitch1 = 'switch'
                    dynamax1 = 0
                    toMove1 = 0
                    nMoveSwitch1 = random.choice([3, 4])
                # derecha... ¿con quien cambio?
                if battle.user.reserved[0].fainted or battle.user.reserved[1].fainted:
                    moveSwitch2 = 'pass'
                    dynamax2 = 0
                    toMove2 = 0
                    nMoveSwitch2 = 0
                else:
                    moveSwitch2 = 'switch'
                    dynamax2 = 0
                    toMove2 =  0
                    # Se toma la decisión...
                    if nMoveSwitch1 == 3:
                        nMoveSwitch2 = 4
                    else:
                        nMoveSwitch2 = 3
        else:
            # Para el Pokémon de la izquierda... ¿no está debilitado?
            if battle.user.active1.fainted == False:
                # Movimiento especial...
                if (battle.request_json).get('active','False')[0].get('moves','False')[0].get('pp','False') == 'False':
                    print("The opposing Shedinja used Phantom Force!")
                    # Está atrapado y tiene que hacer ese movimiento...
                    moveSwitch1 = 'move'
                    nMoveSwitch1 = 1
                    dynamax1 = 0
                    toMove1 = 0
                else:
                    # ¿move o switch?
                    moveSwitch1 = random.choice(['move', 'switch'])
                    if moveSwitch1 == 'move':
                        # ¿Qué movimientos están disponibles?
                        pokemon1move = []
                        if (battle.request_json).get('active','False')[0].get('moves','False')[0].get('disabled','False') == False:
                            pokemon1move.append(1)
                        if (battle.request_json).get('active','False')[0].get('moves','False')[1].get('disabled','False') == False:
                            pokemon1move.append(2)
                        if (battle.request_json).get('active','False')[0].get('moves','False')[2].get('disabled','False') == False:
                            pokemon1move.append(3)
                        if (battle.request_json).get('active','False')[0].get('moves','False')[3].get('disabled','False') == False:
                            pokemon1move.append(4)
                        # ¿Hay al menos algún movimiento para escoger?
                        if len(pokemon1move) > 0:
                            # Sí, hay al menos uno...
                            nMoveSwitch1 = random.choice(pokemon1move)
                            # Ahora que se ha escogido al menos 1 movimiento... ¿es posible hacer dynamax?
                            if battle.user.active1.can_dynamax:
                                logger.warning(str(battle.user.active1.can_dynamax))
                                logger.warning(str(type(battle.user.active1.can_dynamax)))
                                if ((battle.request_json).get('active','False')[0].get('canDynamax','False')):
                                    # Sí es posible...
                                    dynamax1 = random.choice([0, 1])
                                    if dynamax1:
                                        battle.user.active1.can_dynamax = False
                                        battle.user.active2.can_dynamax = False
                                        battle.user.reserved[0].can_dynamax = False
                                        battle.user.reserved[1].can_dynamax = False
                                        # Como va a realizar dynamax...
                                        if (battle.request_json)['active'][0]['maxMoves']['maxMoves'][nMoveSwitch1-1]['target'] == 'adjacentFoe':
                                            toMove1 = random.choice([1, 2])
                                        else: # 'self'
                                            toMove1 = 0
                                    else:
                                        # Ahora, ¿tiene un solo objetivo?
                                        if (battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target'] == 'allAdjacentFoes' or (battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target'] == 'allySide' or (battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target'] == 'self' or (battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target'] == 'all':
                                            toMove1 = 0
                                        elif (battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target'] == 'normal' or (battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target'] == 'any' or (battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target'] == 'selected-pokemon':
                                            if battle.user.active2.fainted == False:
                                                toMove1 = random.choice([-2, 1, 2])
                                            else:
                                                toMove1 = random.choice([1, 2])
                                        else:
                                            logger.error("ACCION NO CONTEMPLADA" + str((battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target']))
                            else:
                                # No es posible...
                                dynamax1 = 0
                                # ¿Actualmente es Dynamax?
                                if constants.DYNAMAX in battle.user.active1.volatile_statuses:
                                    if (battle.request_json)['active'][0]['maxMoves']['maxMoves'][nMoveSwitch1-1]['target'] == 'adjacentFoe':
                                        toMove1 = random.choice([1, 2])
                                    else: # 'self'
                                        toMove1 = 0
                                else:
                                    # Ahora, ¿tiene un solo objetivo?
                                    if (battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target'] == 'allAdjacentFoes' or (battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target'] == 'allySide' or (battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target'] == 'self' or (battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target'] == 'all':
                                        toMove1 = 0
                                    elif (battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target'] == 'normal' or (battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target'] == 'any' or (battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target'] == 'selected-pokemon':
                                        if battle.user.active2.fainted == False:
                                            toMove1 = random.choice([-2, 1, 2])
                                        else:
                                            toMove1 = random.choice([1, 2])
                                    else:
                                        logger.error("ACCION NO CONTEMPLADA" + str((battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target']))
                        else:
                            # No hay ningún movimiento disponible... algo muy poco probable... Struggle se considera como uno... así que solo queda cambiar, sí o sí debe haber al menos 1
                            moveSwitch1 = 'switch'
                            switch3F = battle.user.reserved[0].fainted
                            switch4F = battle.user.reserved[1].fainted
                            if switch3F == False and switch4F == True:
                                # Solo Pokémon 3 puede reemplazar 
                                nMoveSwitch1 = 3
                            elif switch3F == True and switch4F == False:
                                # Solo Pokémon 4 puede reemplazar
                                nMoveSwitch1 = 4
                            else:
                                # Pokémon 3 y 4 pueden reemplazar
                                nMoveSwitch1 = random.choice([3, 4])
                    else:
                        # El Random ha elegido Switch
                        if ((battle.request_json).get('active','False')[0].get('trapped','False') == True or (battle.request_json).get('active','False')[0].get('maybeTrapped','False') == True) or (battle.user.reserved[0].fainted and battle.user.reserved[1].fainted):
                            # Si está atrapado... o los 2 reservados están debilitados... solo queda hacer un movimiento
                            print("Para 0 ... Si está atrapado... o los 2 reservados están debilitados... solo queda hacer un movimiento")
                            moveSwitch1 = 'move'
                            pokemon1move = []
                            if (battle.request_json).get('active','False')[0].get('moves','False')[0].get('disabled','False') == False:
                                pokemon1move.append(1)
                            if (battle.request_json).get('active','False')[0].get('moves','False')[1].get('disabled','False') == False:
                                pokemon1move.append(2)
                            if (battle.request_json).get('active','False')[0].get('moves','False')[2].get('disabled','False') == False:
                                pokemon1move.append(3)
                            if (battle.request_json).get('active','False')[0].get('moves','False')[3].get('disabled','False') == False:
                                pokemon1move.append(4)
                            # Como hay al menos uno...
                            nMoveSwitch1 = random.choice(pokemon1move)
                            # Ahora que se ha escogido al menos 1 movimiento... ¿es posible hacer dynamax?
                            if battle.user.active1.can_dynamax:
                                if ((battle.request_json)['active'][0]['canDynamax']):
                                    # Sí es posible...
                                    dynamax1 = random.choice([0, 1])
                                    if dynamax1:
                                        battle.user.active1.can_dynamax = False
                                        battle.user.active2.can_dynamax = False
                                        battle.user.reserved[0].can_dynamax = False
                                        battle.user.reserved[1].can_dynamax = False
                                        # Como va a realizar dynamax...
                                        if (battle.request_json)['active'][0]['maxMoves']['maxMoves'][nMoveSwitch1-1]['target'] == 'adjacentFoe':
                                            toMove1 = random.choice([1, 2])
                                        else: # 'self'
                                            toMove1 = 0
                                    else:
                                        # Ahora, ¿tiene un solo objetivo?
                                        if (battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target'] == 'allAdjacentFoes' or (battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target'] == 'allySide' or (battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target'] == 'self' or (battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target'] == 'all':
                                            toMove1 = 0
                                        elif (battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target'] == 'normal' or (battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target'] == 'any' or (battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target'] == 'selected-pokemon':
                                            if battle.user.active2.fainted == False:
                                                toMove1 = random.choice([-2, 1, 2])
                                            else:
                                                toMove1 = random.choice([1, 2])
                                        else:
                                            logger.error("ACCION NO CONTEMPLADA" + str((battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target']))
                            else:
                                # No es posible...
                                dynamax1 = 0
                                # ¿Actualmente es Dynamax?
                                if constants.DYNAMAX in battle.user.active1.volatile_statuses:
                                    if (battle.request_json)['active'][0]['maxMoves']['maxMoves'][nMoveSwitch1-1]['target'] == 'adjacentFoe':
                                        toMove1 = random.choice([1, 2])
                                    else: # 'self'
                                        toMove1 = 0
                                else:
                                    # Ahora, ¿tiene un solo objetivo?
                                    if (battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target'] == 'allAdjacentFoes' or (battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target'] == 'allySide' or (battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target'] == 'self' or (battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target'] == 'all':
                                        toMove1 = 0
                                    elif (battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target'] == 'normal' or (battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target'] == 'any' or (battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target'] == 'selected-pokemon':
                                        if battle.user.active2.fainted == False:
                                            toMove1 = random.choice([-2, 1, 2])
                                        else:
                                            toMove1 = random.choice([1, 2])
                                    else:
                                        logger.error("ACCION NO CONTEMPLADA" + str((battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target']))
                        else:
                            # Si no... sí puede hacer un cambio normal
                            switch3F = battle.user.reserved[0].fainted
                            switch4F = battle.user.reserved[1].fainted
                            if switch3F == False and switch4F == True:
                                # Solo Pokémon 3 puede reemplazar 
                                nMoveSwitch1 = 3
                            elif switch3F == True and switch4F == False:
                                # Solo Pokémon 4 puede reemplazar
                                nMoveSwitch1 = 4
                            else:
                                # Pokémon 3 y 4 pueden reemplazar
                                nMoveSwitch1 = random.choice([3, 4])
                                print("CAMBIO ALEATORIO NORMAL PARA 1")
            else:
                moveSwitch1 = 'pass'
                nMoveSwitch1 = 0
                dynamax1 = 0
                toMove1 = 0
            # Para el Pokémon de la derecha... ¿no está debilitado?
            if battle.user.active2.fainted == False:
                # Movimiento especial...
                if (battle.request_json).get('active','False')[1].get('moves','False')[0].get('pp','False') == 'False':
                    print("The opposing Shedinja used Phantom Force!")
                    # Está atrapado y tiene que hacer ese movimiento...
                    moveSwitch2 = 'move'
                    nMoveSwitch2 = 1
                    dynamax2 = 0
                    toMove2 = 0
                else:
                    # ¿move o switch?
                    moveSwitch2 = random.choice(['move', 'switch'])
                    if moveSwitch2 == 'move':
                        # ¿Qué movimientos están disponibles?
                        pokemon2move = []
                        if (battle.request_json).get('active','False')[1].get('moves','False')[0].get('disabled','False') == False:
                            pokemon2move.append(1)
                        if (battle.request_json).get('active','False')[1].get('moves','False')[1].get('disabled','False') == False:
                            pokemon2move.append(2)
                        if (battle.request_json).get('active','False')[1].get('moves','False')[2].get('disabled','False') == False:
                            pokemon2move.append(3)
                        if (battle.request_json).get('active','False')[1].get('moves','False')[3].get('disabled','False') == False:
                            pokemon2move.append(4)
                        # ¿Hay al menos algún movimiento para escoger?
                        if len(pokemon2move) > 0:
                            # Sí, hay al menos uno...
                            nMoveSwitch2 = random.choice(pokemon2move)
                            # Ahora que se ha escogido al menos 1 movimiento... ¿es posible hacer dynamax?
                            if battle.user.active2.can_dynamax:
                                if ((battle.request_json).get('active','False')[1].get('canDynamax','False')):
                                    # Sí es posible...
                                    dynamax2 = random.choice([0, 1])
                                    if dynamax2:
                                        battle.user.active1.can_dynamax = False
                                        battle.user.active2.can_dynamax = False
                                        battle.user.reserved[0].can_dynamax = False
                                        battle.user.reserved[1].can_dynamax = False
                                        # Como va a realizar dynamax...
                                        if (battle.request_json)['active'][1]['maxMoves']['maxMoves'][nMoveSwitch2-1]['target'] == 'adjacentFoe':
                                            toMove2 = random.choice([1, 2])
                                        else: # 'self'
                                            toMove2 = 0
                                    else:
                                        # Ahora, ¿tiene un solo objetivo?
                                        if (battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target'] == 'allAdjacentFoes' or (battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target'] == 'allySide' or (battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target'] == 'self' or (battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target'] == 'all':
                                            toMove2 = 0
                                        elif (battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target'] == 'normal' or (battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target'] == 'any' or (battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target'] == 'selected-pokemon':
                                            if battle.user.active1.fainted == False:
                                                toMove2 = random.choice([-1, 1, 2])
                                            else:
                                                toMove2 = random.choice([1, 2])
                                        else:
                                            logger.error("ACCION NO CONTEMPLADA" + str((battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target']))
                            else:
                                # No es posible...
                                dynamax2 = 0
                                # ¿Actualmente es Dynamax?
                                if constants.DYNAMAX in battle.user.active2.volatile_statuses:
                                    if (battle.request_json)['active'][1]['maxMoves']['maxMoves'][nMoveSwitch2-1]['target'] == 'adjacentFoe':
                                        toMove2 = random.choice([1, 2])
                                    else: # 'self'
                                        toMove2 = 0
                                else:
                                    # Ahora, ¿tiene un solo objetivo?
                                    if (battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target'] == 'allAdjacentFoes' or (battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target'] == 'allySide' or (battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target'] == 'self' or (battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target'] == 'all':
                                        toMove2 = 0
                                    elif (battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target'] == 'normal' or (battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target'] == 'any' or (battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target'] == 'selected-pokemon':
                                        if battle.user.active1.fainted == False:
                                            toMove2 = random.choice([-1, 1, 2])
                                        else:
                                            toMove2 = random.choice([1, 2])
                                    else:
                                        logger.error("ACCION NO CONTEMPLADA" + str((battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target']))
                        else:
                            # No hay ningún movimiento disponible... algo muy poco probable... Struggle se considera como uno... así que solo queda cambiar, sí o sí debe haber al menos 1
                            moveSwitch2 = 'switch'
                            switch3F = battle.user.reserved[0].fainted
                            switch4F = battle.user.reserved[1].fainted
                            if switch3F == False and switch4F == True:
                                # Solo Pokémon 3 puede reemplazar 
                                nMoveSwitch2 = 3
                            elif switch3F == True and switch4F == False:
                                # Solo Pokémon 4 puede reemplazar
                                nMoveSwitch2 = 4
                            else:
                                # Pokémon 3 y 4 pueden reemplazar
                                nMoveSwitch2 = random.choice([3, 4])
                                print("CAMBIO ALEATORIO NORMAL PARA 2")
                    else:
                        # El Random ha elegido switch
                        if ((battle.request_json).get('active','False')[1].get('trapped','False') == True or (battle.request_json).get('active','False')[1].get('maybeTrapped','False') == True) or ((battle.user.reserved[0].fainted or battle.user.reserved[1].fainted) and (nMoveSwitch1 == 3 or nMoveSwitch1 == 4)) or (battle.user.reserved[0].fainted and battle.user.reserved[1].fainted):
                            # Si está atrapado... o los 2 reservados están debilitados... solo queda hacer un movimiento
                            print("Para 1 ... Si está atrapado... o los 2 reservados están debilitados... solo queda hacer un movimiento")
                            moveSwitch2 = 'move'
                            pokemon2move = []
                            if (battle.request_json).get('active','False')[1].get('moves','False')[0].get('disabled') == False:
                                pokemon2move.append(1)
                            if (battle.request_json).get('active','False')[1].get('moves','False')[1].get('disabled') == False:
                                pokemon2move.append(2)
                            if (battle.request_json).get('active','False')[1].get('moves','False')[2].get('disabled') == False:
                                pokemon2move.append(3)
                            if (battle.request_json).get('active','False')[1].get('moves','False')[3].get('disabled') == False:
                                pokemon2move.append(4)
                            # ¿Hay al menos algún movimiento para escoger?
                            if len(pokemon2move) > 0:
                                # Sí, hay al menos uno...
                                nMoveSwitch2 = random.choice(pokemon2move)
                                # Ahora que se ha escogido al menos 1 movimiento... ¿es posible hacer dynamax?
                                if battle.user.active2.can_dynamax:
                                    if ((battle.request_json)['active'][1]['canDynamax']):
                                        # Sí es posible...
                                        dynamax2 = random.choice([0, 1])
                                        if dynamax2:
                                            battle.user.active1.can_dynamax = False
                                            battle.user.active2.can_dynamax = False
                                            battle.user.reserved[0].can_dynamax = False
                                            battle.user.reserved[1].can_dynamax = False
                                            # Como va a realizar dynamax...
                                            if (battle.request_json)['active'][1]['maxMoves']['maxMoves'][nMoveSwitch2-1]['target'] == 'adjacentFoe':
                                                toMove2 = random.choice([1, 2])
                                            else: # 'self'
                                                toMove2 = 0
                                        else:
                                            # Ahora, ¿tiene un solo objetivo?
                                            if (battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target'] == 'allAdjacentFoes' or (battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target'] == 'allySide' or (battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target'] == 'self' or (battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target'] == 'all':
                                                toMove2 = 0
                                            elif (battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target'] == 'normal' or (battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target'] == 'any' or (battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target'] == 'selected-pokemon':
                                                if battle.user.active1.fainted == False:
                                                    toMove2 = random.choice([-1, 1, 2])
                                                else:
                                                    toMove2 = random.choice([1, 2])
                                            else:
                                                logger.error("ACCION NO CONTEMPLADA" + str((battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target']))
                                else:
                                    # No es posible...
                                    dynamax2 = 0
                                    # ¿Actualmente es Dynamax?
                                    if constants.DYNAMAX in battle.user.active2.volatile_statuses:
                                        if (battle.request_json)['active'][1]['maxMoves']['maxMoves'][nMoveSwitch2-1]['target'] == 'adjacentFoe':
                                            toMove2 = random.choice([1, 2])
                                        else: # 'self'
                                            toMove2 = 0
                                    else:
                                        # Ahora, ¿tiene un solo objetivo?
                                        if (battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target'] == 'allAdjacentFoes' or (battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target'] == 'allySide' or (battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target'] == 'self' or (battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target'] == 'all':
                                            toMove2 = 0
                                        elif (battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target'] == 'normal' or (battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target'] == 'any' or (battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target'] == 'selected-pokemon':
                                            if battle.user.active1.fainted == False:
                                                toMove2 = random.choice([-1, 1, 2])
                                            else:
                                                toMove2 = random.choice([1, 2])
                                        else:
                                            logger.error("ACCION NO CONTEMPLADA" + str((battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target']))
                        else:
                            switch3F = battle.user.reserved[0].fainted
                            switch4F = battle.user.reserved[1].fainted
                            if switch3F == False and switch4F == True:
                                # Solo Pokémon 3 puede reemplazar
                                nMoveSwitch2 = 3
                            elif switch3F == True and switch4F == False:
                                # Solo Pokémon 4 puede reemplazar
                                nMoveSwitch2 = 4
                            else:
                                # Pokémon 3 y 4 pueden reemplazar
                                if nMoveSwitch1 == 0:
                                    nMoveSwitch2 = random.choice([3, 4])
                                else:
                                    if nMoveSwitch1 == 4:
                                        nMoveSwitch2 = 3
                                    else:
                                        nMoveSwitch2 = 4
            else:
                moveSwitch2 = 'pass'
                nMoveSwitch2 = 0
                dynamax2 = 0
                toMove2 = 0
        # Se define los Text
        if moveSwitch1 == 'move':
            text1 = await getMoveName(battle, 0, nMoveSwitch1-1, dynamax1)
            text1Aux = text1.lower().replace(' ', '-')
            color1 = await getColorMove(text1Aux)
            text3 = await getToMovePokemon(battle, text1Aux, toMove1)
            color3 = '#FFFFFF'
        elif moveSwitch1 == 'pass':
            text1 = 'Pass'
            color1 = '#FFFFFF'
            text3 = 'Pass'
            color3 = '#FFFFFF'
        elif moveSwitch1 == 'switch':
            text1 = 'Switch to'
            color1 = '#FFFFFF'
            text3 = battle.user.reserved[nMoveSwitch1-3].name
            color3 = '#FFFFFF'
        else:
            logger.error("NO DEBE ENTRAR ACÁ - moveSwitch1 == 'switch'")
        if moveSwitch2 == 'move':
            text2 = await getMoveName(battle, 1, nMoveSwitch2-1, dynamax2)
            text2Aux = text2.lower().replace(' ', '-')
            color2 = await getColorMove(text2Aux)
            text4 = await getToMovePokemon(battle, text2Aux, toMove2)
            color4 = '#FFFFFF'
        elif moveSwitch2 == 'pass':
            text2 = 'Pass'
            color2 = '#FFFFFF'
            text4 = 'Pass'
            color4 = '#FFFFFF'
        elif moveSwitch2 == 'switch':
            text2 = 'Switch to'
            color2 = '#FFFFFF'
            text4 = battle.user.reserved[nMoveSwitch2-3].name
            color4 = '#FFFFFF'
        else:
            logger.error("NO DEBE ENTRAR ACÁ - moveSwitch2 == 'switch'")
        # Se envía al Queue la acción
        lastReward, reward = self.calculateRewards()
        logger.warning([2, text1, color1, text2, color2, text3, color3, text4, color4, lastReward, reward])
        battle.q.put([2, text1, color1, text2, color2, text3, color3, text4, color4, lastReward, reward])
        return battle, moveSwitch1, int(nMoveSwitch1), dynamax1, int(toMove1), moveSwitch2, int(nMoveSwitch2), dynamax2, int(toMove2)