import sys
import json
import math
import requests
import constants

from abc import ABC
from numpy import sqrt
from abc import abstractmethod
from collections import defaultdict
from API.PokeAPI import getPkmnInfo
from multiprocessing import Process, Queue


class Battle(ABC):
    def __init__(self, battle_tag):
        self.battle_tag = battle_tag
        self.user = Battler()
        self.opponent = Battler()
        self.weather = None             # ['None','raindance','sunnyday','sandstorm','hail']
        self.weatherTurn = 0
        self.terrain = None             # ['None','electricterrain','grassyterrain','mistyterrain','psychicterrain']
        self.terrainTurn = 0
        self.trick_room = 0             # [0,4,3,2,1]
        self.turn = 0
        self.started = False
        self.rqid = None
        self.force_switch = False
        self.wait = False
        self.error = False
        self.request_json = None
        self.reward = 0
        self.lastReward = 0
        self.botMode = None
        self.q = Queue()
        self.isTraining = False
        self.cadenaTypes = json.loads(open("API/JSONs/typeChart.json","r").read())
        self.cadenaTP = json.loads(open("API/JSONs/geneticAlgorithm.json","r").read())
        self.moveTypes = json.loads(open("API/JSONs/pkmnMovesTypes.json","r").read())

    # Only Opponent
    async def initialize_team_preview(self, user_json, opponent_pokemon):
        # Insert Pkmn User
        self.user.updatePkmnUser(user_json, first_turn=True)
        # The battle has started
        self.started = True
        # The rqid is obtained
        self.rqid = user_json[constants.RQID]
        # Insert Pkmn Opponent
        for pkmn_string in opponent_pokemon:
            # opponent_pokemon = ["Lapras, L50, M","Kyogre, L50","Zacian-*, L50","Tapu Fini, L50","Zapdos, L50","Mimikyu, L50, F"]
            # pkmn_string = "Lapras, L50, M"
            # Add Types Moves
            name = pkmn_string.partition(",")[0]
            if name.lower() in constants.standardName:
                namePkmn = constants.standardName[name.lower()]
            else:
                namePkmn = name.lower()
            moveDict = self.moveTypes[namePkmn.lower().replace(" ", "-")]
            for key in moveDict:
                if moveDict[key] > 0.75:
                    moveDict[key] = 1
                elif moveDict[key] > 0.50:
                    moveDict[key] = 0.75
                elif moveDict[key] > 0.25:
                    moveDict[key] = 0.50
                else:
                    moveDict[key] = 0.25
            self.opponent.reserved.append(Pokemon(namePkmn, -1, True, None, None, None, None, moveDict))

    @abstractmethod
    async def action_team_preview(self, battle):
        ...

    @abstractmethod
    async def action_battle(self, battle):
        ...        

    def imprimir(self):
        original_stdout = sys.stdout
        with open('reporte.txt', 'a') as f:
            sys.stdout = f
            print()
            print("Estado del Entorno")
            print("------------------")
            print("Turn: " + str(self.turn))
            print("Started: " + str(self.started))
            print("Weather: " + str(self.weather))
            print("Weather Turn: " + str(self.weatherTurn))
            print("Terrain: " + str(self.terrain))
            print("Terrain Turn: " + str(self.terrainTurn))
            print("Trick Room: " + str(self.trick_room))
            print("rqid: " + str(self.rqid))
            print("Force Switch: " + str(self.force_switch))
            print("Wait: " + str(self.wait))
            print()
            print("USUARIO: ")
            self.user.imprimir()
            print()
            print("OPONENTE: ")
            self.opponent.imprimir()
            sys.stdout = original_stdout

    # Calculate the reward for a specific turn
    def calculateRewards(self):
        # Only user
        user = self.formula(self.user.active1)*1.1 + self.formula(self.user.active2) + 0.9*(self.formula(self.user.reserved[0]) + self.formula(self.user.reserved[1]))
        #print("user: {}".format(user))
        #print("user1: {}".format(self.formula(self.user.active1)*1.1))
        #print("user2: {}".format(self.formula(self.user.active2)))
        #print("user3: {}".format(0.9*self.formula(self.user.reserved[0])))
        #print("user4: {}".format(0.9*self.formula(self.user.reserved[1])))
        # Only opponent
        if self.opponent.opponentPkmnConfirmed == 2:
            opponent = self.formula(self.opponent.active1)*1.1 + self.formula(self.opponent.active2) + 0.9*(self.formula(self.opponent.reserved[0]) + self.formula(self.opponent.reserved[1]) + self.formula(self.opponent.reserved[2]) + self.formula(self.opponent.reserved[3]))/2
            #print("opponent1: {}".format(self.formula(self.opponent.active1)*1.1))
            #print("opponent2: {}".format(self.formula(self.opponent.active2)))
            #print("opponent3: {}".format(self.formula(self.opponent.reserved[0])*0.45))
            #print("opponent4: {}".format(self.formula(self.opponent.reserved[1])*0.45))
            #print("opponent5: {}".format(self.formula(self.opponent.reserved[2])*0.45))
            #print("opponent6: {}".format(self.formula(self.opponent.reserved[3])*0.45))
        elif self.opponent.opponentPkmnConfirmed == 4:
            opponent = self.formula(self.opponent.active1)*1.1 + self.formula(self.opponent.active2) + 0.9*(self.formula(self.opponent.reserved[0]) + self.formula(self.opponent.reserved[1]))
        else:   # == 3
            if self.opponent.reserved[0].opponent_confirmed:
                opponent = self.formula(self.opponent.active1)*1.1 + self.formula(self.opponent.active2) + 0.9*(self.formula(self.opponent.reserved[0])) + 0.9*(self.formula(self.opponent.reserved[1]) + self.formula(self.opponent.reserved[2]) + self.formula(self.opponent.reserved[3]))/3
            elif self.opponent.reserved[1].opponent_confirmed:
                opponent = self.formula(self.opponent.active1)*1.1 + self.formula(self.opponent.active2) + 0.9*(self.formula(self.opponent.reserved[1])) + 0.9*(self.formula(self.opponent.reserved[0]) + self.formula(self.opponent.reserved[2]) + self.formula(self.opponent.reserved[3]))/3
            elif self.opponent.reserved[2].opponent_confirmed:
                opponent = self.formula(self.opponent.active1)*1.1 + self.formula(self.opponent.active2) + 0.9*(self.formula(self.opponent.reserved[2])) + 0.9*(self.formula(self.opponent.reserved[0]) + self.formula(self.opponent.reserved[1]) + self.formula(self.opponent.reserved[3]))/3
            elif self.opponent.reserved[3].opponent_confirmed:
                opponent = self.formula(self.opponent.active1)*1.1 + self.formula(self.opponent.active2) + 0.9*(self.formula(self.opponent.reserved[3])) + 0.9*(self.formula(self.opponent.reserved[0]) + self.formula(self.opponent.reserved[1]) + self.formula(self.opponent.reserved[2]))/3
            else:
                opponent = 1000000
        print("opponent: {}".format(opponent))
        # On the first turn, the lastReward is 0
        if not self.force_switch:
            if self.turn > 1:
                self.lastReward = (user - opponent) - self.reward
            else:
                self.lastReward = 0
            # Heart
            self.reward = user - opponent
            return self.lastReward, self.reward
        else:
            return self.lastReward, self.reward

    # Calculate the reward for the user or for the opponent of 1 Pkmn
    def formula(self, pkmn):
        # General Formula - Only 6 variables
        return sqrt((pkmn.ATB*pkmn.ATBoost*pkmn.VEB)/10)*sqrt((pkmn.HPB*pkmn.DEBoost*pkmn.DEB)/10)

class Battler:
    # Constructor
    def __init__(self):
        self.name = None                                # String
        self.account_name = None                        # String
        self.active1 = None                             # class Pokemon or None
        self.active2 = None                             # class Pokemon or None
        self.reserved = []                              # class Pokemon
        self.side_conditions = defaultdict(lambda: 0)   # Tailwind, Stealth Rock, Reflect
        self.isUser = False
        self.opponentPkmnConfirmed = 0

    def getDamageClass(self, move):
        damageClass = json.loads(open("API/JSONs/moves.json","r").read())[move]['category'].lower()
        if damageClass != 'status':
            return move
        else:
            return None

    def updatePkmnUser(self, JSONdict, first_turn):
        if (first_turn):
            # Update data
            self.id = JSONdict["side"]["id"]
            self.account_name = JSONdict["side"]["name"]
            self.isUser = True
            # Insert Pkmn
            name = JSONdict["side"]["pokemon"][0]["details"].partition(",")[0]
            if name.lower() in constants.standardName:
                namePkmn = constants.standardName[name.lower()]
            else:
                namePkmn = name.lower()
            if len(JSONdict["side"]["pokemon"][0]["moves"]) > 1:
                self.reserved.append(Pokemon(namePkmn, JSONdict["side"]["pokemon"][0]["condition"].partition("/")[2],False, self.getDamageClass(JSONdict["side"]["pokemon"][0]["moves"][0].lower().replace(' ', '-')), self.getDamageClass(JSONdict["side"]["pokemon"][0]["moves"][1].lower().replace(' ', '-')), self.getDamageClass(JSONdict["side"]["pokemon"][0]["moves"][2].lower().replace(' ', '-')), self.getDamageClass(JSONdict["side"]["pokemon"][0]["moves"][3].lower().replace(' ', '-')), None))
            else:
                self.reserved.append(Pokemon(namePkmn, JSONdict["side"]["pokemon"][0]["condition"].partition("/")[2],False, self.getDamageClass(JSONdict["side"]["pokemon"][0]["moves"][0].lower().replace(' ', '-')), None, None, None, None))
            name = JSONdict["side"]["pokemon"][1]["details"].partition(",")[0]
            if name.lower() in constants.standardName:
                namePkmn = constants.standardName[name.lower()]
            else:
                namePkmn = name.lower()
            if len(JSONdict["side"]["pokemon"][1]["moves"]) > 1:
                self.reserved.append(Pokemon(namePkmn, JSONdict["side"]["pokemon"][1]["condition"].partition("/")[2],False, self.getDamageClass(JSONdict["side"]["pokemon"][1]["moves"][0].lower().replace(' ', '-')), self.getDamageClass(JSONdict["side"]["pokemon"][1]["moves"][1].lower().replace(' ', '-')), self.getDamageClass(JSONdict["side"]["pokemon"][1]["moves"][2].lower().replace(' ', '-')), self.getDamageClass(JSONdict["side"]["pokemon"][1]["moves"][3].lower().replace(' ', '-')), None))
            else:
                self.reserved.append(Pokemon(namePkmn, JSONdict["side"]["pokemon"][1]["condition"].partition("/")[2],False, self.getDamageClass(JSONdict["side"]["pokemon"][1]["moves"][0].lower().replace(' ', '-')), None, None, None, None))
            name = JSONdict["side"]["pokemon"][2]["details"].partition(",")[0]
            if name.lower() in constants.standardName:
                namePkmn = constants.standardName[name.lower()]
            else:
                namePkmn = name.lower()
            if len(JSONdict["side"]["pokemon"][2]["moves"]) > 1:
                self.reserved.append(Pokemon(namePkmn, JSONdict["side"]["pokemon"][2]["condition"].partition("/")[2],False, self.getDamageClass(JSONdict["side"]["pokemon"][2]["moves"][0].lower().replace(' ', '-')), self.getDamageClass(JSONdict["side"]["pokemon"][2]["moves"][1].lower().replace(' ', '-')), self.getDamageClass(JSONdict["side"]["pokemon"][2]["moves"][2].lower().replace(' ', '-')), self.getDamageClass(JSONdict["side"]["pokemon"][2]["moves"][3].lower().replace(' ', '-')), None))
            else:
                self.reserved.append(Pokemon(namePkmn, JSONdict["side"]["pokemon"][2]["condition"].partition("/")[2],False, self.getDamageClass(JSONdict["side"]["pokemon"][2]["moves"][0].lower().replace(' ', '-')), None, None, None, None))
            name = JSONdict["side"]["pokemon"][3]["details"].partition(",")[0]
            if name.lower() in constants.standardName:
                namePkmn = constants.standardName[name.lower()]
            else:
                namePkmn = name.lower()
            if len(JSONdict["side"]["pokemon"][3]["moves"]) > 1:
                self.reserved.append(Pokemon(namePkmn, JSONdict["side"]["pokemon"][3]["condition"].partition("/")[2],False, self.getDamageClass(JSONdict["side"]["pokemon"][3]["moves"][0].lower().replace(' ', '-')), self.getDamageClass(JSONdict["side"]["pokemon"][3]["moves"][1].lower().replace(' ', '-')), self.getDamageClass(JSONdict["side"]["pokemon"][3]["moves"][2].lower().replace(' ', '-')), self.getDamageClass(JSONdict["side"]["pokemon"][3]["moves"][3].lower().replace(' ', '-')), None))
            else:
                self.reserved.append(Pokemon(namePkmn, JSONdict["side"]["pokemon"][3]["condition"].partition("/")[2],False, self.getDamageClass(JSONdict["side"]["pokemon"][3]["moves"][0].lower().replace(' ', '-')), None, None, None, None))
            name = JSONdict["side"]["pokemon"][4]["details"].partition(",")[0]
            if name.lower() in constants.standardName:
                namePkmn = constants.standardName[name.lower()]
            else:
                namePkmn = name.lower()
            if len(JSONdict["side"]["pokemon"][4]["moves"]) > 1:
                self.reserved.append(Pokemon(namePkmn, JSONdict["side"]["pokemon"][4]["condition"].partition("/")[2],False, self.getDamageClass(JSONdict["side"]["pokemon"][4]["moves"][0].lower().replace(' ', '-')), self.getDamageClass(JSONdict["side"]["pokemon"][4]["moves"][1].lower().replace(' ', '-')), self.getDamageClass(JSONdict["side"]["pokemon"][4]["moves"][2].lower().replace(' ', '-')), self.getDamageClass(JSONdict["side"]["pokemon"][4]["moves"][3].lower().replace(' ', '-')), None))
            else:
                self.reserved.append(Pokemon(namePkmn, JSONdict["side"]["pokemon"][4]["condition"].partition("/")[2],False, self.getDamageClass(JSONdict["side"]["pokemon"][4]["moves"][0].lower().replace(' ', '-')), None, None, None, None))
            name = JSONdict["side"]["pokemon"][5]["details"].partition(",")[0]
            if name.lower() in constants.standardName:
                namePkmn = constants.standardName[name.lower()]
            else:
                namePkmn = name.lower()
            if len(JSONdict["side"]["pokemon"][5]["moves"]) > 1:
                self.reserved.append(Pokemon(namePkmn, JSONdict["side"]["pokemon"][5]["condition"].partition("/")[2],False, self.getDamageClass(JSONdict["side"]["pokemon"][5]["moves"][0].lower().replace(' ', '-')), self.getDamageClass(JSONdict["side"]["pokemon"][5]["moves"][1].lower().replace(' ', '-')), self.getDamageClass(JSONdict["side"]["pokemon"][5]["moves"][2].lower().replace(' ', '-')), self.getDamageClass(JSONdict["side"]["pokemon"][5]["moves"][3].lower().replace(' ', '-')), None))
            else:
                self.reserved.append(Pokemon(namePkmn, JSONdict["side"]["pokemon"][5]["condition"].partition("/")[2],False, self.getDamageClass(JSONdict["side"]["pokemon"][5]["moves"][0].lower().replace(' ', '-')), None, None, None, None))
            # Update Pkmn
            for x in range(len(self.reserved)):
                self.reserved[x].HPB_battle_base = int(JSONdict["side"]["pokemon"][x]["condition"].partition("/")[0].partition(" ")[0])
                self.reserved[x].ATK_battle_base = int(JSONdict["side"]["pokemon"][x]["stats"]["atk"])
                self.reserved[x].DEF_battle_base = int(JSONdict["side"]["pokemon"][x]["stats"]["def"])
                self.reserved[x].SPA_battle_base = int(JSONdict["side"]["pokemon"][x]["stats"]["spa"])
                self.reserved[x].SPD_battle_base = int(JSONdict["side"]["pokemon"][x]["stats"]["spd"])
                self.reserved[x].VEL_battle_base = int(JSONdict["side"]["pokemon"][x]["stats"]["spe"])
                self.reserved[x].HPB = int(JSONdict["side"]["pokemon"][x]["condition"].partition("/")[0].partition(" ")[0])
                self.reserved[x].ATB = max(int(JSONdict["side"]["pokemon"][x]["stats"]["atk"]),int(JSONdict["side"]["pokemon"][x]["stats"]["spa"]))
                self.reserved[x].DEB = int(math.floor((int(JSONdict["side"]["pokemon"][x]["stats"]["def"])+int(JSONdict["side"]["pokemon"][x]["stats"]["spd"]))/2))
                self.reserved[x].VEB = int(JSONdict["side"]["pokemon"][x]["stats"]["spe"])
        else:
            if len(self.reserved) == 4:
                name1 = self.reserved[0].name
                name2 = self.reserved[1].name
                name3 = self.reserved[2].name
                name4 = self.reserved[3].name
                if JSONdict["side"]["pokemon"][2]["details"].partition(",")[0].lower().replace(" ", "-") in constants.standardName:
                    auxPkmname1 = constants.standardName[JSONdict["side"]["pokemon"][2]["details"].partition(",")[0].lower().replace(" ", "-")]
                else:
                    auxPkmname1 = JSONdict["side"]["pokemon"][2]["details"].partition(",")[0].lower().replace(" ", "-")
                if JSONdict["side"]["pokemon"][3]["details"].partition(",")[0].lower().replace(" ", "-") in constants.standardName:
                    auxPkmname2 = constants.standardName[JSONdict["side"]["pokemon"][3]["details"].partition(",")[0].lower().replace(" ", "-")]
                else:
                    auxPkmname2 = JSONdict["side"]["pokemon"][3]["details"].partition(",")[0].lower().replace(" ", "-")                
                if name1 != auxPkmname1 and name1 != auxPkmname2:
                    self.reserved.pop(0)
                if name2 != auxPkmname1 and name2 != auxPkmname2:
                    self.reserved.pop(abs(3-len(self.reserved)))
                if name3 != auxPkmname1 and name3 != auxPkmname2:
                    self.reserved.pop(abs(2-len(self.reserved)))
                if name4 != auxPkmname1 and name4 != auxPkmname2:
                    self.reserved.pop(abs(1-len(self.reserved)))
            # Update Pkmn - Turno 2 en adelante... HPB
            self.active1.HPB = int(JSONdict["side"]["pokemon"][0]["condition"].partition("/")[0].partition(" ")[0])
            self.active2.HPB = int(JSONdict["side"]["pokemon"][1]["condition"].partition("/")[0].partition(" ")[0])
            # Update Pkmn - Turno 2 en adelante... Moves
            if JSONdict.get("active","False") != "False":
                for i in range(len(JSONdict.get("active","False")[0].get("moves","False"))):
                    self.active1.movePP[i] = int(JSONdict.get("active","False")[0].get("moves","False")[i].get("pp","False"))
                for i in range(len(JSONdict.get("active","False")[1].get("moves","False"))):
                    self.active2.movePP[i] = int(JSONdict.get("active","False")[1].get("moves","False")[i].get("pp","False"))
            # Reserved
            self.reserved[0].HPB = int(JSONdict["side"]["pokemon"][2]["condition"].partition("/")[0].partition(" ")[0])
            self.reserved[1].HPB = int(JSONdict["side"]["pokemon"][3]["condition"].partition("/")[0].partition(" ")[0])

    def imprimir(self):
        print()
        print("id: " + str(self.id))
        print("Nombre de la cuenta: " + str(self.account_name))
        print()
        if self.isUser == False:
            print("opponentPkmnConfirmed: " + str(self.opponentPkmnConfirmed))
        print()
        print("Side conditions: ")
        for key, value in self.side_conditions.items():
            print(key, ' : ', value)
        print()
        if self.active1 is not None:
            print("Active 1: ")
            print()
            self.active1.imprimir()
            print()
        else:
            print("Active 1: " + str(None))
            print()
        if self.active2 is not None:
            print("Active 2: ")
            print()
            self.active2.imprimir()
            print()
        else:
            print("Active 2: " + str(None))
            print()
        print("List of Reserved: ")
        print()
        for pkmn in self.reserved:
            pkmn.imprimir()
            print()

class Pokemon:
    # Constructor
    def __init__(self, name: str, max_hp: int, isOpponent: bool, move1: str, move2: str, move3: str, move4: str,  moveDict: dict):
        self.name = name
        self.max_hp = max_hp
        self.type1 = None
        self.type2 = None
        self.status = None                      # ['slp', 'brn', 'frz', 'par', 'psn']
        self.volatile_statuses = []             # dynamax, moveyawn
        self.boosts = defaultdict(lambda: 0)
        if name.startswith('zacian') or name.startswith('zamazenta') or name.startswith('eternatus'):
            self.can_dynamax = False
        else:
            self.can_dynamax = True
        self.isDynamax = False
        self.opponent_confirmed = False
        self.HPB_battle_base = 0
        self.HPB = 0
        self.ATK_battle_base = 0
        self.SPA_battle_base = 0
        self.ATB = 0
        self.ATBoost = 1
        self.SPD_battle_base = 0
        self.DEF_battle_base = 0
        self.DEB = 0
        self.DEBoost = 1
        self.VEL_battle_base = 0
        self.VEB = 0
        self.FRO = 0
        self.ASL = 0
        self.PAR = 0
        self.DRO = 0
        if name == 'Ditto':
            self.isDitto = 1
        else:
            self.isDitto = 0
        self.movePP = [-1, -1, -1, -1]
        self.moveType = defaultdict(lambda: 0)
        if move1 is not None:
            self.moveType[self.findMoveType(move1)] = 1
            if move2 is not None:
                self.moveType[self.findMoveType(move2)] = 1
            if move3 is not None:
                self.moveType[self.findMoveType(move3)] = 1
            if move4 is not None:
                self.moveType[self.findMoveType(move4)] = 1
        else:
            self.moveType = moveDict
        self.getStatsBattle(isOpponent)

    def isFainted(self):
        return self.HPB == 0

    def findMoveType(self, move):
        return json.loads(open("API/JSONs/moves.json","r").read())[move]['type'].lower()

    def imprimir(self):
        print("Name: " + str(self.name))
        print("Type 1: " + str(self.type1))
        print("Type 2: " + str(self.type2))
        print("Max HP: " + str(self.max_hp))
        print("Status: " + str(self.status))
        print("Volatile Status: " + str(self.volatile_statuses))
        print("Boosts: ")
        for key, value in self.boosts.items():
            print(key, ' : ', value)
        print("Can Dynamax: " + str(self.can_dynamax))
        print("Is Dynamax: " + str(self.isDynamax))
        print("Is Ditto: " + str(self.isDitto))
        print("isOpponentConfirmed: " + str(self.opponent_confirmed))
        print("Is Fainted?: {}".format(self.isFainted()))
        print("---STATS---")
        print("HPB: " + str(self.HPB))
        print("ATB: " + str(self.ATB))
        print("DEB: " + str(self.DEB))
        print("VEB: " + str(self.VEB))
        print("---HEALTH---")
        print("FRO: " + str(self.FRO))
        print("ASL: " + str(self.ASL))
        print("PAR: " + str(self.PAR))
        print("DRO: " + str(self.DRO))
        print("---MOVES---")
        print(self.movePP)
        for key, value in self.moveType.items():
            print(key, ' : ', value)

    def HP_BATTLE_CALCULATE(self, HP_base):
        return math.floor(((2*HP_base+constants.IV+(constants.EV/4))*constants.LEVEL)/100) + constants.LEVEL + 10

    def OTHER_STAT_CALCULATE(self, STAT_base, modifier, defense):
        if modifier and defense:
            return math.floor((((2*STAT_base+constants.IV+(0/4))*constants.LEVEL)/100 + 5) * 1)
        else:
            return math.floor((((2*STAT_base+constants.IV+((constants.EV)/4))*constants.LEVEL)/100 + 5) * constants.NATURE)

    def getStatsBattle(self, isOpponent):
        if self.name.lower() in constants.standardName:
            namePkmn = constants.standardName[self.name.lower()]
        else:
            namePkmn = self.name.lower()
        self.type1, self.type2, HP_base, ATK_base, DEF_base, SPA_base, SPD_base, VEL_base = getPkmnInfo(namePkmn)
        if isOpponent:
            # Stats Battle
            self.HPB_battle_base = self.HP_BATTLE_CALCULATE(HP_base)
            self.max_hp = self.HPB_battle_base
            self.ATK_battle_base = self.OTHER_STAT_CALCULATE(ATK_base,0,0)
            self.SPA_battle_base = self.OTHER_STAT_CALCULATE(SPA_base,0,0)
            if DEF_base > SPD_base:
                self.SPD_battle_base = self.OTHER_STAT_CALCULATE(SPD_base,0,1)
                self.DEF_battle_base = self.OTHER_STAT_CALCULATE(DEF_base,1,1)
            else:
                self.DEF_battle_base = self.OTHER_STAT_CALCULATE(DEF_base,0,1)
                self.SPD_battle_base = self.OTHER_STAT_CALCULATE(SPD_base,1,1)
            self.VEL_battle_base = self.OTHER_STAT_CALCULATE(VEL_base,0,0)
            # Cálculo del Reward
            self.HPB = self.HPB_battle_base
            self.ATB = max(int(self.ATK_battle_base),int(self.SPA_battle_base))
            self.DEB = int(math.floor((int(self.DEF_battle_base)+int(self.SPD_battle_base))/2))
            self.VEB = int(self.VEL_battle_base)