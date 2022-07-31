import json
import requests
import math
import constants
import logging
import sys

from multiprocessing import Process, Queue
from collections import defaultdict
from numpy import sqrt 
from abc import ABC
from abc import abstractmethod

logger = logging.getLogger(__name__)

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
        self.turn = False
        self.started = False
        self.rqid = None
        self.force_switch = False
        self.wait = False
        self.request_json = None
        self.reward = 0
        self.lastReward = 0
        self.q = Queue()

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
            self.opponent.reserved.append(Pokemon(pkmn_string.partition(",")[0], 100, True))

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
            print("Weather: " + str(self.weather))
            print("Weather Turn: " + str(self.weatherTurn))
            print("Terrain: " + str(self.terrain))
            print("Terrain Turn: " + str(self.terrainTurn))
            print("Trick Room: " + str(self.trick_room))
            print("Turn: " + str(self.turn))
            print("Started: " + str(self.started))
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

    def calculateRewards(self):
        user = self.formula(self.user.active1)*1.1 + self.formula(self.user.active2) + 0.9*(self.formula(self.user.reserved[0]) + self.formula(self.user.reserved[1]))
        if len(self.opponent.reserved) == 2:
            opponent = self.formula(self.opponent.active1)*1.1 + self.formula(self.opponent.active2) + 0.9*(self.formula(self.opponent.reserved[0]) + self.formula(self.opponent.reserved[1]) + self.formula(self.opponent.reserved[2]) + self.formula(self.opponent.reserved[3]))/2
        else:
            opponent = self.formula(self.opponent.active1)*1.1 + self.formula(self.opponent.active2) + 0.9*(self.formula(self.opponent.reserved[0]) + self.formula(self.opponent.reserved[1]))
        self.lastReward = self.reward
        self.reward = user - opponent
        return self.lastReward, self.reward

    def formula(self, pkmn):
        return sqrt((pkmn.ATB*pkmn.VEB)/10)*sqrt((pkmn.HPB*pkmn.DEB)/10)

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

    def updatePkmnUser(self, JSONdict, first_turn):
        if (first_turn):
            # Update data
            self.id = JSONdict["side"]["id"]
            self.account_name = JSONdict["side"]["name"]
            self.isUser = True
            # Insert Pkmn
            self.reserved.append(Pokemon(JSONdict["side"]["pokemon"][0]["details"].partition(",")[0], JSONdict["side"]["pokemon"][0]["condition"].partition("/")[2],False))
            self.reserved.append(Pokemon(JSONdict["side"]["pokemon"][1]["details"].partition(",")[0], JSONdict["side"]["pokemon"][1]["condition"].partition("/")[2],False))
            self.reserved.append(Pokemon(JSONdict["side"]["pokemon"][2]["details"].partition(",")[0], JSONdict["side"]["pokemon"][2]["condition"].partition("/")[2],False))
            self.reserved.append(Pokemon(JSONdict["side"]["pokemon"][3]["details"].partition(",")[0], JSONdict["side"]["pokemon"][3]["condition"].partition("/")[2],False))
            self.reserved.append(Pokemon(JSONdict["side"]["pokemon"][4]["details"].partition(",")[0], JSONdict["side"]["pokemon"][4]["condition"].partition("/")[2],False))
            self.reserved.append(Pokemon(JSONdict["side"]["pokemon"][5]["details"].partition(",")[0], JSONdict["side"]["pokemon"][5]["condition"].partition("/")[2],False))
            # Update Pkmn
            for x in range(len(self.reserved)):
                self.reserved[x].HPB = JSONdict["side"]["pokemon"][x]["condition"].partition("/")[0].partition(" ")[0]
                self.reserved[x].ATB = max(int(JSONdict["side"]["pokemon"][x]["stats"]["atk"]),int(JSONdict["side"]["pokemon"][x]["stats"]["spa"]))
                self.reserved[x].DEB = int(math.floor((int(JSONdict["side"]["pokemon"][x]["stats"]["def"])+int(JSONdict["side"]["pokemon"][x]["stats"]["spd"]))/2))
                self.reserved[x].VEB = JSONdict["side"]["pokemon"][x]["stats"]["spe"]
        else:
            # Update Pkmn1
            self.active1.HPB = JSONdict["side"]["pokemon"][0]["condition"].partition("/")[0].partition(" ")[0]
            self.active1.ATB = max(int(JSONdict["side"]["pokemon"][0]["stats"]["atk"]),int(JSONdict["side"]["pokemon"][0]["stats"]["spa"]))
            self.active1.DEB = int(math.floor((int(JSONdict["side"]["pokemon"][0]["stats"]["def"])+int(JSONdict["side"]["pokemon"][0]["stats"]["spd"]))/2))
            self.active1.VEB = JSONdict["side"]["pokemon"][0]["stats"]["spe"]
            # Update Pkmn2
            self.active2.HPB = JSONdict["side"]["pokemon"][1]["condition"].partition("/")[0].partition(" ")[0]
            self.active2.ATB = max(int(JSONdict["side"]["pokemon"][1]["stats"]["atk"]),int(JSONdict["side"]["pokemon"][1]["stats"]["spa"]))
            self.active2.DEB = int(math.floor((int(JSONdict["side"]["pokemon"][1]["stats"]["def"])+int(JSONdict["side"]["pokemon"][1]["stats"]["spd"]))/2))
            self.active2.VEB = JSONdict["side"]["pokemon"][1]["stats"]["spe"]
            # Update Pkmn3
            self.reserved[0].HPB = JSONdict["side"]["pokemon"][2]["condition"].partition("/")[0].partition(" ")[0]
            self.reserved[0].ATB = max(int(JSONdict["side"]["pokemon"][2]["stats"]["atk"]),int(JSONdict["side"]["pokemon"][2]["stats"]["spa"]))
            self.reserved[0].DEB = int(math.floor((int(JSONdict["side"]["pokemon"][2]["stats"]["def"])+int(JSONdict["side"]["pokemon"][2]["stats"]["spd"]))/2))
            self.reserved[0].VEB = JSONdict["side"]["pokemon"][2]["stats"]["spe"]
            # Update Pkmn4
            self.reserved[1].HPB = JSONdict["side"]["pokemon"][3]["condition"].partition("/")[0].partition(" ")[0]
            self.reserved[1].ATB = max(int(JSONdict["side"]["pokemon"][3]["stats"]["atk"]),int(JSONdict["side"]["pokemon"][3]["stats"]["spa"]))
            self.reserved[1].DEB = int(math.floor((int(JSONdict["side"]["pokemon"][3]["stats"]["def"])+int(JSONdict["side"]["pokemon"][3]["stats"]["spd"]))/2))
            self.reserved[1].VEB = JSONdict["side"]["pokemon"][3]["stats"]["spe"]

    def imprimir(self):
        print()
        print("id: " + str(self.id))
        print("Nombre de la cuenta: " + str(self.account_name))
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
    def __init__(self, name: str, max_hp: int, isOpponent: int):
        self.name = name
        self.max_hp = max_hp
        self.hp = self.max_hp
        self.fainted = False
        self.status = None                      # ['slp', 'brn', 'frz', 'par', 'psn']
        self.volatile_statuses = []             # dynamax
        self.boosts = defaultdict(lambda: 0)
        if name.startswith('Zacian') or name.startswith('Zamazenta') or name.startswith('Eternatus'):
            self.can_dynamax = False
        else:
            self.can_dynamax = True
        self.opponent_confirmed = False
        self.isDynamax = False
        self.HPB = 0
        self.ATB = 0
        self.DEB = 0
        self.VEB = 0
        self.FRO = 0
        self.ASL = 0
        self.PAR = 0
        self.DRO = 0
        if name == 'Ditto':
            self.isDitto = 1
        else:
            self.isDitto = 0
        if isOpponent:
            self.POKEAPI()

    def imprimir(self):
        print("Name: " + str(self.name))
        print("Max HP: " + str(self.max_hp))
        print("HP: " + str(self.hp))
        print("Fainted: " + str(self.fainted))
        print("Status: " + str(self.status))
        print("Volatile Status: " + str(self.volatile_statuses))
        print("Boosts: ")
        for key, value in self.boosts.items():
            print(key, ' : ', value)
        print("Can Dynamax: " + str(self.can_dynamax))
        print("Is Ditto: " + str(self.isDitto))
        print("HPB: " + str(self.HPB))
        print("ATB: " + str(self.ATB))
        print("DEB: " + str(self.DEB))
        print("VEB: " + str(self.VEB))
        print("FRO: " + str(self.FRO))
        print("ASL: " + str(self.ASL))
        print("PAR: " + str(self.PAR))
        print("DRO: " + str(self.DRO))

    def HP_BATTLE_CALCULATE(self, HP_base):
        return math.floor(((2*HP_base+constants.IV+(constants.EV/4))*constants.LEVEL)/100) + constants.LEVEL + 10

    def OTHER_STAT_CALCULATE(self, STAT_base, modifier, defense):
        if modifier and defense:
            return math.floor((((2*STAT_base+constants.IV+(0/4))*constants.LEVEL)/100 + 5) * 1)
        else:
            return math.floor((((2*STAT_base+constants.IV+((constants.EV/2)/4))*constants.LEVEL)/100 + 5) * constants.NATURE)

    def POKEAPI(self):
        pokeapiPkmn = {'zacian': 'zacian-crowned', 'zacian-*': 'zacian-crowned', 'thundurus': 'thundurus-incarnate',
                        'landorus': 'landorus-incarnate', 'tornadus': 'tornadus-incarnate','indeedee-f': 'indeedee-female',
                        'urshifu': 'urshifu-single-strike', 'urshifu-rapid-strike': 'urshifu-single-strike'}
        
        if self.name.lower() in pokeapiPkmn:
            namePkmn = pokeapiPkmn[self.name.lower()]
        else:
            namePkmn = self.name
        url = 'https://pokeapi.co/api/v2/pokemon/' + namePkmn.lower()
        response = requests.get(url)
        if response.status_code == 200:
            payload = response.json()
            HP_base = payload['stats'][0]['base_stat']
            ATK_base = payload['stats'][1]['base_stat']
            DEF_base = payload['stats'][2]['base_stat']
            SPA_base = payload['stats'][3]['base_stat']
            SPD_base = payload['stats'][4]['base_stat']
            VEL_base = payload['stats'][5]['base_stat']

            self.HPB = self.HP_BATTLE_CALCULATE(HP_base)
            ATK_battle_base = self.OTHER_STAT_CALCULATE(ATK_base,0,0)
            SPA_battle_base = self.OTHER_STAT_CALCULATE(SPA_base,0,0)
            self.ATB = max(ATK_battle_base, SPA_battle_base)
            if DEF_base > SPD_base:
                SPD_battle_base = self.OTHER_STAT_CALCULATE(SPD_base,0,1)
                DEF_battle_base = self.OTHER_STAT_CALCULATE(DEF_base,1,1)
            else:
                DEF_battle_base = self.OTHER_STAT_CALCULATE(DEF_base,0,1)
                SPD_battle_base = self.OTHER_STAT_CALCULATE(SPD_base,1,1)
            self.DEB = int(math.floor((DEF_battle_base + SPD_battle_base)/2))
            self.VEB = self.OTHER_STAT_CALCULATE(VEL_base,0,0)