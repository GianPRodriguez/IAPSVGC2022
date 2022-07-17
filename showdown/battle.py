import itertools
import json
from collections import defaultdict
from collections import namedtuple
from copy import copy
from copy import deepcopy
from abc import ABC
from abc import abstractmethod

import constants
import config
import logging

import data
from data import all_move_json
from data import pokedex
from data.parse_smogon_stats import MOVES_STRING
from data.parse_smogon_stats import SPREADS_STRING
from data.parse_smogon_stats import ABILITY_STRING
from data.parse_smogon_stats import ITEM_STRING
from data.helpers import get_pokemon_sets
from data.helpers import get_mega_pkmn_name
from data.helpers import PASS_ITEMS
from data.helpers import PASS_ABILITIES
from data.helpers import get_all_likely_moves
from data.helpers import get_most_likely_item
from data.helpers import get_most_likely_ability
from data.helpers import get_most_likely_spread
from data.helpers import get_all_possible_moves_for_random_battle

from showdown.engine.objects import State
from showdown.engine.objects import Side
from showdown.engine.objects import Pokemon as TransposePokemon

from showdown.engine.helpers import remove_duplicate_spreads
from showdown.engine.helpers import get_pokemon_info_from_condition
from showdown.engine.helpers import set_makes_sense
from showdown.engine.helpers import normalize_name
from showdown.engine.helpers import calculate_stats


logger = logging.getLogger(__name__)


class Battle(ABC):
    def __init__(self, battle_tag):
        self.battle_tag = battle_tag
        self.user = Battler()
        self.opponent = Battler()
        self.weather = None
        self.field = None
        self.trick_room = False
        self.turn = False
        self.started = False
        self.rqid = None
        self.force_switch = False               #
        self.wait = False                       #
        self.request_json = None


    # TEAM PREVIEW
    def initialize_team_preview(self, user_json, opponent_pokemon):
        # Inicialización de Battler para usuario
        self.user.from_json(user_json, first_turn=True)
        # La Batalla ha iniciado
        self.started = True
        # Se obtiene el rqid
        self.rqid = user_json[constants.RQID]
        # Para el Oponente... solo tenemos el nombre del Pokémon, su HP será 100%
        for pkmn_string in opponent_pokemon:
            # opponent_pokemon = ["Lapras, L50, M","Kyogre, L50","Zacian-*, L50","Tapu Fini, L50","Zapdos, L50","Mimikyu, L50, F"]
            # pkmn_string = "Lapras, L50, M"
            # Aquí deberá ir el MySQL
            self.opponent.reserved.append(Pokemon(pkmn_string.partition(",")[0], 100))


    @abstractmethod
    def find_best_move(self):
        ...

    def imprimir(self):
        self.user.imprimir()
        self.opponent.imprimir()
        print("Weather: " + str(self.weather))
        print("Field:" + str(self.field))
        print("Trick Room: " + str(self.trick_room))
        print("Turn: " + str(self.turn))
        print("Started: " + str(self.started))
        print("rqid: " + str(self.rqid))
        print("Force Switch: " + str(self.force_switch))
        print("Wait: " + str(self.wait))


class Battler:
    # Constructor
    def __init__(self):
        self.active1 = None                             # class Pokemon or None
        self.active2 = None                             # class Pokemon or None
        self.reserved = []                              # class Pokemon
        self.name = None                                # String


    # Dado un JSON, hay que actualizar esta clase... propiamente para el lado del usuario
    # first_turn = 0, 1 y 2
    # 0 = TeamPreview
    # 1 = First Turn
    # 2 = Second Turn en adelante...
    def from_json(self, user_json, first_turn):
        JSONdict = user_json
        # Si es el primer turno...
        if (first_turn):
            self.name = JSONdict["side"]["name"]
            # Pokémon Reservados: Todos
            self.reserved.append(Pokemon(JSONdict["side"]["pokemon"][0]["details"].partition(",")[0], JSONdict["side"]["pokemon"][0]["condition"].partition("/")[2]))
            self.reserved.append(Pokemon(JSONdict["side"]["pokemon"][1]["details"].partition(",")[0], JSONdict["side"]["pokemon"][1]["condition"].partition("/")[2]))
            self.reserved.append(Pokemon(JSONdict["side"]["pokemon"][2]["details"].partition(",")[0], JSONdict["side"]["pokemon"][2]["condition"].partition("/")[2]))
            self.reserved.append(Pokemon(JSONdict["side"]["pokemon"][3]["details"].partition(",")[0], JSONdict["side"]["pokemon"][3]["condition"].partition("/")[2]))
            self.reserved.append(Pokemon(JSONdict["side"]["pokemon"][4]["details"].partition(",")[0], JSONdict["side"]["pokemon"][4]["condition"].partition("/")[2]))
            self.reserved.append(Pokemon(JSONdict["side"]["pokemon"][5]["details"].partition(",")[0], JSONdict["side"]["pokemon"][5]["condition"].partition("/")[2]))
        else:
            # Pokémon Activo 1
            self.active1.hp = JSONdict["side"]["pokemon"][0]["condition"].partition("/")[0]
            # Pokémon Activo 2
            self.active2.hp = JSONdict["side"]["pokemon"][1]["condition"].partition("/")[0]


    def imprimir(self):
        print("Nombre de la cuenta: " + str(self.name))
        self.active1.imprimir()
        print()
        self.active2.imprimir()
        print()
        self.reserved[0].imprimir()
        print()
        self.reserved[1].imprimir()
        print()


class Pokemon:
    # Constructor
    def __init__(self, name: str, max_hp: int):
        self.name = normalize_name(name)
        self.max_hp = max_hp
        self.hp = self.max_hp
        self.fainted = False
        self.status = None                      # A lo más uno
        self.volatile_statuses = []             # Varios
        self.boosts = defaultdict(lambda: 0)    # Diccionario vacío
        self.can_dynamax = True


    # Cambio de forma - Constante
    def forme_change(self, new_pkmn_name):
        ...

    def imprimir(self):
        print("Name: " + str(self.name))
        print("Max HP: " + str(self.max_hp))
        print("HP: " + str(self.hp))
        print("Fainted: " + str(self.fainted))
        print("Status: " + str(self.status))
        print("Volatile Status: " + str(self.volatile_statuses))
        print("Boosts: " + str(self.boosts))
        print("Can Dynamax: " + str(self.can_dynamax))