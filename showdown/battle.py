import sys
import json
import math
import queue
import requests
import constants

from abc import ABC
from numpy import sqrt
from abc import abstractmethod
from collections import defaultdict
from API.PokeAPI import getPkmnInfo

############################################################################################################################################################
# OK -  Información General de la Batalla                                                                                                                ###
############################################################################################################################################################
class Battle(ABC):
    def __init__(self):

        ''' Usuario y Oponente '''
        self.user = Battler()
        self.opponent = Battler()

        ''' Tag Globales '''
        self.battleBO3_tag = None
        self.battleBO3_count = 1
        self.botMode = None
        self.battle_tag = None

        ''' Características del Entorno '''
        self.weather = None             # ['None','raindance','sunnyday','sandstorm','hail']
        self.weatherTurn = 0
        self.terrain = None             # ['None','electricterrain','grassyterrain','mistyterrain','psychicterrain']
        self.terrainTurn = 0
        self.trick_room = 0             # [0,4,3,2,1]

        ''' Estados e Info del Turno '''
        self.turn = 0
        self.force_switch = False
        self.wait = False
        self.error = False
        self.request_json = None
        self.rqid = None

        ''' Recompensas '''
        self.reward = 0
        self.score = 0

############################################################################################################################################################
# OK - Inicializa la Clase Batalla con los datos de los 6 PKMN usuario y 6 oponente en el TEAM PREVIEW, se ejecuta 1 vez por Batalla Individual
############################################################################################################################################################
    async def initialize_team_preview(self, user_json, opponent_pokemon):
        self.user.updatePkmnUser(user_json, first_turn=True) # Inicializa solo los 6 PKMN del Usuario
        self.rqid = user_json[constants.RQID] # The rqid is obtained

        for pkmn_string in opponent_pokemon:
            # pkmn_string = ['Swampert', 'Swampertite', 'Torrent', ['WaveCrash', 'Surf', 'BulkUp', 'FlipTurn'], 'Brave']
            name = pkmn_string[0]
            objectPkmn = pkmn_string[1] if len(pkmn_string) > 1 else None
            abilityPkmn = pkmn_string[2] if len(pkmn_string) > 2 else None
            nature = pkmn_string[4] if len(pkmn_string) > 4 else None

            moves = pkmn_string[3] if len(pkmn_string) > 3 and isinstance(pkmn_string[3], list) else []

            # Extraer hasta 4 movimientos
            move1 = moves[0] if len(moves) > 0 else None
            move2 = moves[1] if len(moves) > 1 else None
            move3 = moves[2] if len(moves) > 2 else None
            move4 = moves[3] if len(moves) > 3 else None

            # Normalizar nombre
            name_lower = 'mimikyu-disguised' if name.lower() == 'mimikyu' else name.lower()
            namePkmn = constants.standardName.get(name_lower, name_lower)

            self.opponent.reserved.append(
                Pokemon(namePkmn, -1, True, objectPkmn, abilityPkmn, move1, move2, move3, move4, nature) # Dentro del Constructor se Autocompleta los datos restantes del PKMN Oponente
            )


############################################################################################################################################################
# OK - Método Abstracto BOT TEAM PREVIEW                                                                                                                 ###
############################################################################################################################################################
    @abstractmethod
    async def action_team_preview(self, battle):
        ...


############################################################################################################################################################
# OK - Método Abstracto BOT TURNO                                                                                                                        ###
############################################################################################################################################################
    @abstractmethod
    async def action_battle(self, battle):
        ...        


############################################################################################################################################################
# OK - Impresión de TODA la Batalla                                                                                                                      ###       
############################################################################################################################################################
    def imprimir(self):
        """Imprime por consola y guarda en 'reporte.txt' el estado completo de la batalla."""
        with open('reporte.txt', 'a', encoding='utf-8') as f:
            print("\n==================================================", file=f)
            print("                ESTADO DEL ENTORNO                ", file=f)
            print("==================================================", file=f)
            print(f"Turno: {self.turn} | rqid: {self.rqid}", file=f)
            print(f"Clima: {self.weather} (Turnos restantes: {self.weatherTurn})", file=f)
            print(f"Terreno: {self.terrain} (Turnos restantes: {self.terrainTurn})", file=f)
            print(f"Trick Room: {self.trick_room}", file=f)
            print(f"Force Switch: {self.force_switch} | Wait: {self.wait} | Error: {self.error}", file=f)
            print(f"Recompensa actual: {self.reward} | Score: {self.score}", file=f)
            
            print("\n--------------------------------------------------", file=f)
            print("                     USUARIO                      ", file=f)
            print("--------------------------------------------------", file=f)
            # Asegúrate de que Battler tenga su propio método imprimir()
            if hasattr(self.user, 'imprimir'):
                # Nota: Si el Battler imprime directamente a stdout, 
                # podrías requerir pasarle el archivo 'f' si deseas que también se guarde allí.
                self.user.imprimir()
            else:
                print(self.user, file=f)

            print("\n--------------------------------------------------", file=f)
            print("                     OPONENTE                     ", file=f)
            print("--------------------------------------------------", file=f)
            if hasattr(self.opponent, 'imprimir'):
                self.opponent.imprimir()
            else:
                print(self.opponent, file=f)
            print("==================================================\n", file=f)


############################################################################################################################################################
# PENDIENTE - Cálculos de las Recompensas                                                                                                                ###
############################################################################################################################################################
    def calculateRewards(self):
        self.lastReward = 100.00
        self.reward = 1000.00
        return self.lastReward, self.reward


############################################################################################################################################################
# OK - 
############################################################################################################################################################
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

###############################################################################################################################################################
# OK - Actualiza TODOS los datos de los PKMN del usuario en base al JSON recibido por cada tipo de |request|
###############################################################################################################################################################
    def updatePkmnUser(self, JSONdict, first_turn):
        if (first_turn): # TEAM PREVIEW
            # Datos Básicos
            self.id = JSONdict["side"]["id"]
            self.account_name = JSONdict["side"]["name"]
            self.isUser = True

            for pkmn_data in JSONdict["side"]["pokemon"]:
                # Normalizar nombre
                raw_name = pkmn_data["details"].partition(",")[0].lower()
                namePkmn = constants.standardName.get(raw_name, raw_name)

                # Extraer datos básicos
                max_hp = pkmn_data["condition"].partition("/")[2]
                item = pkmn_data.get("item", "")
                ability = pkmn_data.get("ability", "")
                
                # Extraer hasta 4 movimientos
                moves = pkmn_data.get("moves", [])
                m1 = moves[0] if len(moves) > 0 else None
                m2 = moves[1] if len(moves) > 1 else None
                m3 = moves[2] if len(moves) > 2 else None
                m4 = moves[3] if len(moves) > 3 else None

                # Crear instancia
                pkmn = Pokemon(namePkmn, max_hp, False, item, ability, m1, m2, m3, m4, None)

                # Asignar stats en combate
                stats = pkmn_data["stats"]
                pkmn.HPA_battle_actual = int(pkmn_data["condition"].partition("/")[0].partition(" ")[0])
                pkmn.ATK_battle_actual = int(stats["atk"])
                pkmn.DEF_battle_actual = int(stats["def"])
                pkmn.SPA_battle_actual = int(stats["spa"])
                pkmn.SPD_battle_actual = int(stats["spd"])
                pkmn.VEL_battle_actual = int(stats["spe"])

                self.reserved.append(pkmn)
        else:
            if len(self.reserved) == 4: # Solo quedan 2 PKMN activos para las rondas
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
                    if JSONdict.get("active","False")[0].get("moves","False")[i].get("pp","False") != 'False':
                        self.active1.movePP[i] = int(JSONdict.get("active","False")[0].get("moves","False")[i].get("pp","False"))
                for i in range(len(JSONdict.get("active","False")[1].get("moves","False"))):
                    if JSONdict.get("active","False")[1].get("moves","False")[i].get("pp","False") != 'False':
                        self.active2.movePP[i] = int(JSONdict.get("active","False")[1].get("moves","False")[i].get("pp","False"))
            # Reserved
            self.reserved[0].HPB = int(JSONdict["side"]["pokemon"][2]["condition"].partition("/")[0].partition(" ")[0])
            self.reserved[1].HPB = int(JSONdict["side"]["pokemon"][3]["condition"].partition("/")[0].partition(" ")[0])


############################################################################################################################################################
# OK - 
############################################################################################################################################################
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

############################################################################################################################################################
# OK - Representa a un Pokémon en combate, ya sea del usuario o del oponente                                                                             ###
############################################################################################################################################################
class Pokemon:
    def __init__(
        self, 
        name: str, 
        max_hp: int, 
        is_opponent: bool, 
        object_pkmn: str, 
        ability_pkmn: str, 
        move1: str, 
        move2: str, 
        move3: str, 
        move4: str, 
        nature: str
    ):
        # Identidad y Configuración Básica
        self.name = name
        self.max_hp = max_hp
        self.nature = nature
        self.is_opponent = is_opponent
        self.objectPkmn = object_pkmn
        self.abilityPkmn = ability_pkmn
        self.is_mega_evo = False
        self.can_mega_evo = False
        self.is_opponent_confirmed = False
        self.is_ditto = True if name == 'Ditto' else False  # Mantenido como entero por compatibilidad si se requiere
        self.is_fainted = False
        
        # Tipos (Se asignan posteriormente)
        self.type1 = None
        self.type2 = None

        # Estados y Modificadores de Combate
        self.non_volatile_statuses = defaultdict(lambda: 0)  # Ej: ['slp', 'brn', 'frz', 'par', 'psn']
        self.volatile_statuses = defaultdict(lambda: 0)      # Ej: 'yawn', etc.
        self.boosts = defaultdict(lambda: 0)                 # Modificadores de estadísticas (-6 a +6)

        # Estadísticas Base (Se toma como Base para la actualización de estadísticas del oponente)
        self.HPB_battle_base = 0
        self.ATK_battle_base = 0
        self.DEF_battle_base = 0
        self.SPA_battle_base = 0
        self.SPD_battle_base = 0
        self.VEL_battle_base = 0

        # Estadísticas Actuales (Se actualizan cada turno para Ambos)
        self.HPA_battle_actual = 0
        self.ATK_battle_actual = 0
        self.DEF_battle_actual = 0
        self.SPA_battle_actual = 0
        self.SPD_battle_actual = 0
        self.VEL_battle_actual = 0

        # Movimientos y PP
        self.moves = [move1, move2, move3, move4]
        self.movesPP = [-1, -1, -1, -1]

        # Actualizar atributos POKEAPI
        self.getStatsBattle(is_opponent)


############################################################################################################################################################
# OK - Actualización Otros Atributos POKEAPI                                                                                                             ###
############################################################################################################################################################
    async def getStatsBattle(self, isOpponent) -> None:
        if isOpponent:
            self.type1, self.type2, self.HPB_battle_base, self.ATK_battle_base, self.DEF_battle_base, self.SPA_battle_actual, self.SPD_battle_actual, self.VEL_battle_actual = await getPkmnInfo(self.name)


############################################################################################################################################################
# OK - Impresión de la clase Pokemon                                                                                                                     ###
############################################################################################################################################################
    def imprimir(self) -> None:
        """Imprime por consola el estado actual del Pokémon de forma organizada."""
        print(f"==================== POKÉMON: {self.name} ====================")
        print(f"Tipo 1: {self.type1} | Tipo 2: {self.type2}")
        print(f"Naturaleza: {self.nature} | HP Máximo: {self.max_hp}")
        print(f"Objeto: {self.objectPkmn} | Habilidad: {self.abilityPkmn}")
        print(f"Es Oponente: {self.is_opponent} | Confirmado: {self.is_opponent_confirmed}")
        print(f"Es Mega Evolución: {self.is_mega_evo} | Es Ditto: {self.is_ditto} | ¿Derrotado?: {self.is_fainted}")
        
        print("\n--- ESTADOS Y MODIFICADORES ---")
        print(f"Estados No Volátiles: {dict(self.non_volatile_statuses)}")
        print(f"Estados Volátiles: {dict(self.volatile_statuses)}")
        
        print("Boosts (Modificadores):")
        if self.boosts:
            for stat, value in self.boosts.items():
                if value != 0:  # Opcional: muestra solo los boosts activos
                    print(f"  {stat}: {value}")
        else:
            print("  Ninguno")

        print("\n--- ESTADÍSTICAS BASE ---")
        print(f"HPB: {self.HPB_battle_base} | ATK: {self.ATK_battle_base} | DEF: {self.DEF_battle_base}")
        print(f"SPA: {self.SPA_battle_base} | SPD: {self.SPD_battle_base} | VEL: {self.VEL_battle_base}")

        print("\n--- ESTADÍSTICAS ACTUALES ---")
        print(f"HPA: {self.HPA_battle_actual} | ATK: {self.ATK_battle_actual} | DEF: {self.DEF_battle_actual}")
        print(f"SPA: {self.SPA_battle_actual} | SPD: {self.SPD_battle_actual} | VEL: {self.VEL_battle_actual}")

        print("\n--- MOVIMIENTOS Y PP ---")
        for i, (move, pp) in enumerate(zip(self.moves, self.movesPP), start=1):
            print(f"  Movimiento {i}: {move} (PP: {pp})")
        print("=========================================================")