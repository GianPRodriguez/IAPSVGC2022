import sys
import math
import queue
import requests
import constants

from abc import ABC
from numpy import sqrt
from abc import abstractmethod
from collections import defaultdict
from API.PokeAPI import getPkmnInfo

import json
from typing import Union, Dict, Any

from teams.team_converter import normalize_name

############################################################################################################################################################
# OK - CLASE BATTLE                                                                                                                                      ###
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
        self.q = queue.Queue()

        ''' Características del Entorno '''
        self.weather = None                     # ['None','raindance','sunnyday','sandstorm','hail']
        self.weatherTurn = 0
        self.terrain = defaultdict(lambda: 0)   # ['electricterrain','grassyterrain','mistyterrain','psychicterrain', 'wonderroom', 'trickroom']

        ''' Estados e Info del Turno '''
        self.turn = 0
        self.active = False
        self.force_switch = False
        self.wait = False
        self.error = False
        self.request_json = None
        self.rqid = None

        ''' Recompensas '''
        self.reward = 0
        self.score = 0


############################################################################################################################################################
# OK - Completa las clases Battler con los datos de los 6 PKMN usuario y 6 oponente en el TEAM PREVIEW, se ejecuta 1 vez por Batalla Individual          ###
############################################################################################################################################################
    async def initialize_team_preview(self, user_json: str, opponent_pokemon: list, natures: list) -> None:
        # Lado Usuario
        first_turn = True
        self.user.updatePkmnUser(user_json, first_turn, natures) # Inicializa solo los 6 PKMN del Usuario
        self.rqid = user_json[constants.RQID] # The rqid is obtained

        # Lado Oponente
        for pkmn_string in opponent_pokemon:
            # pkmn_string = ['Swampert', 'Swampertite', 'Torrent', ['WaveCrash', 'Surf', 'BulkUp', 'FlipTurn'], 'Brave']
            name = pkmn_string[0]
            objectPkmn = normalize_name(pkmn_string[1] if len(pkmn_string) > 1 else None)
            abilityPkmn = normalize_name(pkmn_string[2] if len(pkmn_string) > 2 else None)
            nature = normalize_name(pkmn_string[4] if len(pkmn_string) > 4 else None)

            moves = pkmn_string[3] if len(pkmn_string) > 3 and isinstance(pkmn_string[3], list) else []

            # Extraer hasta 4 movimientos
            move1 = moves[0] if len(moves) > 0 else None
            move2 = moves[1] if len(moves) > 1 else None
            move3 = moves[2] if len(moves) > 2 else None
            move4 = moves[3] if len(moves) > 3 else None

            # Normalizar nombre
            name_lower = 'mimikyu-disguised' if name.lower() == 'mimikyu' else name.lower()
            namePkmn = constants.standardName.get(name_lower, name_lower)

            self.opponent.opponents.append(
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
# OK - Limpiar Nueva Batalla                                                                                                                             ###       
############################################################################################################################################################
    def toClean(self) -> None:
        """
        Limpia y reinicia el estado de la batalla para una nueva ronda (por ejemplo, en un BO3),
        reseteando al usuario, oponente, entorno, turnos y recompensas.
        Mantiene intactos los Tag Globales.
        """
        # =====================================================================
        # 1. Reiniciar Usuario y Oponente
        # =====================================================================
        self.user = Battler()
        self.opponent = Battler()

        # =====================================================================
        # 2. Reiniciar Características del Entorno
        # =====================================================================
        self.weather = None
        self.weatherTurn = 0
        self.terrain = defaultdict(lambda: 0)

        # =====================================================================
        # 3. Reiniciar Estados e Info del Turno
        # =====================================================================
        self.turn = 0
        self.force_switch = False
        self.wait = False
        self.error = False
        self.request_json = None
        self.rqid = None

        # =====================================================================
        # 4. Reiniciar Recompensas
        # =====================================================================
        self.reward = 0
        self.score = 0

        # Nota: Los 'Tag Globales' (self.battleBO3_tag, self.battleBO3_count, 
        # self.botMode, self.battle_tag) NO se tocan y conservan sus valores.

############################################################################################################################################################
# OK - Impresión de TODA la Batalla                                                                                                                      ###       
############################################################################################################################################################
    def imprimir(self):
        """Imprime por consola y guarda en 'reporte.txt' el estado completo de la batalla."""
        with open('reporte.txt', 'a', encoding='utf-8') as f:
            print("**************************************************", file=f)
            print("##################################################", file=f)
            print("                ESTADO DEL ENTORNO                ", file=f)
            print("##################################################", file=f)
            print(f"Turno: {self.turn} | rqid: {self.rqid}", file=f)
            print(f"Clima: {self.weather} (Turnos restantes: {self.weatherTurn})", file=f)
            print(f"Terrenos: {dict(self.terrain)}", file=f)
            print(f"Force Switch: {self.force_switch} | active: {self.active} | Wait: {self.wait} | Error: {self.error}", file=f)
            print(f"Recompensa actual: {self.reward} | Score: {self.score}", file=f)
            
            print("\n--------------------------------------------------", file=f)
            print("                     USUARIO                      ", file=f)
            print("--------------------------------------------------", file=f)
            self.user.imprimir(f)

            print("\n--------------------------------------------------", file=f)
            print("                     OPONENTE                     ", file=f)
            print("--------------------------------------------------", file=f)
            self.opponent.imprimir(f)
            print("**************************************************", file=f)


############################################################################################################################################################
# PENDIENTE - Cálculos de las Recompensas                                                                                                                ###
############################################################################################################################################################
    def calculateRewards(self):
        self.lastReward = 100.00
        self.reward = 1000.00
        return self.lastReward, self.reward


############################################################################################################################################################
# OK - CLASE BATTLER                                                                                                                                     ###
############################################################################################################################################################
class Battler:
    def __init__(self):
        self.id = None                                  # Ejm: p1
        self.account_name = None                        # String
        self.active1 = None                             # class Pokemon or None
        self.active2 = None                             # class Pokemon or None
        self.reserved = []                              # class Pokemon
        self.opponents = []                             # class Pokemon
        self.side_conditions = defaultdict(lambda: 0)   # Tailwind, Stealth Rock, Reflect
        self.isUser = False                             # Se actualiza en True solo cuando es Usuario
        self.opponentPkmnConfirmed = 0                  # Contabilizador de reservados


###############################################################################################################################################################
# OK - Actualiza TODOS los datos de los PKMN del usuario en base al JSON recibido por cada tipo de |request|                                                ###
###############################################################################################################################################################
    def updatePkmnUser(self, JSONdict: Union[str, Dict[str, Any]], first_turn: bool, natures: list) -> None:
        if (first_turn): # TEAM PREVIEW - 6 PKMN
            # Datos Básicos
            self.id = JSONdict["side"]["id"]
            self.account_name = JSONdict["side"]["name"]
            self.isUser = True

            for i, pkmn_data in enumerate(JSONdict["side"]["pokemon"]):
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
                pkmn = Pokemon(namePkmn, max_hp, False, item, ability, m1, m2, m3, m4, natures[i])

                # Asignar stats en combate
                stats = pkmn_data["stats"]
                pkmn.ATK_battle_actual = int(stats["atk"])
                pkmn.DEF_battle_actual = int(stats["def"])
                pkmn.SPA_battle_actual = int(stats["spa"])
                pkmn.SPD_battle_actual = int(stats["spd"])
                pkmn.VEL_battle_actual = int(stats["spe"])

                self.reserved.append(pkmn)
        else:
            if len(self.reserved) == 4: # Solo quedan 2 PKMN activos para las rondas
                # Creamos una lista temporal con todos los Pokémon actuales de la reserva
                current_reserved = list(self.reserved)
                self.reserved = []
                
                # Recorremos estrictamente el JSON en el orden en que vienen (índices 2 y 3)
                for idx in (2, 3):
                    if idx < len(JSONdict["side"]["pokemon"]):
                        pkmn_data = JSONdict["side"]["pokemon"][idx]
                        raw_details = pkmn_data["details"].partition(",")[0].lower().replace(" ", "-")
                        target_name = constants.standardName.get(raw_details, raw_details)
                        
                        # Buscamos en la reserva anterior cuál coincide con este nombre en este orden exacto
                        found_pkmn = None
                        for p in current_reserved:
                            if p is not None and p.name == target_name:
                                found_pkmn = p
                                break
                        if found_pkmn:
                            self.reserved.append(found_pkmn)

            # =========================================================================
            # UPDATE POKÉMON ACTIVO 1 (Izquierda - Índice 0 del JSON)
            # =========================================================================
            pokemon0 = JSONdict["side"]["pokemon"][0]
            cond0 = pokemon0["condition"]
            if "/" in cond0:
                self.active1.HPA_battle_actual = int(cond0.partition("/")[0].partition(" ")[0])
                self.active1.HPB_battle_base = int(cond0.partition("/")[2].partition(" ")[0])
            else:
                self.active1.HPA_battle_actual = 0  # Debilitado (ej. "0 fnt")

            # Estadísticas de combate (Activo 1)
            stats1 = pokemon0["stats"]
            self.active1.ATK_battle_actual = int(stats1["atk"])
            self.active1.DEF_battle_actual = int(stats1["def"])
            self.active1.SPA_battle_actual = int(stats1["spa"])
            self.active1.SPD_battle_actual = int(stats1["spd"])
            self.active1.VEL_battle_actual = int(stats1["spe"])

            # =========================================================================
            # UPDATE POKÉMON ACTIVO 2 (Derecha - Índice 1 del JSON)
            # =========================================================================
            pokemon1 = JSONdict["side"]["pokemon"][1]
            cond1 = pokemon1["condition"]
            if "/" in cond1:
                self.active2.HPA_battle_actual = int(cond1.partition("/")[0].partition(" ")[0])
                self.active2.HPB_battle_base = int(cond1.partition("/")[2].partition(" ")[0])
            else:
                self.active2.HPA_battle_actual = 0  # Debilitado (ej. "0 fnt")
            
            # Estadísticas de combate (Activo 2)
            stats2 = pokemon1["stats"]
            self.active2.ATK_battle_actual = int(stats2["atk"])
            self.active2.DEF_battle_actual = int(stats2["def"])
            self.active2.SPA_battle_actual = int(stats2["spa"])
            self.active2.SPD_battle_actual = int(stats2["spd"])
            self.active2.VEL_battle_actual = int(stats2["spe"])

            # Asegurándonos de que 'active' tenga al menos 2 elementos en la lista
            request_active = JSONdict.get("active", [])
            
            # Para el Pokémon de la izquierda (active1 - índice 0)
            active1_info = request_active[0] if len(request_active) > 0 else {}
            self.active1.can_mega_evo = bool(active1_info.get("canMegaEvo", False))
            
            # Para el Pokémon de la derecha (active2 - índice 1)
            active2_info = request_active[1] if len(request_active) > 1 else {}
            self.active2.can_mega_evo = bool(active2_info.get("canMegaEvo", False))

            # Update Pkmn - Turno 2 en adelante... Moves (PPs)
            # Cambiado por una validación más limpia de existencia de "active"
            if "active" in JSONdict and JSONdict["active"] is not None:
                for i in range(len(JSONdict["active"][0].get("moves", []))):
                    move_data = JSONdict["active"][0]["moves"][i]
                    if move_data.get("pp") is not None:
                        self.active1.movesPP[i] = int(move_data["pp"])
                for i in range(len(JSONdict["active"][1].get("moves", []))):
                    move_data = JSONdict["active"][1]["moves"][i]
                    if move_data.get("pp") is not None:
                        self.active2.movesPP[i] = int(move_data["pp"])

            # =========================================================================
            # UPDATE POKÉMON RESERVA 1 (Index 2 del JSON)
            # =========================================================================
            pokemon2 = JSONdict["side"]["pokemon"][2]
            cond2 = pokemon2["condition"]
            if "/" in cond2:
                self.reserved[0].HPA_battle_actual = int(cond2.partition("/")[0].partition(" ")[0])
                self.reserved[0].HPB_battle_base = int(cond2.partition("/")[2].partition(" ")[0])
            else:
                self.reserved[0].HPA_battle_actual = 0  # Debilitado (ej. "0 fnt")
            
            # Estadísticas de combate (Reserva 1)
            stats3 = pokemon2["stats"]
            self.reserved[0].ATK_battle_actual = int(stats3["atk"])
            self.reserved[0].DEF_battle_actual = int(stats3["def"])
            self.reserved[0].SPA_battle_actual = int(stats3["spa"])
            self.reserved[0].SPD_battle_actual = int(stats3["spd"])
            self.reserved[0].VEL_battle_actual = int(stats3["spe"])

            # =========================================================================
            # UPDATE POKÉMON RESERVA 2 (Index 3 del JSON)
            # =========================================================================
            pokemon3 = JSONdict["side"]["pokemon"][3]
            cond3 = pokemon3["condition"]
            if "/" in cond3:
                self.reserved[1].HPA_battle_actual = int(cond3.partition("/")[0].partition(" ")[0])
                self.reserved[1].HPB_battle_base = int(cond3.partition("/")[2].partition(" ")[0])
            else:
                self.reserved[1].HPA_battle_actual = 0  # Debilitado (ej. "0 fnt")
            
            # Estadísticas de combate (Reserva 2)
            stats4 = pokemon3["stats"]
            self.reserved[1].ATK_battle_actual = int(stats4["atk"])
            self.reserved[1].DEF_battle_actual = int(stats4["def"])
            self.reserved[1].SPA_battle_actual = int(stats4["spa"])
            self.reserved[1].SPD_battle_actual = int(stats4["spd"])
            self.reserved[1].VEL_battle_actual = int(stats4["spe"])


###############################################################################################################################################################
# OK - Impresión de la clase Battler                                                                                                                        ###
###############################################################################################################################################################
    def imprimir(self, f):
        print(f"ID: {self.id} | Cuenta: {self.account_name}", file=f)
        
        if not self.isUser:
            print(f"Pokémon del oponente confirmados: {self.opponentPkmnConfirmed}", file=f)

        # 1. Side conditions en una sola línea si no hay ninguna
        if self.side_conditions:
            print("Side conditions:", file=f)
            for key, value in self.side_conditions.items():
                print(f"  {key} : {value}", file=f)
        else:
            print("Side conditions: Ninguna", file=f)

        print("\n--- POKÉMON ACTIVO 1 ---", file=f)
        if self.active1 is not None:
            self.active1.imprimir(f, not self.isUser)
        else:
            print("  Ninguno (None)", file=f)

        print("\n--- POKÉMON ACTIVO 2 ---", file=f)
        if self.active2 is not None:
            self.active2.imprimir(f, not self.isUser)
        else:
            print("  Ninguno (None)", file=f)

        print("\n--- POKÉMON EN RESERVA ---", file=f)
        if self.reserved:
            for i, pkmn in enumerate(self.reserved, start=1):
                print(f"\n[Reserva {i}]", file=f)
                if pkmn is not None:
                    pkmn.imprimir(f, not self.isUser)
                else:
                    print("  Vacío (None)", file=f)
        else:
            print("  Sin reservas", file=f)

        # 2. Sección de Pokémon Oponentes solo para el bando oponente
        if not self.isUser:
            # Obtenemos la lista usando 'not_confirmed' o 'opponents' según cómo esté definida en tu clase
            lista_oponentes = getattr(self, 'opponents', [])
            
            print("\n--- POKÉMON OPONENTES ---", file=f)
            if lista_oponentes:
                for i, pkmn in enumerate(lista_oponentes, start=1):
                    print(f"\n[Oponente {i}]", file=f)
                    if pkmn is not None:
                        pkmn.imprimir(f, not self.isUser)
                    else:
                        print("  Vacío (None)", file=f)
            else:
                print("  Sin Pokémon oponentes en lista de preview", file=f)

        print("-" * 50, file=f)


############################################################################################################################################################
# OK - CLASE POKEMON                                                                                                                                     ###
############################################################################################################################################################
class Pokemon:
    def __init__(
        self, 
        name: str, 
        HPB_battle_base: int, 
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
        self.nature = nature
        self.is_opponent = is_opponent
        self.objectPkmn = object_pkmn
        self.abilityPkmn = ability_pkmn
        self.is_mega_evo = False
        self.can_mega_evo = False
        self.is_opponent_confirmed = False
        self.is_ditto = True if name == 'ditto' else False
        self.is_zoroark = True if name == 'Zoroark' or name == 'zoroark-hisui' else False
        self.is_fainted = False
        self.is_substitute = False
        
        # Tipos (Se asignan posteriormente)
        self.type1 = None
        self.type2 = None

        # Estados y Modificadores de Combate
        self.non_volatile_statuses = defaultdict(lambda: 0)  # Ej: ['slp', 'brn', 'frz', 'par', 'psn']
        self.volatile_statuses = defaultdict(lambda: 0)      # Ej: 'yawn', etc.
        self.boosts = defaultdict(lambda: 0)                 # Modificadores de estadísticas (-6 a +6)

        # Estadísticas Base - Solo se llena una vez para el Oponente, -1 para el Usuario
        self.HPB_battle_base = HPB_battle_base
        self.ATK_battle_base = -1
        self.DEF_battle_base = -1
        self.SPA_battle_base = -1
        self.SPD_battle_base = -1
        self.VEL_battle_base = -1

        # Estadísticas Actuales - Se actualiza por cada turno, -1 para el Oponente
        self.HPA_battle_actual = HPB_battle_base
        self.ATK_battle_actual = -1
        self.DEF_battle_actual = -1
        self.SPA_battle_actual = -1
        self.SPD_battle_actual = -1
        self.VEL_battle_actual = -1

        # Movimientos y PP
        self.moves = [move1, move2, move3, move4]
        self.movesPP = [-1, -1, -1, -1]

        # Actualizar atributos
        self.getStatsBattle(is_opponent)


############################################################################################################################################################
# OK - Actualización Otros Atributos POKEAPI                                                                                                             ###
############################################################################################################################################################
    def getStatsBattle(self, isOpponent) -> None:
        type1, type2, HPB_battle_base, ATK_battle_base, DEF_battle_base, SPA_battle_base, SPD_battle_base, VEL_battle_base = getPkmnInfo(self.name)
        if isOpponent:
            self.type1, self.type2, self.HPB_battle_base, self.ATK_battle_base, self.DEF_battle_base, self.SPA_battle_base, self.SPD_battle_base, self.VEL_battle_base = type1, type2, HPB_battle_base, ATK_battle_base, DEF_battle_base, SPA_battle_base, SPD_battle_base, VEL_battle_base
            self.HPA_battle_actual = 100
        else:
            self.type1, self.type2 = type1, type2


############################################################################################################################################################
# OK - Impresión de la clase Pokemon                                                                                                                     ###
############################################################################################################################################################
    def imprimir(self, f, isOpponent) -> None:
        """Imprime por consola o archivo el estado actual del Pokémon de forma organizada."""
        print(f"  > POKÉMON: {self.name}", file=f)
        print(f"    Tipo 1: {self.type1} | Tipo 2: {self.type2}", file=f)
        print(f"    Naturaleza: {self.nature} | Objeto: {self.objectPkmn} | Habilidad: {self.abilityPkmn}", file=f)
        if isOpponent:
            print(f"    Es Oponente: {self.is_opponent} | Es Oponente Confirmado: {self.is_opponent_confirmed}", file=f)
        print(f"    Puede Megeevolucionar: {self.can_mega_evo} | Ha Megaevolucionado: {self.is_mega_evo}  ", file=f)
        print(f"    Es Ditto: {self.is_ditto} | Es Zoroark: {self.is_zoroark}", file=f)
        print(f"    ¿Sustituto?: {self.is_substitute} | ¿Debilitado?: {self.is_fainted}", file=f)
        
        print("    [Estados y Modificadores]", file=f)
        print(f"      No Volátiles: {dict(self.non_volatile_statuses)}", file=f)
        print(f"      Volátiles: {dict(self.volatile_statuses)}", file=f)
        print(f"      Boosts Activos: {dict(self.boosts)}", file=f)

        print("    [Puntos de Vida]", file=f)
        print(f"      HPA: {self.HPA_battle_actual} | HPB: {self.HPB_battle_base}", file=f)

        print("    [Estadísticas Base]", file=f)
        print(f"      ATK: {self.ATK_battle_base} | DEF: {self.DEF_battle_base}", file=f)
        print(f"      SPA: {self.SPA_battle_base} | SPD: {self.SPD_battle_base}", file=f)
        print(f"      VEL: {self.VEL_battle_base}", file=f)

        print("    [Estadísticas Actuales]", file=f)
        print(f"      ATK: {self.ATK_battle_actual} | DEF: {self.DEF_battle_actual}", file=f)
        print(f"      SPA: {self.SPA_battle_actual} | SPD: {self.SPD_battle_actual} | ", file=f)
        print(f"      VEL: {self.VEL_battle_actual}", file=f)

        print("    [Movimientos y PP]", file=f)
        for i, (move, pp) in enumerate(zip(self.moves, self.movesPP), start=1):
            print(f"      Movimiento {i}: {move} (PP: {pp})", file=f)