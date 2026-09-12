import sys
import math
import queue
import requests
import constants

from abc import ABC
from abc import abstractmethod
from collections import defaultdict
from API.PokemonData import getPkmnInfo

import json
from typing import Union, Dict, Any

from teams.team_converter import normalize_name


def _fmt_mapping(mapping) -> str:
    """Compacta dicts/defaultdicts vacíos como '-' para reportes legibles."""
    if not mapping:
        return '-'
    data = dict(mapping)
    if not data:
        return '-'
    return ', '.join(f'{k}:{v}' for k, v in data.items())


def _fmt_stat_line(atk, df, spa, spd, spe) -> str:
    return f"Atk {atk} | Def {df} | SpA {spa} | SpD {spd} | Spe {spe}"

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
        """
        Guarda el estado de la batalla para pruebas:
        - logs/latest_state.txt  → solo el estado actual (se sobrescribe; fácil de refrescar)
        - logs/reporte.txt       → historial append con timestamp
        Devuelve un resumen corto para consola.
        """
        import os
        from datetime import datetime

        os.makedirs('logs', exist_ok=True)
        stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        tag = getattr(self, 'battle_tag', None) or getattr(self, 'battleBO3_tag', None) or '?'
        summary = (
            f"[{stamp}] turn={self.turn} rqid={self.rqid} "
            f"{getattr(self.user, 'account_name', '?')} vs {getattr(self.opponent, 'account_name', '?')} "
            f"| weather={self.weather}/{self.weatherTurn} force={self.force_switch} err={self.error}"
        )

        def _write(f, header: bool):
            if header:
                print("=" * 72, file=f)
                print(f"  SNAPSHOT  {stamp}", file=f)
                print(f"  Tag: {tag}  |  BO3 game: {getattr(self, 'battleBO3_count', '?')}", file=f)
                print("=" * 72, file=f)
            else:
                print("=" * 72, file=f)
                print(f"  ESTADO ACTUAL  {stamp}", file=f)
                print(f"  Tag: {tag}", file=f)
                print("=" * 72, file=f)

            print(f"Turno: {self.turn} | rqid: {self.rqid} | modo: {self.botMode}", file=f)
            print(
                f"Clima: {self.weather or '-'} ({self.weatherTurn}) | "
                f"Terreno/campo: {_fmt_mapping(self.terrain)}",
                file=f,
            )
            print(
                f"Flags: force_switch={self.force_switch} active={self.active} "
                f"wait={self.wait} error={self.error}",
                file=f,
            )
            print(f"Reward/Score: {self.reward} / {self.score}", file=f)

            print("\n--- USUARIO ---", file=f)
            self.user.imprimir(f)
            print("\n--- OPONENTE ---", file=f)
            self.opponent.imprimir(f)
            print("=" * 72, file=f)

        with open('logs/latest_state.txt', 'w', encoding='utf-8') as f:
            _write(f, header=False)

        with open('logs/reporte.txt', 'a', encoding='utf-8') as f:
            print(file=f)
            _write(f, header=True)

        return summary


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
            request_pokemon = JSONdict["side"].get("pokemon", [])[:4]
            current_pokemon = [self.active1, self.active2, *self.reserved]
            ordered_pokemon = []

            def request_pokemon_name(pkmn_data):
                raw_name = pkmn_data["details"].partition(",")[0].lower().replace(" ", "-")
                if "-mega" in raw_name:
                    raw_name = raw_name.partition("-mega")[0]
                if raw_name in ("aegislash-blade", "aegislash-shield"):
                    raw_name = "aegislash"
                return constants.standardName.get(raw_name, raw_name)

            def create_pokemon(pkmn_data, name):
                condition = pkmn_data.get("condition", "0/0")
                max_hp = condition.partition("/")[2].partition(" ")[0]
                moves = pkmn_data.get("moves", [])[:4]
                pkmn = Pokemon(
                    name,
                    max_hp,
                    False,
                    pkmn_data.get("item", ""),
                    pkmn_data.get("ability", ""),
                    *(moves + [None] * (4 - len(moves))),
                    None,
                )
                return pkmn

            for pkmn_data in request_pokemon:
                target_name = request_pokemon_name(pkmn_data)
                found_pkmn = next(
                    (
                        pkmn for pkmn in current_pokemon
                        if pkmn is not None and pkmn not in ordered_pokemon
                        and (
                            request_pokemon_name({"details": pkmn.name}) == target_name
                            or (pkmn.is_ditto and target_name == "ditto")
                        )
                    ),
                    None,
                )
                if found_pkmn is None:
                    found_pkmn = create_pokemon(pkmn_data, target_name)
                if found_pkmn is not None:
                    ordered_pokemon.append(found_pkmn)

            if len(ordered_pokemon) >= 2:
                self.active1, self.active2 = ordered_pokemon[:2]
                self.reserved = ordered_pokemon[2:4]

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
            if not (self.active1.is_ditto and self.active1.name != "ditto"):
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
            if not (self.active2.is_ditto and self.active2.name != "ditto"):
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

            # Update Pkmn - Turno 2 en adelante... Moves y PP
            request_active = JSONdict.get("active") or []
            for index, active_pkmn in enumerate((self.active1, self.active2)):
                if active_pkmn is None or index >= len(request_active):
                    continue

                move_data_list = request_active[index].get("moves", [])
                active_pkmn.moves = [
                    move_data.get("id")
                    for move_data in move_data_list[:4]
                ]
                active_pkmn.moves.extend([None] * (4 - len(active_pkmn.moves)))
                active_pkmn.movesPP = [
                    1 if move_data.get("id") == "struggle" else (
                        int(move_data["pp"]) if move_data.get("pp") is not None else -1
                    )
                    for move_data in move_data_list[:4]
                ]
                active_pkmn.movesPP.extend([-1] * (4 - len(active_pkmn.movesPP)))

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
            print(f"Confirmados oponente: {self.opponentPkmnConfirmed}", file=f)
        print(f"Side conditions: {_fmt_mapping(self.side_conditions)}", file=f)

        print("\n  [Activo 1]", file=f)
        if self.active1 is not None:
            self.active1.imprimir(f, not self.isUser)
        else:
            print("    (vacío)", file=f)

        print("\n  [Activo 2]", file=f)
        if self.active2 is not None:
            self.active2.imprimir(f, not self.isUser)
        else:
            print("    (vacío)", file=f)

        print("\n  [Reserva]", file=f)
        if self.reserved:
            for i, pkmn in enumerate(self.reserved, start=1):
                print(f"  -- Reserva {i} --", file=f)
                if pkmn is not None:
                    pkmn.imprimir(f, not self.isUser)
                else:
                    print("    (vacío)", file=f)
        else:
            print("    (sin reservas)", file=f)

        if not self.isUser:
            lista_oponentes = getattr(self, 'opponents', [])
            print("\n  [Lista preview oponentes]", file=f)
            if lista_oponentes:
                for i, pkmn in enumerate(lista_oponentes, start=1):
                    print(f"  -- Preview {i} --", file=f)
                    if pkmn is not None:
                        pkmn.imprimir(f, not self.isUser)
                    else:
                        print("    (vacío)", file=f)
            else:
                print("    (vacía)", file=f)


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
        # Yawn: True desde |-start|move: Yawn hasta dormir / salir / |-end|Yawn.
        # Durante el turno de decisión previo al slp el mon sigue despierto y puede actuar.
        self.is_drowsy = False
        # Perish Song: 0 = sin efecto; 3/2/1 = contador visible de Showdown (| -start|perishN).
        self.perish_count = 0
        
        # Tipos (Se asignan posteriormente)
        self.type1 = None
        self.type2 = None

        # Estados y Modificadores de Combate
        self.non_volatile_statuses = defaultdict(lambda: 0)  # Ej: ['slp', 'brn', 'frz', 'par', 'psn']
        self.volatile_statuses = defaultdict(lambda: 0)      # Ej: 'yawn', 'confusion', 'substitute'
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
        """Impresión compacta orientada a pruebas / depuración de estado."""
        flags = []
        if self.is_fainted:
            flags.append('FAINTED')
        if getattr(self, 'is_drowsy', False):
            flags.append('DROWSY')
        if getattr(self, 'perish_count', 0):
            flags.append(f'PERISH{self.perish_count}')
        if self.is_ditto:
            flags.append('DITTO')
        if self.is_zoroark:
            flags.append('ZOROARK')
        if self.is_mega_evo:
            flags.append('MEGA')
        if self.can_mega_evo:
            flags.append('CAN_MEGA')
        if isOpponent and self.is_opponent_confirmed:
            flags.append('CONFIRMED')

        types = f"{self.type1}" + (f"/{self.type2}" if self.type2 else "")
        hp_pct = ''
        try:
            max_hp = int(float(str(self.HPB_battle_base).split()[0]))
            cur_hp = int(float(str(self.HPA_battle_actual).split()[0]))
            if max_hp > 0:
                hp_pct = f" ({100 * cur_hp / max_hp:.0f}%)"
        except Exception:
            hp_pct = ''

        print(f"    {self.name} [{types}]  HP {self.HPA_battle_actual}/{self.HPB_battle_base}{hp_pct}", file=f)
        print(
            f"      item={self.objectPkmn or '-'} | abi={self.abilityPkmn or '-'} | nat={self.nature or '-'}",
            file=f,
        )
        if flags:
            print(f"      flags: {', '.join(flags)}", file=f)
        print(
            f"      status={_fmt_mapping(self.non_volatile_statuses)} | "
            f"volatile={_fmt_mapping(self.volatile_statuses)} | "
            f"boosts={_fmt_mapping(self.boosts)}",
            file=f,
        )
        print(
            f"      stats: {_fmt_stat_line(self.ATK_battle_actual, self.DEF_battle_actual, self.SPA_battle_actual, self.SPD_battle_actual, self.VEL_battle_actual)}",
            file=f,
        )
        if any(v != -1 for v in (self.ATK_battle_base, self.DEF_battle_base, self.SPA_battle_base, self.SPD_battle_base, self.VEL_battle_base)):
            print(
                f"      base:  {_fmt_stat_line(self.ATK_battle_base, self.DEF_battle_base, self.SPA_battle_base, self.SPD_battle_base, self.VEL_battle_base)}",
                file=f,
            )
        moves = [f"{move}({pp})" for move, pp in zip(self.moves, self.movesPP) if move]
        print(f"      moves: {', '.join(moves) if moves else '-'}", file=f)
