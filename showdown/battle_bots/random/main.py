import math
import random
from API.MoveData import getColorMove
import constants
from showdown.battle import Battle
from showdown.battle_bots.helpers import getMoveName
from showdown.battle_bots.helpers import getToMovePokemon


############################################################################################################################################################
# OK - Implementación del bot aleatorio (Random Bot) para Pokémon Showdown VGC                                                                           ###
############################################################################################################################################################
class BattleBot(Battle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


############################################################################################################################################################
# OK - Determina de manera inteligente y segura qué Pokémon de la reserva  debe entrar al campo a combatir en caso de un cambio (switch)                 ###
############################################################################################################################################################
    def _choose_switch_slot(self, battle: Battle) -> int:

        switch3_fainted = battle.user.reserved[0].is_fainted
        switch4_fainted = battle.user.reserved[1].is_fainted

        # Si el Pokémon del slot 3 está debilitado, obligatoriamente elegimos el slot 4
        if not switch3_fainted and switch4_fainted:
            return 3
        # Si el Pokémon del slot 4 está debilitado, obligatoriamente elegimos el slot 3
        if switch3_fainted and not switch4_fainted:
            return 4
        
        # Si ambos están sanos, seleccionamos uno al azar
        return random.choice([3, 4])


############################################################################################################################################################
# OK - Obtiene la lista de los índices de movimientos que están actualmente disponibles (no deshabilitados) para un Pokémon activo específico.           ###
############################################################################################################################################################
    def _get_available_moves(self, battle: Battle, active_index: int) -> list[int]:
        active_pokemon_data = battle.request_json.get('active', [])
        
        if active_index >= len(active_pokemon_data):
            return []

        # Obtenemos la lista de diccionarios de movimientos del Pokémon en ese slot
        moves_data = active_pokemon_data[active_index].get('moves', [])
        
        available_moves = []
        for i, move_info in enumerate(moves_data):
            # Verificamos que el movimiento no esté deshabilitado (disabled) y tenga PP
            is_disabled = move_info.get('disabled', False)
            pp = move_info.get('pp', 1)  # Si no tiene el campo pp asumimos que tiene (ej. force-moves)
            
            if not is_disabled and pp > 0:
                # Los índices en Showdown para los movimientos van de 1 a 4
                available_moves.append(i + 1)
                
        return available_moves


############################################################################################################################################################
# OK - Escoge los 4 Pokémon de los 6 disponibles de manera aleatoria para el inicio de la batalla                                                        ###
############################################################################################################################################################
    async def action_team_preview(self, battle: Battle) -> tuple[int, ...]:
        # 1. Obtener 4 índices únicos aleatorios entre 1 y 6
        pkmn_indices = random.sample(range(1, 7), 4)

        # 2. Acceso seguro a la lista de Pokémon en el JSON
        pokemon_list = battle.request_json.get('side', {}).get('pokemon', [])

        # 3. Extraer nombres de forma segura usando list comprehension
        names = []
        for idx in pkmn_indices:
            if 0 <= idx - 1 < len(pokemon_list):
                name = pokemon_list[idx - 1].get('details', '').partition(",")[0]
            else:
                name = "Unknown"
            names.append(name)

        # 4. Enviar a la cola para la interfaz gráfica (Módulo de Análisis)
        if getattr(battle, 'q', None) is not None:
            battle.q.put([1, names[0], names[1], names[2], names[3], 0, 0])

        # 5. Retornar los 4 enteros
        return tuple(pkmn_indices) # tupla (5, 2, 1, 4)


############################################################################################################################################################
# OK - Escoge acciones aleatorias válidas para el turno de batalla (Soporta Mega-Evolución VGC)                                                          ###
############################################################################################################################################################
    async def action_battle(self, battle: Battle, _) -> tuple[str, int, int, int, str, int, int, int, float, float]:
        moveSwitch1 = None
        nMoveSwitch1 = 0
        megaEvo1 = 0
        toMove1 = 0

        moveSwitch2 = None
        nMoveSwitch2 = 0
        megaEvo2 = 0
        toMove2 = 0

        request_active = battle.request_json.get('active', [])

        # =====================================================================
        # CASO 1: Cambio Obligatorio (forceSwitch)
        # =====================================================================
        if battle.force_switch:
            force_switches = battle.request_json.get(constants.FORCE_SWITCH, [False, False])
            izquierdaSwitch = force_switches[0] if len(force_switches) > 0 else False
            derechaSwitch = force_switches[1] if len(force_switches) > 1 else False

            if izquierdaSwitch and not derechaSwitch:
                moveSwitch2 = 'pass'
                moveSwitch1 = 'switch'
                nMoveSwitch1 = self._choose_switch_slot(battle)

            elif not izquierdaSwitch and derechaSwitch:
                moveSwitch1 = 'pass'
                moveSwitch2 = 'switch'
                nMoveSwitch2 = self._choose_switch_slot(battle)

            else:  # Ambos Pokémon deben cambiar obligatoriamente
                moveSwitch1 = 'switch'
                moveSwitch2 = 'switch'

                switch3F = battle.user.reserved[0].is_fainted
                switch4F = battle.user.reserved[1].is_fainted

                if switch3F and not switch4F:
                    nMoveSwitch1 = 4
                    moveSwitch2 = 'pass'
                elif not switch3F and switch4F:
                    nMoveSwitch1 = 3
                    moveSwitch2 = 'pass'
                elif not switch3F and not switch4F:
                    nMoveSwitch1 = 3
                    nMoveSwitch2 = 4

        # =====================================================================
        # CASO 2: Turno de Batalla Normal
        # =====================================================================
        else:
            # -----------------------------------------------------------------
            # POKÉMON IZQUIERDA (Active 1)
            # -----------------------------------------------------------------
            if not battle.user.active1.is_fainted:
                active1_info = request_active[0] if len(request_active) > 0 else {}
                is_trapped = active1_info.get(constants.TRAPPED, False) or active1_info.get(constants.MAYBE_TRAPPED, False)
                no_reserves = battle.user.reserved[0].is_fainted and battle.user.reserved[1].is_fainted

                if is_trapped or no_reserves:
                    moveSwitch1 = 'move'
                else:
                    moveSwitch1 = random.choices(['move', 'switch'], weights=(80, 20), k=1)[0]

                if moveSwitch1 == 'move':
                    avail_moves = self._get_available_moves(battle, 0)
                    if avail_moves:
                        nMoveSwitch1 = random.choice(avail_moves)
                        selected_move_info = active1_info.get('moves', [])[nMoveSwitch1 - 1]

                        # Validar si el movimiento carece de 'target' (movimiento de carga/recarga)
                        is_charging_or_recharging_1 = 'target' not in selected_move_info

                        # Selección aleatoria de Mega-Evolución si está disponible y NO está recargando/cargando
                        if not is_charging_or_recharging_1 and active1_info.get('canMegaEvo', False):
                            megaEvo1 = random.choice([0, 1])
                        else:
                            megaEvo1 = 0

                        # Selección de Objetivo (Target)
                        if is_charging_or_recharging_1:
                            toMove1 = 0
                        else:
                            target_type = selected_move_info.get('target', 'normal')

                            if target_type in constants.MOVETYPES_WITHOUT_PARAMETERS:
                                # Movimientos que no requieren seleccionar objetivo
                                toMove1 = 0
                            elif target_type in constants.MOVETYPES_SINGLE_TARGET:
                                # Pueden apuntar a cualquier Pokémon en 2v2
                                if not battle.user.active2.is_fainted:
                                    # Comprobamos si el aliado está vivo
                                    if random.random() < 0.10: 
                                        # 10% de probabilidad de atacar al aliado, 90% a un rival
                                        toMove1 = -2
                                    else:
                                        toMove1 = random.choice([1, 2])
                                else:
                                    # Si el aliado está debilitado, 100% de probabilidad a un rival
                                    toMove1 = random.choice([1, 2])
                            elif target_type == constants.MOVETYPES_ADJACENT_FOE:
                                # Exclusivamente para atacar a un rival
                                toMove1 = random.choice([1, 2])
                            elif target_type == constants.MOVETYPES_ADJACENT_ALLY:
                                # Exclusivamente a un aliado (ej. Refuerzo)
                                toMove1 = -1
                            elif target_type == constants.MOVETYPES_ALLY_OR_SELF:
                                # 50% de probabilidad al aliado (-2), 50% a sí mismo (0)
                                if not battle.user.active2.is_fainted:
                                    toMove1 = random.choice([-2, 0])
                                else:
                                    # Si el aliado está debilitado, se lo aplica a sí mismo 100%
                                    toMove1 = 0
                            else:
                                print("Se ha recibido un targetType LEFT desconocido: {}".format(target_type));
                                raise ValueError("Se ha recibido un targetType LEFT desconocido")
                    else:
                        moveSwitch1 = 'switch'
                        nMoveSwitch1 = self._choose_switch_slot(battle)
                else:
                    nMoveSwitch1 = self._choose_switch_slot(battle)
            else:
                moveSwitch1 = 'pass'

            # -----------------------------------------------------------------
            # POKÉMON DERECHA (Active 2)
            # -----------------------------------------------------------------
            if not battle.user.active2.is_fainted:
                active2_info = request_active[1] if len(request_active) > 1 else {}
                is_trapped = active2_info.get('trapped', False) or active2_info.get('maybeTrapped', False)
                no_reserves = battle.user.reserved[0].is_fainted and battle.user.reserved[1].is_fainted

                # Evitamos elegir cambiar al mismo slot reservado que el active1
                reserved_slot_used = (moveSwitch1 == 'switch')

                if is_trapped or no_reserves or reserved_slot_used:
                    moveSwitch2 = 'move'
                else:
                    moveSwitch2 = random.choices(['move', 'switch'], weights=(80, 20), k=1)[0]

                if moveSwitch2 == 'move':
                    avail_moves = self._get_available_moves(battle, 1)
                    if avail_moves:
                        nMoveSwitch2 = random.choice(avail_moves)
                        selected_move_info = active2_info.get('moves', [])[nMoveSwitch2 - 1]

                        # Validar si el movimiento carece de 'target' (movimiento de carga/recarga)
                        is_charging_or_recharging_2 = 'target' not in selected_move_info

                        # Selección aleatoria de Mega-Evolución si no está cargando, canMegaEvo es True y el lado 1 no megaevolucionó
                        if not is_charging_or_recharging_2 and active2_info.get('canMegaEvo', False) and megaEvo1 == 0:
                            megaEvo2 = random.choice([0, 1])
                        else:
                            megaEvo2 = 0

                        # Selección de Objetivo (Target)
                        if is_charging_or_recharging_2:
                            toMove2 = 0
                        else:
                            target_type = selected_move_info.get('target', 'normal')

                            if target_type in constants.MOVETYPES_WITHOUT_PARAMETERS:
                                # Movimientos que no requieren seleccionar objetivo
                                toMove2 = 0
                            elif target_type in constants.MOVETYPES_SINGLE_TARGET:
                                # Pueden apuntar a cualquier Pokémon en 2v2
                                if not battle.user.active1.is_fainted:
                                    # Comprobamos si el aliado está vivo
                                    if random.random() < 0.10: 
                                        # 10% de probabilidad de atacar al aliado, 90% a un rival
                                        toMove2 = -1
                                    else:
                                        toMove2 = random.choice([1, 2])
                                else:
                                    # Si el aliado está debilitado, 100% de probabilidad a un rival
                                    toMove2 = random.choice([1, 2])
                            elif target_type == constants.MOVETYPES_ADJACENT_FOE:
                                # Exclusivamente para atacar a un rival
                                toMove2 = random.choice([1, 2])
                            elif target_type == constants.MOVETYPES_ADJACENT_ALLY:
                                # Exclusivamente a un aliado (ej. Refuerzo)
                                toMove2 = -1
                            elif target_type == constants.MOVETYPES_ALLY_OR_SELF:
                                # 50% de probabilidad al aliado (-1), 50% a sí mismo (0)
                                if not battle.user.active1.is_fainted:
                                    toMove2 = random.choice([-1, 0])
                                else:
                                    # Si el aliado está debilitado, se lo aplica a sí mismo 100%
                                    toMove2 = 0
                            else:
                                print("Se ha recibido un targetType RIGHT desconocido: {}".format(target_type));
                                raise ValueError("Se ha recibido un targetType RIGHT desconocido")
                        
                    else:
                        moveSwitch2 = 'switch'
                        nMoveSwitch2 = 4 if nMoveSwitch1 == 3 else 3
                else:
                    if moveSwitch1 == 'switch':
                        nMoveSwitch2 = 4 if nMoveSwitch1 == 3 else 3
                    else:
                        nMoveSwitch2 = self._choose_switch_slot(battle)
            else:
                moveSwitch2 = 'pass'

        # =====================================================================
        # Formateo de descripciones para interfaz y cálculo de recompensas
        # =====================================================================
        if moveSwitch1 == 'move':
            text1 = await getMoveName(battle, 0, nMoveSwitch1 - 1)
            text1Aux = text1.lower().replace(' ', '-')
            color1 = await getColorMove(text1Aux)
            text3 = await getToMovePokemon(battle, toMove1)
            color3 = '#FFFFFF'
        elif moveSwitch1 == 'pass':
            text1, color1, text3, color3 = 'Pass', '#FFFFFF', 'Pass', '#FFFFFF'
        elif moveSwitch1 == 'switch':
            text1, color1 = 'Switch to', '#FFFFFF'
            text3 = battle.user.reserved[nMoveSwitch1 - 3].name if len(battle.user.reserved) > (nMoveSwitch1 - 3) else 'Pkmn'
            color3 = '#FFFFFF'

        if moveSwitch2 == 'move':
            text2 = await getMoveName(battle, 1, nMoveSwitch2 - 1)
            text2Aux = text2.lower().replace(' ', '-')
            color2 = await getColorMove(text2Aux)
            text4 = await getToMovePokemon(battle, toMove2)
            color4 = '#FFFFFF'
        elif moveSwitch2 == 'pass':
            text2, color2, text4 = 'Pass', '#FFFFFF', 'Pass'
            color4 = '#FFFFFF'
        elif moveSwitch2 == 'switch':
            text2, color2 = 'Switch to', '#FFFFFF'
            text4 = battle.user.reserved[nMoveSwitch2 - 3].name if len(battle.user.reserved) > (nMoveSwitch2 - 3) else 'Pkmn'
            color4 = '#FFFFFF'

        lastReward, reward = self.calculateRewards()
        delta = math.floor(lastReward)
        total = math.floor(reward)
        if delta > 0:
            color5 = "#325928"
        elif delta >= 0:
            color5 = "#565828"
        else:
            color5 = "#582831"
        color6 = "#325928" if total > 0 else "#582831"

        # Formato UI Módulo 2: [2, t1,c1,t2,c2,t3,c3,t4,c4, delta, c5, reward, c6]
        if getattr(battle, 'q', None) is not None:
            battle.q.put([
                2, text1, color1, text2, color2, text3, color3, text4, color4,
                delta, color5, total, color6,
            ])

        return (
            moveSwitch1, 
            int(nMoveSwitch1), 
            int(megaEvo1), 
            int(toMove1), 
            moveSwitch2, 
            int(nMoveSwitch2), 
            int(megaEvo2), 
            int(toMove2),
            float(lastReward),
            float(reward)
        )