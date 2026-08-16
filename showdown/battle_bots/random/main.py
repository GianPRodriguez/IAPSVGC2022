import random
from API.PokeAPI import getColorMove
import constants
from showdown.battle import Battle
from showdown.battle_bots.helpers import getMoveName
from showdown.battle_bots.helpers import getToMovePokemon

'''
    MEJORAS PENDIENTES:
    - USAR CONSTANTES
    - EFECTO DE GOTHITELLE
'''

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

        switch3_fainted = battle.user.reserved[0].isFainted()
        switch4_fainted = battle.user.reserved[1].isFainted()

        # Si el Pokémon del slot 3 está debilitado, obligatoriamente elegimos el slot 4
        if not switch3_fainted and switch4_fainted:
            return 3
        # Si el Pokémon del slot 4 está debilitado, obligatoriamente elegimos el slot 3
        if switch3_fainted and not switch4_fainted:
            return 4
        
        # Si ambos están sanos (o debilitados), seleccionamos uno al azar
        return random.choice([3, 4])


############################################################################################################################################################
# OK - Obtiene la lista de los índices de movimientos que están actualmente disponibles (no deshabilitados) para un Pokémon activo específico.           ###
############################################################################################################################################################
    def _get_available_moves(self, battle: Battle, active_index: int) -> list[int]:
        # Obtenemos la lista de movimientos desde el JSON de request actual del servidor de Showdown
        active_pokemon_data = battle.request_json.get('active', [])
        if active_index >= len(active_pokemon_data):
            return []


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

        # 4. Enviar a la cola para la interfaz gráfica
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
            force_switches = battle.request_json.get("forceSwitch", [False, False])
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

                switch3F = battle.user.reserved[0].isFainted()
                switch4F = battle.user.reserved[1].isFainted()

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
            if not battle.user.active1.isFainted():
                active1_info = request_active[0] if len(request_active) > 0 else {}
                is_trapped = active1_info.get('trapped', False) or active1_info.get('maybeTrapped', False)
                no_reserves = battle.user.reserved[0].isFainted() and battle.user.reserved[1].isFainted()

                if is_trapped or no_reserves:
                    moveSwitch1 = 'move'
                else:
                    moveSwitch1 = random.choice(['move', 'switch'])

                if moveSwitch1 == 'move':
                    avail_moves = self._get_available_moves(battle, 0)
                    if avail_moves:
                        nMoveSwitch1 = random.choice(avail_moves)
                        selected_move_info = active1_info.get('moves', [])[nMoveSwitch1 - 1]

                        # Selección aleatoria de Mega-Evolución si está disponible
                        if active1_info.get('canMegaEvo', False):
                            megaEvo1 = random.choice([0, 1])

                        # Selección de Objetivo (Target)
                        target_type = selected_move_info.get('target', 'normal')

                        if target_type in constants.MOVETYPES_WITHOUT_PARAMETERS:
                            # Movimientos que no requieren seleccionar objetivo
                            toMove1 = 0
                        elif target_type in ('normal', 'any'):
                            # Pueden apuntar a cualquier Pokémon en 2v2
                            if not battle.user.active2.isFainted():
                                # Comprobamos si el aliado está vivo
                                if random.random() < 0.10: 
                                    # 10% de probabilidad de atacar al aliado, 90% a un rival
                                    toMove1 = -2
                                else:
                                    toMove1 = random.choice([1, 2])
                            else:
                                # Si el aliado está debilitado, 100% de probabilidad a un rival
                                toMove1 = random.choice([1, 2])
                        elif target_type == 'adjacentFoe':
                            # Exclusivamente para atacar a un rival
                            toMove1 = random.choice([1, 2])
                        elif target_type == 'adjacentAlly':
                            # Exclusivamente a un aliado (ej. Refuerzo)
                            toMove1 = -1
                        elif target_type == 'adjacentAllyOrSelf':
                            # 50% de probabilidad al aliado (-2), 50% a sí mismo (0)
                            if not battle.user.active2.isFainted():
                                toMove1 = random.choice([-2, 0])
                            else:
                                # Si el aliado está debilitado, se lo aplica a sí mismo 100%
                                toMove1 = 0
                        else:
                            # Fallback de seguridad en caso de que llegue un dato corrupto
                            toMove1 = 0
                        
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
            if not battle.user.active2.isFainted():
                active2_info = request_active[1] if len(request_active) > 1 else {}
                is_trapped = active2_info.get('trapped', False) or active2_info.get('maybeTrapped', False)
                no_reserves = battle.user.reserved[0].isFainted() and battle.user.reserved[1].isFainted()

                # Evitamos elegir cambiar al mismo slot reservado que el active1
                reserved_slot_used = (moveSwitch1 == 'switch')

                if is_trapped or no_reserves or reserved_slot_used:
                    moveSwitch2 = 'move'
                else:
                    moveSwitch2 = random.choice(['move', 'switch'])

                if moveSwitch2 == 'move':
                    avail_moves = self._get_available_moves(battle, 1)
                    if avail_moves:
                        nMoveSwitch2 = random.choice(avail_moves)
                        selected_move_info = active2_info.get('moves', [])[nMoveSwitch2 - 1]

                        # Selección aleatoria de Mega-Evolución si está disponible
                        if active2_info.get('canMegaEvo', False):
                            megaEvo2 = random.choice([0, 1])
                        
                        # Selección de Objetivo (Target)
                        target_type = selected_move_info.get('target', 'normal')

                        if target_type in constants.MOVETYPES_WITHOUT_PARAMETERS:
                            # Movimientos que no requieren seleccionar objetivo
                            toMove2 = 0
                        elif target_type in ('normal', 'any'):
                            # Pueden apuntar a cualquier Pokémon en 2v2
                            if not battle.user.active1.isFainted():
                                # Comprobamos si el aliado está vivo
                                if random.random() < 0.10: 
                                    # 10% de probabilidad de atacar al aliado, 90% a un rival
                                    toMove2 = -1
                                else:
                                    toMove2 = random.choice([1, 2])
                            else:
                                # Si el aliado está debilitado, 100% de probabilidad a un rival
                                toMove2 = random.choice([1, 2])
                        elif target_type == 'adjacentFoe':
                            # Exclusivamente para atacar a un rival
                            toMove2 = random.choice([1, 2])
                        elif target_type == 'adjacentAlly':
                            # Exclusivamente a un aliado (ej. Refuerzo)
                            toMove2 = -1
                        elif target_type == 'adjacentAllyOrSelf':
                            # 50% de probabilidad al aliado (-1), 50% a sí mismo (0)
                            if not battle.user.active1.isFainted():
                                toMove2 = random.choice([-1, 0])
                            else:
                                # Si el aliado está debilitado, se lo aplica a sí mismo 100%
                                toMove2 = 0
                        else:
                            # Fallback de seguridad en caso de que llegue un dato corrupto
                            toMove1 = 0
                        
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
            text1 = await getMoveName(battle, 0, nMoveSwitch1 - 1, megaEvo1)
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
            text2 = await getMoveName(battle, 1, nMoveSwitch2 - 1, megaEvo2)
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
        battle.q.put([2, text1, color1, text2, color2, text3, color3, text4, color4, lastReward, reward])

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