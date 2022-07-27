import logging
import asyncio
import config
import constants
import requests


logger = logging.getLogger(__name__)


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
    maxMoves = {'maxstrike': 'Max Strike',
                          'maxknuckle': 'Max Knuckle',
                          'maxairstream': 'Max Airstream',
                          'maxooze': 'Max Ooze',
                          'maxquake': 'Max Quake',
                          'maxrockfall': 'Max Rockfall',
                          'maxflutterby': 'Max Flutterby',
                          'maxphantasm': 'Max Phantasm',
                          'maxsteelspike': 'Max Steelspike',
                          'maxflare': 'Max Flare',
                          'maxgeyser': 'Max Geyser',
                          'maxovergrowth': 'Max Overgrowth',
                          'maxlightning': 'Max Lightning',
                          'maxmindstorm': 'Max Mindstorm',
                          'maxhailstorm': 'Max Hailstorm',
                          'maxwyrmwind': 'Max Wyrmwind',
                          'maxdarkness': 'Max Darkness',
                          'maxstarfall': 'Max Starfall',
                          'maxguard': 'Max Guard'}
    if position == 0:
        if dynamax == 1 or battle.user.active1.isDynamax == True:
            maxMoveWS = battle.request_json.get('active','False')[position].get('maxMoves','False').get('maxMoves','False')[nMoveSwitch].get('move', 'False')
            if maxMoveWS in maxMoves:
                return maxMoves[maxMoveWS]
            else:
                logger.info("NO DEBE ENTRAR ACÁ - if dynamax == 1 or battle.user.active1.isDynamax == True:")
        else:
            return battle.request_json.get('active','False')[position].get('moves','False')[nMoveSwitch].get('move', 'False')
    elif position == 1:
        if dynamax == 1 or battle.user.active2.isDynamax == True:
            maxMoveWS = battle.request_json.get('active','False')[position].get('maxMoves','False').get('maxMoves','False')[nMoveSwitch].get('move', 'False')
            if maxMoveWS in maxMoves:
                return maxMoves[maxMoveWS]
            else:
                logger.info("NO DEBE ENTRAR ACÁ - elif position == 1:")
        else:
            return battle.request_json.get('active','False')[position].get('moves','False')[nMoveSwitch].get('move', 'False')
    else:
        logger.info("NO DEBE ENTRAR ACÁ - async def getMoveName(battle, position, nMoveSwitch, dynamax):")

async def getColorMove(move):
    url = 'https://pokeapi.co/api/v2/move/'
    colorType = {'normal': '#A8A77A',
                          'fire': '#EE8130',
                          'water': '#6390F0',
                          'electric': '#F7D02C',
                          'grass': '#7AC74C',
                          'ice': '#96D9D6',
                          'fighting': '#C22E28',
                          'poison': '#A33EA1',
                          'ground': '#E2BF65',
                          'flying': '#A98FF3',
                          'psychic': '#F95587',
                          'bug': '#A6B91A',
                          'rock': '#B6A136',
                          'ghost': '#735797',
                          'dragon': '#6F35FC',
                          'dark': '#705746',
                          'steel': '#B7B7CE',
                          'fairy': '#D685AD'}
    maxMoves = {'maxstrike': 'max-strike',
                          'maxknuckle': 'max-knuckle',
                          'maxairstream': 'max-airstream',
                          'maxooze': 'max-ooze',
                          'maxquake': 'max-quake',
                          'maxrockfall': 'max-rockfall',
                          'maxflutterby': 'max-flutterby',
                          'maxphantasm': 'max-phantasm',
                          'maxsteelspike': 'max-steelspike',
                          'maxflare': 'max-flare',
                          'maxgeyser': 'max-geyser',
                          'maxovergrowth': 'max-overgrowth',
                          'maxlightning': 'max-lightning',
                          'maxmindstorm': 'max-mindstorm',
                          'maxhailstorm': 'max-hailstorm',
                          'maxwyrmwind': 'max-wyrmwind',
                          'maxdarkness': 'max-darkness',
                          'maxstarfall': 'max-starfall',
                          'maxguard': 'max-guard'}
    if move in maxMoves:
        move = maxMoves[move]
    urlMove = url + move
    response = requests.get(urlMove)
    if response.status_code == 200:
        payload = response.json()
        return colorType[payload['type']['name']]
    else:
        return '#FFFFFF'

async def getToMovePokemon(battle, textAux, toMove):
    url = 'https://pokeapi.co/api/v2/move/'
    maxMoves = {'maxstrike': 'max-strike',
                          'maxknuckle': 'max-knuckle',
                          'maxairstream': 'max-airstream',
                          'maxooze': 'max-ooze',
                          'maxquake': 'max-quake',
                          'maxrockfall': 'max-rockfall',
                          'maxflutterby': 'max-flutterby',
                          'maxphantasm': 'max-phantasm',
                          'maxsteelspike': 'max-steelspike',
                          'maxflare': 'max-flare',
                          'maxgeyser': 'max-geyser',
                          'maxovergrowth': 'max-overgrowth',
                          'maxlightning': 'max-lightning',
                          'maxmindstorm': 'max-mindstorm',
                          'maxhailstorm': 'max-hailstorm',
                          'maxwyrmwind': 'max-wyrmwind',
                          'maxdarkness': 'max-darkness',
                          'maxstarfall': 'max-starfall',
                          'maxguard': 'max-guard'}
    if textAux in maxMoves:
        textAux = maxMoves[textAux]
    urlMove = url + textAux
    response = requests.get(urlMove)
    if response.status_code == 200:
        payload = response.json()
        typeToMove = payload['target']['name']
        if typeToMove == 'user':
            return 'User'
        elif typeToMove == 'all-opponents':
            return 'All Opponents'
        elif typeToMove == 'all-other-pokemon':
            return 'All Other Pokémon'
        elif typeToMove == 'entire-field':
            return 'Entire Field'
        elif typeToMove == 'users-field':
            return 'Users Field'
        else: #selected-pokemon-me-first #selected-pokemon
            if toMove == 1:
                return battle.opponent.active1.name
            elif toMove == 2:
                return battle.opponent.active2.name
            elif toMove == -1:
                return battle.user.active1.name
            else:
                return battle.user.active2.name
    else:
        return 'Nada'