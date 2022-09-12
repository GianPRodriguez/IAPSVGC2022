import json
import requests
import constants

urlPokeAPI = 'https://pokeapi.co/api/v2/'

def getMoveColor(move):
    if move in constants.maxMoves:
        move = constants.maxMoves[move]
    urlMove = urlPokeAPI + 'move/' +  move
    response = requests.get(urlMove)
    if response.status_code == 200:
        payload = response.json()
        return constants.colorType[payload['type']['name']]
    else:
        return '#FFFFFF'

def getPkmnImg(listPkmn):
    listImg = []
    for pkmn in listPkmn:
        pkmn = pkmn.lower().replace(' ', '-')
        if pkmn in constants.standardName:
            pkmn = constants.standardName[pkmn]
        url = urlPokeAPI + 'pokemon/' +  pkmn
        response = requests.get(url)
        if response.status_code == 200:
            payload = response.json()
            img = payload['sprites']['other']['official-artwork']['front_default']
            listImg.append(img)
    return listImg[0], listImg[1], listImg[2], listImg[3], listImg[4], listImg[5]

def getMoveTarget(battle, textAux, toMove):
    if textAux in constants.maxMoves:
        textAux = constants.maxMoves[textAux]
    urlMove = urlPokeAPI + 'move/' +  textAux
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

def getPkmnInfo(namePkmn):
    url = urlPokeAPI + 'pokemon/' + namePkmn.replace(' ', '-')
    response = requests.get(url)
    if response.status_code == 200:
        payload = response.json()
        # Stats Base
        HP_base = payload['stats'][0]['base_stat']
        ATK_base = payload['stats'][1]['base_stat']
        DEF_base = payload['stats'][2]['base_stat']
        SPA_base = payload['stats'][3]['base_stat']
        SPD_base = payload['stats'][4]['base_stat']
        VEL_base = payload['stats'][5]['base_stat']
        pkmnType1 = payload.get('types','Null')[0].get('type','Null').get('name','Null')
        if len(payload.get('types','Null')) == 1:
            pkmnType2 = 'Null'
        else:
            pkmnType2 = payload.get('types','Null')[1].get('type','Null').get('name','Null')
        return pkmnType1, pkmnType2, HP_base, ATK_base, DEF_base, SPA_base, SPD_base, VEL_base
    return 'Nada', 'Nada', 'Nada', 'Nada', 'Nada', 'Nada', 'Nada', 'Nada'