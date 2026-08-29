import json
import requests
import constants
import os

urlPokeAPI = 'https://pokeapi.co/api/v2/'

############################################################################################################################################################
# OK - Define Color del Movimiento                                                                                                                       ###
############################################################################################################################################################
async def getColorMove(move: str) -> str:
    url = 'https://pokeapi.co/api/v2/move/'

    urlMove = url + move
    response = requests.get(urlMove)
    if response.status_code == 200:
        payload = response.json()
        move_type = payload.get('type', {}).get('name', '')
        return constants.COLOR_TYPE.get(move_type, '#FFFFFF')
    else:
        return '#FFFFFF'


############################################################################################################################################################
# SIN USO                                                                                                                                                ###
############################################################################################################################################################
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


############################################################################################################################################################
# OK - Actualización Otros Atributos                                                                                                                     ###
############################################################################################################################################################
def getPkmnInfo(name_pkmn: str):

    json_path = "API/JSONs/champions_info_stats.json"
    
    try:
        if not os.path.exists(json_path):
            print(f"[ERROR] No se encontró el archivo de estadísticas en: {json_path}")
            return None, None, 0, 0, 0, 0, 0, 0

        with open(json_path, 'r', encoding='utf-8') as f:
            stats_data = json.load(f)

        # Buscamos el Pokémon intentando matchear el nombre exacto o de forma flexible (case-insensitive)
        matched_key = None
        for key in stats_data.keys():
            if key.lower() == name_pkmn.lower().strip():
                matched_key = key
                break
        
        if matched_key:
            pkmn_info = stats_data[matched_key]
            type1 = pkmn_info.get("tipo1")
            type2 = pkmn_info.get("tipo2")
            hp = pkmn_info.get("hp", 0)
            atk = pkmn_info.get("ataque", 0)
            defe = pkmn_info.get("defensa", 0)
            spa = pkmn_info.get("ataque_especial", 0)
            spd = pkmn_info.get("defensa_especial", 0)
            vel = pkmn_info.get("velocidad", 0)
            
            return type1, type2, hp, atk, defe, spa, spd, vel
        else:
            print(f"[AVISO] Pokémon '{name_pkmn}' no encontrado en champions_info_stats.json")
            
    except Exception as e:
        print(f"[ERROR] Falló la lectura de champions_info_stats.json para '{name_pkmn}': {e}")

    # Retorno por defecto si ocurre algún error o no se encuentra
    return None, None, 0, 0, 0, 0, 0, 0