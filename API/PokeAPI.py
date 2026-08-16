import json
import requests
import constants

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
async def getPkmnInfo(namePkmn: str):
    """
    Obtiene los tipos y estadísticas base de un Pokémon desde la PokéAPI.
    Retorna: (pkmnType1, pkmnType2, HP_base, ATK_base, DEF_base, SPA_base, SPD_base, VEL_base)
    """
    # Formatear el nombre para que coincida con el estándar de la PokéAPI (minúsculas y guiones)
    formatted_name = namePkmn.strip().lower().replace(' ', '-').replace('.', '').replace("'", "")
    url = f"{urlPokeAPI}pokemon/{formatted_name}"
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            payload = response.json()
            
            # Extracción segura de estadísticas base usando un diccionario o lista por comprensión
            stats = {stat['stat']['name']: stat['base_stat'] for stat in payload.get('stats', [])}
            hp_base = stats.get('hp', 0)
            atk_base = stats.get('attack', 0)
            def_base = stats.get('defense', 0)
            spa_base = stats.get('special-attack', 0)
            spd_base = stats.get('special-defense', 0)
            vel_base = stats.get('speed', 0)
            
            # Extracción segura de tipos
            types_list = payload.get('types', [])
            pkmnType1 = types_list[0]['type']['name'] if len(types_list) > 0 else None
            pkmnType2 = types_list[1]['type']['name'] if len(types_list) > 1 else None
            
            return pkmnType1, pkmnType2, hp_base, atk_base, def_base, spa_base, spd_base, vel_base
            
    except requests.exceptions.RequestException as e:
        print(f"Error de red al consultar la PokéAPI para '{namePkmn}': {e}")
    except (KeyError, IndexError, TypeError) as e:
        print(f"Error al procesar el JSON de la PokéAPI para '{namePkmn}': {e}")

    # Retorno por defecto en caso de fallo (usando None en lugar de 'Nada' para mantener consistencia numérica)
    return None, None, 0, 0, 0, 0, 0, 0