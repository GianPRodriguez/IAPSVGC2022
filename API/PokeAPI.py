import requests
import constants

urlPokeAPI = 'https://pokeapi.co/api/v2/'

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