import io
import json
import time
import config
import logging
import requests
import websockets
import asyncio

class LoginError(Exception):
    pass

class SaveReplayError(Exception):
    pass

class PSWebsocketClient:
    websocket = None
    address = None
    login_uri = None
    username = None
    password = None
    last_challenge_time = 0

    @classmethod
    async def create(cls, username, password, address):
        self = PSWebsocketClient()
        self.username = username
        self.password = password
        self.address = "ws://{}/showdown/websocket".format(address)
        self.websocket = await websockets.connect(self.address, ping_interval=None)
        self.login_uri = "https://play.pokemonshowdown.com/action.php"
        self.last_message = None
        return self

    async def join_room(self, room_name):
        message = "/join {}".format(room_name)
        await self.send_message('', [message])
        #logger.info("Joined room '{}'".format(room_name))

    async def receive_message(self):
        self.last_message = await self.websocket.recv()
        return self.last_message

    async def receive_message2(self):
        time.sleep(1.0)
        try:
            await asyncio.wait_for(self.receive_message(), timeout=12.0)
            message = self.last_message
        except asyncio.TimeoutError:
            message = None
        return message

    async def send_message(self, room, message_list):
        message = room + "|" + "|".join(message_list) #<class 'list'>
        #logger.debug("Enviando mensaje hacia el websocket: {}".format(message))
        try:
            await self.websocket.send(message)
        except websockets.ConnectionClosed as exc:
            print(exc)
        self.last_message = message

    async def send_message2(self, room, message_list):
        time.sleep(1.2)
        message = room + "|" + "|".join(message_list)
        try:
            await self.websocket.send(message)
        except websockets.ConnectionClosed as exc:
            print(exc)
        self.last_message = message

    async def get_id_and_challstr(self):
        while True:
            message = await self.receive_message()
            # message = "|challstr|4|a6e6e29652941bc1b872a2c34cb0b4e6fdd83355bd3b8787d1519685b29bfb5d267f479f4be9f9fe554605c6c9842160194586f55644f532d991560f817c5939bf7961319b23de0803bac5f6faa2935e395c0f28da55b4c052ece76f0f64e5acdfea8691ac2607012b0d742704920c9fe3c6bc13730b121d6459b84f03d399b8"
            split_message = message.split('|')
            if split_message[1] == 'challstr':
                # split_message[2] = "4" && split_message[3] = "a6e6e29652941bc1b872a2c34cb0b4e6fdd83355bd3b8787d1519685b29bfb5d267f479f4be9f9fe554605c6c9842160194586f55644f532d991560f817c5939bf7961319b23de0803bac5f6faa2935e395c0f28da55b4c052ece76f0f64e5acdfea8691ac2607012b0d742704920c9fe3c6bc13730b121d6459b84f03d399b8"
                return split_message[2], split_message[3]

    async def login(self, training):
        if not training:
            print("[INFO]     Iniciando sesión...")
        client_id, challstr = await self.get_id_and_challstr()
        # Si se ha registrado una contraseña... ocurre siempre
        if self.password:
            response = requests.post(
                self.login_uri,
                data={
                    'act': 'login',
                    'name': self.username,
                    'pass': self.password,
                    'challstr': "|".join([client_id, challstr])
                }
            )
        else:
            response = requests.post(
                self.login_uri,
                data={
                    'act': 'getassertion',
                    'userid': self.username,
                    'challstr': '|'.join([client_id, challstr]),
                }
            )
        # Debo recibir un status_code, 200 OK
        if response.status_code == 200:
            if self.password:
                response_json = json.loads(response.text[1:])
                if not response_json['actionsuccess']:
                    return -1
                    #raise LoginError("Could not log-in")
                # Obtengo el assertion
                assertion = response_json.get('assertion')
            else:
                assertion = response.text
            # Mensaje
            message = ["/trn " + self.username + ",0," + assertion]
            # "|/trn PopplioPeru98,0,6e414fd79bcc5014f9306dbfbb3d205ab0d5bf178cd3de477b77d8f29624078b152bd9bc54d66afecc73225373f2a92ba24080a35b31325bc2c8b40e5a85dcd2f98b280d9ae42106f309e635f7ab2182856dac6325c5b1b4d184075e7d27ab29b83168cd9e13d7fd3bc71f970b15a76e88a6dbddd17eccdae08456c2ab106a37,popplioperu98,4,1657584912,sim3.psim.us,d14e9223aceb95938c2e3b4c1650426759,768fe0d69ee003cc,aa84d633ecbd9433;54aeedd16494c78287c2d851bf25ee65c7c200f190137d1942bfb1bad2b2edb0a529430602dae99d4edcc0f0ab1d53b82e3b390f37e229e77f306cd8e5ed4677059607d8bcd1ddf1959a4d36d8ddf07598a0521310c2ddc18d3a1dc19e477ea6d0b5e3978a4c3615148bb911774f98a7febfa08c7ffd313e0ad513d06f71aea3755dd51909b6911a9d8b299507acfe3d9ced301014e490e915b4abf93b5aa480b531fd1c91249937eef120120c446ad6a5746f15dd8d7556682a7c7f9a0952223ebd7a293fa0eaaf1b02ed53cb2bbcf4725eddd633325fc93769cadc45bf4ff63cf0a3184689403e191cb8f1b69a4f338d0b3d5e97eca568058d384c302bf6f7e52ee35678009ce28d58ce23825baed8d1168ebcd5530c2140de603bc5bea46e3bc9b70f1858b5e13a39fba1e52f8cf952257a8ae338a660445d7a4c7d07665dee9b17f3aa99d204d0af76468fd89786a70c6f458997ee69cecfd21c2d046ac102397651dd3287a866d1cbe99d10a173d50262960d7df389fc653bc04a09b227f378f6567c307728ddc1de0298727948a66002690928d3927de2ce1d20f97b25d784a3893be816932ef63f58763e081eede2ee8653dc6896e99e9761120b6d93f844fc11918c1420c751a839a891460f5d0fa771d6cfd59431ec8e82090fcfcb467fa2bff310c2d47badc3293a8611bcf7bab28f9576a930d4328b1e3437f45d"
            if not training:
                print("[INFO]     ¡Sesión iniciada con éxito!")
            await self.send_message('', message)
            return 0
        else:
            #logger.error("Could not log-in\nDetails:\n{}".format(response.content))
            return -2
            #raise LoginError("Could not log-in")

    async def update_team(self, team):
        message = ["/utm {}".format(team)]
        # "|/utm Incineroar||aguavberry|intimidate|fakeout,flareblitz,snarl,uturn|Adamant|252,56,160,,40,|||||]Kyogre||choicespecs|drizzle|waterspout,originpulse,icebeam,thunder|Modest|4,,,252,,252||,0,,,,|||]Tornadus||lifeorb|prankster|tailwind,darkpulse,hurricane,icywind|Modest|4,,,252,,252|M|,0,,,,|||]Zacian-Crowned||rustedsword|intrepidsword|behemothblade,closecombat,playrough,sacredsword|Adamant|252,252,,,4,|||||]Indeedee-F||expertbelt|psychicsurge|allyswitch,followme,dazzlinggleam,psychic|Calm|252,,184,,72,|F|,0,,,,|||]Alcremie||focussash|sweetveil|decorate,substitute,dazzlinggleam,psychic|Calm|252,,112,,144,|F|,0,,,,|||"
        await self.send_message('', message)

    async def challenge_user(self, user_to_challenge, battle_format, team, training):
        if time.time() - self.last_challenge_time < 10:
            #logger.info("Sleeping for 10 seconds because last challenge was less than 10 seconds ago")
            await io.sleep(10)
        await self.update_team(team)
        message = ["/challenge {},{}".format(user_to_challenge, battle_format)]
        if not training:
            print("[INFO]     Enfrentándose a {}...".format(user_to_challenge))
        await self.send_message('', message)
        self.last_challenge_time = time.time()

    async def accept_challenge(self, battle_format, team, room_name, training, q):
        if room_name is not None:
            await self.join_room(room_name)
        if not training:
            print("[INFO]     Esperando una batalla en formato {}".format(battle_format))
        await self.update_team(team)
        username = None
        contador = 0
        while username is None:
            time.sleep(1.5)
            msg = await self.receive_message2()
            if msg is not None:
                split_msg = msg.split('|')
                if (len(split_msg) == 9):
                    if (len(split_msg) == 9 and split_msg[1] == "pm" and split_msg[4].startswith("/challenge") and split_msg[5] == battle_format):
                        username = split_msg[2].strip()
            contador += 1
            #print("contador: {}".format(contador))
            if contador == 8 and username is None:
                #print("q.put([-1])")
                #q.put([-1]) ???
                contador = 0
        #print("q.put([0])")
        #q.put([0])
        message = ["/accept " + username]
        await self.send_message('', message)

    async def search_for_match(self, team):
        print("[INFO]     Buscando una batalla en formato {}...".format(config.pokemon_mode))
        await self.update_team(team)
        message = ["/search {}".format(config.pokemon_mode)]
        # "/search gen8vgc2022"
        await self.send_message("", message)

    async def leave_battle(self, battle_tag, save_replay=False):
        if save_replay:
            await self.save_replay(battle_tag)

        message = ["/leave {}".format(battle_tag)]
        await self.send_message('', message)

        while True:
            msg = await self.receive_message()
            if battle_tag in msg and 'deinit' in msg:
                return

    async def save_replay(self, battle_tag):
        message = ["/savereplay"]
        await self.send_message(battle_tag, message)

        while True:
            msg = await self.receive_message()
            if msg.startswith("|queryresponse|savereplay|"):
                obj = json.loads(msg.replace("|queryresponse|savereplay|", ""))
                log = obj['log']
                identifier = obj['id']
                post_response = requests.post(
                    "https://play.pokemonshowdown.com/~~showdown/action.php?act=uploadreplay",
                    data={
                        "log": log,
                        "id": identifier
                    }
                )
                if post_response.status_code != 200:
                    raise SaveReplayError("POST para guardar la repetición no devolvió un código 200: {}".format(post_response.content))
                break

    async def close(self):
        """Cierra la conexión websocket de forma limpia"""
        if self.websocket:
            await self.websocket.close()