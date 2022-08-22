import json
import requests
import math
import constants
import logging
import sys

from multiprocessing import Process, Queue
from collections import defaultdict
from numpy import sqrt 
from abc import ABC
from abc import abstractmethod

logger = logging.getLogger(__name__)

class Battle(ABC):
    def __init__(self, battle_tag):
        self.battle_tag = battle_tag
        self.user = Battler()
        self.opponent = Battler()
        self.weather = None             # ['None','raindance','sunnyday','sandstorm','hail']
        self.weatherTurn = 0
        self.terrain = None             # ['None','electricterrain','grassyterrain','mistyterrain','psychicterrain']
        self.terrainTurn = 0
        self.trick_room = 0             # [0,4,3,2,1]
        self.turn = 0
        self.started = False
        self.rqid = None
        self.force_switch = False
        self.wait = False
        self.request_json = None
        self.reward = 0
        self.lastReward = 0
        self.botMode = None
        self.q = Queue()
        self.cadenaTP = json.loads('[{"0":{"0":0,"1":0,"2":0,"3":0,"4":0,"5":1,"6":0,"7":1,"8":1,"9":0,"10":1,"11":1}},{"1":{"0":0,"1":0,"2":0,"3":0,"4":0,"5":1,"6":0,"7":1,"8":1,"9":1,"10":1,"11":0}},{"2":{"0":0,"1":0,"2":0,"3":0,"4":0,"5":1,"6":1,"7":0,"8":0,"9":1,"10":1,"11":1}},{"3":{"0":0,"1":0,"2":0,"3":0,"4":0,"5":1,"6":1,"7":0,"8":1,"9":1,"10":0,"11":1}},{"4":{"0":0,"1":0,"2":0,"3":0,"4":0,"5":1,"6":1,"7":1,"8":0,"9":1,"10":1,"11":0}},{"5":{"0":0,"1":0,"2":0,"3":0,"4":0,"5":1,"6":1,"7":1,"8":1,"9":0,"10":0,"11":1}},{"6":{"0":0,"1":0,"2":0,"3":0,"4":1,"5":0,"6":0,"7":1,"8":0,"9":1,"10":1,"11":1}},{"7":{"0":0,"1":0,"2":0,"3":0,"4":1,"5":0,"6":0,"7":1,"8":1,"9":1,"10":0,"11":1}},{"8":{"0":0,"1":0,"2":0,"3":0,"4":1,"5":0,"6":1,"7":1,"8":0,"9":1,"10":0,"11":1}},{"9":{"0":0,"1":0,"2":0,"3":0,"4":1,"5":1,"6":0,"7":1,"8":0,"9":1,"10":1,"11":0}},{"10":{"0":0,"1":0,"2":0,"3":0,"4":1,"5":1,"6":0,"7":1,"8":1,"9":0,"10":0,"11":1}},{"11":{"0":0,"1":0,"2":0,"3":0,"4":1,"5":1,"6":1,"7":0,"8":0,"9":1,"10":0,"11":1}},{"12":{"0":0,"1":0,"2":0,"3":1,"4":0,"5":0,"6":0,"7":1,"8":1,"9":0,"10":1,"11":1}},{"13":{"0":0,"1":0,"2":0,"3":1,"4":0,"5":0,"6":0,"7":1,"8":1,"9":1,"10":1,"11":0}},{"14":{"0":0,"1":0,"2":0,"3":1,"4":0,"5":0,"6":1,"7":0,"8":0,"9":1,"10":1,"11":1}},{"15":{"0":0,"1":0,"2":0,"3":1,"4":0,"5":0,"6":1,"7":0,"8":1,"9":1,"10":0,"11":1}},{"16":{"0":0,"1":0,"2":0,"3":1,"4":0,"5":0,"6":1,"7":1,"8":0,"9":1,"10":1,"11":0}},{"17":{"0":0,"1":0,"2":0,"3":1,"4":0,"5":0,"6":1,"7":1,"8":1,"9":0,"10":0,"11":1}},{"18":{"0":0,"1":0,"2":0,"3":1,"4":0,"5":1,"6":0,"7":0,"8":1,"9":0,"10":1,"11":1}},{"19":{"0":0,"1":0,"2":0,"3":1,"4":0,"5":1,"6":0,"7":0,"8":1,"9":1,"10":1,"11":0}},{"20":{"0":0,"1":0,"2":0,"3":1,"4":0,"5":1,"6":1,"7":0,"8":0,"9":0,"10":1,"11":1}},{"21":{"0":0,"1":0,"2":0,"3":1,"4":0,"5":1,"6":1,"7":0,"8":1,"9":1,"10":0,"11":0}},{"22":{"0":0,"1":0,"2":0,"3":1,"4":0,"5":1,"6":1,"7":1,"8":0,"9":0,"10":1,"11":0}},{"23":{"0":0,"1":0,"2":0,"3":1,"4":0,"5":1,"6":1,"7":1,"8":1,"9":0,"10":0,"11":0}},{"24":{"0":0,"1":0,"2":0,"3":1,"4":1,"5":0,"6":0,"7":0,"8":0,"9":1,"10":1,"11":1}},{"25":{"0":0,"1":0,"2":0,"3":1,"4":1,"5":0,"6":0,"7":0,"8":1,"9":1,"10":0,"11":1}},{"26":{"0":0,"1":0,"2":0,"3":1,"4":1,"5":0,"6":0,"7":1,"8":0,"9":0,"10":1,"11":1}},{"27":{"0":0,"1":0,"2":0,"3":1,"4":1,"5":0,"6":0,"7":1,"8":1,"9":1,"10":0,"11":0}},{"28":{"0":0,"1":0,"2":0,"3":1,"4":1,"5":0,"6":1,"7":1,"8":0,"9":0,"10":0,"11":1}},{"29":{"0":0,"1":0,"2":0,"3":1,"4":1,"5":0,"6":1,"7":1,"8":0,"9":1,"10":0,"11":0}},{"30":{"0":0,"1":0,"2":0,"3":1,"4":1,"5":1,"6":0,"7":0,"8":0,"9":1,"10":1,"11":0}},{"31":{"0":0,"1":0,"2":0,"3":1,"4":1,"5":1,"6":0,"7":0,"8":1,"9":0,"10":0,"11":1}},{"32":{"0":0,"1":0,"2":0,"3":1,"4":1,"5":1,"6":0,"7":1,"8":0,"9":0,"10":1,"11":0}},{"33":{"0":0,"1":0,"2":0,"3":1,"4":1,"5":1,"6":0,"7":1,"8":1,"9":0,"10":0,"11":0}},{"34":{"0":0,"1":0,"2":0,"3":1,"4":1,"5":1,"6":1,"7":0,"8":0,"9":0,"10":0,"11":1}},{"35":{"0":0,"1":0,"2":0,"3":1,"4":1,"5":1,"6":1,"7":0,"8":0,"9":1,"10":0,"11":0}},{"36":{"0":0,"1":0,"2":1,"3":0,"4":0,"5":0,"6":0,"7":1,"8":0,"9":1,"10":1,"11":1}},{"37":{"0":0,"1":0,"2":1,"3":0,"4":0,"5":0,"6":0,"7":1,"8":1,"9":1,"10":0,"11":1}},{"38":{"0":0,"1":0,"2":1,"3":0,"4":0,"5":0,"6":1,"7":1,"8":0,"9":1,"10":0,"11":1}},{"39":{"0":0,"1":0,"2":1,"3":0,"4":0,"5":1,"6":0,"7":0,"8":0,"9":1,"10":1,"11":1}},{"40":{"0":0,"1":0,"2":1,"3":0,"4":0,"5":1,"6":0,"7":0,"8":1,"9":1,"10":0,"11":1}},{"41":{"0":0,"1":0,"2":1,"3":0,"4":0,"5":1,"6":0,"7":1,"8":0,"9":0,"10":1,"11":1}},{"42":{"0":0,"1":0,"2":1,"3":0,"4":0,"5":1,"6":0,"7":1,"8":1,"9":1,"10":0,"11":0}},{"43":{"0":0,"1":0,"2":1,"3":0,"4":0,"5":1,"6":1,"7":1,"8":0,"9":0,"10":0,"11":1}},{"44":{"0":0,"1":0,"2":1,"3":0,"4":0,"5":1,"6":1,"7":1,"8":0,"9":1,"10":0,"11":0}},{"45":{"0":0,"1":0,"2":1,"3":0,"4":1,"5":1,"6":0,"7":0,"8":0,"9":1,"10":0,"11":1}},{"46":{"0":0,"1":0,"2":1,"3":0,"4":1,"5":1,"6":0,"7":1,"8":0,"9":0,"10":0,"11":1}},{"47":{"0":0,"1":0,"2":1,"3":0,"4":1,"5":1,"6":0,"7":1,"8":0,"9":1,"10":0,"11":0}},{"48":{"0":0,"1":0,"2":1,"3":1,"4":0,"5":0,"6":0,"7":1,"8":0,"9":1,"10":1,"11":0}},{"49":{"0":0,"1":0,"2":1,"3":1,"4":0,"5":0,"6":0,"7":1,"8":1,"9":0,"10":0,"11":1}},{"50":{"0":0,"1":0,"2":1,"3":1,"4":0,"5":0,"6":1,"7":0,"8":0,"9":1,"10":0,"11":1}},{"51":{"0":0,"1":0,"2":1,"3":1,"4":0,"5":1,"6":0,"7":0,"8":0,"9":1,"10":1,"11":0}},{"52":{"0":0,"1":0,"2":1,"3":1,"4":0,"5":1,"6":0,"7":0,"8":1,"9":0,"10":0,"11":1}},{"53":{"0":0,"1":0,"2":1,"3":1,"4":0,"5":1,"6":0,"7":1,"8":0,"9":0,"10":1,"11":0}},{"54":{"0":0,"1":0,"2":1,"3":1,"4":0,"5":1,"6":0,"7":1,"8":1,"9":0,"10":0,"11":0}},{"55":{"0":0,"1":0,"2":1,"3":1,"4":0,"5":1,"6":1,"7":0,"8":0,"9":0,"10":0,"11":1}},{"56":{"0":0,"1":0,"2":1,"3":1,"4":0,"5":1,"6":1,"7":0,"8":0,"9":1,"10":0,"11":0}},{"57":{"0":0,"1":0,"2":1,"3":1,"4":1,"5":0,"6":0,"7":0,"8":0,"9":1,"10":0,"11":1}},{"58":{"0":0,"1":0,"2":1,"3":1,"4":1,"5":0,"6":0,"7":1,"8":0,"9":0,"10":0,"11":1}},{"59":{"0":0,"1":0,"2":1,"3":1,"4":1,"5":0,"6":0,"7":1,"8":0,"9":1,"10":0,"11":0}},{"60":{"0":0,"1":1,"2":0,"3":0,"4":0,"5":0,"6":0,"7":1,"8":1,"9":0,"10":1,"11":1}},{"61":{"0":0,"1":1,"2":0,"3":0,"4":0,"5":0,"6":0,"7":1,"8":1,"9":1,"10":1,"11":0}},{"62":{"0":0,"1":1,"2":0,"3":0,"4":0,"5":0,"6":1,"7":0,"8":0,"9":1,"10":1,"11":1}},{"63":{"0":0,"1":1,"2":0,"3":0,"4":0,"5":0,"6":1,"7":0,"8":1,"9":1,"10":0,"11":1}},{"64":{"0":0,"1":1,"2":0,"3":0,"4":0,"5":0,"6":1,"7":1,"8":0,"9":1,"10":1,"11":0}},{"65":{"0":0,"1":1,"2":0,"3":0,"4":0,"5":0,"6":1,"7":1,"8":1,"9":0,"10":0,"11":1}},{"66":{"0":0,"1":1,"2":0,"3":0,"4":0,"5":1,"6":0,"7":0,"8":1,"9":0,"10":1,"11":1}},{"67":{"0":0,"1":1,"2":0,"3":0,"4":0,"5":1,"6":0,"7":0,"8":1,"9":1,"10":1,"11":0}},{"68":{"0":0,"1":1,"2":0,"3":0,"4":0,"5":1,"6":1,"7":0,"8":0,"9":0,"10":1,"11":1}},{"69":{"0":0,"1":1,"2":0,"3":0,"4":0,"5":1,"6":1,"7":0,"8":1,"9":1,"10":0,"11":0}},{"70":{"0":0,"1":1,"2":0,"3":0,"4":0,"5":1,"6":1,"7":1,"8":0,"9":0,"10":1,"11":0}},{"71":{"0":0,"1":1,"2":0,"3":0,"4":0,"5":1,"6":1,"7":1,"8":1,"9":0,"10":0,"11":0}},{"72":{"0":0,"1":1,"2":0,"3":0,"4":1,"5":0,"6":0,"7":0,"8":0,"9":1,"10":1,"11":1}},{"73":{"0":0,"1":1,"2":0,"3":0,"4":1,"5":0,"6":0,"7":0,"8":1,"9":1,"10":0,"11":1}},{"74":{"0":0,"1":1,"2":0,"3":0,"4":1,"5":0,"6":0,"7":1,"8":0,"9":0,"10":1,"11":1}},{"75":{"0":0,"1":1,"2":0,"3":0,"4":1,"5":0,"6":0,"7":1,"8":1,"9":1,"10":0,"11":0}},{"76":{"0":0,"1":1,"2":0,"3":0,"4":1,"5":0,"6":1,"7":1,"8":0,"9":0,"10":0,"11":1}},{"77":{"0":0,"1":1,"2":0,"3":0,"4":1,"5":0,"6":1,"7":1,"8":0,"9":1,"10":0,"11":0}},{"78":{"0":0,"1":1,"2":0,"3":0,"4":1,"5":1,"6":0,"7":0,"8":0,"9":1,"10":1,"11":0}},{"79":{"0":0,"1":1,"2":0,"3":0,"4":1,"5":1,"6":0,"7":0,"8":1,"9":0,"10":0,"11":1}},{"80":{"0":0,"1":1,"2":0,"3":0,"4":1,"5":1,"6":0,"7":1,"8":0,"9":0,"10":1,"11":0}},{"81":{"0":0,"1":1,"2":0,"3":0,"4":1,"5":1,"6":0,"7":1,"8":1,"9":0,"10":0,"11":0}},{"82":{"0":0,"1":1,"2":0,"3":0,"4":1,"5":1,"6":1,"7":0,"8":0,"9":0,"10":0,"11":1}},{"83":{"0":0,"1":1,"2":0,"3":0,"4":1,"5":1,"6":1,"7":0,"8":0,"9":1,"10":0,"11":0}},{"84":{"0":0,"1":1,"2":0,"3":1,"4":0,"5":0,"6":0,"7":0,"8":1,"9":0,"10":1,"11":1}},{"85":{"0":0,"1":1,"2":0,"3":1,"4":0,"5":0,"6":0,"7":0,"8":1,"9":1,"10":1,"11":0}},{"86":{"0":0,"1":1,"2":0,"3":1,"4":0,"5":0,"6":1,"7":0,"8":0,"9":0,"10":1,"11":1}},{"87":{"0":0,"1":1,"2":0,"3":1,"4":0,"5":0,"6":1,"7":0,"8":1,"9":1,"10":0,"11":0}},{"88":{"0":0,"1":1,"2":0,"3":1,"4":0,"5":0,"6":1,"7":1,"8":0,"9":0,"10":1,"11":0}},{"89":{"0":0,"1":1,"2":0,"3":1,"4":0,"5":0,"6":1,"7":1,"8":1,"9":0,"10":0,"11":0}},{"90":{"0":0,"1":1,"2":0,"3":1,"4":1,"5":0,"6":0,"7":0,"8":0,"9":0,"10":1,"11":1}},{"91":{"0":0,"1":1,"2":0,"3":1,"4":1,"5":0,"6":0,"7":0,"8":1,"9":1,"10":0,"11":0}},{"92":{"0":0,"1":1,"2":0,"3":1,"4":1,"5":0,"6":1,"7":1,"8":0,"9":0,"10":0,"11":0}},{"93":{"0":0,"1":1,"2":0,"3":1,"4":1,"5":1,"6":0,"7":0,"8":0,"9":0,"10":1,"11":0}},{"94":{"0":0,"1":1,"2":0,"3":1,"4":1,"5":1,"6":0,"7":0,"8":1,"9":0,"10":0,"11":0}},{"95":{"0":0,"1":1,"2":0,"3":1,"4":1,"5":1,"6":1,"7":0,"8":0,"9":0,"10":0,"11":0}},{"96":{"0":0,"1":1,"2":1,"3":0,"4":0,"5":0,"6":0,"7":0,"8":0,"9":1,"10":1,"11":1}},{"97":{"0":0,"1":1,"2":1,"3":0,"4":0,"5":0,"6":0,"7":0,"8":1,"9":1,"10":0,"11":1}},{"98":{"0":0,"1":1,"2":1,"3":0,"4":0,"5":0,"6":0,"7":1,"8":0,"9":0,"10":1,"11":1}},{"99":{"0":0,"1":1,"2":1,"3":0,"4":0,"5":0,"6":0,"7":1,"8":1,"9":1,"10":0,"11":0}},{"100":{"0":0,"1":1,"2":1,"3":0,"4":0,"5":0,"6":1,"7":1,"8":0,"9":0,"10":0,"11":1}},{"101":{"0":0,"1":1,"2":1,"3":0,"4":0,"5":0,"6":1,"7":1,"8":0,"9":1,"10":0,"11":0}},{"102":{"0":0,"1":1,"2":1,"3":0,"4":0,"5":1,"6":0,"7":0,"8":0,"9":0,"10":1,"11":1}},{"103":{"0":0,"1":1,"2":1,"3":0,"4":0,"5":1,"6":0,"7":0,"8":1,"9":1,"10":0,"11":0}},{"104":{"0":0,"1":1,"2":1,"3":0,"4":0,"5":1,"6":1,"7":1,"8":0,"9":0,"10":0,"11":0}},{"105":{"0":0,"1":1,"2":1,"3":0,"4":1,"5":1,"6":0,"7":0,"8":0,"9":0,"10":0,"11":1}},{"106":{"0":0,"1":1,"2":1,"3":0,"4":1,"5":1,"6":0,"7":0,"8":0,"9":1,"10":0,"11":0}},{"107":{"0":0,"1":1,"2":1,"3":0,"4":1,"5":1,"6":0,"7":1,"8":0,"9":0,"10":0,"11":0}},{"108":{"0":0,"1":1,"2":1,"3":1,"4":0,"5":0,"6":0,"7":0,"8":0,"9":1,"10":1,"11":0}},{"109":{"0":0,"1":1,"2":1,"3":1,"4":0,"5":0,"6":0,"7":0,"8":1,"9":0,"10":0,"11":1}},{"110":{"0":0,"1":1,"2":1,"3":1,"4":0,"5":0,"6":0,"7":1,"8":0,"9":0,"10":1,"11":0}},{"111":{"0":0,"1":1,"2":1,"3":1,"4":0,"5":0,"6":0,"7":1,"8":1,"9":0,"10":0,"11":0}},{"112":{"0":0,"1":1,"2":1,"3":1,"4":0,"5":0,"6":1,"7":0,"8":0,"9":0,"10":0,"11":1}},{"113":{"0":0,"1":1,"2":1,"3":1,"4":0,"5":0,"6":1,"7":0,"8":0,"9":1,"10":0,"11":0}},{"114":{"0":0,"1":1,"2":1,"3":1,"4":0,"5":1,"6":0,"7":0,"8":0,"9":0,"10":1,"11":0}},{"115":{"0":0,"1":1,"2":1,"3":1,"4":0,"5":1,"6":0,"7":0,"8":1,"9":0,"10":0,"11":0}},{"116":{"0":0,"1":1,"2":1,"3":1,"4":0,"5":1,"6":1,"7":0,"8":0,"9":0,"10":0,"11":0}},{"117":{"0":0,"1":1,"2":1,"3":1,"4":1,"5":0,"6":0,"7":0,"8":0,"9":0,"10":0,"11":1}},{"118":{"0":0,"1":1,"2":1,"3":1,"4":1,"5":0,"6":0,"7":0,"8":0,"9":1,"10":0,"11":0}},{"119":{"0":0,"1":1,"2":1,"3":1,"4":1,"5":0,"6":0,"7":1,"8":0,"9":0,"10":0,"11":0}},{"120":{"0":1,"1":0,"2":0,"3":0,"4":0,"5":0,"6":0,"7":1,"8":0,"9":1,"10":1,"11":1}},{"121":{"0":1,"1":0,"2":0,"3":0,"4":0,"5":0,"6":0,"7":1,"8":1,"9":1,"10":0,"11":1}},{"122":{"0":1,"1":0,"2":0,"3":0,"4":0,"5":0,"6":1,"7":1,"8":0,"9":1,"10":0,"11":1}},{"123":{"0":1,"1":0,"2":0,"3":0,"4":0,"5":1,"6":0,"7":0,"8":0,"9":1,"10":1,"11":1}},{"124":{"0":1,"1":0,"2":0,"3":0,"4":0,"5":1,"6":0,"7":0,"8":1,"9":1,"10":0,"11":1}},{"125":{"0":1,"1":0,"2":0,"3":0,"4":0,"5":1,"6":0,"7":1,"8":0,"9":0,"10":1,"11":1}},{"126":{"0":1,"1":0,"2":0,"3":0,"4":0,"5":1,"6":0,"7":1,"8":1,"9":1,"10":0,"11":0}},{"127":{"0":1,"1":0,"2":0,"3":0,"4":0,"5":1,"6":1,"7":1,"8":0,"9":0,"10":0,"11":1}},{"128":{"0":1,"1":0,"2":0,"3":0,"4":0,"5":1,"6":1,"7":1,"8":0,"9":1,"10":0,"11":0}},{"129":{"0":1,"1":0,"2":0,"3":0,"4":1,"5":1,"6":0,"7":0,"8":0,"9":1,"10":0,"11":1}},{"130":{"0":1,"1":0,"2":0,"3":0,"4":1,"5":1,"6":0,"7":1,"8":0,"9":0,"10":0,"11":1}},{"131":{"0":1,"1":0,"2":0,"3":0,"4":1,"5":1,"6":0,"7":1,"8":0,"9":1,"10":0,"11":0}},{"132":{"0":1,"1":0,"2":0,"3":1,"4":0,"5":0,"6":0,"7":0,"8":0,"9":1,"10":1,"11":1}},{"133":{"0":1,"1":0,"2":0,"3":1,"4":0,"5":0,"6":0,"7":0,"8":1,"9":1,"10":0,"11":1}},{"134":{"0":1,"1":0,"2":0,"3":1,"4":0,"5":0,"6":0,"7":1,"8":0,"9":0,"10":1,"11":1}},{"135":{"0":1,"1":0,"2":0,"3":1,"4":0,"5":0,"6":0,"7":1,"8":1,"9":1,"10":0,"11":0}},{"136":{"0":1,"1":0,"2":0,"3":1,"4":0,"5":0,"6":1,"7":1,"8":0,"9":0,"10":0,"11":1}},{"137":{"0":1,"1":0,"2":0,"3":1,"4":0,"5":0,"6":1,"7":1,"8":0,"9":1,"10":0,"11":0}},{"138":{"0":1,"1":0,"2":0,"3":1,"4":0,"5":1,"6":0,"7":0,"8":0,"9":0,"10":1,"11":1}},{"139":{"0":1,"1":0,"2":0,"3":1,"4":0,"5":1,"6":0,"7":0,"8":1,"9":1,"10":0,"11":0}},{"140":{"0":1,"1":0,"2":0,"3":1,"4":0,"5":1,"6":1,"7":1,"8":0,"9":0,"10":0,"11":0}},{"141":{"0":1,"1":0,"2":0,"3":1,"4":1,"5":1,"6":0,"7":0,"8":0,"9":0,"10":0,"11":1}},{"142":{"0":1,"1":0,"2":0,"3":1,"4":1,"5":1,"6":0,"7":0,"8":0,"9":1,"10":0,"11":0}},{"143":{"0":1,"1":0,"2":0,"3":1,"4":1,"5":1,"6":0,"7":1,"8":0,"9":0,"10":0,"11":0}},{"144":{"0":1,"1":0,"2":1,"3":1,"4":0,"5":0,"6":0,"7":0,"8":0,"9":1,"10":0,"11":1}},{"145":{"0":1,"1":0,"2":1,"3":1,"4":0,"5":0,"6":0,"7":1,"8":0,"9":0,"10":0,"11":1}},{"146":{"0":1,"1":0,"2":1,"3":1,"4":0,"5":0,"6":0,"7":1,"8":0,"9":1,"10":0,"11":0}},{"147":{"0":1,"1":0,"2":1,"3":1,"4":0,"5":1,"6":0,"7":0,"8":0,"9":0,"10":0,"11":1}},{"148":{"0":1,"1":0,"2":1,"3":1,"4":0,"5":1,"6":0,"7":0,"8":0,"9":1,"10":0,"11":0}},{"149":{"0":1,"1":0,"2":1,"3":1,"4":0,"5":1,"6":0,"7":1,"8":0,"9":0,"10":0,"11":0}},{"150":{"0":1,"1":1,"2":0,"3":0,"4":0,"5":0,"6":0,"7":1,"8":0,"9":1,"10":1,"11":0}},{"151":{"0":1,"1":1,"2":0,"3":0,"4":0,"5":0,"6":0,"7":1,"8":1,"9":0,"10":0,"11":1}},{"152":{"0":1,"1":1,"2":0,"3":0,"4":0,"5":0,"6":1,"7":0,"8":0,"9":1,"10":0,"11":1}},{"153":{"0":1,"1":1,"2":0,"3":0,"4":0,"5":1,"6":0,"7":0,"8":0,"9":1,"10":1,"11":0}},{"154":{"0":1,"1":1,"2":0,"3":0,"4":0,"5":1,"6":0,"7":0,"8":1,"9":0,"10":0,"11":1}},{"155":{"0":1,"1":1,"2":0,"3":0,"4":0,"5":1,"6":0,"7":1,"8":0,"9":0,"10":1,"11":0}},{"156":{"0":1,"1":1,"2":0,"3":0,"4":0,"5":1,"6":0,"7":1,"8":1,"9":0,"10":0,"11":0}},{"157":{"0":1,"1":1,"2":0,"3":0,"4":0,"5":1,"6":1,"7":0,"8":0,"9":0,"10":0,"11":1}},{"158":{"0":1,"1":1,"2":0,"3":0,"4":0,"5":1,"6":1,"7":0,"8":0,"9":1,"10":0,"11":0}},{"159":{"0":1,"1":1,"2":0,"3":0,"4":1,"5":0,"6":0,"7":0,"8":0,"9":1,"10":0,"11":1}},{"160":{"0":1,"1":1,"2":0,"3":0,"4":1,"5":0,"6":0,"7":1,"8":0,"9":0,"10":0,"11":1}},{"161":{"0":1,"1":1,"2":0,"3":0,"4":1,"5":0,"6":0,"7":1,"8":0,"9":1,"10":0,"11":0}},{"162":{"0":1,"1":1,"2":0,"3":1,"4":0,"5":0,"6":0,"7":0,"8":0,"9":1,"10":1,"11":0}},{"163":{"0":1,"1":1,"2":0,"3":1,"4":0,"5":0,"6":0,"7":0,"8":1,"9":0,"10":0,"11":1}},{"164":{"0":1,"1":1,"2":0,"3":1,"4":0,"5":0,"6":0,"7":1,"8":0,"9":0,"10":1,"11":0}},{"165":{"0":1,"1":1,"2":0,"3":1,"4":0,"5":0,"6":0,"7":1,"8":1,"9":0,"10":0,"11":0}},{"166":{"0":1,"1":1,"2":0,"3":1,"4":0,"5":0,"6":1,"7":0,"8":0,"9":0,"10":0,"11":1}},{"167":{"0":1,"1":1,"2":0,"3":1,"4":0,"5":0,"6":1,"7":0,"8":0,"9":1,"10":0,"11":0}},{"168":{"0":1,"1":1,"2":0,"3":1,"4":0,"5":1,"6":0,"7":0,"8":0,"9":0,"10":1,"11":0}},{"169":{"0":1,"1":1,"2":0,"3":1,"4":0,"5":1,"6":0,"7":0,"8":1,"9":0,"10":0,"11":0}},{"170":{"0":1,"1":1,"2":0,"3":1,"4":0,"5":1,"6":1,"7":0,"8":0,"9":0,"10":0,"11":0}},{"171":{"0":1,"1":1,"2":0,"3":1,"4":1,"5":0,"6":0,"7":0,"8":0,"9":0,"10":0,"11":1}},{"172":{"0":1,"1":1,"2":0,"3":1,"4":1,"5":0,"6":0,"7":0,"8":0,"9":1,"10":0,"11":0}},{"173":{"0":1,"1":1,"2":0,"3":1,"4":1,"5":0,"6":0,"7":1,"8":0,"9":0,"10":0,"11":0}},{"174":{"0":1,"1":1,"2":1,"3":0,"4":0,"5":0,"6":0,"7":0,"8":0,"9":1,"10":0,"11":1}},{"175":{"0":1,"1":1,"2":1,"3":0,"4":0,"5":0,"6":0,"7":1,"8":0,"9":0,"10":0,"11":1}},{"176":{"0":1,"1":1,"2":1,"3":0,"4":0,"5":0,"6":0,"7":1,"8":0,"9":1,"10":0,"11":0}},{"177":{"0":1,"1":1,"2":1,"3":0,"4":0,"5":1,"6":0,"7":0,"8":0,"9":0,"10":0,"11":1}},{"178":{"0":1,"1":1,"2":1,"3":0,"4":0,"5":1,"6":0,"7":0,"8":0,"9":1,"10":0,"11":0}},{"179":{"0":1,"1":1,"2":1,"3":0,"4":0,"5":1,"6":0,"7":1,"8":0,"9":0,"10":0,"11":0}}]')
        self.cadenaTypes = json.loads('{"normal":{"rock":0.5,"ghost":0,"steel":0.5},"fire":{"fire":0.5,"water":0.5,"grass":2,"ice":2,"bug":2,"rock":0.5,"dragon":0.5,"steel":2},"water":{"fire":2,"water":0.5,"grass":0.5,"ground":2,"rock":2,"dragon":0.5},"grass":{"fire":0.5,"water":2,"grass":0.5,"poison":0.5,"ground":2,"flying":0.5,"bug":0.5,"rock":2,"dragon":0.5,"steel":0.5},"electric":{"water":2,"grass":0.5,"electric":0.5,"ground":0,"flying":2,"dragon":0.5},"ice":{"fire":0.5,"water":0.5,"grass":2,"ice":0.5,"ground":2,"flying":2,"dragon":2,"steel":0.5},"fighting":{"normal":2,"ice":2,"poison":0.5,"flying":0.5,"psychic":0.5,"bug":0.5,"rock":2,"ghost":0,"dark":2,"steel":2,"fairy":0.5},"poison":{"grass":2,"poison":0.5,"ground":0.5,"rock":0.5,"ghost":0.5,"steel":0,"fairy":2},"ground":{"fire":2,"grass":0.5,"electric":2,"poison":2,"flying":0,"bug":0.5,"rock":2,"steel":2},"flying":{"grass":2,"electric":0.5,"fighting":2,"bug":2,"rock":0.5,"steel":0.5},"psychic":{"fighting":2,"poison":2,"psychic":0.5,"dark":0,"steel":0.5},"bug":{"fire":0.5,"grass":2,"fighting":0.5,"poison":0.5,"flying":0.5,"Psychic":2,"ghost":0.5,"dark":2,"steel":0.5,"fairy":0.5},"rock":{"fire":2,"ice":2,"fighting":0.5,"ground":0.5,"flying":2,"bug":2,"steel":0.5},"ghost":{"normal":0,"psychic":2,"ghost":2,"dark":0.5},"dragon":{"dragon":2,"steel":0.5,"fairy":0},"dark":{"fighting":0.5,"psychic":2,"ghost":2,"dark":0.5,"fairy":0.5},"steel":{"fire":0.5,"water":0.5,"electric":0.5,"ice":2,"rock":2,"steel":0.5,"fairy":2},"fairy":{"fire":0.5,"fighting":2,"poison":0.5,"dragon":2,"dark":2,"steel":0.5}}')
        self.moveTypes = json.loads('{"incineroar":{"normal":0.99948,"fire":0.99477,"dark":0.38806,"bug":0.01792},"zacian-crowned":{"steel":0.99985,"fighting":0.75735,"fairy":0.53830,"normal":0.28244},"kyogre":{"water":0.98305,"ice":0.87910,"electric":0.57364, "normal":0.01600},"thundurus-incarnate":{"electric":0.84966,"flying":0.84930,"fighting":0.53147,"dark":0.15029,"steel":0.08253,"bug":0.01515},"grimmsnarl":{"fairy":0.50748,"dark":0.40074,"normal":0.08012},"groudon":{"ground":0.99691,"ghost":0.59048,"fire":0.54518,"rock":0.50500,"steel":0.12149},"charizard":{"flying":0.96079,"fire":0.94681,"rock":0.58226,"grass":0.15463},"whimsicott":{"fairy":0.60003,"grass":0.32054},"calyrex-shadow":{"ghost":0.99999,"ground":0.53899,"psychic":0.53577,"dark":0.17781,"grass":0.10682},"gastrodon":{"ground":0.99529,"ice":0.89135},"regieleki":{"electric":0.89196,"flying":0.45273,"normal":0.25126,"dark":0.12138},"rillaboom":{"grass":0.99917,"normal":0.96769,"ground":0.76241,"bug":0.44455,"dark":0.20077},"amoonguss":{"bug":0.68530,"grass":0.14045,"dark":0.12953,"ghost":0.04077},"landorus-therian":{"ground":0.95870,"flying":0.94409,"rock":0.83253,"bug":0.09015,"grass":0.07710},"venusaur":{"normal":0.77553,"grass":0.66611,"ground":0.56556,"poison":0.31885},"porygon2":{"dark":0.89516,"ice":0.06786,"ghost":0.03614},"calyrex-ice":{"ice":0.99975,"ground":0.99178,"grass":0.01267},"tornadus-incarnate":{"ice":0.92301,"flying":0.83549,"normal":0.11633},"kartana":{"grass":0.99955,"steel":0.99597,"flying":0.96584,"fighting":0.93301},"yveltal":{"dark":0.76655,"flying":0.71501,"fire":0.15597},"palkia":{"dragon":0.91704,"water":0.82666,"ground":0.78957,"electric":0.23010,"fire":0.17872},"lunala":{"ghost":0.99963,"rock":0.97268},"indeedee-female":{"psychic":0.76492,"fire":0.13001,"ghost":0.06076},"shedinja":{"ghost":0.98947},"tapu-fini":{"fairy":0.86173,"water":0.29690,"ghost":0.13964,"ice":0.09851},"eternatus":{"dragon":0.57631,"fire":0.57257,"poison":0.48843,"steel":0.02353},"solgaleo":{"steel":0.99950,"ground":0.97452,"electric":0.63119,"rock":0.52669,"psychic":0.23064,"fire":0.01563},"umbreon":{"dark":0.99986},"zapdos":{"flying":0.99898,"electric":0.60315,"fire":0.35606,"normal":0.13465},"urshifu-single-strike":{"dark":0.99964,"fighting":0.99805,"bug":0.13374,"rock":0.06665},"blastoise":{"water":0.90052,"normal":0.62051,"ice":0.44852},"bronzong":{"fighting":0.98581,"steel":0.95946},"coalossal":{"rock":0.99350,"fire":0.97287,"grass":0.96308},"moltress-galar":{"dark":0.55628,"flying":0.52932},"urshifu-rapid-strike":{"water":0.99952,"fighting":0.98925},"slowbro":{"fighting":0.99947},"dialga":{"ground":0.90317,"dragon":0.80304,"steel":0.50633,"ice":0.46159,"electric":0.09783,"rock":0.08115,"fighting":0.02609},"golisopod":{"dark":0.99984,"bug":0.96867,"water":0.88996},"volcarona":{"bug":0.97433,"fire":0.91492},"kyurem-white":{"ground":0.97957,"dragon":0.90674,"ice":0.77572},"xerneas":{"fairy":0.89169,"normal":0.09098,"electric":0.06522},"mimikyu":{"ghost":0.96468,"fairy":0.45538},"weezing":{"poison":0.93348},"clefairy":{"fairy":0.10725,"ice":0.09159,"fire":0.05082},"ditto":{},"sableye":{"dark":0.62486,"normal":0.53571,"ghost":0.24517},"regigigas":{"normal":0.97398,"ground":0.93212,"dark":0.77335,"ice":0.43212,"electric":0.06825},"celesteela":{"fire":0.90462,"rock":0.16451,"steel":0.15958,"flying":0.08736},"gothitelle":{"normal":0.99349,"dark":0.07034},"ho-oh":{"flying":0.93616,"fire":0.92701,"ground":0.20401},"entei":{"fire":0.96213,"rock":0.74683,"ground":0.43392,"dark":0.25977,"steel":0.12998},"torkoal":{"fire":0.61526,"fighting":0.37560,"grass":0.32326,"ground":0.08439},"ferrothorn":{"fighting":0.84104,"grass":0.41931,"steel":0.27793,"dark":0.15127},"reshiram":{"dragon":0.98735,"ground":0.97316,"fire":0.93143,"grass":0.21074},"hydreigon":{"dark":0.72964,"rock":0.31429,"dragon":0.31421,"ground":0.25124,"water":0.14286,"flying":0.13643},"zekrom":{"electric":0.84636,"flying":0.80690,"dragon":0.56911,"ground":0.13966,"fighting":0.02913,"dark":0.02913},"snorlax":{"dark":0.74537,"normal":0.46931,"ground":0.38384,"steel":0.11932,"fighting":0.11409,"dark":0.06813},"raichu":{"normal":0.95996,"electric":0.59259,"grass":0.07859},"dragapult":{"dragon":0.55360,"water":0.50625,"ghost":0.49876,"flying":0.33810,"dark":0.09101,"steel":0.06246},"suicune":{"water":0.99299,"dark":0.97535,"ground":0.82199},"zygarde-50":{"ground":0.96458,"steel":0.33615,"dragon":0.15979,"rock":0.08899,"fighting":0.06994},"landorus-incarnate":{"ground":0.97678,"flying":0.83237,"rock":0.77128,"poison":0.18759,"grass":0.08481},"liepard":{"normal":0.99675,"dark":0.68821,"ghost":0.10000,"fairy":0.10000},"rayquaza":{"dragon":0.60498,"ground":0.60018,"flying":0.51700,"normal":0.20780,"fire":0.20475,"rock":0.14844,"ice":0.05789,"electric":0.05252,"grass":0.02632},"dusclops":{"ghost":0.64186,"rock":0.11869},"kingdra":{"dragon":0.89370,"flying":0.88471,"water":0.61108,"ice":0.22185},"milotic":{"water":0.85046,"ice":0.15513},"lugia":{"ground":0.92190,"flying":0.91851,"water":0.20569,"ghost":0.19479},"rotom-heat":{"fire":0.99064,"electric":0.93097,"dark":0.07701},"pikachu":{"normal":0.96999,"electric":0.86026,"ground":0.79406,"dark":0.78705,"water":0.12046,"grass":0.10953},"alcremie":{"fairy":0.61558,"fire":0.19801},"comfey":{"fairy":0.85647,"grass":0.19090,"bug":0.10802},"lapras":{"water":0.91695,"ice":0.86277,"electric":0.11329},"magearna":{"steel":0.15337,"fairy":0.11196,"electric":0.02301,"fighting":0.01209},"zeraora":{"normal":0.35450,"electric":0.22575,"dark":0.02469,"fighting":0.02293},"victini":{"electric":0.06417,"fighting":0.05615,"bug":0.05348,"grass":0.02168,"fire":0.01012},"melmetal":{"steel":0.24153,"ground":0.04661,"electric":0.01695,"fighting":0.01546,"rock":0.01031},"volcanion":{"water":0.15479,"fire":0.03870,"grass":0.01018},"thundurus":{"electric":0.84966,"flying":0.84930,"fighting":0.53147,"dark":0.15029},"mew":{"dark":0.15823,"bug":0.06329,"steel":0.04430,"ground":0.04430,"ghost":0.02532,"flying":0.02532,"rock":0.02532,"ice":0.01899,"fire":0.01899,"fighting":0.01266,"electric":0.01266,"psychic":0.01266},"marshadow":{"ghost":0.34177,"fighting":0.13924,"rock":0.02532},"tornadus":{"ice":0.92301,"flying":0.83549},"zamazenta-crowned":{"steel":0.99552,"dark":0.59504,"fighting":0.24559},"necrozmamane":{"steel":0.91706,"ground":0.56901,"psychic":0.23231,"rock":0.14656,"ghost":0.05518,"fighting":0.04793},"kyurem":{"dragon":0.15942,"ice":0.11594,"ground":0.07246},"zarude":{"grass":0.20968,"dark":0.12903,"rock":0.09677,"ground":0.03226,"bug":0.03226},"kyurem-black":{"electric":0.99720,"ice":0.57461,"dragon":0.47653,"ground":0.32532},"fletchinder":{"fire":0.42000},"wingull":{"ice":0.18000,"bug":0.12000,"water":0.12000,"normal":0.06000},"zarude-dada":{"grass":0.20408,"rock":0.10204,"fighting":0.08163,"dark":0.04082},"keldeo-resolute":{"fighting":0.18182,"water":0.18182,"flying":0.09091},"cinderace":{"fire":0.99042,"dark":0.80190,"fighting":0.69290,"flying":0.34984,"poison":0.12923,"steel":0.06811},"zygarde-complete":{"ground":0.72093,"rock":0.11628,"dragon":0.06977},"gitatina-origin":{"ghost":0.94829,"dragon":0.47158,"flying":0.26240},"jirachi":{"psychic":0.04878},"zygarde":{"ground":0.96458,"steel":0.33615,"dragon":0.15979,"rock":0.08899,"fighting":0.06994},"oranguru":{"dark":0.52592,"psychic":0.18964},"alcremie":{"fairy":0.61558,"fire":0.19801},"mimikyu-busted":{"ghost":0.47619,"fairy":0.14286},"diance":{"rock":0.19048,"steel":0.14286},"weezing-galar":{"fairy":0.37500,"ghost":0.18750},"silvally":{"dark":0.25000}}')

    # Only Opponent
    async def initialize_team_preview(self, user_json, opponent_pokemon):
        # Insert Pkmn User
        self.user.updatePkmnUser(user_json, first_turn=True)
        # The battle has started
        self.started = True
        # The rqid is obtained
        self.rqid = user_json[constants.RQID]
        # Insert Pkmn Opponent
        pokeapiPkmn = {'zacian': 'zacian-crowned', 'zacian-*': 'zacian-crowned', 'thundurus': 'thundurus-incarnate',
                        'landorus': 'landorus-incarnate', 'tornadus': 'tornadus-incarnate','indeedee-f': 'indeedee-female',
                        'urshifu': 'urshifu-single-strike', 'necrozma-dusk-mane': 'necrozma-dusk', 'zamazenta': 'zamazenta-crowned', 'zamazenta-*': 'zamazenta-crowned'}
        for pkmn_string in opponent_pokemon:
            # opponent_pokemon = ["Lapras, L50, M","Kyogre, L50","Zacian-*, L50","Tapu Fini, L50","Zapdos, L50","Mimikyu, L50, F"]
            # pkmn_string = "Lapras, L50, M"
            # Add Types Moves
            name = pkmn_string.partition(",")[0]
            if name.lower() in pokeapiPkmn:
                namePkmn = pokeapiPkmn[name.lower()]
            else:
                namePkmn = name
            moveDict = self.moveTypes[namePkmn.lower()]
            self.opponent.reserved.append(Pokemon(name, -1, True, None, None, None, None, moveDict))

    @abstractmethod
    async def action_team_preview(self, battle):
        ...

    @abstractmethod
    async def action_battle(self, battle):
        ...        

    def imprimir(self):
        original_stdout = sys.stdout
        with open('reporte.txt', 'a') as f:
            sys.stdout = f
            print()
            print("Estado del Entorno")
            print("------------------")
            print("Turn: " + str(self.turn))
            print("Started: " + str(self.started))
            print("Weather: " + str(self.weather))
            print("Weather Turn: " + str(self.weatherTurn))
            print("Terrain: " + str(self.terrain))
            print("Terrain Turn: " + str(self.terrainTurn))
            print("Trick Room: " + str(self.trick_room))
            print("rqid: " + str(self.rqid))
            print("Force Switch: " + str(self.force_switch))
            print("Wait: " + str(self.wait))
            print()
            print("USUARIO: ")
            self.user.imprimir()
            print()
            print("OPONENTE: ")
            self.opponent.imprimir()
            sys.stdout = original_stdout

    # Calculate the reward for a specific turn
    def calculateRewards(self):
        # Only user
        user = self.formula(self.user.active1)*1.1 + self.formula(self.user.active2) + 0.9*(self.formula(self.user.reserved[0]) + self.formula(self.user.reserved[1]))
        # Only opponent
        if self.opponent.opponentPkmnConfirmed == 2:
            opponent = self.formula(self.opponent.active1)*1.1 + self.formula(self.opponent.active2) + 0.9*(self.formula(self.opponent.reserved[0]) + self.formula(self.opponent.reserved[1]) + self.formula(self.opponent.reserved[2]) + self.formula(self.opponent.reserved[3]))/2
        elif self.opponent.opponentPkmnConfirmed == 4:
            opponent = self.formula(self.opponent.active1)*1.1 + self.formula(self.opponent.active2) + 0.9*(self.formula(self.opponent.reserved[0]) + self.formula(self.opponent.reserved[1]))
        else:   # == 3
            if self.opponent.reserved[0].opponent_confirmed:
                opponent = self.formula(self.opponent.active1)*1.1 + self.formula(self.opponent.active2) + 0.9*(self.formula(self.opponent.reserved[0])) + 0.9*(self.formula(self.opponent.reserved[1]) + self.formula(self.opponent.reserved[2]) + self.formula(self.opponent.reserved[3]))/3
            elif self.opponent.reserved[1].opponent_confirmed:
                opponent = self.formula(self.opponent.active1)*1.1 + self.formula(self.opponent.active2) + 0.9*(self.formula(self.opponent.reserved[1])) + 0.9*(self.formula(self.opponent.reserved[0]) + self.formula(self.opponent.reserved[2]) + self.formula(self.opponent.reserved[3]))/3
            elif self.opponent.reserved[2].opponent_confirmed:
                opponent = self.formula(self.opponent.active1)*1.1 + self.formula(self.opponent.active2) + 0.9*(self.formula(self.opponent.reserved[2])) + 0.9*(self.formula(self.opponent.reserved[0]) + self.formula(self.opponent.reserved[1]) + self.formula(self.opponent.reserved[3]))/3
            elif self.opponent.reserved[3].opponent_confirmed:
                opponent = self.formula(self.opponent.active1)*1.1 + self.formula(self.opponent.active2) + 0.9*(self.formula(self.opponent.reserved[3])) + 0.9*(self.formula(self.opponent.reserved[0]) + self.formula(self.opponent.reserved[1]) + self.formula(self.opponent.reserved[2]))/3
            else:
                logger.error("def calculateRewards(self):")
                opponent = 1000000
        # On the first turn, the lastReward is 0
        if self.turn > 1:
            self.lastReward = (user - opponent) - self.reward
        else:
            self.lastReward = 0
        # Heart
        self.reward = user - opponent
        return self.lastReward, self.reward

    # Calculate the reward for the user or for the opponent of 1 Pkmn
    def formula(self, pkmn):
        # General Formula - Only 6 variables
        return sqrt((pkmn.ATB*pkmn.ATBoost*pkmn.VEB)/10)*sqrt((pkmn.HPB*pkmn.DEBoost*pkmn.DEB)/10)

class Battler:
    # Constructor
    def __init__(self):
        self.name = None                                # String
        self.account_name = None                        # String
        self.active1 = None                             # class Pokemon or None
        self.active2 = None                             # class Pokemon or None
        self.reserved = []                              # class Pokemon
        self.side_conditions = defaultdict(lambda: 0)   # Tailwind, Stealth Rock, Reflect
        self.isUser = False
        self.opponentPkmnConfirmed = 0

    def updatePkmnUser(self, JSONdict, first_turn):
        if (first_turn):
            # Update data
            self.id = JSONdict["side"]["id"]
            self.account_name = JSONdict["side"]["name"]
            self.isUser = True
            # Insert Pkmn
            self.reserved.append(Pokemon(JSONdict["side"]["pokemon"][0]["details"].partition(",")[0], JSONdict["side"]["pokemon"][0]["condition"].partition("/")[2],False, JSONdict["side"]["pokemon"][0]["moves"][0], JSONdict["side"]["pokemon"][0]["moves"][1], JSONdict["side"]["pokemon"][0]["moves"][2], JSONdict["side"]["pokemon"][0]["moves"][3], None))
            self.reserved.append(Pokemon(JSONdict["side"]["pokemon"][1]["details"].partition(",")[0], JSONdict["side"]["pokemon"][1]["condition"].partition("/")[2],False, JSONdict["side"]["pokemon"][1]["moves"][0], JSONdict["side"]["pokemon"][1]["moves"][1], JSONdict["side"]["pokemon"][1]["moves"][2], JSONdict["side"]["pokemon"][1]["moves"][3], None))
            self.reserved.append(Pokemon(JSONdict["side"]["pokemon"][2]["details"].partition(",")[0], JSONdict["side"]["pokemon"][2]["condition"].partition("/")[2],False, JSONdict["side"]["pokemon"][2]["moves"][0], JSONdict["side"]["pokemon"][2]["moves"][1], JSONdict["side"]["pokemon"][2]["moves"][2], JSONdict["side"]["pokemon"][2]["moves"][3], None))
            self.reserved.append(Pokemon(JSONdict["side"]["pokemon"][3]["details"].partition(",")[0], JSONdict["side"]["pokemon"][3]["condition"].partition("/")[2],False, JSONdict["side"]["pokemon"][3]["moves"][0], JSONdict["side"]["pokemon"][3]["moves"][1], JSONdict["side"]["pokemon"][3]["moves"][2], JSONdict["side"]["pokemon"][3]["moves"][3], None))
            self.reserved.append(Pokemon(JSONdict["side"]["pokemon"][4]["details"].partition(",")[0], JSONdict["side"]["pokemon"][4]["condition"].partition("/")[2],False, JSONdict["side"]["pokemon"][4]["moves"][0], JSONdict["side"]["pokemon"][4]["moves"][1], JSONdict["side"]["pokemon"][4]["moves"][2], JSONdict["side"]["pokemon"][4]["moves"][3], None))
            self.reserved.append(Pokemon(JSONdict["side"]["pokemon"][5]["details"].partition(",")[0], JSONdict["side"]["pokemon"][5]["condition"].partition("/")[2],False, JSONdict["side"]["pokemon"][5]["moves"][0], JSONdict["side"]["pokemon"][5]["moves"][1], JSONdict["side"]["pokemon"][5]["moves"][2], JSONdict["side"]["pokemon"][5]["moves"][3], None))
            # Update Pkmn
            for x in range(len(self.reserved)):
                self.reserved[x].HPB_battle_base = int(JSONdict["side"]["pokemon"][x]["condition"].partition("/")[0].partition(" ")[0])
                self.reserved[x].ATK_battle_base = int(JSONdict["side"]["pokemon"][x]["stats"]["atk"])
                self.reserved[x].DEF_battle_base = int(JSONdict["side"]["pokemon"][x]["stats"]["def"])
                self.reserved[x].SPA_battle_base = int(JSONdict["side"]["pokemon"][x]["stats"]["spa"])
                self.reserved[x].SPD_battle_base = int(JSONdict["side"]["pokemon"][x]["stats"]["spd"])
                self.reserved[x].VEL_battle_base = int(JSONdict["side"]["pokemon"][x]["stats"]["spe"])
                self.reserved[x].HPB = int(JSONdict["side"]["pokemon"][x]["condition"].partition("/")[0].partition(" ")[0])
                self.reserved[x].ATB = max(int(JSONdict["side"]["pokemon"][x]["stats"]["atk"]),int(JSONdict["side"]["pokemon"][x]["stats"]["spa"]))
                self.reserved[x].DEB = int(math.floor((int(JSONdict["side"]["pokemon"][x]["stats"]["def"])+int(JSONdict["side"]["pokemon"][x]["stats"]["spd"]))/2))
                self.reserved[x].VEB = int(JSONdict["side"]["pokemon"][x]["stats"]["spe"])
        else:
            if len(self.reserved) == 4:
                for pkmn in self.reserved:
                    if pkmn.name != JSONdict["side"]["pokemon"][2]["details"].partition(",")[0] and pkmn.name != JSONdict["side"]["pokemon"][3]["details"].partition(",")[0]:
                        self.reserved.remove(pkmn)
            # Update Pkmn - Turno 2 en adelante... HPB
            self.active1.HPB = int(JSONdict["side"]["pokemon"][0]["condition"].partition("/")[0].partition(" ")[0])
            self.active2.HPB = int(JSONdict["side"]["pokemon"][1]["condition"].partition("/")[0].partition(" ")[0])
            # Update Pkmn - Turno 2 en adelante... Moves
            if JSONdict.get("active","False") != "False":
                for i in range(len(JSONdict.get("active","False")[0].get("moves","False"))):
                    self.active1.movePP[i] = int(JSONdict.get("active","False")[0].get("moves","False")[i].get("pp","False"))
                for i in range(len(JSONdict.get("active","False")[1].get("moves","False"))):
                    self.active2.movePP[i] = int(JSONdict.get("active","False")[1].get("moves","False")[i].get("pp","False"))
            # Reserved
            self.reserved[0].HPB = int(JSONdict["side"]["pokemon"][2]["condition"].partition("/")[0].partition(" ")[0])
            self.reserved[1].HPB = int(JSONdict["side"]["pokemon"][3]["condition"].partition("/")[0].partition(" ")[0])

    def imprimir(self):
        print()
        print("id: " + str(self.id))
        print("Nombre de la cuenta: " + str(self.account_name))
        print()
        if self.isUser == False:
            print("opponentPkmnConfirmed: " + str(self.opponentPkmnConfirmed))
        print()
        print("Side conditions: ")
        for key, value in self.side_conditions.items():
            print(key, ' : ', value)
        print()
        if self.active1 is not None:
            print("Active 1: ")
            print()
            self.active1.imprimir()
            print()
        else:
            print("Active 1: " + str(None))
            print()
        if self.active2 is not None:
            print("Active 2: ")
            print()
            self.active2.imprimir()
            print()
        else:
            print("Active 2: " + str(None))
            print()
        print("List of Reserved: ")
        print()
        for pkmn in self.reserved:
            pkmn.imprimir()
            print()

class Pokemon:
    # Constructor
    def __init__(self, name: str, max_hp: int, isOpponent: bool, move1: str, move2: str, move3: str, move4: str,  moveDict: dict):
        self.name = name
        self.max_hp = max_hp
        self.type1 = None
        self.type2 = None
        self.status = None                      # ['slp', 'brn', 'frz', 'par', 'psn']
        self.volatile_statuses = []             # dynamax, moveyawn
        self.boosts = defaultdict(lambda: 0)
        if name.startswith('Zacian') or name.startswith('Zamazenta') or name.startswith('Eternatus'):
            self.can_dynamax = False
        else:
            self.can_dynamax = True
        self.isDynamax = False
        self.opponent_confirmed = False
        self.HPB_battle_base = 0
        self.HPB = 0
        self.ATK_battle_base = 0
        self.SPA_battle_base = 0
        self.ATB = 0
        self.ATBoost = 1
        self.SPD_battle_base = 0
        self.DEF_battle_base = 0
        self.DEB = 0
        self.DEBoost = 1
        self.VEL_battle_base = 0
        self.VEB = 0
        self.FRO = 0
        self.ASL = 0
        self.PAR = 0
        self.DRO = 0
        if name == 'Ditto':
            self.isDitto = 1
        else:
            self.isDitto = 0
        self.movePP = [-1, -1, -1, -1]
        self.moveType = defaultdict(lambda: 0)
        if move1 is not None:
            self.moveType[self.findMoveType(move1)] = 1
            self.moveType[self.findMoveType(move2)] = 1
            self.moveType[self.findMoveType(move3)] = 1
            self.moveType[self.findMoveType(move4)] = 1
        else:
            self.moveType = moveDict
        self.POKEAPI(isOpponent)

    def isFainted(self):
        return self.HPB == 0

    def findMoveType(self, move):
        url = 'https://play.pokemonshowdown.com/data/moves.json'
        response = requests.get(url)
        if response.status_code == 200:
            payload = response.json()
            return payload[move]['type'].lower()
        else:
            return None

    def imprimir(self):
        print("Name: " + str(self.name))
        print("Type 1: " + str(self.type1))
        print("Type 2: " + str(self.type2))
        print("Max HP: " + str(self.max_hp))
        print("Status: " + str(self.status))
        print("Volatile Status: " + str(self.volatile_statuses))
        print("Boosts: ")
        for key, value in self.boosts.items():
            print(key, ' : ', value)
        print("Can Dynamax: " + str(self.can_dynamax))
        print("Is Dynamax: " + str(self.isDynamax))
        print("Is Ditto: " + str(self.isDitto))
        print("isOpponentConfirmed: " + str(self.opponent_confirmed))
        print("Is Fainted?: {}".format(self.isFainted()))
        print("---STATS---")
        print("HPB: " + str(self.HPB))
        print("ATB: " + str(self.ATB))
        print("DEB: " + str(self.DEB))
        print("VEB: " + str(self.VEB))
        print("---HEALTH---")
        print("FRO: " + str(self.FRO))
        print("ASL: " + str(self.ASL))
        print("PAR: " + str(self.PAR))
        print("DRO: " + str(self.DRO))
        print("---MOVES---")
        print(self.movePP)
        for key, value in self.moveType.items():
            print(key, ' : ', value)

    def HP_BATTLE_CALCULATE(self, HP_base):
        return math.floor(((2*HP_base+constants.IV+(constants.EV/4))*constants.LEVEL)/100) + constants.LEVEL + 10

    def OTHER_STAT_CALCULATE(self, STAT_base, modifier, defense):
        if modifier and defense:
            return math.floor((((2*STAT_base+constants.IV+(0/4))*constants.LEVEL)/100 + 5) * 1)
        else:
            return math.floor((((2*STAT_base+constants.IV+((constants.EV)/4))*constants.LEVEL)/100 + 5) * constants.NATURE)

    def POKEAPI(self, isOpponent):
        pokeapiPkmn = {'zacian': 'zacian-crowned', 'zacian-*': 'zacian-crowned', 'thundurus': 'thundurus-incarnate',
                        'landorus': 'landorus-incarnate', 'tornadus': 'tornadus-incarnate','indeedee-f': 'indeedee-female',
                        'urshifu': 'urshifu-single-strike', 'necrozma-dusk-mane': 'necrozma-dusk', 'zamazenta': 'zamazenta-crowned', 'zamazenta-*': 'zamazenta-crowned'}
        
        if self.name.lower() in pokeapiPkmn:
            namePkmn = pokeapiPkmn[self.name.lower()]
        else:
            namePkmn = self.name
        url = 'https://pokeapi.co/api/v2/pokemon/' + namePkmn.lower().replace(' ', '-')
        response = requests.get(url)
        if response.status_code == 200:
            payload = response.json()
            self.type1 = payload.get('types','Null')[0].get('type','Null').get('name','Null')
            if len(payload.get('types','Null')) == 1:
                self.type2 = 'Null'
            else:
                self.type2 = payload.get('types','Null')[1].get('type','Null').get('name','Null')
            if isOpponent:
                # Stats Base
                HP_base = payload['stats'][0]['base_stat']
                ATK_base = payload['stats'][1]['base_stat']
                DEF_base = payload['stats'][2]['base_stat']
                SPA_base = payload['stats'][3]['base_stat']
                SPD_base = payload['stats'][4]['base_stat']
                VEL_base = payload['stats'][5]['base_stat']
                # Stats Battle
                self.HPB_battle_base = self.HP_BATTLE_CALCULATE(HP_base)
                self.max_hp = self.HPB_battle_base
                self.ATK_battle_base = self.OTHER_STAT_CALCULATE(ATK_base,0,0)
                self.SPA_battle_base = self.OTHER_STAT_CALCULATE(SPA_base,0,0)
                if DEF_base > SPD_base:
                    self.SPD_battle_base = self.OTHER_STAT_CALCULATE(SPD_base,0,1)
                    self.DEF_battle_base = self.OTHER_STAT_CALCULATE(DEF_base,1,1)
                else:
                    self.DEF_battle_base = self.OTHER_STAT_CALCULATE(DEF_base,0,1)
                    self.SPD_battle_base = self.OTHER_STAT_CALCULATE(SPD_base,1,1)
                self.VEL_battle_base = self.OTHER_STAT_CALCULATE(VEL_base,0,0)
                # Cálculo del Reward
                self.HPB = self.HPB_battle_base
                self.ATB = max(int(self.ATK_battle_base),int(self.SPA_battle_base))
                self.DEB = int(math.floor((int(self.DEF_battle_base)+int(self.SPD_battle_base))/2))
                self.VEB = int(self.VEL_battle_base)