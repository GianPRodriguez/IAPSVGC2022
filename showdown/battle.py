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

    async def initialize_team_preview(self, user_json, opponent_pokemon):
        # Insert Pkmn User
        self.user.updatePkmnUser(user_json, first_turn=True)
        # The battle has started
        self.started = True
        # The rqid is obtained
        self.rqid = user_json[constants.RQID]
        # Insert Pkmn Opponent
        for pkmn_string in opponent_pokemon:
            # opponent_pokemon = ["Lapras, L50, M","Kyogre, L50","Zacian-*, L50","Tapu Fini, L50","Zapdos, L50","Mimikyu, L50, F"]
            # pkmn_string = "Lapras, L50, M"
            self.opponent.reserved.append(Pokemon(pkmn_string.partition(",")[0], -1, True))

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
            print("Weather: " + str(self.weather))
            print("Weather Turn: " + str(self.weatherTurn))
            print("Terrain: " + str(self.terrain))
            print("Terrain Turn: " + str(self.terrainTurn))
            print("Trick Room: " + str(self.trick_room))
            print("Turn: " + str(self.turn))
            print("Started: " + str(self.started))
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

    def calculateRewards(self):
        user = self.formula(self.user.active1)*1.1 + self.formula(self.user.active2) + 0.9*(self.formula(self.user.reserved[0]) + self.formula(self.user.reserved[1]))
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
        if self.turn > 1:
            self.lastReward = self.reward
        else:
            self.lastReward = 0
        self.reward = user - opponent
        return self.reward-self.lastReward, self.reward

    def formula(self, pkmn):
        return sqrt((pkmn.ATB*pkmn.VEB)/10)*sqrt((pkmn.HPB*pkmn.DEB)/10)

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
            self.reserved.append(Pokemon(JSONdict["side"]["pokemon"][0]["details"].partition(",")[0], JSONdict["side"]["pokemon"][0]["condition"].partition("/")[2],False))
            self.reserved.append(Pokemon(JSONdict["side"]["pokemon"][1]["details"].partition(",")[0], JSONdict["side"]["pokemon"][1]["condition"].partition("/")[2],False))
            self.reserved.append(Pokemon(JSONdict["side"]["pokemon"][2]["details"].partition(",")[0], JSONdict["side"]["pokemon"][2]["condition"].partition("/")[2],False))
            self.reserved.append(Pokemon(JSONdict["side"]["pokemon"][3]["details"].partition(",")[0], JSONdict["side"]["pokemon"][3]["condition"].partition("/")[2],False))
            self.reserved.append(Pokemon(JSONdict["side"]["pokemon"][4]["details"].partition(",")[0], JSONdict["side"]["pokemon"][4]["condition"].partition("/")[2],False))
            self.reserved.append(Pokemon(JSONdict["side"]["pokemon"][5]["details"].partition(",")[0], JSONdict["side"]["pokemon"][5]["condition"].partition("/")[2],False))
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
    def __init__(self, name: str, max_hp: int, isOpponent: bool):
        self.name = name
        self.max_hp = max_hp
        self.type1 = None
        self.type2 = None
        self.fainted = False
        self.status = None                      # ['slp', 'brn', 'frz', 'par', 'psn']
        self.volatile_statuses = []             # dynamax
        self.boosts = defaultdict(lambda: 0)
        if name.startswith('Zacian') or name.startswith('Zamazenta') or name.startswith('Eternatus'):
            self.can_dynamax = False
        else:
            self.can_dynamax = True
        self.opponent_confirmed = False
        self.isDynamax = False
        self.HPB_battle_base = 0
        self.HPB = 0
        self.ATK_battle_base = 0
        self.SPA_battle_base = 0
        self.ATB = 0
        self.SPD_battle_base = 0
        self.DEF_battle_base = 0
        self.DEB = 0
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
        self.POKEAPI(isOpponent)

    def imprimir(self):
        print("Name: " + str(self.name))
        print("Max HP: " + str(self.max_hp))
        print("Fainted: " + str(self.fainted))
        print("Status: " + str(self.status))
        print("Volatile Status: " + str(self.volatile_statuses))
        print("Boosts: ")
        for key, value in self.boosts.items():
            print(key, ' : ', value)
        print("Can Dynamax: " + str(self.can_dynamax))
        print("Is Ditto: " + str(self.isDitto))
        print("HPB: " + str(self.HPB))
        print("ATB: " + str(self.ATB))
        print("DEB: " + str(self.DEB))
        print("VEB: " + str(self.VEB))
        print("FRO: " + str(self.FRO))
        print("ASL: " + str(self.ASL))
        print("PAR: " + str(self.PAR))
        print("DRO: " + str(self.DRO))
        print("isOpponentConfirmed: " + str(self.opponent_confirmed))
        print(self.movePP)

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
                        'urshifu': 'urshifu-single-strike', 'urshifu-rapid-strike': 'urshifu-single-strike'}
        
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