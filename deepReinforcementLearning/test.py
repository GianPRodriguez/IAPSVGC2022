import os
import sys
import gym
import time
import queue
import random
import numpy as np
import multiprocessing
import tensorflow as tf

from os import path
from threading import Thread
from datetime import datetime
from collections import deque
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
sys.path.append(parent_directory)

from run import AsyncDownload  

listAgentTeamname = ["Eduardo Cunha","Guillermo Castilla","James Baek"]

def msgToArray(moveSwitch1, nMoveSwitch1, dynamax1, toMove1, moveSwitch2, nMoveSwitch2, dynamax2, toMove2):
	#print("msgToArray: {} {} {} {} {} {} {} {}".format(moveSwitch1, nMoveSwitch1, dynamax1, toMove1, moveSwitch2, nMoveSwitch2, dynamax2, toMove2))
	arreglo = []
	if moveSwitch1 == 'move':
		arreglo.append(0)
		arreglo.append(0)
		arreglo.append(1)
	elif moveSwitch1 == 'switch':
		arreglo.append(0)
		arreglo.append(1)
		arreglo.append(0)
	elif moveSwitch1 == 'pass':
		arreglo.append(1)
		arreglo.append(0)
		arreglo.append(0)
	else:
		print("Error 1 - msgToArray")
	if nMoveSwitch1 == '' or nMoveSwitch1 == 0:
		arreglo.append(0)
		arreglo.append(0)
		arreglo.append(0)
		arreglo.append(0)
		arreglo.append(1)
	elif nMoveSwitch1 == '1' or nMoveSwitch1 == 1:
		arreglo.append(0)
		arreglo.append(0)
		arreglo.append(0)
		arreglo.append(1)
		arreglo.append(0)
	elif nMoveSwitch1 == '2' or nMoveSwitch1 == 2:
		arreglo.append(0)
		arreglo.append(0)
		arreglo.append(1)
		arreglo.append(0)
		arreglo.append(0)
	elif nMoveSwitch1 == '3' or nMoveSwitch1 == 3:
		arreglo.append(0)
		arreglo.append(1)
		arreglo.append(0)
		arreglo.append(0)
		arreglo.append(0)
	elif nMoveSwitch1 == '4' or nMoveSwitch1 == 4:
		arreglo.append(1)
		arreglo.append(0)
		arreglo.append(0)
		arreglo.append(0)
		arreglo.append(0)
	else:
		print("Error 2 - msgToArray")
	if dynamax1 == '' or dynamax1 == 0:
		arreglo.append(0)
	elif dynamax1 == 'dynamax' or dynamax1 == 1:
		arreglo.append(1)
	else:
		print("Error 3 - msgToArray")
	if toMove1 == '-2' or toMove1 == -2:
		arreglo.append(0)
		arreglo.append(0)
		arreglo.append(0)
		arreglo.append(1)
	elif toMove1 == '' or toMove1 == 0:
		arreglo.append(0)
		arreglo.append(0)
		arreglo.append(1)
		arreglo.append(0)
	elif toMove1 == '1' or toMove1 == 1:
		arreglo.append(0)
		arreglo.append(1)
		arreglo.append(0)
		arreglo.append(0)
	elif toMove1 == '2' or toMove1 == 2:
		arreglo.append(1)
		arreglo.append(0)
		arreglo.append(0)
		arreglo.append(0)
	else:
		print("Error 4 - msgToArray")
	if moveSwitch2 == 'move':
		arreglo.append(0)
		arreglo.append(0)
		arreglo.append(1)
	elif moveSwitch2 == 'switch':
		arreglo.append(0)
		arreglo.append(1)
		arreglo.append(0)
	elif moveSwitch2 == 'pass':
		arreglo.append(1)
		arreglo.append(0)
		arreglo.append(0)
	else:
		print("Error 5 - msgToArray")
	if nMoveSwitch2 == '' or nMoveSwitch2 == 0:
		arreglo.append(0)
		arreglo.append(0)
		arreglo.append(0)
		arreglo.append(0)
		arreglo.append(1)
	elif nMoveSwitch2 == '1' or nMoveSwitch2 == 1:
		arreglo.append(0)
		arreglo.append(0)
		arreglo.append(0)
		arreglo.append(1)
		arreglo.append(0)
	elif nMoveSwitch2 == '2' or nMoveSwitch2 == 2:
		arreglo.append(0)
		arreglo.append(0)
		arreglo.append(1)
		arreglo.append(0)
		arreglo.append(0)
	elif nMoveSwitch2 == '3' or nMoveSwitch2 == 3:
		arreglo.append(0)
		arreglo.append(1)
		arreglo.append(0)
		arreglo.append(0)
		arreglo.append(0)
	elif nMoveSwitch2 == '4' or nMoveSwitch2 == 4:
		arreglo.append(1)
		arreglo.append(0)
		arreglo.append(0)
		arreglo.append(0)
		arreglo.append(0)
	else:
		print("Error 6 - msgToArray")
	if dynamax2 == '' or dynamax2 == 0:
		arreglo.append(0)
	elif dynamax2 == 'dynamax' or dynamax2 == 1:
		arreglo.append(1)
	else:
		print("Error 7 - msgToArray")
	if toMove2 == '-1' or toMove2 == -1:
		arreglo.append(0)
		arreglo.append(0)
		arreglo.append(0)
		arreglo.append(1)
	elif toMove2 == '' or toMove2 == 0:
		arreglo.append(0)
		arreglo.append(0)
		arreglo.append(1)
		arreglo.append(0)
	elif toMove2 == '1' or toMove2 == 1:
		arreglo.append(0)
		arreglo.append(1)
		arreglo.append(0)
		arreglo.append(0)
	elif toMove2 == '2' or toMove2 == 2:
		arreglo.append(1)
		arreglo.append(0)
		arreglo.append(0)
		arreglo.append(0)
	else:
		print("Error 8 - msgToArray")
	return arreglo

def NeuralNetworkOutput(listNeuralNetworkOutput):
    # Parámetros
    moveSwitch1 = None
    nMoveSwitch1 = 0
    dynamax1 = 0
    toMove1 = 0
    moveSwitch2 = None
    nMoveSwitch2 = 0
    dynamax2 = 0
    toMove2 = 0
    # 1 to 3
    if listNeuralNetworkOutput[0] >= listNeuralNetworkOutput[1] and listNeuralNetworkOutput[0] >= listNeuralNetworkOutput[2]:
        moveSwitch1 = 'move'
    elif listNeuralNetworkOutput[1] >= listNeuralNetworkOutput[0] and listNeuralNetworkOutput[1] >= listNeuralNetworkOutput[2]:
        moveSwitch1 = 'switch'
    elif listNeuralNetworkOutput[2] >= listNeuralNetworkOutput[0] and listNeuralNetworkOutput[2] >= listNeuralNetworkOutput[1]:
        moveSwitch1 = 'pass'
    else:
        print("ERROR 1 - NeuralNetworkOutput")
    # 4 to 8
    if listNeuralNetworkOutput[3] >= listNeuralNetworkOutput[4] and listNeuralNetworkOutput[3] >= listNeuralNetworkOutput[5] and listNeuralNetworkOutput[3] >= listNeuralNetworkOutput[6] and listNeuralNetworkOutput[3] >= listNeuralNetworkOutput[7]:
        nMoveSwitch1 = ''
    elif listNeuralNetworkOutput[4] >= listNeuralNetworkOutput[3] and listNeuralNetworkOutput[4] >= listNeuralNetworkOutput[5] and listNeuralNetworkOutput[4] >= listNeuralNetworkOutput[6] and listNeuralNetworkOutput[4] >= listNeuralNetworkOutput[7]:
        nMoveSwitch1 = '1'
    elif listNeuralNetworkOutput[5] >= listNeuralNetworkOutput[3] and listNeuralNetworkOutput[5] >= listNeuralNetworkOutput[4] and listNeuralNetworkOutput[5] >= listNeuralNetworkOutput[6] and listNeuralNetworkOutput[5] >= listNeuralNetworkOutput[7]:
        nMoveSwitch1 = '2'
    elif listNeuralNetworkOutput[6] >= listNeuralNetworkOutput[3] and listNeuralNetworkOutput[6] >= listNeuralNetworkOutput[4] and listNeuralNetworkOutput[6] >= listNeuralNetworkOutput[5] and listNeuralNetworkOutput[6] >= listNeuralNetworkOutput[7]:
        nMoveSwitch1 = '3'
    elif listNeuralNetworkOutput[7] >= listNeuralNetworkOutput[3] and listNeuralNetworkOutput[7] >= listNeuralNetworkOutput[4] and listNeuralNetworkOutput[7] >= listNeuralNetworkOutput[5] and listNeuralNetworkOutput[7] >= listNeuralNetworkOutput[6]:
        nMoveSwitch1 = '4'
    else:
        print("ERROR 2 - NeuralNetworkOutput")
    # 9
    if listNeuralNetworkOutput[8] > 0.5:
        dynamax1 = 'dynamax'
    else:
        dynamax1 = ''
    # 10 to 13
    if listNeuralNetworkOutput[9] >= listNeuralNetworkOutput[10] and listNeuralNetworkOutput[9] >= listNeuralNetworkOutput[11] and listNeuralNetworkOutput[9] >= listNeuralNetworkOutput[12]:
        toMove1 = '-2'
    elif listNeuralNetworkOutput[10] >= listNeuralNetworkOutput[9] and listNeuralNetworkOutput[10] >= listNeuralNetworkOutput[11] and listNeuralNetworkOutput[10] >= listNeuralNetworkOutput[12]:
        toMove1 = ''
    elif listNeuralNetworkOutput[11] >= listNeuralNetworkOutput[9] and listNeuralNetworkOutput[11] >= listNeuralNetworkOutput[10] and listNeuralNetworkOutput[11] >= listNeuralNetworkOutput[12]:
        toMove1 = '1'
    elif listNeuralNetworkOutput[12] >= listNeuralNetworkOutput[9] and listNeuralNetworkOutput[12] >= listNeuralNetworkOutput[10] and listNeuralNetworkOutput[12] >= listNeuralNetworkOutput[11]:
        toMove1 = '2'
    else:
        print("ERROR 3 - NeuralNetworkOutput")
    # 14 to 16
    if listNeuralNetworkOutput[13] >= listNeuralNetworkOutput[14] and listNeuralNetworkOutput[13] >= listNeuralNetworkOutput[15]:
        moveSwitch2 = 'move'
    elif listNeuralNetworkOutput[14] >= listNeuralNetworkOutput[13] and listNeuralNetworkOutput[14] >= listNeuralNetworkOutput[15]:
        moveSwitch2 = 'switch'
    elif listNeuralNetworkOutput[15] >= listNeuralNetworkOutput[13] and listNeuralNetworkOutput[15] >= listNeuralNetworkOutput[14]:
        moveSwitch2 = 'pass'
    else:
        print("ERROR 4 - NeuralNetworkOutput")
    # 17 to 21
    if listNeuralNetworkOutput[16] >= listNeuralNetworkOutput[17] and listNeuralNetworkOutput[16] >= listNeuralNetworkOutput[18] and listNeuralNetworkOutput[16] >= listNeuralNetworkOutput[19] and listNeuralNetworkOutput[16] >= listNeuralNetworkOutput[20]:
        nMoveSwitch2 = ''
    elif listNeuralNetworkOutput[17] >= listNeuralNetworkOutput[16] and listNeuralNetworkOutput[17] >= listNeuralNetworkOutput[18] and listNeuralNetworkOutput[17] >= listNeuralNetworkOutput[19] and listNeuralNetworkOutput[17] >= listNeuralNetworkOutput[20]:
        nMoveSwitch2 = '1'
    elif listNeuralNetworkOutput[18] >= listNeuralNetworkOutput[16] and listNeuralNetworkOutput[18] >= listNeuralNetworkOutput[17] and listNeuralNetworkOutput[18] >= listNeuralNetworkOutput[19] and listNeuralNetworkOutput[18] >= listNeuralNetworkOutput[20]:
        nMoveSwitch2 = '2'
    elif listNeuralNetworkOutput[19] >= listNeuralNetworkOutput[16] and listNeuralNetworkOutput[19] >= listNeuralNetworkOutput[17] and listNeuralNetworkOutput[19] >= listNeuralNetworkOutput[18] and listNeuralNetworkOutput[19] >= listNeuralNetworkOutput[20]:
        nMoveSwitch2 = '3'
    elif listNeuralNetworkOutput[20] >= listNeuralNetworkOutput[16] and listNeuralNetworkOutput[20] >= listNeuralNetworkOutput[17] and listNeuralNetworkOutput[20] >= listNeuralNetworkOutput[18] and listNeuralNetworkOutput[20] >= listNeuralNetworkOutput[19]:
        nMoveSwitch2 = '4'
    else:
        print("ERROR 5 - NeuralNetworkOutput")
    # 22
    if listNeuralNetworkOutput[21] > 0.5:
        dynamax2 = 'dynamax'
    else:
        dynamax2 = ''
    # 23 to 26
    if listNeuralNetworkOutput[22] >= listNeuralNetworkOutput[23] and listNeuralNetworkOutput[22] >= listNeuralNetworkOutput[24] and listNeuralNetworkOutput[22] >= listNeuralNetworkOutput[25]:
        toMove2 = '-1'
    elif listNeuralNetworkOutput[23] >= listNeuralNetworkOutput[22] and listNeuralNetworkOutput[23] >= listNeuralNetworkOutput[24] and listNeuralNetworkOutput[23] >= listNeuralNetworkOutput[25]:
        toMove2 = ''
    elif listNeuralNetworkOutput[24] >= listNeuralNetworkOutput[22] and listNeuralNetworkOutput[24] >= listNeuralNetworkOutput[23] and listNeuralNetworkOutput[24] >= listNeuralNetworkOutput[25]:
        toMove2 = '1'
    elif listNeuralNetworkOutput[25] >= listNeuralNetworkOutput[22] and listNeuralNetworkOutput[25] >= listNeuralNetworkOutput[23] and listNeuralNetworkOutput[25] >= listNeuralNetworkOutput[24]:
        toMove2 = '2'
    else:
        print("ERROR 6 - NeuralNetworkOutput")
    # Formateo
    message = "/choose {} {} {} {}, {} {} {} {}".format(moveSwitch1, nMoveSwitch1, dynamax1, toMove1, moveSwitch2, nMoveSwitch2, dynamax2, toMove2)
    return [message], moveSwitch1, nMoveSwitch1, dynamax1, toMove1, moveSwitch2, nMoveSwitch2, dynamax2, toMove2

class DDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size		# Tamaño del vector que representa el estado
        self.action_size = action_size      # Tamaño del vector que representa la acción
        self.memory = deque(maxlen=64000)   # Tamaño de la memoria del agente
        self.gamma = 0.95                   # Determina cuánto se preocupan los agentes de aprendizaje por refuerzo por las recompensas en un futuro lejano en relación con las del futuro inmediato
        self.learning_rate = 0.001          # La tasa de aprendizaje sería la longitud del paso que da cada vez que decide cambiar de posición, probar tambien con 0.0001
        
        self.epsilon = 1.0 					# Factor de exploration inicial
        self.epsilon_min = 0.01 			# Factor de exploration minimo
        self.epsilon_decay = 0.998 			# Factor de decaimiento del factor de exploracion: 1*0.998*0.998*...
        self.model = self._build_model() 	# Construye el modelo neuronal a entrenar 

        self.q = queue.Queue()
        self.q50 = queue.Queue()
        self.qm50 = queue.Queue()
        self.q150 = queue.Queue()

        # Construye un 2do modelo neuronal (target_model) para ayudar a estimar el target del 1er modelo (model)   
        self.target_model = self._build_model()
        # Actualiza el target_model copiando los pesos de model
        self.update_target_model()
       

    def _build_model(self):
    	# Informa que las capas que se van agregar son secuenciales
        model = Sequential()
        # Capa de Entrada: 219 neuronas - Capa Oculta 1: 219 neuronas
        model.add(Dense(219, input_dim=self.state_size, activation='relu'))
        # Capa Oculta 2: 111 neuronas
        model.add(Dense(111, activation='relu'))
        # Capa Oculta 3: 54 neuronas
        model.add(Dense(54, activation='relu'))
        # Capa Oculta 4: 38 neuronas
        model.add(Dense(38, activation='relu')) 
        # Capa de Salida: 26 neuronas
        model.add(Dense(self.action_size, activation='linear')) 
        # La funcion de perdida es el error cuadratico medio (mse)
        model.compile(loss='mse', optimizer = Adam(learning_rate=self.learning_rate)) 
        return model
    
    # Metodo para copiar los pesos de model a target_model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # Metodo para guardar una transicion del agente (experiencia): (estado, accion, reward resultante, nuevo estado, done)
    # Done es un flag que indica que el entorno cayo en un estado terminal, la batalla ha finalizado
    def remember(self, state, action, reward, next_state, done): 
        self.memory.append((state, action, reward, next_state, done))

    # Retorna una accion.  
    def get_action(self, state):
    	if np.random.rand() <= 0.7:
    		# retorna una accion aleatoria con probabilidad self.epsilon
    		# Solicito msg random
    		self.q50.put([50,True])
    		print("50 ENVIADO")
    		# Ahora, el Thread ha leido el 50... Espero la respuesta
    		while True:
    			if self.qm50.qsize() > 0:
		    		if self.qm50.queue[0][0] == -50:
		    			cola = self.qm50.get(0)
		    			print("{} RECIBIDO".format(cola[0]))
		    			if cola[0] == -50:
		    				# Obtengo la respuesta
		    				msg = cola[1]
		    				break
		    		else:
		    			print("ERROR {} RECIBIDO P2".format(cola[0]))
    		moveSwitch1 = cola[2]
    		nMoveSwitch1 = cola[3]
    		dynamax1 = cola[4]
    		toMove1 = cola[5]
    		moveSwitch2 = cola[6]
    		nMoveSwitch2 = cola[7]
    		dynamax2 = cola[8]
    		toMove2 = cola[9]
    		return msg, moveSwitch1, nMoveSwitch1, dynamax1, toMove1, moveSwitch2, nMoveSwitch2, dynamax2, toMove2
    	# Obtiene los q valores predichos por el modelo para cada accion
    	self.q50.put([50,False])
    	print("50 ENVIADO (2)")
    	action_values = self.model.predict(state)
    	# Retorna el mensaje de acción, que pudiera significar una jugada no válida
    	return NeuralNetworkOutput(action_values[0])

    # Metodo para actualizar los pesos de model con una muestra (minibatch) de la memoria de experiencias de tamao batch_size
    def replay(self, batch_size): 
        # obtiene una muestra de su memoria de experiencias
        minibatch = random.sample(self.memory, batch_size) 

        # Obtiene arrays de  states, accions, rewards, next_estates  del minibatch
        states = np.array([i[0] for i in minibatch])    
        actions = np.array([i[1] for i in minibatch])    
        rewards = np.array([i[2] for i in minibatch])    
        next_states = np.array([i[3] for i in minibatch])   
        dones = np.array([i[4] for i in minibatch])        
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)
        
        # Calcula el target Q-value para la accion ejecutada en cada experiencia del batch 
        Qvals_next_states = self.target_model.predict_on_batch(next_states)  # predice Qvals de next_state con el modelo target
        Targets_actions = rewards + self.gamma*(np.amax(Qvals_next_states, axis=1))*(1-dones) # si es terminal -> Target = reward
        #print("Targets_actions: {}".format(Targets_actions))

        # Construye el target completo de Q-values para los states de cada experiencia del batch
        Targets = self.model.predict_on_batch(states)  # los target Q-values son los que predice el modelo actual, salvo el correspondiente a la accion tomada
        ind = np.array([i for i in range(batch_size)])   
        #print("ind: {}".format(ind))
        #print("actions: {}".format(actions))
        Targets[[ind], [actions]] = Targets_actions   # actualiza el target Q-value de la accion ejecutada  

        # Ajusta el modelo  con pares  [states, Targets] del minibatch
        self.model.fit(states, Targets, epochs=1, verbose=0)

        # si no esta en el valor minimo del factor de exploracion -> hace un decaimiento del factor de exploracion
        if self.epsilon > self.epsilon_min: 
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def addQueue(self, q, q50, qm50, q150):
    	self.q = q
    	self.q50 = q50
    	self.qm50 = qm50
    	self.q150 = q150

def inicio():
	# Número de Batallas: 80000
	EPISODES = 1
	# Tamaño del lote: 64 = 8 batallas * 8 turnos
	batch_size = 3
	# Neuronas de entrada
	state_size = 219
	# Neuronas de salida
	action_size = 26
	# Se crea al agente
	agente = DDQNAgent(state_size, action_size)
	# Lista para guardar la evolucion del score
	score_list = []
	# Lista para guardar la evolución del factor de exploración
	epsilon_list = []
	# listAction
	listAction = []
	# W / L
	win = 0
	loss = 0
	# Para cada Batalla...
	for e in range(EPISODES):
		q1 = queue.Queue()
		q2 = queue.Queue()
		q50 = queue.Queue()
		qm50 = queue.Queue()
		q150 = queue.Queue()
		# Inicio Batalla - Agente Random
		opponentThread = Thread(target=AsyncDownload, args=(q2,q2,q2,q2,"Deep1ODDQN","123456",listAgentTeamname[e%len(listAgentTeamname)],"ACCEPT_CHALLENGE","Deep1UDDQN",True,False,1,))
		opponentThread.start()
		# Inicio Batalla - Agente Inteligente
		userThread = Thread(target=AsyncDownload, args=(q1,q50,qm50,q150,"Deep1UDDQN","123456","Eduardo Cunha","CHALLENGE_USER","Deep1ODDQN",True,True,1,))
		userThread.start()
		agente.addQueue(q1, q50, qm50, q150)
		interacciones = 0
		while True:
			try:
				cola = q1.get(0)
				print("{} RECIBIDO".format(cola[0]))
				if cola[0] == 10:
					# Recibo el array de estado actual del entorno, invariante si es el primer turno o cualquiera
					listNeuralNetworkInput = cola[1]
					# Imprimo en un archivo como validacion, por el momento
					#nameFile = "deepReinforcementLearning/reportes/Deep1UDDQN.txt"
					#if path.exists(nameFile):
					#	with open(nameFile, 'a') as outfile:
					#		outfile.write("{}\n".format(listNeuralNetworkInput))
					#else:
					#	with open(nameFile, 'w') as outfile:
					#		outfile.write("{}\n".format(listNeuralNetworkInput))
					# Imprimo en un archivo de reporte el estado actual del entrenamiento
					nameFile = "deepReinforcementLearning/reportes/0.txt"
					# Inicio Parámetros
					numBattle = str(e).zfill(5)
					turnBattle = str(cola[2]).zfill(3)
					if int(cola[3]) >= 0:
						reward = str(cola[3]).zfill(4)	#xxxx
						reward = "+{}".format(reward)	#+xxxx
					else:
						reward = str(cola[3]).zfill(5)	#-xxxx
					if int(cola[4]) >= 0:
						score = str(cola[4]).zfill(5)	#xxxxx
						score = "+{}".format(score)		#+xxxxx
					else:
						score = str(cola[4]).zfill(6)	#-xxxxx
					dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
					# Fin Parámetros
					if path.exists(nameFile):
						with open(nameFile, 'a') as outfile:
							outfile.write("{}\t{}\t{}\t{}\t{}\n".format(numBattle, turnBattle, reward, score, dt_string))
							print("{}\t{}\t{}\t{}\t{}".format(numBattle, turnBattle, reward, score, dt_string))
					else:
						with open(nameFile, 'w') as outfile:
							outfile.write("{}\t{}\t{}\t{}\t{}\n".format(numBattle, turnBattle, reward, score, dt_string))
							print("{}\t{}\t{}\t{}\t{}".format(numBattle, turnBattle, reward, score, dt_string))
					# Es momento de realizar alguna acción...
					# Formato de Input de la Red Neuronal
					#print("listNeuralNetworkInput: {}".format(listNeuralNetworkInput))
					state = np.reshape(listNeuralNetworkInput, [1, state_size])

					# Ejecuta la acción en el entorno y obtiene, en el siguiente turno: nuevo estado, reward y flag done (si es estado terminal, o sea batalla terminada), uso q
					if cola[5]: # No es turno final
						# Obtiene una accion del agente
						action, moveSwitch1, nMoveSwitch1, dynamax1, toMove1, moveSwitch2, nMoveSwitch2, dynamax2, toMove2 = agente.get_action(state)
						q150.put([150,action])
						print("150 ENVIADO (1) (2)")
					else:
						action = None
					interacciones += 1
					# A partir del segundo turno o turno 1.2
					if interacciones >= 2:
						# Almacena esta experiencia en la memoria del agente, OK
						agente.remember(lastState, lastAction, int(lastReward)*1.0, state, False)
						# Si hay suficiente experiencias en memoria del agente -> ajusta su modelo neuronal 
						if len(agente.memory) > batch_size:
							agente.replay(batch_size)
					# Last
					lastState = state
					array = msgToArray(moveSwitch1, nMoveSwitch1, dynamax1, toMove1, moveSwitch2, nMoveSwitch2, dynamax2, toMove2)
					if array in listAction:
						index = listAction.index(array)
						actionEncoded = index
						#print("old action")
					else:
						listAction.append(array)
						index = listAction.index(array)
						actionEncoded = index
						#print("new action: {}".format(listAction))
					lastAction = actionEncoded
					lastReward = reward
				elif cola[0] == -10:
					# Fin de la batalla
					print("Fin de la Batalla")
					break
				else:
					print("ERROR {} RECIBIDO P1".format(cola[0]))
			except queue.Empty:
				...
		# Actualiza el target_model al final de cada batalla
		agente.update_target_model()
		# 
		score_list.append(score)
		# 
		epsilon_list.append(agente.epsilon)
		#print("Fin de la Batalla")	# Por ahora mantenerlo
		nameFile = "deepReinforcementLearning/reportes/wl.txt"
		# W or L
		if cola[1] == "Neuro2UNeat":
			win += 1
		else:
			loss += 1
		# Fin Parámetros
		if path.exists(nameFile):
			with open(nameFile, 'a') as outfile:
				outfile.write("{}\t{}\t{}\t{}\t{}\n".format(numBattle, win, loss, score, dt_string))
		else:
			with open(nameFile, 'w') as outfile:
				outfile.write("{}\t{}\t{}\t{}\t{}\n".format(numBattle, win, loss, score, dt_string))


if __name__ == "__main__":
    inicio()