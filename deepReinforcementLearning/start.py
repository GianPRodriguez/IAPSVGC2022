import os
import sys
import gym
import time
import json
import queue
import random
import numpy as np
import multiprocessing
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt

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

DICCIONARIO = json.loads(open("API/JSONs/movesANN.json","r").read())
TEAMS = ["Eduardo Cunha","Guillermo Castilla","James Baek","Paul Chua","Calvin Foster","Meaghan Rattle","Davide Carrer","David Koutesh"]
STATE = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 64.0, 187.0, 141.0, 106.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 144.0, 146.0, 222.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 167.0, 126.0, 179.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 158.0, 182.0, 180.0, 209.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 141.0, 366.0, 155.0, 179.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.75, 0.0, 0.0, 1.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 126.0, 238.0, 176.0, 222.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.75, 0.0, 0.75, 0.0, 0.0, 1.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]


class DDQNAgent:
    def __init__(self, state_size, action_size, model):
        self.state_size = state_size				# Tamaño del vector que representa el estado
        self.action_size = action_size      		# Tamaño del vector que representa la acción
        self.memory = deque(maxlen=64000)   		# Tamaño de la memoria del agente
        self.gamma = 0.95                   		# Determina cuánto se preocupan los agentes de aprendizaje por refuerzo por las recompensas en un futuro lejano en relación con las del futuro inmediato
        self.learning_rate = 0.001          		# La tasa de aprendizaje sería la longitud del paso que da cada vez que decide cambiar de posición, probar tambien con 0.0001
        
        self.epsilon = 1.00 						# Factor de exploration inicial
        self.epsilon_min = 0.01 					# Factor de exploration minimo
        self.epsilon_decay = 0.998 					# Factor de decaimiento del factor de exploracion: 1*0.998*0.998*...
        if model is None:
        	self.model = self._build_model() 		# Construye el modelo neuronal a entrenar 
        else:
        	self.model = model

        self.q = queue.Queue()
        self.q50 = queue.Queue()
        self.qm50 = queue.Queue()
        self.q150 = queue.Queue()
        self.accion = False

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
        # Capa de Salida: 219 neuronas
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
    	# 0.5 de probabilidad de retornar una accion aleatoria
    	if np.random.rand() <= self.epsilon:
    		self.accion = False
    		# Solicito msg random
    		self.q50.put([50,True])
    		#print("50 ENVIADO")
    		#print("RANDOM:")
    		# Ahora, el Thread ha leido el 50... Espero la respuesta
    		while True:
    			if self.qm50.qsize() > 0:
    				if self.qm50.queue[0][0] == -50:
    					cola = self.qm50.get(0)
    					#print("{} RECIBIDO".format(cola[0]))
    					if cola[0] == -50:
    						# Obtengo la respuesta
    						msg = cola[1]
    						break
    				else:
    					print("ERROR {} RECIBIDO P2".format(cola[0]))
    		action = ''.join(msg[0]).replace("/choose ","")
    		try:
    			action_value = int(list(DICCIONARIO.keys()) [list(DICCIONARIO.values()).index(action)])
    		except:
    			print("\n\n\n\nACCIÓN INVÁLIDA: -{}-\n\n\n".format(action))
    			action_value = 0
    		return action_value
    	# Obtiene los q valores predichos por el modelo para cada accion
    	self.accion = True
    	self.q50.put([50,False])
    	#print("50 ENVIADO (2)")
    	#print("ANN:")
    	action_value = self.model.predict(state)
    	# action_value is a number, 0 to 218
    	action_value = int(np.argmax(action_value))
    	# Retorno el indice de una acción
    	return action_value

    # Metodo para actualizar los pesos de model con una muestra (minibatch) de la memoria de experiencias de tamaño batch_size
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
        #print("ind: {}".format(ind)
        #print("actions: {}".format(actions))
        Targets[[ind], [actions]] = Targets_actions   # actualiza el target Q-value de la accion ejecutada  

        # Ajusta el modelo  con pares  [states, Targets] del minibatch
        self.model.fit(states, Targets, epochs=1, verbose=0)

        # si no esta en el valor minimo del factor de exploracion -> hace un decaimiento del factor de exploracion
        if self.epsilon > self.epsilon_min: 
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_model(name)

    def save(self, name):
        self.model.save(name)

    def addQueue(self, q, q50, qm50, q150):
    	self.q = q
    	self.q50 = q50
    	self.qm50 = qm50
    	self.q150 = q150

def inicio():
	# Lista de Rewards
	score_list = []
	# Número de Batallas:
	EPISODES = 500
	# Tamaño del lote: 500 interacciones
	batch_size = 50
	# Neuronas de entrada
	state_size = 219
	# Neuronas de salida
	action_size = 219
	# Se crea al agente
	#agente = DDQNAgent(state_size, action_size, None)
	agente = DDQNAgent(state_size, action_size, tf.keras.models.load_model("deepReinforcementLearning/modelos/neat-checkpoint-5.0/"))
	# Para cada Batalla...
	for e in range(EPISODES):
		q1 = queue.Queue()
		q2 = queue.Queue()
		q50 = queue.Queue()
		qm50 = queue.Queue()
		q150 = queue.Queue()
		# Inicio Batalla - Agente Random
		opponentThread = Thread(target=AsyncDownload, args=(q2,q2,q2,q2,"Deep1ODDQN","123456",TEAMS[e%len(TEAMS)],"ACCEPT_CHALLENGE","Deep1UDDQN",True,False,1,))
		opponentThread.start()
		# Inicio Batalla - Agente Inteligente
		bucle1 = True
		while bucle1:
			bucle2 = True
			#print("INICIANDO USUARIO...")
			userThread = None
			time.sleep(1.0)
			userThread = Thread(target=AsyncDownload, args=(q1,q50,qm50,q150,"Deep1UDDQN","123456","Training Team","CHALLENGE_USER","Deep1ODDQN",True,True,1,))
			userThread.start()
			agente.addQueue(q1, q50, qm50, q150)
			# ¿Recibo confirmación del oponente que he comenzado la batalla?
			while bucle2:
				if q2.qsize() > 0:
					cola = q2.get(0)
					if cola[0] == 0:
						#print("USUARIO INICIADO:")
						qm50.put([0])
						bucle1 = False
						bucle2 = False
					elif cola[0] == -1:
						qm50.put([-1])
						bucle2 = False
					else:
						...
		interacciones = 0
		batallaEnCurso = True
		while batallaEnCurso:
			try:
				cola = q1.get(0)
				if cola[0] == 10:
					# Recibo el array de estado actual del entorno
					listNeuralNetworkInput = cola[1]
					# Formato de Input de la Red Neuronal
					state = np.reshape(listNeuralNetworkInput, [1, state_size])
					# Obtengo el indice de la acción
					action_value = agente.get_action(state)
					# Y el mensaje de la acción
					action = '/choose ' + DICCIONARIO[str(action_value)]
					# Envío la acción en formato de mensaje
					q150.put([150,action])
					interacciones += 1
					# A partir de la segunda interacción, donde ya tengo la recompensa...
					if interacciones >= 2:
						# Almacena esta experiencia en la memoria del agente, False al final... no es estado terminal
						if agente.accion:
							score_list.append(int(cola[3]))
							print("ACCION PROPIA AGREGADA: {}".format(len(score_list)))
							if len(score_list) == 500:
								print("SE HA ALCANZADO LAS 500 INTERACCIONES PROPIAS")
								print("\nSCORE:\n\n{}".format(score_list))
								return
						agente.remember(lastState, lastAction, int(cola[3])*1.0, state, False)
						if int(cola[3]) == -9999 and len(agente.memory) > batch_size:
							agente.replay(batch_size)
					# Almaceno las variables para ser usadas en la siguiente interacción
					lastState = state
					lastAction = action_value
				elif cola[0] == -10:
					# Fin de la batalla... almaceno la última experiencia en la memoria del agente, True al final, es estado terminal
					listNeuralNetworkInput = cola[1]
					state = np.reshape(listNeuralNetworkInput, [1, state_size])
					if agente.accion and int(cola[3]) != 0:
						score_list.append(int(cola[3]))
						print("ACCION PROPIA AGREGADA: {}".format(len(score_list)))
						if len(score_list) == 500:
							print("SE HA ALCANZADO LAS 500 INTERACCIONES PROPIAS")
							print("\nSCORE:\n\n{}".format(score_list))
							return
					agente.remember(lastState, lastAction, int(cola[3])*1.0, state, True)
					batallaEnCurso = False
				else:
					print("ERROR {} RECIBIDO P1".format(cola[0]))
				# Reporte en CMD
				numBattle = str(e).zfill(5)
				turnBattle = str(cola[2]).zfill(3)
				if int(cola[3]) >= 0:
					reward = str(cola[3]).zfill(5)
					reward = "+{}".format(reward)
				else:
					reward = str(cola[3]).zfill(6)
				if int(cola[4]) >= 0:
					score = str(cola[4]).zfill(6)
					score = "+{}".format(score)
				else:
					score = str(cola[4]).zfill(7)
				dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
				print("{}\t{}\t{}\t{}\t{}".format(numBattle, turnBattle, reward, score, dt_string))
			except queue.Empty:
				...
		# Si hay suficiente experiencias en memoria del agente... ajusta su modelo neuronal 
		if len(agente.memory) > batch_size:
			agente.replay(batch_size)
		# Archivos de Registro - WinLoss$.txt
		numBattle = str(e).zfill(5)
		indexOT = str(e%len(TEAMS))
		scoreBatalla = cola[4]
		win, loss = 0, 0
		if int(scoreBatalla) >= 0:
			win = str(1)
		else:
			loss = str(1)
		if int(scoreBatalla) >= 0:
			scoreBatalla = str(scoreBatalla).zfill(6)
			scoreBatalla = "+{}".format(scoreBatalla)
		else:
			scoreBatalla = str(scoreBatalla).zfill(7)
		dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
		rutaWinLoss = "./deepReinforcementLearning/reportes/WinLoss/" + str(0) + ".txt"
		with open(rutaWinLoss, 'a') as WLScore:
			WLScore.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(numBattle, win, loss, indexOT, scoreBatalla, dt_string))
		# Archivos de Registro - Score$.txt
		rutaScore = "./deepReinforcementLearning/reportes/Score/" + str(0) + ".txt"
		with open(rutaScore, 'a') as fScore:
			fScore.write("{}\n".format(scoreBatalla))
		# Archivos de Registro - InvalidActions$.txt
		perInvalidActions = cola[5]
		rutaInvalidActions = "./deepReinforcementLearning/reportes/InvalidActions/" + str(0) + ".txt"
		with open(rutaInvalidActions, 'a') as IAScore:
			IAScore.write("{}\n".format(perInvalidActions))
		# Actualiza el target_model al final de cada batalla
		agente.update_target_model()
		# Guardo una copia de la red neuronal cada 05 batallas - ejemplo 1
		if (int(e)+1)%10 == 0:	# (int(e)+1)%10 == 0
			rutaSave = "deepReinforcementLearning/modelos/neat-checkpoint-" + str((int(e)+1)/10) + "/"
			agente.save(rutaSave)
	# Fin del bucle...
	print("--- FIN DEL PROGRAMA ---")
	# Prueba pequeña:
	state = np.reshape(STATE, [1, state_size])
	action_values = agente.model.predict(state)
	# action_value is a number, 0 to 218
	action_value = int(np.argmax(action_values))
	action = '/choose ' + DICCIONARIO[str(action_value)]
	print("Índice: {} Acción: {}".format(action_value, action))
	# Gráfica
	print("\nSCORE:\n\n{}".format(score_list))
	return


if __name__ == "__main__":
    inicio()