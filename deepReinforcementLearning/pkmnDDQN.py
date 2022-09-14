import random
import gym
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

class DDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size      # tamao de un estado (numero de atributos que representan un estado)
        self.action_size = action_size    # tamao del vector de acciones 
        self.memory = deque(maxlen=3000)  # define la memoria del agente (3000 registros como maximo)
        self.gamma = 0.95                 # discount rate
        self.learning_rate = 0.001        # taza de aprendizaje 
        
        self.epsilon = 1.0               # factor de exploration inicial
        self.epsilon_min = 0.01          # factor de exploration minimo
        self.epsilon_decay = 0.998       # factor de decaimiento del factor de exploracion
        self.model = self._build_model() # construye el modelo neuronal a entrenar 
        
        # construye un 2do modelo neuronal (target_model) para ayudar a estimar el target del 1er modelo (model)   
        self.target_model = self._build_model() 
        self.update_target_model()  # actualiza el target_model copiando los pesos de model
       

    def _build_model(self):
        # Define y compila un modelo de red neuronal de 3 capas: state_size entradas X 20 neuronas X 20 neuronas x action_size neuronas de salida
        model = Sequential()   # Informa que las capas que se van agregar son secuenciales
        model.add(Dense(20, input_dim=self.state_size, activation='relu')) # 1ra capa de 20 neuronas, cada neurona recibe state_size entradas (4 para CartPole), activacion relu
        model.add(Dense(20, activation='relu')) # 2da capa de 20 neuronas, funcion de activacion relu
        model.add(Dense(self.action_size, activation='linear')) # 3ra capa (salida) de action_size neuronas (2 para CartPole)
       
        model.compile(loss='mse', optimizer = Adam(learning_rate=self.learning_rate)) # la funcion de perdida es el error cuadratico medio (mse)
        return model
    
    # Metodo para copiar los pesos de model a target_model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # metodo para guardar una transicion del agente (experiencia): (estado, accion, reward resultante, nuevo estado, done)
    # done es un flag que indica que el entorno cayo en un estado terminal
    def remember(self, state, action, reward, next_state, done): 
        self.memory.append((state, action, reward, next_state, done))

    # retorna una accion.  
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:  # retorna una accion aleatoria con probabilidad self.epsilon
            return random.randrange(self.action_size)
        action_values = self.model.predict(state) # obtiene los q valores predichos por el modelo para cada accion
        return np.argmax(action_values[0])  # retorna la accion con el maximo q-valor predicho

    # metodo para actualizar los pesos de model con una muestra (minibatch) de la memoria de experiencias de tamao batch_size
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
        
        # Construye el target completo de Q-values para los states de cada experiencia del batch
        Targets = self.model.predict_on_batch(states)  # los target Q-values son los que predice el modelo actual, salvo el correspondiente a la accion tomada
        ind = np.array([i for i in range(batch_size)])
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

def run():
    # La batalla no está terminada
    done = False
    # Tamaño del lote: 64 = 8 batallas * 8 turnos
    batch_size = 4
    # Número de Batallas: 80000
    EPISODES = 2
    
    # Neuronas de entrada
    state_size = 183
    # Neuronas de salida
    action_size = 26
    # Se crea al agente
    agent = DDQNAgent(state_size, action_size)
    
    # Lista para guardar la evolucion del score
    score_list = []
    # Lista para guardar la evolución del factor de exploración
    epsilon_list = []

    # Para cada Batalla...
    for e in range(EPISODES):

        # Inicio Batalla - Agente Inteligente
        userThread = Thread(target=AsyncDownload, args=(q,agentUsername,agentPassword,agentTeamname,agentBotmode,opponentUsername,True,1,))
        userThread.start()
        # Inicio Batalla - Agente Random
        opponentThread = Thread(target=AsyncDownload, args=(None,opponentUsername,opponentPassword,opponentTeamname,opponentBotmode,agentUsername,True,1,))
        opponentThread.start()

        # Correcto
        score = 0

        # Obteniendo valores del Queue
        if userThread.is_alive():
            # check the thread every 100ms
            self.after(100, lambda: self.monitor(q))
            try:
                # Get Queue
                self.cola = q.get(0)
                if self.cola[0] == 0:
                    # Obtengo el state inicial
                    ...
                    # [ 0.02468106  0.00889922 -0.04296796 -0.02061151], valores iniciales del primer turno
                    state = self.cola[1]
                    ...
                    # [[ 0.02468106,  0.00889922, -0.04296796, -0.02061151]], formateo valores neuronales primer turno
                    state = np.reshape(state, [1, state_size])
                    ...
                    # Obtiene una accion del agente, uso q
                    action = agent.get_action(state)
                    ...

                    # Ejecuta la acción en el entorno y obtiene: nuevo estado, reward y flag done (si es estado terminal, o sea batalla terminada), uso q
                    env.step(action)
                elif self.cola[0] == 1:
                    # Obtengo next_state, reward, done, _
                    next_state = self.cola[1]
                    reward = self.cola[2]
                    done = self.cola[3]
                    ...
                    # Si es estado terminal el reward es -10, aqui debería ser escoger accion prohibida
                    reward = reward if not done else -10
                    # [[ 0.02485905, -0.18558103, -0.04338019,  0.25821091]], formateo nuevo estado valores neuronales
                    next_state = np.reshape(next_state, [1, state_size])
                    # Almacena esta experiencia en la memoria del agente, OK
                    agent.remember(state, action, reward, next_state, done)
                    # Actualiza el estado actual al nuevo estado y el reward histórico llamado SCORE
                    state = next_state
                    score += reward
                    
                    # Si hay suficiente experiencias en memoria del agente -> ajusta su modelo neuronal 
                    if len(agent.memory) > batch_size:
                        agent.replay(batch_size)
                    # Si es estado terminal, terminó la batalla, imprime resultados del trial. El escore del trial es el numero de pasos que logro ejecutar el agente
                    if done:
                        print("episode: {}/{}, score: {}, e: {:.2}".format(e, EPISODES, step, agent.epsilon))
            except queue.Empty:
                # None
                ...
        else:
            ...
        # Actualiza el target_model al final de cada batalla
        agent.update_target_model()
        # 
        score_list.append(score)
        # 
        epsilon_list.append(agent.epsilon)

    # ...
    agent.model.save("saved_model/")

if __name__ == "__main__":
    run()