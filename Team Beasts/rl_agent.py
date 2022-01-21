import random
import numpy as np
from keras import Sequential
from collections import deque
from keras.layers import Dense
from keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
import math


class DQN:

    """ Deep Q Network """
    def __init__(self):

        self.action_space = 3
        self.state_space = 18
        self.epsilon = 1 
        self.gamma = .95
        #self.batch_size = params['batch_size'] 
        self.batch_size = 10
        self.epsilon_min = .01 
        self.epsilon_decay = .995 
        self.learning_rate = .00025
        self.layer_sizes = [128, 128, 128]
        self.memory = deque(maxlen=2500)
        self.model = self.build_model()
        self.prev_state = {}
        self.action = -1

    def build_model(self):
        model = Sequential()
        for i in range(len(self.layer_sizes)):
            if i == 0:
                model.add(Dense(self.layer_sizes[i], input_shape=(self.state_space,), activation='relu'))
            else:
                model.add(Dense(self.layer_sizes[i], activation='relu'))
        model.add(Dense(self.action_space, activation='softmax'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        model.trainable = True
        return model


    def remember(self, state, action, reward, next_state, done):
        if len(state) == 0:
            return
        if len(next_state) == 0:
            return
        self.memory.append((state, action, reward, next_state, done))
        #print(self.memory)


    def act(self, state):

        if np.random.rand() <= self.epsilon or len(state) == 0:
            return random.randrange(self.action_space)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])


    def replay(self):
        #print(len(self.memory) < self.batch_size)
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        #print(minibatch)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)
        targets = rewards + self.gamma*(np.amax(self.model.predict_on_batch(next_states), axis=1))*(1-dones)
        targets_full = self.model.predict_on_batch(states)

        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
# Function to plot result
def plot_result(data_name_array, run):
    plt_agent_sweeps = []
    
    fig, ax = plt.subplots(figsize=(8,6))
    max_list = []
    min_list = []
    for data_name in data_name_array:
        data = data_name_array[data_name]

        # smoothed_sum_reward = smooth(data=sum_reward_data, k=k)
        max_list.append(max(data))
        min_list.append(min(data))
        plot_x_range = np.arange(1,len(data)+1)
        ax.plot(plot_x_range, data, label=data_name)

    max_to_hundred = int(math.ceil(max(max_list) / 100.0)) * 100
    min_to_hundred = int(math.floor(min(min_list) / 100.0)) * 100
    
    ax.legend(fontsize = 13)
    ax.set_title("Learning Curve", fontsize = 15)
    ax.set_xlabel('Episodes', fontsize = 14)
    ax.set_ylabel("Reward", rotation=0, labelpad=40, fontsize = 14)
    ax.set_ylim([min_to_hundred, max_to_hundred])
    plt.savefig(f"plot_{run}.png") 
    return 
'''if __name__ == '__main__':

    params = dict()
    params['name'] = None
    params['epsilon'] = 1
    params['gamma'] = .95
    params['batch_size'] = 500
    params['epsilon_min'] = .01
    params['epsilon_decay'] = .995
    params['learning_rate'] = 0.00025
    params['layer_sizes'] = [128, 128, 128]'''
    
