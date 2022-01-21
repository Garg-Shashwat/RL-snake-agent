from keras.backend import equal
from environment import Vasuki

from gym import Env
from gym.spaces import Discrete, Box, MultiDiscrete
from nqueens import Queen

import numpy as np
import random
import os

import matplotlib
import matplotlib.pyplot as plt

import cv2

from collections import namedtuple, deque
from itertools import count
from base64 import b64encode

from keras.models import load_model
from rl_agent import DQN
from rl_agent import plot_result
# ----------------------------------------------------- #

class Runner():
    def __init__(self, model_A, model_B, checkpoint):
        # Path to store the Video
        self.checkpoint = checkpoint
        # Defining the Environment
        config = {'n': 8, 'rewards': {'Food': 4, 'Movement': -1, 'Illegal': -2}, 'game_length': 100} # Should not change for evaluation
        self.env = Vasuki(**config)
        self.runs = 1000
        # Trained Policies
        self.model_A = model_A # Loaded model with weights
        self.model_B = model_B # Loaded model with weights
        # Results
        self.winner = {'Player_A': 0, 'Player_B': 0}
        self.score={}
        self.scoreA=[]
        self.scoreB=[]

    def reset(self):
        self.winner = {'Player_A': 0, 'Player_B': 0}

    def evaluate_A(self, info,reward,done):
        
        # pos = info['agentA']["state"]
        # pos2 = info["agentB"]["state"]
        # head = info["agentA"]["head"]
        # points = self.env.live_foodspawn_space
        # points.sort(key = lambda K: (pos[0]-K[0])**2 + (pos[1]-K[1])**2)
        # # state.append(int(points[0][0]==pos2[0] and points[0][1]==pos2[1]))
        # state = [int(pos[0]==0), int(pos[0]==7), int(pos[1]==0), int(pos[1]==7), int(pos[0]>pos2[0]), int(pos[0]<pos2[0]), int(pos[1]>pos2[1]), int(pos[1]<pos2[1]), int(info["agentA"]["score"]>info["agentB"]["score"]),int(head==0), int(head==1), int(head==2), int(head==3),int(pos[0]>points[0][0]), int(pos[0]<points[0][0]), int(pos[1]>points[0][1]), int(pos[1]<points[0][1])]
        # state.append(int(((pos[0]-pos2[0])**2 + (pos[1]-pos2[1])**2) > ((pos[0]-points[0][0])**2 + (pos[1]-points[0][1])**2)))
        
       
        
        state = [0]*18
        state = np.reshape(state, (1, 18))
        if len(info) != 0:
            pos = info['agentA']["state"]
            pos2 = info["agentB"]["state"]
            head = info["agentA"]["head"]
            points = self.env.live_foodspawn_space
            sorted(points , key = lambda K: (pos[0]-K[0])**2 + (pos[1]-K[1])**2)
            # state.append(int(points[0][0]==pos2[0] and points[0][1]==pos2[1]))
            state = [int(pos[0]==0), int(pos[0]==7), int(pos[1]==0), int(pos[1]==7), int(pos[0]>pos2[0]), int(pos[0]<pos2[0]), int(pos[1]>pos2[1]), int(pos[1]<pos2[1]), int(info["agentA"]["score"]>info["agentB"]["score"]),int(head==0), int(head==1), int(head==2), int(head==3),int(pos[0]>points[0][0]), int(pos[0]<points[0][0]), int(pos[1]>points[0][1]), int(pos[1]<points[0][1])]
            state.append(int(((pos[0]-pos2[0])**2 + (pos[1]-pos2[1])**2) > ((pos[0]-points[0][0])**2 + (pos[1]-points[0][1])**2)))
            state = np.reshape(state, (1, 18))
            #print( self.model_A.prev_state)
            # print("yytyyy")
        '''if len(self.model_A.prev_state) != 0:
            self.model_A.remember(self.model_A.prev_state, self.model_A.action,reward,state,done)
            self.model_A.replay()
            
        
        # if state == None:
        self.model_A.action = self.model_A.act(state)
        #print(state)
        self.model_A.prev_state = state
        #print( self.model_A.prev_state)
        
        #print(action_A)'''
        result = self.model_A.predict(state)
        return np.argmax(result[0])# Action in {0, 1, 2}

    def evaluate_B(self, info,reward,done):
        # if state == None:
        # pos = info["agentB"]["state"]
        # pos2 = info["agentA"]["state"]
        # head = info["agentB"]["head"]
        # points = self.env.live_foodspawn_space
        # points.sort(key = lambda K: (pos[0]-K[0])**2 + (pos[1]-K[1])**2)
        # # state.append(int(points[0][0]==pos2[0] and points[0][1]==pos2[1]))
        # state = [int(pos[0]==0), int(pos[0]==7), int(pos[1]==0), int(pos[1]==7), int(pos[0]>pos2[0]), int(pos[0]<pos2[0]), int(pos[1]>pos2[1]), int(pos[1]<pos2[1]), int(info["agentA"]["score"]>info["agentB"]["score"]),int(head==0), int(head==1), int(head==2), int(head==3),int(pos[0]>points[0][0]), int(pos[0]<points[0][0]), int(pos[1]>points[0][1]), int(pos[1]<points[0][1])]
        # state.append(int(((pos[0]-pos2[0])**2 + (pos[1]-pos2[1])**2) > ((pos[0]-points[0][0])**2 + (pos[1]-points[0][1])**2)))
        
       
        
        state = []
        if self.model_B.action != -1:
            pos = info["agentB"]["state"]
            pos2 = info["agentA"]["state"]
            head = info["agentB"]["head"]
            points = self.env.live_foodspawn_space
            sorted(points, key = lambda K: (pos[0]-K[0])**2 + (pos[1]-K[1])**2)
            # state.append(int(points[0][0]==pos2[0] and points[0][1]==pos2[1]))
            state = [int(pos[0]==0), int(pos[0]==7), int(pos[1]==0), int(pos[1]==7), int(pos[0]>pos2[0]), int(pos[0]<pos2[0]), int(pos[1]>pos2[1]), int(pos[1]<pos2[1]), int(info["agentA"]["score"]>info["agentB"]["score"]),int(head==0), int(head==1), int(head==2), int(head==3),int(pos[0]>points[0][0]), int(pos[0]<points[0][0]), int(pos[1]>points[0][1]), int(pos[1]<points[0][1])]
            state.append(int(((pos[0]-pos2[0])**2 + (pos[1]-pos2[1])**2) > ((pos[0]-points[0][0])**2 + (pos[1]-points[0][1])**2)))
            state = np.reshape(state, (1, self.model_B.state_space))
        
        if len(self.model_B.prev_state) != 0:
            self.model_B.remember(self.model_B.prev_state, self.model_B.action,reward,state,done)
            self.model_B.replay()
            
        
        # if state == None:
        self.model_B.action = self.model_B.act(state)
        self.model_B.prev_state = state
        
        
        #print(action_A)
        return self.model_B.action # Action in {0, 1, 2}

    def visualize(self, run):
        self.env.reset()
        done = False
        video = []
        info = {}
        rewardA = 0
        rewardB = 0
        while not done:
            # Actions based on the current state using the learned policy 
            # print(info)
            actionA = self.evaluate_A(info,rewardA,done)
            actionB = self.evaluate_B(info,rewardB,done)
            action = {'actionA': actionA, 'actionB': actionB}
            rewardA, rewardB, done, info = self.env.step(action)
            # print(type(info))
            # Rendering the enviroment to generate the simulation
            if len(self.env.history)>1:
                state = self.env.render(actionA, actionB)
                encoded, _ = self.env.encode()
                state = np.array(state, dtype=np.uint8)
                video.append(state)
        # Recording the Winner
        self.model_A.action = -1
        self.model_B.action = -1
        self.scoreA.append(self.env.agentA['score'])
        self.scoreB.append(self.env.agentB['score'])
        if self.env.agentA['score'] > self.env.agentB['score']:
            self.winner['Player_A'] += 1
        elif self.env.agentB['score'] > self.env.agentA['score']:
            self.winner['Player_B'] += 1
        # Generates a video simulation of the game
        if run%20==0:
            print("Saving Models")
            self.model_B.model.save(f"model_B_{run}",save_format='tf')
            print("Saving Video")
            aviname = os.path.join(self.checkpoint, f"game_{run}.avi")
            mp4name = os.path.join(self.checkpoint, f"game_{run}.mp4")
            w, h, _ = video[0].shape
            out = cv2.VideoWriter(aviname, cv2.VideoWriter_fourcc(*'DIVX'), 2, (h, w))
            for state in video:
                assert state.shape==(256,512,3)
                out.write(state)
            cv2.destroyAllWindows()
            os.popen("ffmpeg -i {input} {output}".format(input=aviname, output=mp4name))
            print("Getting Results")
            self.score["Agent_A"] = self.scoreA
            self.score["Agent_B"] = self.scoreB
            self.score["Average"] = (np.array(self.scoreA) + np.array(self.scoreB)) / 2.0
            print(self.score)
            plot_result(self.score, run)
            # os.popen("rm -f {input}".format(input=aviname))

    def arena(self):
        # Pitching the Agents against each other
        for run in range(1, self.runs+1, 1):
            self.visualize(run)
            print("Run: " + str(run) + " finished")
        return self.winner

if __name__ == '__main__':
    m1 = load_model(f'MODEL_A.h5')
    m1.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    m2 = DQN()
    m3 = DQN()
    obj = Runner(m1, m3, "./")
    print(obj.arena())
    