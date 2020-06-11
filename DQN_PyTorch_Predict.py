
import gym
import sys
import os
import copy
import torch
from torch.autograd import Variable
import random
import matplotlib.pyplot as plt
from PIL import Image
#from IPython.display import clear_output
import math
import torchvision.transforms as T
import numpy as np

import time
# from TrafficLightFlow import *
from TrafficLightDoubleLane import *
from dqn_cp_pytorch import DQN, DQN_double, DQN_replay, plot_res
from flow.envs.base import Env
from flow.core.util import emission_to_csv


# Number of states

# i=0: 

def LoadModel(dqnModel, name):
    dqnModel.model.load_state_dict(torch.load(name))
    return dqnModel
    

def prediction(env: Env, numEpisodes: int, getAction: lambda state, time: int):
    """ Prediction of a pytorch model"""
     # used to store
    info_dict = {
        "returns": [],
        "velocities": [],
        "outflows": [],
    }

    final = []
    for episode in range(numEpisodes):
        state = env.reset()
        done = False
        tls_state = []
        total = 0
        t = 0
        vel = []
        while not done:
            t+=1
            action = getAction(state, t)
            
            # Take action and extract results
            next_state, reward, done, _ = env.step(action)
            state = next_state
            
            # Update reward
            total += reward
            id = env.k.traffic_light.get_ids()[0]
            tls_state.append(env.k.traffic_light.get_state(id))
            
            
            # Compute the velocity speeds and cumulative returns.
            veh_ids = env.k.vehicle.get_ids()
            vel.append(np.mean(env.k.vehicle.get_speed(veh_ids)))
            
            if done:
                break
        # Store the information from the run in info_dict.
        outflow = env.k.vehicle.get_outflow_rate(int(500))
        info_dict["returns"].append(total)
        info_dict["velocities"].append(np.mean(vel))
        info_dict["outflows"].append(outflow)
        print(f'TLS State: {tls_state}')

        # Add to the final reward
        final.append(total)
        print("episode: {}, total reward: {}".format(episode, total))
        
        #plot_res(final,'Prediction')

    # Print the averages/std for all variables in the info_dict.
    for key in info_dict.keys():
        print("Average, std {}: {}, {}".format(
            key, np.mean(info_dict[key]), np.std(info_dict[key])))
    
    ConvertToCSV(env)

    return final

def staticBaselinePrediction(state, t):
    return t%20 == 0

def ConvertToCSV(env):
    dir_path = env.sim_params.emission_path
    emission_filename = \
        "{0}-emission.xml".format(env.network.name)
    emission_path = os.path.join(dir_path, emission_filename)
    # convert the emission file into a csv
    emission_to_csv(emission_path)

def modelBasedPrediction(model):
    def getAction(state, t): 
        q_values = model.predict(state)
        return torch.argmax(q_values).item()

    return getAction

def getDQNModel(env):  
    n_state = env.observation_space.shape[0]
    # Number of actions
    n_action = env.action_space.n
    # Number of hidden nodes in the DQN
    n_hidden = 50
    # Learning rate
    lr = 0.002
    dqn_model = DQN_double(n_state, n_action, n_hidden, lr)
    return LoadModel(dqn_model, 'DQL_Replay_dense_500.pkl')

if __name__ == "__main__":

    num_episodes = 1
    InFlowProbs = [(0.05, 0.05,0.15,0.15), (0.1, 0.1,0.1,0.1), (0.15, 0.1,0.15,0.1)]
    render = True
    if sys.argv[1] == 'baseline':
        print('Running Traffic Flow using Baseline static actions')
        
        env = GetTrafficLightEnv(InFlowProbs[0],render=render,evaluate=False)
        prediction(env, num_episodes, staticBaselinePrediction)
        # for inflowProbs in InFlowProbs:
        #     env = GetTrafficLightEnv(inflowProbs,render=render,evaluate=False)
        #     prediction(env, num_episodes, staticBaselinePrediction)
    else:
        print('Running prediction using model')
        #for inflowProbs in InFlowProbs:
        env = GetTrafficLightEnv(inflowProbs,render=render,evaluate=False)
        dqn_model = getDQNModel(env)
        prediction(env, num_episodes, modelBasedPrediction(dqn_model))



