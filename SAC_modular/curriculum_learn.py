# In this file we will do HER, but with a curriculum as our start states.
# How best to implement? Every second run - init somewhere along your demo states?
# in that case each of our RL envs will needs a function, init with observation vector, init with goal.
# first thing to attack is 'set init function, making that general'.
import pybullet as p
import os
import math
import pickle
import socket
import time
import gym 
import ur5_RL
from tqdm import tqdm
import numpy as np
from natsort import natsorted, ns

env = gym.make('ur5_RL_objects-v0')
env.render(mode='human')
env.reset()


path= '../../ur5_RL/ur5_RL/envs/play_data/set_9'

# collect all file paths
moments = [x[0] for x in os.walk(path)][1:]
# sort them according to timestamp
moments = natsorted(moments, alg=ns.IGNORECASE)

def load_data_into_memory(moments):

    observations = []
    actions  = []
    imgs = []

    for s in tqdm(moments):
        act = np.load(s+'/act.npy')
        actions.append(act)
        obs = np.load(s+'/obs.npy')
        observations.append(obs)
        
        
    return np.array(observations).astype(float), np.array(actions).astype(float)

observations, actions = load_data_into_memory(moments)

while(1):
	env.reset()
	index= np.random.randint(0,len(observations))
	env.initialize_start_pos(observations[index])
	#env.reset_goal_pos(observations[index+50][19:22])
	time.sleep(0.5)
	pass