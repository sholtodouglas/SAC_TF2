from common import *
from latent import *
import tensorflow as tf
import gym
import pointMass
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
plt.figure()
import numpy as np

tfd = tfp.distributions

LAYER_SIZE = 128
LATENT_DIM = 6
P_DROPOUT = 0.2
ACHEIVED_GOAL_INDEX = 2  # point up to which we care about the goal

MAX_SEQ_LEN = 60
OBS_GOAL_INDEX = 4
#TODO FIX episode return

env = gym.make('pointMass-v0')

idx = 4000
length = 70#MAX_SEQ_LEN
REPLAN_HORIZON = 70
observations= np.load(drive_path+'collected_data/10000GAILpointMass-v0_Hidden_128l_2expert_obs_.npy').astype('float32')[:,0:OBS_GOAL_INDEX] # Don't include the goal in the obs
actions = np.load(drive_path+'collected_data/10000GAILpointMass-v0_Hidden_128l_2expert_actions.npy').astype('float32')
OBS_DIM = observations.shape[1]
ACT_DIM = actions.shape[1]

#extension = 'saved_models/Z_learning_0.01'
extension = 'saved_models/Z_learning_0.005enc2plan6'

encoder = TRAJECTORY_ENCODER_LSTM(LAYER_SIZE, LATENT_DIM, P_DROPOUT)
actor = ACTOR(LAYER_SIZE, ACT_DIM, P_DROPOUT)
planner = PLANNER(LAYER_SIZE, LATENT_DIM, P_DROPOUT)
encoder, actor, planner = load_weights(extension, 8, MAX_SEQ_LEN, OBS_DIM, ACT_DIM,ACHEIVED_GOAL_INDEX, LATENT_DIM,  encoder, actor, planner)



trajectory_obs, trajectory_acts = np.expand_dims(observations[idx:idx+length], 0), np.expand_dims(actions[idx:idx+length], 0)
mu_enc, s_enc = encoder(trajectory_obs[:,::2,:], trajectory_acts[:,::2,:])
encoder_normal = tfd.Normal(mu_enc,s_enc)
z = tf.squeeze(mu_enc) #tf.squeeze(encoder_normal.sample())
s_g = tf.squeeze(trajectory_obs[:,-1,:ACHEIVED_GOAL_INDEX])
s_i = tf.squeeze(trajectory_obs[:,0,:])

paths = []
#s_i = tf.constant([0.0,0.0,0.0,0.0])
for i in range(0,10):
    #s_g = tf.constant((np.random.rand(2)*6 - 3).astype('float32'))
    mu_plan, s_plan = planner(tf.expand_dims(s_i, axis=0), tf.expand_dims(s_g, axis=0))
    planner_normal = tfd.Normal(mu_plan, s_plan)
    #plan = tf.squeeze(planner_normal.sample()) * (np.random.rand(1)*3-1)
    plan = tf.squeeze(encoder_normal.sample())

    # start from the middle, random plans.
    #plan = tf.squeeze(np.random.rand(12).astype('float32') * 20 - 1)



    episodes, n_steps = rollout_trajectories(REPLAN_HORIZON, env, max_ep_len=REPLAN_HORIZON, actor=actor.get_deterministic_action,
                                             exp_name='point', z=plan, s_g=s_g, start_state=s_i, return_episode=True,
                                             goal_based=True, render =True)
    planner_obs, planner_acts = episode_to_trajectory(episodes['episodes'][0], representation_learning=True)

    #s_i = tf.squeeze(planner_obs[REPLAN_HORIZON//5,:])
    #plt.scatter(np.squeeze(planner_obs)[:REPLAN_HORIZON//5, 0], np.squeeze(planner_obs)[:REPLAN_HORIZON//5, 1], s=10)
    #s_i = tf.constant([0.0,0.0,0.0,0.0])


    paths.append(planner_obs)

plt.scatter(np.squeeze(trajectory_obs)[:,0], np.squeeze(trajectory_obs)[:,1], s= 25, c= 'black' )
for path in paths:
   plt.scatter(np.squeeze(path)[:, 0], np.squeeze(path)[:, 1], s = 10)
plt.scatter(np.array([s_g[0]]),np.array([s_g[1]]),s =20, marker = 'x')
plt.show()




