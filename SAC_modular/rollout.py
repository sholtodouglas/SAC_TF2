import gym
import reach2D
import numpy as np
from SAC import *

def rollout(test_env, algorithm, max_ep_len = 5000):
  env.render(mode='human')

  o,r, d, ep_len = test_env.reset(), 0, False, 0
  env.activate_movable_goal()
  env.camera_adjust()
  actions = []
  observations = []
  while not(d or (ep_len == max_ep_len)):
      # Take deterministic actions at test time 
      a = algorithm.get_action(o, True)
      o, r, d, _ = test_env.step(a)
      test_env.render(mode='human')
      ep_len += 1
      actions.append(a)
      observations.append(o)
  np.save('saved_models/expert_actions'+str(max_ep_len),np.array(actions))
  np.save('saved_models/expert_obs'+str(max_ep_len),np.array(observations))
      

ENV_NAME = 'reacher2D-v0'
env = gym.make(ENV_NAME) #wrapping the env to render as a video
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
experiment_name = 'reacher2D-v0_Hidden_128l_2'
SAC = SAC_model(env, obs_dim, act_dim, [128,128],load = True, exp_name = experiment_name)

  
rollout(env, SAC)