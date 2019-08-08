import gym
import reach2D
import numpy as np
from SAC import *
from train import *
import pointMass
from gym import wrappers

def episode_to_trajectory(episode, include_goal = False, flattened = False):
	# episode arrives as a list of o, a, r, o2, d
	# trajectory is two lists, one of o s, one of a s. 
	observations = []
	actions = []
	for transition in episode:
		o, a, r, o2, d = transition
		if flattened:
			observations.append(o)
		else:
			if include_goal:
				observations.append(np.concatenate(o['observation'], o['desired_goal']))
			else:
				observations.append(o['observation'])
		actions.append(a)

	return np.array(observations), np.array(actions)



ENV_NAME = 'pointMass-v0'#'reacher2D-v0'

#ENV_NAME = 'Pendulum-v0'
env = gym.make(ENV_NAME) 

env = wrappers.FlattenDictWrapper(env, dict_keys=['observation', 'desired_goal'])

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
#experiment_name = 'seg_reacher2D-v0_Hidden_128l_2'
#experiment_name = 'pos_cntrl_seg_reacher2D-v0_Hidden_128l_2'
#experiment_name = 'pos_cntrl_exp_pointMass-v0_Hidden_128l_2'
#experiment_name = 'no_reset_vel_pointMass-v0_Hidden_128l_2'
experiment_name = 'HER_pointMass-v0_Hidden_128l_2'
#experiment_name = 'GAILpointMass-v0_Hidden_128l_2'
#experiment_name = 'GAILreacher2D-v0_Hidden_128l_2'
env.activate_movable_goal()


SAC = SAC_model(env, obs_dim, act_dim, [128,128],load = True, exp_name = experiment_name)

episodes, n_steps = rollout_trajectories(n_steps = 10000,env = env, max_ep_len = 10000 , actor = SAC.actor.get_deterministic_action, train = False, render = True, exp_name = experiment_name, return_episode = True)

action_buff = []
observation_buff = []
for ep in episodes:
	observations, actions = episode_to_trajectory(ep, flattened = True)
	action_buff.append(actions)
	observation_buff.append((observations))
np.save('collected_data/'+str(n_steps)+experiment_name+'expert_actions',np.concatenate(action_buff))
np.save('collected_data/'+str(n_steps)+experiment_name+'expert_obs_',np.concatenate(observation_buff))

# if train encoder z = enc(T) - train with policy reco loss.
# then we can do trajectory based GAIL
# f(T|Z)
# but then we need something picking a desired z. 
# hindsight wise with a batch - okay you made this trajectory, well with that z then thats expert. 
# could we do play data in GAIL?

############ OBSERVATION FORMAT #############
# finger_tip_pose[0], #fingertip x
# finger_tip_pose[1], #fingertip y
# target_x,
# target_y,
# self.to_target_vec[0],
# self.to_target_vec[1],
# np.cos(theta),
# np.sin(theta),
# self.theta_dot,
# self.gamma,
# self.gamma_dot,