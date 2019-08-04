import gym
import reach2D
import numpy as np
from SAC import *
from train import *
import pointMass


ENV_NAME = 'pointMass-v0'#'reacher2D-v0'

#ENV_NAME = 'Pendulum-v0'
env = gym.make(ENV_NAME) 
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
#experiment_name = 'seg_reacher2D-v0_Hidden_128l_2'
#experiment_name = 'pos_cntrl_seg_reacher2D-v0_Hidden_128l_2'
#experiment_name = 'pos_cntrl_exp_pointMass-v0_Hidden_128l_2'
#experiment_name = 'no_reset_vel_pointMass-v0_Hidden_128l_2'
experiment_name = 'GAILpointMass-v0_Hidden_128l_2'
#experiment_name = 'GAILpointMass-v0_Hidden_128l_2'
#experiment_name = 'GAILreacher2D-v0_Hidden_128l_2'
env.activate_movable_goal()

SAC = SAC_model(env, obs_dim, act_dim, [128,128],load = True, exp_name = experiment_name)

rollout_trajectories(n_steps = 10000,env = env, max_ep_len = 10000 , actor = SAC.actor.get_deterministic_action, train = False, render = True, collect_trajectories = True, exp_name = experiment_name)


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