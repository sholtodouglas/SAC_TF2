import numpy as np
import tensorflow as tf
import gym
import time
import pybullet
import reach2D
from SAC import *


# collects a n_steps steps for the replay buffer.
def rollout_trajectories(n_steps,env, max_ep_len = 200, actor = None, replay_buffer = None, summary_writer = None, current_total_steps = 0, render = False, train = True, collect_trajectories = False, exp_name = None):

  # reset the environment


  ###################  quick fix for the need for this to activate rendering pre env reset.  ################### 
   ###################  MUST BE A BETTER WAY? Env realising it needs to change pybullet client?  ################### 
  if 'reacher'  in exp_name:
    pybullet = True
  else:
    pybullet = False

  if pybullet:
    if render:
      # have to do it beforehand to connect up the pybullet GUI
      env.render(mode='human')

  ###################  ###################  ###################  ###################  ################### 


  o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

  # if we want to store expert actions
  if collect_trajectories:
    actions = []
    observations = []

  for t in range(n_steps):


      if actor == 'random':
        a = env.action_space.sample()
      else:
        a = actor(o)
      # Step the env
      o2, r, d, _ = env.step(a)
      
      if render:
        env.render(mode='human')
      if collect_trajectories:
        actions.append(a)
        observations.append(o)

      ep_ret += r
      ep_len += 1

      # Ignore the "done" signal if it comes from hitting the time
      # horizon (that is, when it's an artificial terminal signal
      # that isn't based on the agent's state)
      d = False if ep_len==max_ep_len else d

      # Store experience to replay buffer
      if train:
        replay_buffer.store(o, a, r, o2, d)

      # Super critical, easy to overlook step: make sure to update 
      # most recent observation!
      o = o2
      # if either we've ended an episdoe, collected all the steps or have reached max ep len and 
      # thus need to log ep reward and reset
      if d or (ep_len == max_ep_len) or (t == (n_steps-1)):
          if summary_writer:
            with summary_writer.as_default():
              if train:
                print('Frame: ', t+current_total_steps, ' Return: ', ep_ret)
                tf.summary.scalar('Episode_return', ep_ret, step=t+current_total_steps)
              else:
                print('Test Frame: ', t+current_total_steps, ' Return: ', ep_ret)
                tf.summary.scalar('Test_Episode_return', ep_ret, step=t+current_total_steps)
          # reset the env if there are still steps to collect
          if t < n_steps -1:
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
          # else it will be the end of the loop and of the function. 


  if collect_trajectories:
    np.save('collected_data/expert_actions_'+exp_name+str(n_steps),np.array(actions))
    np.save('collected_data/expert_obs_'+exp_name+str(n_steps),np.array(observations))


  return n_steps



##########################################################################################################################################################################################################################################

# This is our training loop.
def training_loop(env_fn,  ac_kwargs=dict(), seed=0, 
        steps_per_epoch=5000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=5000, 
        max_ep_len=1000, save_freq=1, load = False, exp_name = "Experiment_1", render = False):
    
  tf.random.set_seed(seed)
  np.random.seed(seed)
  env, test_env = env_fn(), env_fn()
  # Get Env dimensions
  obs_dim = env.observation_space.shape[0]
  act_dim = env.action_space.shape[0]
  SAC = SAC_model(env, obs_dim, act_dim, ac_kwargs['hidden_sizes'],lr, gamma, alpha, polyak,  load, exp_name)
  # Experience buffer
  replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
  
  #Logging 
  start_time = time.time()
  train_log_dir = 'logs/sub/' + exp_name+':'+str(start_time) + '/stochastic'
  summary_writer = tf.summary.create_file_writer(train_log_dir)

  def update_models(model, replay_buffer, steps, batch_size):
    for j in range(steps):
        batch = replay_buffer.sample_batch(batch_size)
        LossPi, LossQ1, LossQ2, LossV, Q1Vals, Q2Vals, VVals, LogPi = model.train_step(batch)

  # now collect epsiodes
  total_steps = steps_per_epoch * epochs
  steps_collected = 0

  # collect some initial random steps to initialise
  random_steps = 5000
  steps_collected  += rollout_trajectories(n_steps = random_steps,env = env, max_ep_len = max_ep_len, actor = 'random', replay_buffer = replay_buffer, summary_writer = summary_writer, exp_name = exp_name)
  update_models(SAC, replay_buffer, steps = random_steps, batch_size = batch_size)

  # now act with our actor, and alternately collect data, then train.
  while steps_collected < total_steps:
    # collect an episode
    steps_collected  += rollout_trajectories(n_steps = max_ep_len,env = env, max_ep_len = max_ep_len, actor = SAC.get_action, replay_buffer = replay_buffer, summary_writer=summary_writer, current_total_steps = steps_collected, exp_name = exp_name)
    # take than many training steps
    update_models(SAC, replay_buffer, steps = max_ep_len, batch_size = batch_size)

    # if an epoch has elapsed, save and test.
    if steps_collected  > 0 and steps_collected  % steps_per_epoch == 0:
        SAC.save_weights()
        # Test the performance of the deterministic version of the agent.
        rollout_trajectories(n_steps = max_ep_len*10,env = test_env, max_ep_len = max_ep_len, actor = SAC.get_deterministic_action, summary_writer=summary_writer, current_total_steps = steps_collected, train = False, render = True, exp_name = exp_name)






# MODIFIABLE VARIBALES TODO PROPERLY PUT THIS IN A CLASS
#ENV_NAME='reacher2D-v0'
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Pendulum-v0')
    parser.add_argument('--hid', type=int, default=128)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--max_ep_len', type=int, default=200)
    parser.add_argument('--exp_name', type=str, default='experiment_1')
    parser.add_argument('--load', type=bool, default=False)
    parser.add_argument('--render', type=bool, default=False)
    args = parser.parse_args()

    experiment_name = args.env+'_Hidden_'+str(args.hid)+'l_'+str(args.l)

    training_loop(lambda : gym.make(args.env), 
      ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
      gamma=args.gamma, seed=args.seed, epochs=args.epochs, load = args.load, exp_name = experiment_name, max_ep_len = args.max_ep_len, render = args.render)



# we want to try goal conditioned GAIL on 2D reacher. 


