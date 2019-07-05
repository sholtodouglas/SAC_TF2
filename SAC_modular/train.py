import numpy as np
import tensorflow as tf
import gym
import time
import pybullet
import reach2D
from SAC import *

# Have this as our actual loop.
def sac(env_fn,  ac_kwargs=dict(), seed=0, 
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
  train_log_dir = 'logs/gradient_tape/' + exp_name+':'+str(start_time) + '/stochastic'
  train_summary_writer = tf.summary.create_file_writer(train_log_dir)

 
  def test_agent(n=10):
        for j in range(n):
            test_env.render(mode='human')
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            test_env.camera_adjust()
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time 
                o, r, d, _ = test_env.step(SAC.get_action(o, True))
                ep_ret += r
                ep_len += 1
            with train_summary_writer.as_default():
              tf.summary.scalar('test_episode_return', ep_ret, step=t+j)

  if render:
    
    env.render(mode='human')
  o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
  env.camera_adjust()
  total_steps = steps_per_epoch * epochs

  


  # Main loop: collect experience in env and update/log each epoch
  for t in range(total_steps):

      """
      Until start_steps have elapsed, randomly sample actions
      from a uniform distribution for better exploration. Afterwards, 
      use the learned policy. 
      """
      if t > start_steps:
          a = SAC.get_action(o)
      else:
          a = env.action_space.sample()

      # Step the env
      o2, r, d, _ = env.step(a)
      ep_ret += r
      ep_len += 1

      # Ignore the "done" signal if it comes from hitting the time
      # horizon (that is, when it's an artificial terminal signal
      # that isn't based on the agent's state)
      d = False if ep_len==max_ep_len else d

      # Store experience to replay buffer
      replay_buffer.store(o, a, r, o2, d)

      # Super critical, easy to overlook step: make sure to update 
      # most recent observation!
      o = o2

      if d or (ep_len == max_ep_len):
          """
          Perform all SAC updates at the end of the trajectory.
          This is a slight difference from the SAC specified in the
          original paper.
          """
          for j in range(ep_len):
              batch = replay_buffer.sample_batch(batch_size)

              LossPi, LossQ1, LossQ2, LossV, Q1Vals, Q2Vals, VVals, LogPi = SAC.train_step(batch)

              # logger.store(LossPi=LossPi, LossQ1=LossQ1, LossQ2=LossQ2,
              #              LossV=LossV, Q1Vals=Q1Vals, Q2Vals=Q2Vals,
              #              VVals=VVals, LogPi=LogPi)

          # logger.store(EpRet=ep_ret, EpLen=ep_len)
          with train_summary_writer.as_default():
            print('Frame: ', t, ' Return: ', ep_ret)
            tf.summary.scalar('episode_return', ep_ret, step=t)

          if render:
            env.camera_adjust()
            env.render(mode='human')
          o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
          env.camera_adjust()


      # End of epoch wrap-up
      if t > 0 and t % steps_per_epoch == 0:
          epoch = t // steps_per_epoch


          SAC.save_weights()

          # Test the performance of the deterministic version of the agent.
          test_agent()





# MODIFIABLE VARIBALES TODO PROPERLY PUT THIS IN A CLASS
ENV_NAME='reacher2D-v0'
hid =128
l=2
gamma=0.999
seed=0
epochs=50
max_ep_len = 200 # for reacher, 1000 for not. 
experiment_name = ENV_NAME+'_Hidden_'+str(hid)+'l_'+str(l)


sac(lambda : gym.make(ENV_NAME), 
    ac_kwargs=dict(hidden_sizes=[hid]*l),
    gamma=gamma, seed=seed, epochs=epochs, load = True, exp_name = experiment_name, max_ep_len = max_ep_len, render = False)