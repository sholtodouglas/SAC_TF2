import numpy as np
import tensorflow as tf
import gym
import time
import pybullet
import reach2D
from SAC import *
from train import *


observations= np.load('collected_data/expert_obs_Pendulum-v0_Hidden_32l_21000.npy').astype('float32')
actions = np.load('collected_data/expert_actions_Pendulum-v0_Hidden_32l_21000.npy').astype('float32')

train_length = int(0.9*(len(observations)))
print(train_length)
train_obs = observations[:train_length,:]
train_acts = actions[:train_length,:]
valid_obs = observations[train_length:,:]
valid_acts = actions[train_length:,:]

print(train_obs.shape)
print(valid_obs.shape)

#act_limit = 1.0 # for reacher env
act_limit = 2.0 # for Pendulum
start_time = time.time()
train_log_dir = 'logs/'+ 'BC:'+str(start_time)
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
policy  = mlp_gaussian_policy(act_dim = actions.shape[1], act_limit = act_limit, hidden_sizes = [128,128])
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
max_ep_len = 200

ENV_NAME='Pendulum-v0'
env = gym.make(ENV_NAME)

# do like this so we can sub into our trajectory rollerouter.
def get_action(o):
	
	mu,_,_ = policy(o.reshape(1,-1))
	
	return mu[0]


# Behavioural clone this mf.
@tf.function
def train_step(obs, expert_act):
    with tf.GradientTape() as tape:
        mu,_,_ = policy(obs)
        
        #print(mu.shape)
        BC_loss = tf.reduce_sum(tf.losses.MSE(mu, expert_act))
        #print(BC_loss)
        
    BC_gradients = tape.gradient(BC_loss, policy.trainable_variables)
    optimizer.apply_gradients(zip(BC_gradients, policy.trainable_variables))
    return BC_loss
    
    
def test_step(obs,expert_act):
  
    mu,_,_ = policy(obs)

    #print(mu.shape)
    BC_loss = tf.reduce_sum(tf.losses.MSE(mu, expert_act))
    return BC_loss

# Training Loop
train_steps = 100000
batch_size = 512


  
for t in range(train_steps):
    indexes = np.random.choice(train_obs.shape[0], 512)
    batch_obs = train_obs[indexes, :]
    batch_acts = train_acts[indexes,:]
    BC_loss = train_step(batch_obs, batch_acts)
    with train_summary_writer.as_default():
         tf.summary.scalar('BC_MSE_loss',BC_loss, step=t)
        
    if t % 1000 ==0:
      rollout_trajectories(n_steps = max_ep_len,env = env, max_ep_len = max_ep_len, actor = get_action, summary_writer=train_summary_writer, current_total_steps = t,train = False, render = False)
      
      indexes = np.random.choice(valid_obs.shape[0], 512)
      batch_obs = valid_obs[indexes, :]
      batch_acts = valid_acts[indexes,:]
      l = test_step(batch_obs, batch_acts)
      print(t,l)
      with train_summary_writer.as_default():
         tf.summary.scalar('validation_BC_MSE_loss',l, step=t)
