import numpy as np
import tensorflow as tf
import gym
import time
import datetime
from tqdm import tqdm
from tensorflow.keras.layers import Dense#, Flatten, Conv2D,Bidirectional, LSTM, Dropout
from tensorflow.keras import Model
import os
print(tf.__version__)

import pybullet
import reach2D

#@title Building Blocks and Probability Func{ display-mode: "form" }
EPS = 1e-8

def mlp(hidden_sizes=[32,], activation='relu', output_activation=None):
    model = tf.keras.Sequential()
    for layer_size in hidden_sizes[:-1]:
      model.add(Dense(layer_size, activation=activation))
    # Add the last layer with no activation
    model.add(Dense(hidden_sizes[-1], activation=output_activation))
    return model


def count_vars(model):
    return sum([np.prod(var.shape.as_list()) for var in model.trainable_variables])

def gaussian_likelihood(x, mu, log_std):
    
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(input_tensor=pre_sum, axis=1)

def clip_but_pass_gradient(x, l=-1., u=1.):
    clip_up = tf.cast(x > u, tf.float32)
    clip_low = tf.cast(x < l, tf.float32)
    return x + tf.stop_gradient((u - x)*clip_up + (l - x)*clip_low)
  
def apply_squashing_func(mu, pi, logp_pi):
    mu = tf.tanh(mu)
    pi = tf.tanh(pi)
    
    # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
    logp_pi -= tf.reduce_sum(input_tensor=tf.math.log(clip_but_pass_gradient(1 - pi**2, l=0, u=1) + 1e-6), axis=1)
    return mu, pi, logp_pi


    #@title Policies { display-mode: "form" }

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class mlp_gaussian_policy(Model):

  def __init__(self, act_dim, act_limit, hidden_sizes=[400,300], activation = 'relu', output_activation=None):
    super(mlp_gaussian_policy, self).__init__()
    self.mlp = mlp(list(hidden_sizes), activation, activation)
    self.mu = Dense(act_dim, activation=output_activation)
    self.log_std = Dense(act_dim, activation='tanh')
    self.act_limit = act_limit

  def call(self, inputs):
    x = self.mlp(inputs)
    mu = self.mu(x)
    """
    Because algorithm maximizes trade-off of reward and entropy,
    entropy must be unique to state---and therefore log_stds need
    to be a neural network output instead of a shared-across-states
    learnable parameter vector. But for deep Relu and other nets,
    simply sticking an activationless dense layer at the end would
    be quite bad---at the beginning of training, a randomly initialized
    net could produce extremely large values for the log_stds, which
    would result in some actions being either entirely deterministic
    or too random to come back to earth. Either of these introduces
    numerical instability which could break the algorithm. To 
    protect against that, we'll constrain the output range of the 
    log_stds, to lie within [LOG_STD_MIN, LOG_STD_MAX]. This is 
    slightly different from the trick used by the original authors of
    SAC---they used tf.clip_by_value instead of squashing and rescaling.
    I prefer this approach because it allows gradient propagation
    through log_std where clipping wouldn't, but I don't know if
    it makes much of a difference.
    """
    log_std = self.log_std(x)
    log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
    std = tf.exp(log_std)
    pi = mu + tf.random.normal(tf.shape(input=mu)) * std
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    
    
    # I suppose just put this in here as the ops would overwrite - means theres less reuse but eh, won't kill us to have a slightly different policy func for each algo. 
    mu, pi, logp_pi = apply_squashing_func(mu, pi, logp_pi)
    
    # make sure actions are in correct range
    action_scale = self.act_limit
    mu *= action_scale
    pi *= action_scale

    return mu, pi, logp_pi
  
#@title Replay Buffer { display-mode: "form" }
# End Core, begin SAC.
class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

#@title SAC Model{ display-mode: "form" }
class SAC_model():
  
  def __init__(self, env, obs_dim, act_dim, hidden_sizes,lr = 0.003,gamma = None, alpha = None, polyak = None,  load = False, exp_name = 'Exp1'):
    self.env = env
    self.pi_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    self.value_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    self.gamma = gamma
    self.alpha = alpha
    self.polyak = polyak
    self.load = load
    self.exp_name = exp_name
    self.create_networks(obs_dim, act_dim, hidden_sizes)

    
    
   
  def build_models(self,actor, q_func1, q_func2, value, value_targ, batch_size, obs_dim, act_dim):
    # run arbitrary data through the models to build them to the correct dimensions.
    actor(tf.constant(np.zeros((batch_size,obs_dim)).astype('float32')))
    q_func1(tf.constant(np.zeros((batch_size,obs_dim+act_dim)).astype('float32')))
    q_func2(tf.constant(np.zeros((batch_size,obs_dim+act_dim)).astype('float32')))
    value(tf.constant(np.zeros((batch_size,obs_dim)).astype('float32')))
    value_targ(tf.constant(np.zeros((batch_size,obs_dim)).astype('float32')))
    
    
  
    
  def create_networks(self,obs_dim, act_dim, hidden_sizes = [32,32], batch_size = 100, activation = 'relu'):
    
    # Get Env dimensions
    obs_dim = self.env.observation_space.shape[0]
    act_dim = self.env.action_space.shape[0]
    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = self.env.action_space.high[0]
    
    
    self.actor = mlp_gaussian_policy(act_dim, act_limit, hidden_sizes, activation, None)
    # Create two q functions
    self.q_func1 = mlp(hidden_sizes+[1], activation, None)
    self.q_func2 = mlp(hidden_sizes+[1], activation, None)
    # create value and target value functions
    self.value = mlp(hidden_sizes+[1], activation, None)
    
    #collect them for saving/loading
    self.models = {'actor':self.actor, 'q1':self.q_func1, 'q2':self.q_func2, 'value':self.value}
    
    self.value_targ = mlp(hidden_sizes+[1], activation, None)
  
    #build the models by passing through arbitrary data.
    self.build_models(self.actor, self.q_func1, self.q_func2, self.value, self.value_targ, batch_size, obs_dim, act_dim)
    
    
    if self.load:
      
        self.load_weights()
      
      
    # Initializing targets to match main variables
    for v_main, v_targ in zip(self.value.trainable_variables, self.value_targ.trainable_variables):
      tf.compat.v1.assign(v_targ, v_main) 
    
    # Count variables
    var_counts = tuple(count_vars(model) for model in 
                       [self.actor, self.q_func1, self.q_func2, self.value])
    print(('\nNumber of parameters: \t pi: %d, \t' + \
           'q1: %d, \t q2: %d, \t v: %d')%var_counts)
    
    
    
    
    
  @tf.function
  def train_step(self, batch):
    with tf.GradientTape() as actor_tape, tf.GradientTape() as value_tape:
      x = batch['obs1']
      x2 = batch['obs2']
      a = batch['acts']
      r = batch['rews']
      d = batch['done']

      mu, pi, logp_pi = self.actor(x)

      q1 = tf.squeeze(self.q_func1(tf.concat([x,a], axis=-1)))
      q1_pi = tf.squeeze(self.q_func1(tf.concat([x,pi], axis=-1)))
      q2 = tf.squeeze(self.q_func2(tf.concat([x,a], axis=-1)))
      q2_pi = tf.squeeze(self.q_func2(tf.concat([x,pi], axis=-1)))
      v = tf.squeeze(self.value(x))
      v_targ = tf.squeeze(self.value_targ(x2))

      # Min Double-Q:
      min_q_pi = tf.minimum(q1_pi, q2_pi)

      # Targets for Q and V regression
      q_backup = tf.stop_gradient(r + self.gamma*(1-d)*v_targ)
      v_backup = tf.stop_gradient(min_q_pi - self.alpha * logp_pi)

      # Soft actor-critic losses
      pi_loss = tf.reduce_mean(input_tensor=self.alpha * logp_pi - q1_pi)
      q1_loss = 0.5 * tf.reduce_mean(input_tensor=(q_backup - q1)**2)
      q2_loss = 0.5 * tf.reduce_mean(input_tensor=(q_backup - q2)**2)
      v_loss = 0.5 * tf.reduce_mean(input_tensor=(v_backup - v)**2)
      value_loss = q1_loss + q2_loss + v_loss

    # Policy train step
    # (has to be separate from value train step, because q1_pi appears in pi_loss)
    pi_gradients = actor_tape.gradient(pi_loss, self.actor.trainable_variables)
    self.pi_optimizer.apply_gradients(zip(pi_gradients, self.actor.trainable_variables))

    # Value train step
    value_variables = self.q_func1.trainable_variables + self.q_func2.trainable_variables + self.value.trainable_variables
    value_gradients = value_tape.gradient(value_loss, value_variables)
    #One notable byproduct of eager execution is that tf.control_dependencies() is no longer required, as all lines of code execute in order (within a tf.function, code with side effects execute in the order written).
    # Therefore, should no longer need controldependencies here. 
    self.value_optimizer.apply_gradients(zip(value_gradients, value_variables))

    # Polyak averaging for target variables
    for v_main, v_targ in zip(self.value.trainable_variables, self.value_targ.trainable_variables):
      tf.compat.v1.assign(v_targ, self.polyak*v_targ + (1-self.polyak)*v_main) 

    return pi_loss, q1_loss, q2_loss, v_loss, q1, q2, v, logp_pi
  
  
  def get_action(self, o, deterministic=False):
    mu, pi, logp_pi = self.actor(o.reshape(1,-1))
    return  mu[0] if deterministic else pi[0]
  
  
  def load_weights(self):
    try:
      print('Loading in network weights...')

      for name,model in self.models.items():
        model.load_weights('saved_models/'+self.exp_name+'/'+name+'.h5')

      print('Loaded.')
    except:
        print("Failed to load weights.")



  def save_weights(self):
      try:
          os.mkdirs('saved_models/'+self.exp_name)
      except Exception as e:
          #print(e)
          pass

      for name,model in self.models.items():
        model.save_weights('saved_models/'+self.exp_name+'/'+name+'.h5')
      print("Model Saved at ", self.exp_name)
  

 