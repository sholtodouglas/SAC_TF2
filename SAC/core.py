import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense#, Flatten, Conv2D,Bidirectional, LSTM, Dropout
import os

# Core Functions
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


"""
Policies
"""

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
  
 
def build_models(actor, q_func1, q_func2, value, value_targ, batch_size, obs_dim, act_dim):
    # run arbitrary data through the models to build them to the correct dimensions.
    actor(tf.constant(np.zeros((batch_size,obs_dim)).astype('float32')))
    q_func1(tf.constant(np.zeros((batch_size,obs_dim+act_dim)).astype('float32')))
    q_func2(tf.constant(np.zeros((batch_size,obs_dim+act_dim)).astype('float32')))
    value(tf.constant(np.zeros((batch_size,obs_dim)).astype('float32')))
    value_targ(tf.constant(np.zeros((batch_size,obs_dim)).astype('float32')))
    
    # Initializing targets to match main variables
    for v_main, v_targ in zip(value.trainable_variables, value_targ.trainable_variables):
      tf.compat.v1.assign(v_targ, v_main) 
  


def load_weights(models, exp_name):
    try:
      print('Loading in network weights...')

      for name,model in models.items():
        model.load_weights('saved_models/'+exp_name+'/'+name+'.h5')

      print('Loaded.')
    except:
        print("Failed to load weights.")



def save_weights(models, exp_name):
      try:

          os.mkdir('saved_models/'+exp_name)
      except Exception as e:
          print(e)
          pass

      for name,model in models.items():
        model.save_weights('saved_models/'+exp_name+'/'+name+'.h5')
      print("Model Saved at ", exp_name)
  
