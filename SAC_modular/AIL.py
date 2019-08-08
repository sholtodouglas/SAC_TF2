import numpy as np
import tensorflow as tf
import gym
import time
import pybullet
import reach2D
from SAC import *
from train import *
from GAIL import discriminator_train_step

train_test_split = 0.9
MAX_SEQ_LEN = 30
MIN_SEQ_LEN = 16
# This could mess us up if we have an LSTM based actor?
FRAME_SKIP = 2 # frame interval to decimate sequences at. cant frame skip if we want to do determin
BATCH_SIZE = 256
LAYER_SIZE = 128 
LATENT_DIM = 12
in_dim = OBS_DIM*2+LATENT_DIM #s, z, s_g.
P_DROPOUT = 0.2
BETA = 0.05
OBS_GOAL_INDEX = 4 # index from which the goal is in the obs vector
EPOCHS = 1000
extension = 'Z_learning_B005'
observations= np.load(drive_path+'collected_data/10000GAILpointMass-v0_Hidden_128l_2expert_obs_.npy').astype('float32')[:,0:OBS_GOAL_INDEX] # Don't include the goal in the obs
actions = np.load(drive_path+'collected_data/10000GAILpointMass-v0_Hidden_128l_2expert_actions.npy').astype('float32')
OBS_DIM = observations.shape[1]
ACT_DIM = actions.shape[1]
train_len = int(len(observations)*train_test_split)
train_obs_subset = observations[:train_len,:]
train_acts_subset = actions[:train_len,:]
valid_obs_subset = observations[train_len:,:]
valid_acts_subset = actions[train_len:,:]
train_len = len(train_obs_subset)-MAX_SEQ_LEN*FRAME_SKIP
valid_len = len(valid_obs_subset)-MAX_SEQ_LEN*FRAME_SKIP
################################################################################################################################################################
												# DATASET 
####################################################################################################################################################################################
def data_generator(actions,subset):

      if subset == b'Train':
            set_len = train_len
            obs_set = train_obs_subset
            act_set = train_acts_subset
      if subset == b'Valid':
            set_len = valid_len
            obs_set = valid_obs_subset
            act_set = valid_acts_subset
            
      for idx in range(0, set_len):
          # yield the observation randomly between min and max sequence length.
          length = np.random.randint(MIN_SEQ_LEN*FRAME_SKIP, (MAX_SEQ_LEN*FRAME_SKIP))
          
          if length % 2 != 0:
            length -= 1
          
          
          obs_padding = np.zeros((MAX_SEQ_LEN-length//FRAME_SKIP, OBS_DIM))

          
          
          padded_obs = np.concatenate((obs_set[idx:idx+length:FRAME_SKIP], obs_padding), axis = 0)

          act_padding = np.zeros((MAX_SEQ_LEN-length//FRAME_SKIP, ACT_DIM))
          padded_act = np.concatenate((act_set[idx:idx+length:FRAME_SKIP], act_padding), axis = 0)

          # ones to length of actions, zeros for the rest to mask out loss. 
          mask = np.concatenate((np.ones((length//FRAME_SKIP, ACT_DIM)),act_padding), axis = 0 ) 
          
          if len(padded_obs) != MAX_SEQ_LEN:
            print(idx, length, len(padded_obs))
            
          if len(padded_act) != MAX_SEQ_LEN:
            print(idx, length, len(padded_act))
         

          yield (padded_obs, padded_act, mask, length//FRAME_SKIP)

dataset = tf.data.Dataset.from_generator(data_generator, (tf.float32, tf.float32, tf.float32, tf.int32), args = (actions, 'Train'))
valid_dataset = tf.data.Dataset.from_generator(data_generator, (tf.float32, tf.float32, tf.float32, tf.int32), args = (actions, 'Valid'))


MULTIPLIER = 2 # number more sequences that we analyse than we expressly ned for a litte more diversity
min_seqs = BATCH_SIZE/((MIN_SEQ_LEN+MAX_SEQ_LEN)/MULTIPLIER) # on average the number of sequences we will need to draw to get batch_num transitions
seqs = int(min_seqs*2) # get 2x that number so we have a little more diversity.
expert_set = iter(dataset.shuffle(train_len).batch(seqs).repeat(50)) # unlikely to go through 50 iterations of the dataset, if we do reset it!
########################################################################################################################################################################################################
										#Models
########################################################################################################################################################################################################


#@title Models
class TRAJECTORY_ENCODER_LSTM(Model):
  def __init__(self, LAYER_SIZE, LATENT_DIM, P_DROPOUT):
    super(TRAJECTORY_ENCODER_LSTM, self).__init__()

    self.bi_lstm = Bidirectional(CuDNNLSTM(LAYER_SIZE, return_sequences=True), merge_mode=None)
    self.mu = Dense(LATENT_DIM)
    self.scale = Dense(LATENT_DIM, activation='softplus')
    self.dropout1 = tf.keras.layers.Dropout(P_DROPOUT)

  def call(self, obs, acts, training = False):
    x = tf.concat([obs,acts], axis = 2) # concat observations and actions together.
    x = self.bi_lstm(x)
    x = self.dropout1(x, training=training)
    bottom = x[0][:,-1, :] # Take the last element of the bottom row
    top = x[1][:,0,:] # Take the first elemetn of the top row cause Bidirectional, top row goes backward.
    x = tf.concat([bottom, top], axis = 1)
    mu = self.mu(x)
    s = self.scale(x)

    return mu, s
  
  
# this actor function is more complex than necessary because it selves dual purpose
# as an actor here, learnt with supervised learning and also we want to be able to directly pass
# it into our SAC implementation. 
LOG_STD_MAX = 2
LOG_STD_MIN = -20
class ACTOR(Model):
  def __init__(self, LAYER_SIZE, ACT_DIM, P_DROPOUT):
    super(ACTOR, self).__init__()
    self.l1 = Dense(LAYER_SIZE, activation = 'relu', name = 'layer1')
    self.l2 = Dense(LAYER_SIZE, activation = 'relu', name = 'layer2')
    self.mu =  Dense(ACT_DIM, name='mu')
    self.log_std = Dense(ACT_DIM, activation='tanh', name = 'log_std')
    self.dropout1 = tf.keras.layers.Dropout(P_DROPOUT)
    
    
  def gaussian_likelihood(self,x, mu, log_std):
    EPS = 1e-8
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(input_tensor=pre_sum, axis=1)

  def clip_but_pass_gradient(self,x, l=-1., u=1.):
    clip_up = tf.cast(x > u, tf.float32)
    clip_low = tf.cast(x < l, tf.float32)
    return x + tf.stop_gradient((u - x)*clip_up + (l - x)*clip_low)

  def apply_squashing_func(self,mu, pi, logp_pi):
    # TODO: Tanh makes the gradients bad - we don't necessarily wan't to tanh these - lets confirm later
    # 
    mu = tf.tanh(mu)
    pi = tf.tanh(pi)

    # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
    logp_pi -= tf.reduce_sum(input_tensor=tf.math.log(self.clip_but_pass_gradient(1 - pi**2, l=0, u=1) + 1e-6), axis=1)
    return mu, pi, logp_pi
    
  def call(self, s, z = None, s_g = None, training= False):
    
    # check if the user has fed in z and s_g
    # if not, means they're passing s,z,s_g as one vector through s - this will be in the case of this
    # actor being used in our typical RL algorithms
    if z != None and s_g != None: 
      B = z.shape[0] #dynamically get batch size
      if len(s.shape) == 3:
        x = tf.concat([s, z, s_g], axis = 2) # (BATCHSIZE)
      else:
        x = tf.concat([s, z, s_g], axis = 0) # (BATCHSIZE,  OBS+OBS+LATENT)
        x= tf.expand_dims(x, 0) # make it (1, OBS+OBS+LATENT)
    else:
      x = s
      
      
      
    x = self.l1(x)
    x = self.dropout1(x, training = training)
    x = self.l2(x)
    mu = self.mu(x)
    log_std = self.log_std(x)
    log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
    std = tf.exp(log_std)
    
    pi = mu + tf.random.normal(tf.shape(input=mu)) * std
    logp_pi = self.gaussian_likelihood(pi, mu, log_std)
    
    # Equivalent to this, should we change over to TFP properly at some point?
    # For some reason it doesn't work as well on pendulum. Weird.
    pdf = tfd.Normal(loc=mu,scale=std)
    # logp_pi = tf.reduce_sum(pdf.log_prob(pi))
    
    mu, pi, logp_pi = self.apply_squashing_func(mu, pi, logp_pi)
    return mu, pi, logp_pi, std, pdf
  
  def get_deterministic_action(self, o):
    # o should be s,z,s_g concatted together
    o = tf.expand_dims(o, axis = 0)
    mu,_,_,_,_ = self.call(o)
  
    return mu[0]
  
  def get_stochastic_action(self, o):
    o = tf.expand_dims(o, axis = 0)
    _,pi,_,_,_ = self.call(o)
  
    return pi[0]
    
  
  
encoder = TRAJECTORY_ENCODER_LSTM(LAYER_SIZE, LATENT_DIM, P_DROPOUT)
actor = ACTOR(LAYER_SIZE, ACT_DIM, P_DROPOUT)
load_weights(extension)
########################################################################################################################################################################################################



def assign_variables(net1, net2):
  for main, targ in zip(net1.trainable_variables, net2.trainable_variables):
    tf.compat.v1.assign(targ, main) 


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
        self.demo_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done, demo_next_obs = None):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.demo_buf[self.ptr] = demo_next_obs
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs],
                    demo=self.demo_buf[idxs],
                    idxs=idxs)

    def purge(self,idxs, batch_size):
        # purge transitions which are very obviously generated.
        # only purge if we have enough that we won't go under a batch size
        if self.ptr - len(idxs) > batch_size:
            np.delete(self.obs1_buf, idxs)
            np.delete(self.obs2_buf, idxs)
            np.delete(self.acts_buf, idxs)
            np.delete(self.rews_buf, idxs)
            np.delete(self.done_buf, idxs)
            self.ptr = self.ptr - len(idxs)
            self.size = self.size - len(idxs)
            
            





class mlp_gail_discriminator(Model):

  def __init__(self, hidden_sizes=[32,32,32], activation = 'relu'):
    super(mlp_gail_discriminator, self).__init__()
    self.mlp = mlp(list(hidden_sizes), activation, activation)
    self.prob = Dense(1, activation='sigmoid')


  def call(self, obs, acts):
    x = tf.concat([obs,acts], axis = -1)
    x = self.mlp(x)
    prob = self.prob(x)
    return prob


def draw_expert_batch(batch_size):
    try:
      obs,acts, mask, lengths = expert_set.next()
    except: 
      expert_set = iter(dataset.shuffle(train_len).batch(seqs).repeat(50)) #  reset it and keep iterating, woo!
      obs,acts, mask, lengths = expert_set.next()

    s_i = obs[:,0,:]

    range_lens = tf.expand_dims(tf.range(tf.shape(lengths)[0]), 1)
    expanded_lengths = tf.expand_dims(lengths-1,1)# lengths must be subtracted by 1 to become indices.

    s_g = tf.gather_nd(obs, tf.concat((range_lens, expanded_lengths),1)) # get the actual last element of the sequencs.


    # Encode the trajectory
    mu_enc, s_enc = encoder(obs, acts, training = True)
    encoder_normal = tfd.Normal(mu_enc,s_enc)
    z = encoder_normal.sample()

    s_g_dim = s_g.shape[-1]
    s_g = tf.tile(s_g, [1,MAX_SEQ_LEN])
    s_g = tf.reshape(s_g, [-1, MAX_SEQ_LEN,s_g_dim ]) 

    z = tf.tile(z, [1, MAX_SEQ_LEN])
    z = tf.reshape(z, [-1, MAX_SEQ_LEN, LATENT_DIM]) 
    o = tf.concat([obs,z,s_g], axis = -1)

    two_D_mask = tf.squeeze(mask[:,:,0]) # this gives us a batch x timesteps matrix with 1s where there are timesteps and 0s where there is padding
    indexes = tf.where(two_D_mask > 0) # gets us a list of the timesteps which are real not padding. 
    batch_o = tf.gather_nd(o, indexes[::MULTIPLIER])
    batch_a = tf.gather_nd(acts, indexes[::MULTIPLIER])
    return {'obs':batch_o, 'acts':batch_a}






@tf.function
def BC_step(expert_batch, policy, optimizer):
    obs, expert_act = expert_batch['obs'], expert_batch['acts']
    with tf.GradientTape() as tape:
        mu,_,_ = policy(obs)
        BC_loss = tf.reduce_sum(tf.losses.MSE(mu, expert_act))
    BC_gradients = tape.gradient(BC_loss, policy.trainable_variables)
    optimizer.apply_gradients(zip(BC_gradients, policy.trainable_variables))
    return BC_loss

# def test_step(expert_batch, policy):
#     obs, expert_act = expert_batch['obs'], expert_batch['acts']
#     mu,_,_ = policy(obs)
#     BC_loss = tf.reduce_sum(tf.losses.MSE(mu, expert_act))
#     return BC_loss

def pretrain_BC(model, BC_optimizer, batch_size):
    print('Beginning Pretraining')
    for i in range(0,10000):
        expert_batch = sample_expert_transitions(batch_size)
        BC_loss = BC_step(expert_batch, model.actor, BC_optimizer)
        if i % 5000 == 0:
            print(BC_loss) 
    print('Ending Pretraining')




def training_loop(env_fn,  ac_kwargs=dict(), seed=0, 
        steps_per_epoch=5000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=BATCH_SIZE, start_steps=5000, 
        max_ep_len=1000, save_freq=1, load = False, exp_name = "Experiment_1", render = False, discrim_req_acc = 0.7, BC = False, DEMO_EUCLIDEAN_REWARD = False):

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
    train_log_dir = 'logs/' + str(discrim_req_acc)+exp_name+':'+str(start_time)
    summary_writer = tf.summary.create_file_writer(train_log_dir)

    discriminator = mlp_gail_discriminator()
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    if BC:
        BC_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)


    def update_models(model, replay_buffer, steps, batch_size, current_step = None):
        agent_accuracy = 0
        # until the discriminator is trained to sufficiently distinguish correct transitions.
        print('Updating Discriminator')
        while agent_accuracy < discrim_req_acc or expert_acurracy < discrim_req_acc:
            batch = replay_buffer.sample_batch(batch_size)
            expert_batch = sample_expert_transitions(batch_size)
            _,expert_acurracy,agent_accuracy = discriminator_train_step(batch, expert_batch, discriminator, discriminator_optimizer, replay_buffer, batch_size, discrim_req_acc)
            print(expert_acurracy, agent_accuracy)

        # now update SAC
        print('Updating Policy')
        for j in range(steps):
            batch = replay_buffer.sample_batch(batch_size)
            batch_obs, batch_acts = batch['obs1'], batch['acts']
            agent_probs = discriminator(batch_obs,batch_acts)
            agent_reward = (tf.math.log(agent_probs+1e-8)-(tf.math.log(1-agent_probs+1e-8))).numpy().squeeze().astype('float32') #            # use GAIL reward instead of environment reward
            batch['rews'] = agent_reward

            LossPi, LossQ1, LossQ2, LossV, Q1Vals, Q2Vals, VVals, LogPi = model.train_step(batch)
            if BC:
                # Use BC to accelerate GAIL convergence
                 
                expert_batch = sample_expert_transitions(batch_size)
                BC_loss = BC_step(expert_batch, model.actor, BC_optimizer)
                with summary_writer.as_default():
                    tf.summary.scalar('BC_MSE_loss',BC_loss, step=current_step+j)



    # now collect epsiodes
    total_steps = steps_per_epoch * epochs
    steps_collected = 0

    #pretrain with BC
    #pretrain_BC(SAC, BC_optimizer, batch_size)
    # collect some initial random steps to initialise
    random_steps = 5000
    steps_collected  += rollout_trajectories(n_steps = random_steps,env = env, max_ep_len = max_ep_len, actor = 'random', replay_buffer = replay_buffer, summary_writer = summary_writer, exp_name= exp_name)


    update_models(SAC, replay_buffer, steps = random_steps, batch_size = batch_size, current_step = steps_collected)

    # now act with our actor, and alternately collect data, then train.
    while steps_collected < total_steps:
    # collect an episode
        steps_collected  += rollout_trajectories(n_steps = max_ep_len,env = env, max_ep_len = max_ep_len, actor = SAC.actor.get_stochastic_action, replay_buffer = replay_buffer, summary_writer=summary_writer, current_total_steps = steps_collected, exp_name= exp_name)
        # take than many training steps
        update_models(SAC, replay_buffer, steps = max_ep_len, batch_size = batch_size, current_step = steps_collected)

        # if an epoch has elapsed, save and test.
        if steps_collected  > 0 and steps_collected  % steps_per_epoch == 0:
            SAC.save_weights()
            # Test the performance of the deterministic version of the agent.
            rollout_trajectories(n_steps = max_ep_len*10,env = test_env, max_ep_len = max_ep_len, actor = SAC.actor.get_deterministic_action, summary_writer=summary_writer, current_total_steps = steps_collected, train = False, render = True, exp_name= exp_name)


# MODIFIABLE VARIBALES TODO PROPERLY PUT THIS IN A CLASS
#ENV_NAME='reacher2D-v0'
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='reacher2D-v0')
    parser.add_argument('--hid', type=int, default=128)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--max_ep_len', type=int, default=200)
    parser.add_argument('--exp_name', type=str, default='GAIL')
    parser.add_argument('--load', type=bool, default=False)
    parser.add_argument('--render', type=bool, default=False)
    parser.add_argument('--discrim_req_acc', type = float, default = 0.7)
    parser.add_argument('--BC', type = bool, default = False)
    args = parser.parse_args()

    experiment_name = 'GAIL'+args.env+'_Hidden_'+str(args.hid)+'l_'+str(args.l)

    training_loop(lambda : gym.make(args.env), 
      ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
      gamma=args.gamma, seed=args.seed, epochs=args.epochs, load = args.load, exp_name = experiment_name, max_ep_len = args.max_ep_len, render = args.render, discrim_req_acc = args.discrim_req_acc, BC = args.BC)