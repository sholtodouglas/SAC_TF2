

import numpy as np
import tensorflow as tf
import gym
import pybullet
import pointMass #  the act of importing registers the env.
import ur5_RL
import time
from common import *
from SAC import *
import copy
import psutil
import multiprocessing as mp
from tqdm import tqdm
from natsort import natsorted, ns
from latent import *

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
#TODO Answer why reward scaling makes such a damn difference?

############################################################################################################
#Her with additional support for representation learning
############################################################################################################


# Agree with the stable baselines guys, HER is best implemented as a wrapper on the replay buffer.


# this is what we're working with at the moment.

# transitions arrive as -  obs, act, rew, next_obs, done
# but in HER, we need entire episodes.
# Ok, so one function for store episode, and that stores a bunch of transitions, either with strategy future or final.
# yeah so instead of storing transitions, store episdoes. Done? Then sample transitons the same way.
# how do we handle obs? take it as a dict at the ep stage, then when we're sampling for SAC .. flattened? Reward the same
# yeah. I reckon.

class HERReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, env, obs_dim, act_dim, size, n_sampled_goal = 4, goal_selection_strategy = 'future'):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.n_sampled_goal = n_sampled_goal
        self.env = env
        self.goal_selection_strategy = goal_selection_strategy
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

        # could be a goal, or both goal and z!
    def sample_achieved(self, transitions, transition_idx, strategy = 'future', encoder = None):
        if strategy == 'future':
            selected_idx = np.random.choice(np.arange(transition_idx + 1, len(transitions)))
            selected_transition = transitions[selected_idx]
            # here is where we get the obs and acts for the sequence up till there.
            # and here is where we will encode it, and get a nice z.
        elif strategy == 'final':
            selected_transition = transitions[-1]

        goal = selected_transition[0]['achieved_goal']
        return goal # and in future, z.


    # pass in an encoder if we desired reencoding of our representation learning trajectories.
    def store_hindsight_episode(self, episode, encoder = None):
        # ok, our episode comes in as a sequence of obs, acts. But we also want the actual rewards.
        # So really what we want is a sequence of transitions. Is that how we want to collect our episodes?

        # but we also want to be able to immediately convert our expert obs into our replay buffer?
        # so that they can be used immediately.
        # additionally, we want to be able to recompute z?

        # for the most part, when we store an obs we are storing o and d_g. # o, a, r, o2, d
        # when representation learning, we store o, z, d_g. from o,a,r,o2,d,z
        # remember, neural net inference is cheap, so maybe we can encode on the fly?


        for transition_idx, transition in enumerate(episode):

            if encoder != None:
                o, a, r, o2, d, z = transition
                o  = np.concatenate([o['observation'], z, o['desired_goal']])
                o2 = np.concatenate([o2['observation'], z, o2['desired_goal']])
            else:
                o, a, r, o2, d = transition
                o = np.concatenate([o['observation'], o['desired_goal']])
                o2 = np.concatenate([o2['observation'], o2['desired_goal']])

            self.store(o, a, r, o2, d)

            if transition_idx == len(episode)-1:
                selection_strategy = 'final'
            else:
                selection_strategy = self.goal_selection_strategy

            sampled_achieved_goals = [self.sample_achieved(episode, transition_idx, selection_strategy) for _ in range(self.n_sampled_goal)]


            for goal in sampled_achieved_goals:

                if encoder != None:
                    o, a, r, o2, d, z = copy.deepcopy(transition)
                else:
                    o, a, r, o2, d = copy.deepcopy(transition)

                o['desired_goal'] = goal
                o2['desired_goal'] = goal

                r = self.env.compute_reward(goal, o2['desired_goal'], info = None) #i.e 1 for the most part


                if encoder != None:
                    o = np.concatenate([o['observation'], z, o['desired_goal']])
                    o2 = np.concatenate([o2['observation'], z, o2['desired_goal']])
                else:
                    o = np.concatenate([o['observation'], o['desired_goal']])
                    o2 = np.concatenate([o2['observation'], o2['desired_goal']])

                self.store(o, a, r, o2, d)





MAX_SEQ_LEN = 200
MIN_SEQ_LEN = 30
train_test_split = 0.9
curriculum_learn = False
BC_repeat = 50
EPOCHS = 10
AG_IDXS = [19,22] # only for UR5
if curriculum_learn:
    observations = np.load('collected_data/demo_o.npy')
    actions = np.load('collected_data/demo_a.npy')
    OBS_DIM = observations.shape[1]
    ACT_DIM = actions.shape[1]
    train_len = int(len(observations) * train_test_split)
    train_obs_subset = observations[:train_len, :]
    train_acts_subset = actions[:train_len, :]
    valid_obs_subset = observations[train_len:, :]
    valid_acts_subset = actions[train_len:, :]
    train_len = len(train_obs_subset) - MAX_SEQ_LEN
    valid_len = len(valid_obs_subset) - MAX_SEQ_LEN

    dataloader = Dataloader(observations, actions, MIN_SEQ_LEN, MAX_SEQ_LEN)
    dataset = tf.data.Dataset.from_generator(dataloader.data_generator, (tf.float32, tf.float32, tf.float32, tf.int32),
                                             args=('-','Train'))
    valid_dataset = tf.data.Dataset.from_generator(dataloader.data_generator, (tf.float32, tf.float32, tf.float32, tf.int32),
                                                   args=('-','Valid'))


    OBS_DIM = dataloader.OBS_DIM
    ACT_DIM = dataloader.ACT_DIM

def sample_curriculum(observations):
    index= np.random.randint(0,len(observations)-MAX_SEQ_LEN)
    length = np.random.randint(MIN_SEQ_LEN, MAX_SEQ_LEN)
    s_i = observations[index]
    s_g = observations[index+length][AG_IDXS[0]:AG_IDXS[1]] #only on UR5 at the moment
    return s_i, s_g

def fill_replay_buffer_with_expert_transitions(replay_buffer):
    # we have obs, acts
    # demo acts are absolute, so must convert them to relative.
    # we need o[observation], achieved goal\, [desired goal]
    # desired goal is final achieved goal of play data
    # get each from o with proper indexing
    # get action, subtract appropraite o consider that a is transformed with relative
    # r is 1
    # d = False
    episode = []
    index= np.random.randint(0,len(observations)-MAX_SEQ_LEN)
    length = np.random.randint(MAX_SEQ_LEN//2, MAX_SEQ_LEN)
    s_i = observations[index]
    s_g = observations[index+length][AG_IDXS[0]:AG_IDXS[1]] #only on UR5 at the moment
    for i in range(length-1):
        transition = []
        o = {'observation':observations[index+i],'achieved_goal':observations[index+i][AG_IDXS[0]:AG_IDXS[1]], 'desired_goal':s_g}
        a = actions[index+i]
        r = 1
        o2 = {'observation':observations[index+i+1],'achieved_goal':observations[index+i+1][AG_IDXS[0]:AG_IDXS[1]], 'desired_goal':s_g}
        d = False
        transition = [o,a,r,o2,d]
        episode.append(transition)

    replay_buffer.store_hindsight_episode(episode)

# @title Models

def compute_loss(obs, acts, s_g, mask, lengths,training=False, actor = None):
    AVG_SEQ_LEN = obs.shape[1]
    CURR_BATCH_SIZE = obs.shape[0]

    IMI = 0
    OBS_pred_loss = 0

    s_g_dim = s_g.shape[-1]
    s_g = tf.tile(s_g, [1, MAX_SEQ_LEN])
    s_g = tf.reshape(s_g, [-1, MAX_SEQ_LEN, s_g_dim])

    o_in = tf.concat((obs,s_g), axis  = 2)
    o_in = tf.reshape(o_in,[CURR_BATCH_SIZE*MAX_SEQ_LEN, OBS_DIM+s_g.shape[-1]])
    mu, _, _, _, _ = actor(o_in)

    # mu will be B,T,A. Acts B,T,A. Mask is also B,T,A.
    acts = tf.reshape(acts, [CURR_BATCH_SIZE*MAX_SEQ_LEN, ACT_DIM])
    mask = tf.reshape(mask, [CURR_BATCH_SIZE*MAX_SEQ_LEN, ACT_DIM])
    loss = tf.reduce_mean(tf.losses.MAE(mu * mask, acts * mask))


    return loss


def BC_train_step(obs, acts, mask, lengths, optimizer, actor):
    with tf.GradientTape() as tape:
        # obs and acts are a trajectory, so get intial and goal
        s_i = obs[:, 0, :]

        range_lens = tf.expand_dims(tf.range(tf.shape(lengths)[0]), 1)
        expanded_lengths = tf.expand_dims(lengths - 1, 1)  # lengths must be subtracted by 1 to become indices.

        s_g = tf.gather_nd(obs,
                           tf.concat((range_lens, expanded_lengths), 1))  # get the actual last element of the sequencs.
        s_g = s_g[:, AG_IDXS[0]:AG_IDXS[1]]


        lengths = tf.cast(lengths, tf.float32)
        loss = compute_loss(obs, acts, s_g, mask, lengths,actor = actor)

        # find and apply gradients with total loss
    actor_vars = [v for v in actor.trainable_variables if 'log_std' not in v.name]
    gradients = tape.gradient(loss, actor_vars)
    optimizer.apply_gradients(zip(gradients,actor_vars))

    # return values for diagnostics
    return loss

def log_BC_metrics(summary_writer, IMI, steps):
    with summary_writer.as_default():
        tf.summary.scalar('imi_loss', IMI, step=steps)


class VAE_Encoder(Model):
    def __init__(self, LAYER_SIZE, LATENT_DIM):
        super(VAE_Encoder, self).__init__()
        # self.flatten = Flatten()
        self.d1 = Dense(LAYER_SIZE, activation=tf.nn.leaky_relu)
        self.d2 = Dense(LAYER_SIZE, activation=tf.nn.leaky_relu)
        self.mu = Dense(LATENT_DIM)
        self.scale = Dense(LATENT_DIM, activation='softplus')

    def call(self, x, acts=None, training=False):
        # x = self.flatten(x)

        x = self.d1(x)
        x = self.d2(x)

        mu = self.mu(x)
        s = self.scale(x)
        return mu,s
def load_autoencoder(extension, BATCH_SIZE, OBS_DIM, encoder):
    print('Loading in network weights...')
    # load some sample data to initialise the model
    #         load_set = iter(self.dataset.shuffle(self.TRAIN_LEN).batch(self.BATCH_SIZE))
    if IMAGE:
        obs = tf.zeros((BATCH_SIZE, 48, 48, 3))
    else:
        obs = tf.zeros((BATCH_SIZE, OBS_DIM))
    o,_ = encoder(obs)

    print('Models Initalised')
    encoder.load_weights(extension + '/encoder.h5')
    print('Weights loaded.')
    return encoder

AUTOENCODE = True
if AUTOENCODE:
    IMAGE = False
    LAYER_SIZE = 64
    LATENT_DIM = 12
    encoder = VAE_Encoder(LAYER_SIZE, LATENT_DIM)
    encoder = load_autoencoder('saved_models/manifold_learning_states_baseline12', LAYER_SIZE, 8, encoder)

# This is our training loop.
def training_loop(env_fn,  ac_kwargs=dict(), seed=0,
        steps_per_epoch=10000, epochs=100, replay_size=int(1e6), gamma=0.99,
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=3000,
        max_ep_len=300, save_freq=1, load = False, exp_name = "Experiment_1", render = False, strategy = 'future', num_cpus = 'max'):

  print('Begin')
  tf.random.set_seed(seed)
  np.random.seed(seed)


  print('Pretestenv')
  test_env = env_fn()
  print('test_env-',test_env)
  num_cpus = psutil.cpu_count(logical=False)
  env = env_fn()
  #pybullet needs the GUI env to be reset first for our noncollision stuff to work.
  obs_dim = env.observation_space.spaces['observation'].shape[0] + env.observation_space.spaces['desired_goal'].shape[0]
  act_dim = env.action_space.shape[0]
  if AUTOENCODE:
      env.set_state_representation(encoder)
      test_env.set_state_representation(encoder)
      obs_dim = LATENT_DIM  + env.observation_space.spaces['desired_goal'].shape[0]



  if render:
    print('Rendering Test Rollouts')
    test_env.render(mode='human')
  test_env.reset()



  # Get Env dimensions


  SAC = SAC_model(env, obs_dim, act_dim, ac_kwargs['hidden_sizes'],lr, gamma, alpha, polyak,  load, exp_name)
  # Experience buffer
  replay_buffer = HERReplayBuffer(env, obs_dim, act_dim, replay_size, n_sampled_goal = 4, goal_selection_strategy = strategy)
  if curriculum_learn:
    BC_set = iter(dataset.shuffle(train_len).batch(5).repeat(steps_per_epoch*epochs//train_len))

  #Logging
  start_time = time.time()
  train_log_dir = 'logs/' + exp_name+str(int(start_time))
  summary_writer = tf.summary.create_file_writer(train_log_dir)

  def update_models(model, replay_buffer, steps, batch_size):
    for j in range(steps):
        batch = replay_buffer.sample_batch(batch_size)
        LossPi, LossQ1, LossQ2, LossV, Q1Vals, Q2Vals, VVals, LogPi = model.train_step(batch)

  # now collect epsiodes
  total_steps = steps_per_epoch * epochs
  steps_collected = 0
  BC_steps = 0
  epoch_ticker = 0

  s_i, s_g = None, None


  # for s in range(100000):
  #     obs, acts, mask, lengths = BC_set.next()
  #     l  =  BC_train_step(obs, acts, mask, lengths, SAC.pi_optimizer, SAC.actor)
  #     log_BC_metrics(summary_writer, l, steps_collected+s)
  #     print(s, l)
  #     if s % 1000 ==0:
  #         for i in range(0,2):
  #             s_i, s_g = sample_curriculum(observations)
  #
  #             rollout_trajectories(n_steps = max_ep_len,env = test_env, start_state=s_i,s_g = s_g,max_ep_len = max_ep_len, actor = SAC.actor.get_deterministic_action, summary_writer=summary_writer, current_total_steps = steps_collected+i*max_ep_len, train = False, render = render, exp_name = exp_name, return_episode = True, goal_based = True)




  if not load:
  # collect some initial random steps to initialise
    if curriculum_learn:

        for i in range(0,20):
            s_i, s_g = sample_curriculum(observations)
            # lots of short episodes, the intention is to get lots of different exposure to object interaction states
            episodes = rollout_trajectories(n_steps = max_ep_len//4,env = env, start_state=s_i,max_ep_len = max_ep_len//4, actor = 'random', summary_writer = summary_writer, exp_name = exp_name, return_episode = True, goal_based = True)
            steps_collected += episodes['n_steps']
            [replay_buffer.store_hindsight_episode(e) for e in episodes['episodes']]
            fill_replay_buffer_with_expert_transitions(replay_buffer)



    else:
        episodes = rollout_trajectories(n_steps = start_steps,env = env, start_state=s_i,max_ep_len = max_ep_len, actor = 'random', summary_writer = summary_writer, exp_name = exp_name, return_episode = True, goal_based = True)
        steps_collected += episodes['n_steps']
        [replay_buffer.store_hindsight_episode(e) for e in episodes['episodes']]
  # episodes, steps  = rollout_trajectories(n_steps = start_steps,env = env, max_ep_len = max_ep_len, actor = 'random', summary_writer = summary_writer, exp_name = exp_name, return_episode = True, goal_based = True)
    # steps_collected += steps
    # [replay_buffer.store_hindsight_episode(episode) for episode in episodes]
    update_models(SAC, replay_buffer, steps = steps_collected, batch_size = batch_size)



  # now act with our actor, and alternately collect data, then train.
  while steps_collected < total_steps:
    # collect an episode
    # episodes, steps   = rollout_trajectories(n_steps = max_ep_len,env = env, max_ep_len = max_ep_len, actor = SAC.actor.get_stochastic_action, summary_writer=summary_writer, current_total_steps = steps_collected, exp_name = exp_name, return_episode = True, goal_based = True)
    # steps_collected += steps

    # # take than many training steps
    # [replay_buffer.store_hindsight_episode(episode) for episode in episodes]
    if curriculum_learn:
        s_i, s_g = sample_curriculum(observations)
        fill_replay_buffer_with_expert_transitions(replay_buffer)

        for i in range(BC_repeat):
            obs, acts, mask, lengths = BC_set.next()
            l  =  BC_train_step(obs, acts, mask, lengths, SAC.pi_optimizer, SAC.actor)
            log_BC_metrics(summary_writer, l, BC_steps+i)
            BC_steps += 1



    episodes = rollout_trajectories(n_steps = max_ep_len,env = env,start_state=s_i, max_ep_len = max_ep_len, actor = SAC.actor.get_stochastic_action, summary_writer=summary_writer, current_total_steps = steps_collected, exp_name = exp_name, return_episode = True, goal_based = True)
    steps_collected += episodes['n_steps']
    [replay_buffer.store_hindsight_episode(e) for e in episodes['episodes']]

    update_models(SAC, replay_buffer, steps = max_ep_len, batch_size = batch_size)



    # if an epoch has elapsed, save and test.
    if steps_collected >= epoch_ticker:
        SAC.save_weights()
        # Test the performance of the deterministic version of the agent.
        if curriculum_learn:
            for i in range(0,5):
                s_i, s_g = sample_curriculum(observations)

                rollout_trajectories(n_steps = max_ep_len,env = test_env, start_state=s_i,max_ep_len = max_ep_len, actor = SAC.actor.get_deterministic_action, summary_writer=summary_writer, current_total_steps = steps_collected+i*max_ep_len, train = False, render = render, exp_name = exp_name, return_episode = True, goal_based = True)
        else:
            rollout_trajectories(n_steps = max_ep_len*5,env = test_env, max_ep_len = max_ep_len, actor = SAC.actor.get_deterministic_action, summary_writer=summary_writer, current_total_steps = steps_collected, train = False, render = render, exp_name = exp_name, return_episode = True, goal_based = True)
        epoch_ticker += steps_per_epoch



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='pointMass-v0')
    parser.add_argument('--hid', type=int, default=128)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--max_ep_len', type=int, default=200) # fetch reach learns amazingly if 50, but not if 200 -why?
    parser.add_argument('--exp_name', type=str, default='experiment_1')
    parser.add_argument('--load', type=bool, default=False)
    parser.add_argument('--render', type=bool, default=False)
    parser.add_argument('--strategy', type=str, default='future')


    args = parser.parse_args()

    experiment_name = 'HER2_linear_state_rep_sparse_reward_'+args.env+'_Hidden_'+str(args.hid)+'l_'+str(args.l)

    training_loop(lambda : gym.make(args.env),
      ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
      gamma=args.gamma, seed=args.seed, epochs=args.epochs, load = False, exp_name = experiment_name, max_ep_len = args.max_ep_len, render = True, strategy = args.strategy)
