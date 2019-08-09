import numpy as np
import tensorflow as tf
import time
import datetime
from tqdm import tqdm
from tensorflow.keras.layers import Dense, Flatten, Conv2D,Bidirectional, LSTM, Dropout

from tensorflow.keras import Model
from tensorflow.keras.models import  Sequential
from tqdm import tqdm, tqdm_notebook
from HER import HERReplayBuffer
import tensorflow_probability as tfp

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions
tfb = tfp.bijectors
import os
print(tf.__version__)

import pybullet
import reach2D
import pointMass
from common import *
import copy
from SAC import *
import traceback

# All magic numbers here
#@title Definitions
train_test_split = 0.9
MAX_SEQ_LEN = 30
MIN_SEQ_LEN = 16
# TODO Consider if this could mess us up if we have an LSTM based actor?
FRAME_SKIP = 2 #frame interval to decimate sequences at. cant frame skip if we want to do determin
BATCH_SIZE = 256
LAYER_SIZE = 128
LATENT_DIM = 12
P_DROPOUT = 0.2
BETA = 0.05
OBS_GOAL_INDEX = 4  # index from which the goal is in the obs vector
ACHEIVED_GOAL_INDEX = 2  # point up to which we care about the goal
EPOCHS = 1000
extension = 'saved_models/Z_learning_B005'

observations = np.load('collected_data/10000HER_pointMass-v0_Hidden_128l_2expert_obs_.npy').astype('float32')[:, 0:OBS_GOAL_INDEX]  # Don't include the goal in the obs
actions = np.load('collected_data/10000HER_pointMass-v0_Hidden_128l_2expert_actions.npy').astype('float32')
OBS_DIM = observations.shape[1]
ACT_DIM = actions.shape[1]
train_len = int(len(observations) * train_test_split)
train_obs_subset = observations[:train_len, :]
train_acts_subset = actions[:train_len, :]
valid_obs_subset = observations[train_len:, :]
valid_acts_subset = actions[train_len:, :]
train_len = len(train_obs_subset) - MAX_SEQ_LEN * FRAME_SKIP
valid_len = len(valid_obs_subset) - MAX_SEQ_LEN * FRAME_SKIP


def data_generator(actions, subset):
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
        length = np.random.randint(MIN_SEQ_LEN * FRAME_SKIP, (MAX_SEQ_LEN * FRAME_SKIP))

        if length % 2 != 0:
            length -= 1

        obs_padding = np.zeros((MAX_SEQ_LEN - length // FRAME_SKIP, OBS_DIM))

        padded_obs = np.concatenate((obs_set[idx:idx + length:FRAME_SKIP], obs_padding), axis=0)

        act_padding = np.zeros((MAX_SEQ_LEN - length // FRAME_SKIP, ACT_DIM))
        padded_act = np.concatenate((act_set[idx:idx + length:FRAME_SKIP], act_padding), axis=0)

        # ones to length of actions, zeros for the rest to mask out loss.
        mask = np.concatenate((np.ones((length // FRAME_SKIP, ACT_DIM)), act_padding), axis=0)

        if len(padded_obs) != MAX_SEQ_LEN:
            print(idx, length, len(padded_obs))

        if len(padded_act) != MAX_SEQ_LEN:
            print(idx, length, len(padded_act))

        yield (padded_obs, padded_act, mask, length // FRAME_SKIP)


dataset = tf.data.Dataset.from_generator(data_generator, (tf.float32, tf.float32, tf.float32, tf.int32),
                                         args=(actions, 'Train'))
valid_dataset = tf.data.Dataset.from_generator(data_generator, (tf.float32, tf.float32, tf.float32, tf.int32),
                                               args=(actions, 'Valid'))


# @title Models
class TRAJECTORY_ENCODER_LSTM(Model):
    def __init__(self, LAYER_SIZE, LATENT_DIM, P_DROPOUT):
        super(TRAJECTORY_ENCODER_LSTM, self).__init__()

        self.bi_lstm = Bidirectional(LSTM(LAYER_SIZE, return_sequences=True, activation ='tanh',recurrent_activation ='sigmoid', recurrent_dropout = 0, use_bias = True), merge_mode=None)
        self.mu = Dense(LATENT_DIM)
        self.scale = Dense(LATENT_DIM, activation='softplus')
        self.dropout1 = tf.keras.layers.Dropout(P_DROPOUT)

                                                
    def call(self, obs, acts, training=False):
        x = tf.concat([obs, acts], axis=2)  # concat observations and actions together.
        x = self.bi_lstm(x)
        x = self.dropout1(x, training=training)
        bottom = x[0][:, -1, :]  # Take the last element of the bottom row
        top = x[1][:, 0, :]  # Take the first elemetn of the top row cause Bidirectional, top row goes backward.
        x = tf.concat([bottom, top], axis=1)
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
        self.l1 = Dense(LAYER_SIZE, activation='relu', name='layer1')
        self.l2 = Dense(LAYER_SIZE, activation='relu', name='layer2')
        self.mu = Dense(ACT_DIM, name='mu')
        self.log_std = Dense(ACT_DIM, activation='tanh', name='log_std')
        self.dropout1 = tf.keras.layers.Dropout(P_DROPOUT)

    def gaussian_likelihood(self, x, mu, log_std):
        EPS = 1e-8
        pre_sum = -0.5 * (((x - mu) / (tf.exp(log_std) + EPS)) ** 2 + 2 * log_std + np.log(2 * np.pi))
        return tf.reduce_sum(input_tensor=pre_sum, axis=1)

    def clip_but_pass_gradient(self, x, l=-1., u=1.):
        clip_up = tf.cast(x > u, tf.float32)
        clip_low = tf.cast(x < l, tf.float32)
        return x + tf.stop_gradient((u - x) * clip_up + (l - x) * clip_low)

    def apply_squashing_func(self, mu, pi, logp_pi):
        # TODO: Tanh makes the gradients bad - we don't necessarily wan't to tanh these - lets confirm later
        #
        mu = tf.tanh(mu)
        pi = tf.tanh(pi)

        # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
        logp_pi -= tf.reduce_sum(input_tensor=tf.math.log(self.clip_but_pass_gradient(1 - pi ** 2, l=0, u=1) + 1e-6),
                                 axis=1)
        return mu, pi, logp_pi

    def call(self, s, z=None, s_g=None, training=False):

        # check if the user has fed in z and s_g
        # if not, means they're passing s,z,s_g as one vector through s - this will be in the case of this
        # actor being used in our typical RL algorithms
        if z != None and s_g != None:
            B = z.shape[0]  # dynamically get batch size
            if len(s.shape) == 3:
                x = tf.concat([s, z, s_g], axis=2)  # (BATCHSIZE)
            else:
                x = tf.concat([s, z, s_g], axis=0)  # (BATCHSIZE,  OBS+OBS+LATENT)
                x = tf.expand_dims(x, 0)  # make it (1, OBS+OBS+LATENT)
        else:
            x = s

        x = self.l1(x)
        x = self.dropout1(x, training=training)
        x = self.l2(x)
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        std = tf.exp(log_std)

        pi = mu + tf.random.normal(tf.shape(input=mu)) * std
        logp_pi = self.gaussian_likelihood(pi, mu, log_std)

        # Equivalent to this, should we change over to TFP properly at some point?
        # For some reason it doesn't work as well on pendulum. Weird.
        pdf = tfd.Normal(loc=mu, scale=std)
        # logp_pi = tf.reduce_sum(pdf.log_prob(pi))

        mu, pi, logp_pi = self.apply_squashing_func(mu, pi, logp_pi)
        return mu, pi, logp_pi, std, pdf

    def get_deterministic_action(self, o):
        # o should be s,z,s_g concatted together
        o = tf.expand_dims(o, axis=0)
        mu, _, _, _, _ = self.call(o)

        return mu[0]

    def get_stochastic_action(self, o):
        o = tf.expand_dims(o, axis=0)
        _, pi, _, _, _ = self.call(o)

        return pi[0]


# @title Training and Test Step
# Ok recall that the gist of the loop is, get a bunch of trajectories, take the first as s_i, the last as s_g. Encode full trajectory as z.

# Training Step

# @tf.function
def train_step(obs, acts, BETA, mask, lengths):
    with tf.GradientTape() as tape:
        # obs and acts are a trajectory, so get intial and goal
        s_i = obs[:, 0, :]

        range_lens = tf.expand_dims(tf.range(tf.shape(lengths)[0]), 1)
        expanded_lengths = tf.expand_dims(lengths - 1, 1)  # lengths must be subtracted by 1 to become indices.

        s_g = tf.gather_nd(obs,tf.concat((range_lens, expanded_lengths), 1))  # get the actual last element of the sequencs.
        s_g = s_g[:, :ACHEIVED_GOAL_INDEX]
        # Encode the trajectory
        mu_enc, s_enc = encoder(obs, acts, training=True)
        encoder_normal = tfd.Normal(mu_enc, s_enc)
        z = encoder_normal.sample()

        lengths = tf.cast(lengths, tf.float32)
        loss, IMI, info_kl = compute_loss(encoder_normal, z, obs, acts, s_g, BETA, mu_enc, s_enc, mask, lengths,
                                          training=True)

        # find and apply gradients with total loss
    actor_vars = [v for v in actor.trainable_variables if 'log_std' not in v.name]
    gradients = tape.gradient(loss, encoder.trainable_variables + actor_vars)
    optimizer.apply_gradients(zip(gradients, encoder.trainable_variables + actor_vars))

    # return values for diagnostics
    return IMI, info_kl


def test_step(obs, acts, mask, lengths):
    # obs and acts are a trajectory, so get intial and goal
    s_i = obs[:, 0, :]
    range_lens = tf.expand_dims(tf.range(tf.shape(lengths)[0]), 1)
    expanded_lengths = tf.expand_dims(lengths - 1, 1)  # lengths must be subtracted by 1 to become indices.
    s_g = tf.gather_nd(obs,
                       tf.concat((range_lens, expanded_lengths), 1))  # get the actual last element of the sequencs.
    s_g = s_g[:, :ACHEIVED_GOAL_INDEX]

    # Encode the trajectory
    mu_enc, s_enc = encoder(obs, acts, training=True)
    encoder_normal = tfd.Normal(mu_enc, s_enc)
    z = encoder_normal.sample()
    lengths = tf.cast(lengths, tf.float32)
    loss, IMI, info_kl = compute_loss(encoder_normal, z, obs, acts, s_g, BETA, mu_enc, s_enc, mask, lengths,
                                      training=True)

    # return values for diagnostics
    return IMI, info_kl


# @title Loss Computation and Model Save/Load
def compute_loss(normal_enc, z, obs, acts, s_g, BETA, mu_enc, s_enc, mask, lengths, training=False):
    AVG_SEQ_LEN = obs.shape[1]
    CURR_BATCH_SIZE = obs.shape[0]
    # Automatically averaged over batch size i.e. SUM_OVER_BATCH_SIZE
    std_normal = tfd.Normal(0, 1)
    batch_avg_mean = tf.reduce_mean(mu_enc,
                                    axis=0)  # m_enc will batch_size, latent_dim. We want average mean across the batches so we end up with a latent dim size avg_mean_vector. Each dimension of the latent dim should be mean 0 avg across the batch, but individually can be different.
    batch_avg_s = tf.reduce_mean(s_enc, axis=0)
    batch_avg_normal = tfd.Normal(batch_avg_mean, batch_avg_s)
    info_kl = tf.reduce_sum(tfd.kl_divergence(batch_avg_normal, std_normal))

    IMI = 0
    OBS_pred_loss = 0

    s_g_dim = s_g.shape[-1]
    s_g = tf.tile(s_g, [1, MAX_SEQ_LEN])
    s_g = tf.reshape(s_g, [-1, MAX_SEQ_LEN, s_g_dim])
    z = tf.tile(z, [1, MAX_SEQ_LEN])
    z = tf.reshape(z, [-1, MAX_SEQ_LEN, LATENT_DIM])  # so that both end up as BATCH, SEQ, DIM

    mu, _, _, _, pdf = actor(obs, z, s_g, training=training)

    #     log_prob_actions = -pdf.log_prob(acts[:,:,:ACT_DIM]) # batchsize, Maxseqlen, actions,

    #     masked_log_probs = log_prob_actions*mask[:,:,:ACT_DIM] # should zero out all masked elements.
    #     avg_batch_wise_sum = tf.reduce_sum(masked_log_probs, axis = (1,2)) / lengths
    #     IMI = tf.reduce_mean(avg_batch_wise_sum) / AVG_SEQ_LEN / CURR_BATCH_SIZE

    # mu will be B,T,A. Acts B,T,A. Mask is also B,T,A.

    IMI = tf.reduce_mean(tf.losses.MAE(mu * mask, acts * mask))

    loss = IMI + BETA * info_kl
    return loss, IMI, info_kl


def load_weights(extension):
    print('Loading in network weights...')
    # load some sample data to initialise the model
    #         load_set = iter(self.dataset.shuffle(self.TRAIN_LEN).batch(self.BATCH_SIZE))
    obs = tf.zeros((BATCH_SIZE, MAX_SEQ_LEN, OBS_DIM))
    acts = tf.zeros((BATCH_SIZE, MAX_SEQ_LEN, ACT_DIM))
    mask = acts
    lengths = tf.cast(tf.ones(BATCH_SIZE), tf.int32)

    _, _ = test_step(obs, acts, mask, lengths)

    print('Models Initalised')
    encoder.load_weights(extension + '/encoder.h5')
    actor.load_weights(extension + '/actor.h5')
    print('Weights loaded.')

class LatentHERReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, env, obs_dim, act_dim, size, n_sampled_goal = 4, goal_selection_strategy = 'future', goal_dist = 0.5, z_dist = 2):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.n_sampled_goal = n_sampled_goal
        self.env = env
        self.goal_selection_strategy = goal_selection_strategy
        self.goal_dist = goal_dist
        self.z_dist = z_dist
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

    def compute_latent_reward(self, achieved_goal, desired_goal, achieved_z, desired_z):
        distance = np.sum(abs(achieved_goal-desired_goal))
        z_dist = np.sum(abs(achieved_z - desired_z))

        r = 0
        if distance <= self.goal_dist and z_dist <= self.z_dist:
            r = 1
        return r


        # could be a goal, or both goal and z!
    def sample_achieved(self, transitions, transition_idx, strategy = 'future', encoder = None):
        if strategy == 'future':
            selected_idx = np.random.choice(np.arange(transition_idx + 1, len(transitions)))
            selected_transition = transitions[selected_idx]


        elif strategy == 'final':
            selected_transition = transitions[-1]

        goal = selected_transition[0]['achieved_goal']
        # here is where we get the obs and acts for the sequence up till there.
        if encoder != None:
        # and here is where we will encode it, and get a nice z.
            sub_sequence = transitions[0:selected_idx + 1]  # up to and including selected index
            seq_obs, seq_acts = episode_to_trajectory(sub_sequence, representation_learning=True)
            # encoding of the path up until that goal.
            z = tf.squeeze(encoder(np.expand_dims(seq_obs[::FRAME_SKIP, :], axis=0),
                                   np.expand_dims(seq_acts[::FRAME_SKIP, :], axis=0)))

            return (goal, z)
        return goal


    # pass in an encoder if we desired reencoding of our representation learning trajectories.
    def store_hindsight_episode(self, episode, encoder = None):

        # get the z we were meant to follow.
        if encoder != None: 
            _,_,_,_,_,desired_z = episode[0]

            # now find the z we actually followed
            seq_obs, seq_acts = episode_to_trajectory(episode, representation_learning=True)
            achieved_z = tf.squeeze(encoder(np.expand_dims(seq_obs[::FRAME_SKIP, :], axis=0),
                                   np.expand_dims(seq_acts[::FRAME_SKIP, :], axis=0)))

        for transition_idx, transition in enumerate(episode):
            
            if encoder != None:
                o, a, _, o2, d, _ = transition
                o  = np.concatenate([o['observation'], desired_z, o['desired_goal']])
                o2 = np.concatenate([o2['observation'], desired_z, o2['desired_goal']])
                # r from the environment won't be accurate, we need to recompute as sparse
                # not only in terms of the goal, but also the latent vector.
                r = compute_latent_reward(self, o['achieved_goal'], o['desired_goal'], achieved_z, desired_z)
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

            # hindsight sub in both the goal and z. So compute the achieved goal and intended z up till that goal.
            for sample in sampled_achieved_goals:
                
                if encoder != None:
                    o, a, r, o2, d, _ = copy.deepcopy(transition)
                    (goal,z) = sample
                else:
                    o, a, r, o2, d = copy.deepcopy(transition)
                    goal = sample

                o['desired_goal'] = goal
                o2['desired_goal'] = goal

                # with our method here, our reward is based off following z and acheiving the goal.
                
                if encoder != None:
                    o = np.concatenate([o['observation'], z, o['desired_goal']])
                    o2 = np.concatenate([o2['observation'], z, o2['desired_goal']])
                    r = compute_latent_reward(self, goal, o2['desired_goal'], z, z)
                else:
                    r = self.env.compute_reward(goal, o2['desired_goal'], info = None)
                    o = np.concatenate([o['observation'], o['desired_goal']])
                    o2 = np.concatenate([o2['observation'], o2['desired_goal']])


                self.store(o, a, r, o2, d)


def sample_expert_trajectory(train_set, encoder):

    obs, acts, mask, lengths = train_set.next()
    s_i = obs[:, 0, :]
    range_lens = tf.expand_dims(tf.range(tf.shape(lengths)[0]), 1)
    expanded_lengths = tf.expand_dims(lengths - 1, 1)  # lengths must be subtracted by 1 to become indices.

    s_g = tf.gather_nd(obs,
                       tf.concat((range_lens, expanded_lengths), 1))  # get the actual last element of the sequencs.
    s_g = s_g[:, :ACHEIVED_GOAL_INDEX]
    # Encode the trajectory
    mu_enc, s_enc = encoder(obs, acts, training=True)
    encoder_normal = tfd.Normal(mu_enc, s_enc)
    z = encoder_normal.sample()
    return s_i, z, s_g, obs, acts, mu_enc, lengths


def log_metrics(summary_writer, current_total_steps, episodes, obs, acts, z, encoder, length, train = True):
  rollout_obs, rollout_acts = episode_to_trajectory(episodes[0], representation_learning = False)     #TODO make this true
  print(rollout_obs.shape, rollout_acts.shape)
  print(rollout_obs.dtype, rollout_acts.dtype)
  rollout_mu_z, rollout_s_z = encoder(np.expand_dims(rollout_obs[::FRAME_SKIP,:],axis= 0), np.expand_dims(rollout_acts[::FRAME_SKIP,:], axis = 0))
  
  z_dist = np.sum(abs(z - np.squeeze(rollout_mu_z)))
  euclidean_dist = np.sum(abs(rollout_obs[:,:2] - np.squeeze(obs, axis=0)[:length,:2]))
  with summary_writer.as_default():
      current_total_steps = int(current_total_steps)
      if train:
          print('Frame: ', current_total_steps, ' Z_distance: ', z_dist, ' Euclidean Distance: ', euclidean_dist)
          tf.summary.scalar('Z_distance', int(z_dist), step=current_total_steps)
          tf.summary.scalar('Euclidean Distance', euclidean_dist, step=current_total_steps)
      else:
          print('Test Frame: ', current_total_steps, ' Z_distance: ', z_dist, ' Euclidean Distance: ', euclidean_dist)
          tf.summary.scalar('Test Z_distance', z_dist, step=current_total_steps)
          tf.summary.scalar('Test Euclidean Distance', euclidean_dist, step=current_total_steps)

encoder = TRAJECTORY_ENCODER_LSTM(LAYER_SIZE, LATENT_DIM, P_DROPOUT)
actor = ACTOR(LAYER_SIZE, ACT_DIM, P_DROPOUT)
load_weights(extension)
#
# This is our training loop.
def training_loop(env_fn, ac_kwargs=dict(), seed=0,
                  steps_per_epoch=2000, epochs=100, replay_size=int(1e6), gamma=0.99,
                  polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=3000,
                  max_ep_len=1000, save_freq=1, load=False, exp_name="Experiment_1", render=False, strategy='future'):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    env, test_env = env_fn(), env_fn()

    env.set_sparse_reward()
    # Get Env dimensions
    in_dim = env.observation_space.spaces['observation'].shape[0] + env.observation_space.spaces['desired_goal'].shape[0]
    act_dim = env.action_space.shape[0]
    SAC = SAC_model(env, in_dim, act_dim, ac_kwargs['hidden_sizes'], lr, gamma, alpha, polyak, load, exp_name)

    # update the actor to our boi's weights.
    SAC.actor = ACTOR(LAYER_SIZE, act_dim, P_DROPOUT)
    SAC.build_models(BATCH_SIZE, in_dim, act_dim)
    #assign_variables(actor, SAC.actor)
    SAC.models['actor'] = SAC.actor

    # Experience buffer
    replay_buffer = LatentHERReplayBuffer(env, in_dim, act_dim, replay_size, n_sampled_goal=4,
                                    goal_selection_strategy=strategy)

    # Logging

    start_time = time.time()
    train_log_dir = 'logs/sub/' + exp_name + ':' + str(start_time) + '/stochastic'
    summary_writer = tf.summary.create_file_writer(train_log_dir)

    def update_models(model, replay_buffer, steps, batch_size):
        for j in range(steps):
            batch = replay_buffer.sample_batch(batch_size)
            LossPi, LossQ1, LossQ2, LossV, Q1Vals, Q2Vals, VVals, LogPi = model.train_step(batch)

    # now collect epsiodes
    total_steps = steps_per_epoch * epochs
    steps_collected = 0


    sample_size = 20
    train_set = iter(dataset.shuffle(valid_len).batch(sample_size).repeat(EPOCHS))
    # sample a bunch of expert trajectories to try to imitate
    s_i, z, s_g, obs, acts, _, lengths = sample_expert_trajectory(train_set, encoder)
    for r in range(sample_size):
        episodes, steps = rollout_trajectories(n_steps=lengths[r], env=env, max_ep_len=lengths[r], actor=SAC.actor.get_deterministic_action,
                                               start_state=s_i[r], s_g = s_g[r], exp_name=exp_name, return_episode=True,
                                               goal_based=True, current_total_steps=steps_collected)
        steps_collected += steps
        [replay_buffer.store_hindsight_episode(episode) for episode in episodes]
        log_metrics(summary_writer, steps_collected, episodes, obs[r], acts[r], z[r], encoder,lengths[r], train = True)

    # now update after
    update_models(SAC, replay_buffer, steps=steps_collected, batch_size=batch_size)

    # now act with our actor, and alternately collect data, then train.
    train_set = iter(dataset.shuffle(valid_len).batch(1).repeat(EPOCHS))

    while steps_collected < total_steps:
        # collect an episode
        s_i, z, s_g, obs, acts, _, lengths = sample_expert_trajectory(train_set, encoder)
        episodes, steps = rollout_trajectories(n_steps=lengths[0], env=env, max_ep_len=lengths[0],
                                               actor=SAC.actor.get_deterministic_action, start_state = s_i[0], s_g= s_g[0],
                                               current_total_steps=steps_collected, exp_name=exp_name,
                                               return_episode=True, goal_based=True)
        steps_collected += steps
        # take than many training steps
        [replay_buffer.store_hindsight_episode(episode) for episode in episodes]
        update_models(SAC, replay_buffer, steps=max_ep_len, batch_size=batch_size)
        log_metrics(summary_writer, steps_collected, episodes, obs[0], acts[0], z[0], encoder, lengths[0])
        # we also want to compute z distance and euclidean distance of trajectories, and summary record them here.

        # if an epoch has elapsed, save and test.
        
        if steps_collected > 0 and steps_collected % steps_per_epoch == 0:
            SAC.save_weights()
            # Test the performance of the deterministic version of the agent.
            for i in range(0,10):
                s_i, _, s_g, obs, acts, mu_z, lengths = sample_expert_trajectory(train_set, encoder)
                episodes, steps = rollout_trajectories(n_steps=lengths[0], env=test_env, max_ep_len=lengths[0],
                                                       actor=SAC.actor.get_deterministic_action,start_state = s_i[0], s_g=s_g[0],
                                                       current_total_steps=steps_collected, exp_name=exp_name,
                                                       return_episode=True, goal_based=True, train = False, render = True)
                log_metrics(summary_writer, steps_collected, episodes, obs[0], acts[0], mu_z[0], encoder, lengths[0], train = False)
            # we also want to compute z distance and euclidean distance of trajectories, and summary record them here.

            ## Z distance, euclidean distance?




















            # TODO Pray?

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='pointMass-v0')
    parser.add_argument('--hid', type=int, default=128)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--max_ep_len', type=int, default=200)
    parser.add_argument('--exp_name', type=str, default='experiment_1')
    parser.add_argument('--load', type=bool, default=False)
    parser.add_argument('--render', type=bool, default=False)
    parser.add_argument('--strategy', type=str, default='future')

    args = parser.parse_args()

    experiment_name = 'HER_' + args.env + '_Hidden_' + str(args.hid) + 'l_' + str(args.l)

    training_loop(lambda: gym.make(args.env),
                  ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
                  gamma=args.gamma, seed=args.seed, epochs=args.epochs, load=args.load, exp_name=experiment_name,
                  max_ep_len=args.max_ep_len, render=args.render, strategy=args.strategy)




