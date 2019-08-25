import numpy as np
import tensorflow as tf
import time
import datetime
from tqdm import tqdm

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
from latent import *
from SAC import *
import traceback

# All magic numbers here
# @title Definitions
train_test_split = 0.9
MAX_SEQ_LEN = 60
MIN_SEQ_LEN = 16
BATCH_SIZE = 64
LAYER_SIZE = 128
LATENT_DIM = 6
P_DROPOUT = 0.2
BETA = 0.05
OBS_GOAL_INDEX = 4  # index from which the goal is in the obs vector
ACHIEVED_GOAL_INDEX = 2  # point up to which we care about the goal
EPOCHS = 1000
#extension = 'saved_models/Z_learning_B005'
#extension = 'saved_models/Z_learning_0.01'
#extension = 'saved_models/Z_learning_0.005enc2plan6'
#extension = 'saved_models/OBSRECONSTRUCTION_MLP'
# observations = np.load('collected_data/10000HER_pointMass-v0_Hidden_128l_2expert_obs_.npy').astype('float32')[:,
#                0:OBS_GOAL_INDEX]  # Don't include the goal in the obs
# actions = np.load('collected_data/10000HER_pointMass-v0_Hidden_128l_2expert_actions.npy').astype('float32')

observations = np.load('collected_data/30000HER_pointMass-v0_Hidden_128l_2expert_obs_.npy').astype(
    'float32')[:, 0:OBS_GOAL_INDEX]  # Don't include the goal in the obs
actions = np.load('collected_data/30000HER_pointMass-v0_Hidden_128l_2expert_actions.npy').astype('float32')


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
# @title Models

# @title Training and Test Step
# Ok recall that the gist of the loop is, get a bunch of trajectories, take the first as s_i, the last as s_g. Encode full trajectory as z.

# Training Step

def train_step(obs, acts, BETA, mask, lengths):
    with tf.GradientTape() as tape, tf.GradientTape() as planner_tape:
        # obs and acts are a trajectory, so get intial and goal
        s_i = obs[:, 0, :]

        range_lens = tf.expand_dims(tf.range(tf.shape(lengths)[0]), 1)
        expanded_lengths = tf.expand_dims(lengths - 1, 1)  # lengths must be subtracted by 1 to become indices.

        s_g = tf.gather_nd(obs,
                           tf.concat((range_lens, expanded_lengths), 1))  # get the actual last element of the sequencs.
        s_g = s_g[:, :ACHIEVED_GOAL_INDEX]
        # Encode the trajectory
        mu_enc, s_enc = encoder(obs[:,::2,:], acts[:,::2,:], training=True)
        encoder_normal = tfd.Normal(mu_enc, s_enc)
        z = encoder_normal.sample()

        # Produce a plan from the inital and goal state
        mu_plan, s_plan = planner(s_i, s_g, training=True)
        planner_normal = tfd.Normal(mu_plan, s_plan)
        zp = planner_normal.sample()

        lengths = tf.cast(lengths, tf.float32)
        loss, IMI, KL, info_kl = compute_loss(encoder_normal, z, obs, acts, s_g, BETA, mu_enc, s_enc, mask, lengths,
                                              planner_normal, mu_plan, s_plan, training=True)
        # loss_r = IMI  + BETA*info_kl
        # KL = KL
        # find and apply gradients with total loss

    actor_vars = [v for v in actor.trainable_variables if 'log_std' not in v.name]
    gradients = tape.gradient(loss, encoder.trainable_variables + actor_vars + planner.trainable_variables)
    optimizer.apply_gradients(zip(gradients, encoder.trainable_variables + actor_vars + planner.trainable_variables))

    #     planner_gradients = planner_tape.gradient(KL, planner.trainable_variables)
    #     planner_optimizer.apply_gradients(zip(planner_gradients, planner.trainable_variables))

    # return values for diagnostics
    return IMI, KL, info_kl


def test_step(obs, acts, mask, lengths, use_planner=True):
    # obs and acts are a trajectory, so get intial and goal
    s_i = obs[:, 0, :]
    range_lens = tf.expand_dims(tf.range(tf.shape(lengths)[0]), 1)
    expanded_lengths = tf.expand_dims(lengths - 1, 1)  # lengths must be subtracted by 1 to become indices.
    s_g = tf.gather_nd(obs,
                       tf.concat((range_lens, expanded_lengths), 1))  # get the actual last element of the sequencs.
    s_g = s_g[:, :ACHIEVED_GOAL_INDEX]

    # Encode the trajectory
    mu_enc, s_enc = encoder(obs[:,::2,:], acts[:,::2,:], training=True)
    encoder_normal = tfd.Normal(mu_enc, s_enc)
    z = encoder_normal.sample()

    mu_plan, s_plan = planner(s_i, s_g)
    planner_normal = tfd.Normal(mu_plan, s_plan)
    zp = planner_normal.sample()

    z = zp  # use the planner for test rollouts

    lengths = tf.cast(lengths, tf.float32)
    loss, IMI, KL, info_kl = compute_loss(encoder_normal, z, obs, acts, s_g, BETA, mu_enc, s_enc, mask, lengths,
                                          planner_normal, mu_plan, s_plan, training=True)

    # return values for diagnostics
    return IMI, KL, info_kl


def compute_kernel(x, y):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
    tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
    return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)

# @title Loss Computation and Model Save/Load
def compute_loss(normal_enc, z, obs, acts, s_g, BETA, mu_enc, s_enc, mask, lengths, normal_plan, mu_plan, s_plan,
                 training=False):
    AVG_SEQ_LEN = obs.shape[1]
    CURR_BATCH_SIZE = obs.shape[0]
    # Automatically averaged over batch size i.e. SUM_OVER_BATCH_SIZE
    true_samples = tf.random.normal(tf.stack([CURR_BATCH_SIZE, LATENT_DIM]))

    std_normal = tfd.Normal(0, 1)
    batch_avg_mean = tf.reduce_mean(mu_enc,
                                    axis=0)  # m_enc will batch_size, latent_dim. We want average mean across the batches so we end up with a latent dim size avg_mean_vector. Each dimension of the latent dim should be mean 0 avg across the batch, but individually can be different.
    batch_avg_s = tf.reduce_mean(s_enc, axis=0)
    batch_avg_normal = tfd.Normal(batch_avg_mean, batch_avg_s)
    # info_kl = tf.reduce_sum(tfd.kl_divergence(batch_avg_normal, std_normal))

    info_kl = compute_mmd(true_samples, normal_enc.sample())

    # reverse
    KL = tf.reduce_sum(tfd.kl_divergence(normal_enc, normal_plan)) / CURR_BATCH_SIZE

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

    loss = IMI + BETA * KL  # +BETA*info_kl +  # #
    return loss, IMI, KL, info_kl


def BC_train_step(obs, acts, BETA, mask, lengths, optimizer, actor):
    with tf.GradientTape() as tape:
        # obs and acts are a trajectory, so get intial and goal
        s_i = obs[:, 0, :]

        range_lens = tf.expand_dims(tf.range(tf.shape(lengths)[0]), 1)
        expanded_lengths = tf.expand_dims(lengths - 1, 1)  # lengths must be subtracted by 1 to become indices.

        s_g = tf.gather_nd(obs,
                           tf.concat((range_lens, expanded_lengths), 1))  # get the actual last element of the sequencs.
        s_g = s_g[:, :ACHIEVED_GOAL_INDEX]
        # Encode the trajectory
        mu_enc, s_enc = encoder(obs[:,::2,:], acts[:,::2,:], training=True)
        encoder_normal = tfd.Normal(mu_enc, s_enc)
        z = encoder_normal.sample()

        lengths = tf.cast(lengths, tf.float32)
        loss, IMI, info_kl = compute_loss(encoder_normal, z, obs, acts, s_g, BETA, mu_enc, s_enc, mask, lengths,
                                          training=True, act = actor)

        # find and apply gradients with total loss
    actor_vars = [v for v in actor.trainable_variables if 'log_std' not in v.name]
    gradients = tape.gradient(loss, actor_vars)
    optimizer.apply_gradients(zip(gradients,actor_vars))

    # return values for diagnostics
    return IMI, info_kl

def log_BC_metrics(summary_writer, IMI, info_KL, steps):
    with summary_writer.as_default():
        tf.summary.scalar('imi_loss', IMI, step=steps)
        tf.summary.scalar('info_kl_loss', info_KL, step=steps)


class LatentHERReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, env, obs_dim, act_dim, size, n_sampled_goal=4, goal_selection_strategy='future', goal_dist=0.5,
                 z_dist=0.1):
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
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def compute_latent_reward(self,env, achieved_goal, desired_goal, achieved_z, desired_z):
        distance = np.sum(abs(achieved_goal - desired_goal))
        z_dist = np.sum(abs(achieved_z - desired_z))

        r = -(z_dist)
        if distance <= self.goal_dist:
            r += 1
        return r

        # could be a goal, or both goal and z!

    def sample_achieved(self, transitions, transition_idx, strategy='future'):
        if strategy == 'future':
            selected_idx = np.random.choice(np.arange(transition_idx + 1, len(transitions)))
            selected_transition = transitions[selected_idx]


        elif strategy == 'final':
            selected_transition = transitions[-1]

        goal = selected_transition[0]['achieved_goal']
        # here is where we get the obs and acts for the sequence up till there.

        return goal

    # pass in an encoder if we desired reencoding of our representation learning trajectories.
    def store_hindsight_episode(self, episode, latent=False, encoder=None, baseline_actor = None):

        # get the z we were meant to follow.
        if encoder:
            # now find the z we actually followed
            seq_obs, seq_acts = episode_to_trajectory(episode, representation_learning=latent)
            achieved_mu, achieved_s = encoder(np.expand_dims(seq_obs, axis=0)[:,::2,:],
                                            np.expand_dims(seq_acts, axis=0)[:,::2,:])
            achieved_z = tf.squeeze(achieved_mu)

        for transition_idx, transition in enumerate(episode):

            if latent:
                o, a, r, o2, d, desired_z = transition

                if encoder:
                    r = self.compute_latent_reward(self.env,o['achieved_goal'], o['desired_goal'], achieved_z, desired_z)

                obs = np.concatenate([o['observation'], desired_z, o['desired_goal']])
                obs2 = np.concatenate([o2['observation'], desired_z, o2['desired_goal']])
                # r from the environment won't be accurate, we need to recompute as sparse
                # not only in terms of the goal, but also the latent vector.
            else:
                o, a, r, o2, d = transition
                obs = np.concatenate([o['observation'], o['desired_goal']])
                obs2 = np.concatenate([o2['observation'], o2['desired_goal']])


            if baseline_actor:
                obs = np.concatenate([obs, o['baseline_action']])
                obs2 = np.concatenate([obs2, o2['baseline_action']])

            self.store(obs, a, r, obs2, d)

            if transition_idx == len(episode) - 1:
                selection_strategy = 'final'
            else:
                selection_strategy = self.goal_selection_strategy


            sampled_achieved_goals = [self.sample_achieved(episode, transition_idx, selection_strategy) for _
                                      in range(self.n_sampled_goal)]

            # hindsight sub in both the goal and z. So compute the achieved goal and intended z up till that goal.
            for goal in sampled_achieved_goals:

                if latent:
                    o, a, r, o2, d, z = copy.deepcopy(transition)

                else:
                    o, a, r, o2, d = copy.deepcopy(transition)

                if encoder:
                    z = achieved_z # reasonable approximation
                    r = self.compute_latent_reward(self.env, goal, goal, achieved_z, achieved_z)
                else:
                    r = self.env.compute_reward(goal, goal, info=None)

                o['desired_goal'] = goal
                o2['desired_goal'] = goal

                if latent:
                    obs = np.concatenate([o['observation'], z, o['desired_goal']])
                    obs2 = np.concatenate([o2['observation'], z, o2['desired_goal']])
                else:
                    obs = np.concatenate([o['observation'], o['desired_goal']])
                    obs2 = np.concatenate([o2['observation'], o2['desired_goal']])

                if baseline_actor:
                    obs = np.concatenate([obs, o['baseline_action']])
                    obs2 = np.concatenate([obs2, o2['baseline_action']])

                self.store(obs, a, r, obs2, d)


    def store_episodes_batchwise(self,episodes,latent=False, encoder=None ):
        # we want to take in all the episodes
        # then for each of them compute their length
        # then for each index choose n_samples goal indices up to that length.
        # then arrange each of those sequences in a big n_samples*n_transitions, lengths, obs_dim+act_dim.
        # chuck that through encoder, get a n_samples*n_transitions, maxlen(padded), obs_dim+act_dim
        # simulataneously store the goal of each of those indices.
        # now have equal length arrays of goals, zs, and tranitions.

        raise NotImplementedError


def sample_expert_trajectory(train_set, encoder):
    obs, acts, mask, lengths = train_set.next()
    s_i = obs[:, 0, :]
    range_lens = tf.expand_dims(tf.range(tf.shape(lengths)[0]), 1)
    expanded_lengths = tf.expand_dims(lengths - 1, 1)  # lengths must be subtracted by 1 to become indices.

    s_g = tf.gather_nd(obs,
                       tf.concat((range_lens, expanded_lengths), 1))  # get the actual last element of the sequencs.
    s_g = s_g[:, :ACHIEVED_GOAL_INDEX]
    # Encode the trajectory
    # TODO fix this - Fix what? Damn you past Sholto
    mu_enc, s_enc = encoder(obs[:,::2,:], acts[:,::2,:], training=True)
    encoder_normal = tfd.Normal(mu_enc, s_enc)
    z = encoder_normal.sample()

    return s_i, z, s_g, obs, acts, mu_enc, lengths


def log_metrics(summary_writer, current_total_steps, episodes, obs, acts, z, length, encoder=None, train=True, latent = False):
    rollout_obs, rollout_acts = episode_to_trajectory(episodes[0],
                                                      representation_learning=latent)
    if encoder:
        rollout_mu_z, rollout_s_z = encoder(np.expand_dims(rollout_obs, axis=0)[:,::2,:], np.expand_dims(rollout_acts, axis=0)[:,::2,:])
        z_dist = np.mean(abs(z - np.squeeze(rollout_mu_z)))
    euclidean_dist = np.mean(abs(rollout_obs[:length, :2] - np.squeeze(obs, axis=0)[:length, :2]))
    with summary_writer.as_default():
        current_total_steps = int(current_total_steps)
        if train:
            print('Frame: ', current_total_steps, 'Euclidean Distance: ', euclidean_dist)
            if encoder: tf.summary.scalar('Z_distance', z_dist, step=current_total_steps)
            tf.summary.scalar('Euclidean Distance', euclidean_dist, step=current_total_steps)
        else:
            print('Test Frame: ', current_total_steps, ' Euclidean Distance: ', euclidean_dist)
            if encoder: tf.summary.scalar('Test Z_distance', z_dist, step=current_total_steps)
            tf.summary.scalar('Test Euclidean Distance', euclidean_dist, step=current_total_steps)




# encoder = TRAJECTORY_ENCODER_LSTM(LAYER_SIZE, LATENT_DIM, P_DROPOUT)
# actor = ACTOR(LAYER_SIZE, ACT_DIM, P_DROPOUT)
# planner = PLANNER(LAYER_SIZE, LATENT_DIM, P_DROPOUT)
# encoder, actor, planner = load_weights(extension, BATCH_SIZE, MAX_SEQ_LEN, OBS_DIM, ACT_DIM,ACHIEVED_GOAL_INDEX, LATENT_DIM,  encoder, actor, planner)

#
# encoder = VAE_Encoder(LATENT_DIM, MAX_SEQ_LEN, decimate_factor = 2) # Divide SEQLEN by two as we are decimating inputs.
# decoder = VAE_Decoder(MAX_SEQ_LEN, ACHIEVED_GOAL_INDEX)
# encoder, decoder = MLP_OBS_load_weights(extension, BATCH_SIZE, MAX_SEQ_LEN, OBS_DIM, ACT_DIM,ACHIEVED_GOAL_INDEX, LATENT_DIM, encoder, decoder)

extension = 'saved_models/Z_learning_0.005enc2plan6'

encoder = TRAJECTORY_ENCODER_LSTM(LAYER_SIZE, LATENT_DIM, P_DROPOUT)
actor = ACTOR(LAYER_SIZE, ACT_DIM, P_DROPOUT)
planner = PLANNER(LAYER_SIZE, LATENT_DIM, P_DROPOUT)
encoder, actor, planner = load_weights(extension, 8, MAX_SEQ_LEN, OBS_DIM, ACT_DIM,ACHIEVED_GOAL_INDEX, LATENT_DIM,  encoder, actor, planner)

#
# This is our training loop.
def training_loop(env_fn,  ac_kwargs=dict(), seed=0, 
        steps_per_epoch=2000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=3000, 
        max_ep_len=1000, save_freq=1, load = False, exp_name = "Experiment_1", render = False, strategy = 'future'):
    
  tf.random.set_seed(seed)
  np.random.seed(seed)
  env, test_env = env_fn(), env_fn()
  try:
    env.set_sparse_reward()
  except:
      print('Env already uses sparse rewards.')

  hindsight_encoder = None #encoder # None # turn off encoder, just have reward based on if it reaches the goal, like normal HER.
  latent =  True
  lstm_actor = actor.get_deterministic_action # a basline latent trained LSTM policy which can act as a baselien with which to sum the actions of our RL policy.


  in_dim = env.observation_space.spaces['observation'].shape[0] + env.observation_space.spaces['desired_goal'].shape[0]
  act_dim = env.action_space.shape[0]
  if latent:
       in_dim += LATENT_DIM
  if lstm_actor:
      in_dim += act_dim # because it should take into account what decision the LSTM baseline policy made.


  SAC = SAC_model(env, in_dim, act_dim, ac_kwargs['hidden_sizes'], lr, gamma, alpha, polyak, load, exp_name)

  # update the actor to our boi's weights.
  SAC.actor = ACTOR(LAYER_SIZE, act_dim, P_DROPOUT)
  SAC.build_models(BATCH_SIZE, in_dim, act_dim)

  # if latent and lstm_actor is None: # if using latent input, and we don't have an LSTM actor (which we cannot assign to the main SAC policy).
  #     assign_variables(actor, SAC.actor)        # won't work if not latent as loaded in actors are latent conditioned.
  SAC.models['actor'] = SAC.actor



  replay_buffer = LatentHERReplayBuffer(env, in_dim, act_dim, replay_size, n_sampled_goal = 4, goal_selection_strategy = strategy)
  
  
  #Logging 
  start_time = time.time()
  train_log_dir = 'logs/' + exp_name+':'+str(start_time) + '/stochastic'
  summary_writer = tf.summary.create_file_writer(train_log_dir)

  def update_models(model, replay_buffer, steps, batch_size, bc_set = None, actor= None, collected_steps = None, summary_writer = None):
    for j in range(steps):
        batch = replay_buffer.sample_batch(batch_size)
        LossPi, LossQ1, LossQ2, LossV, Q1Vals, Q2Vals, VVals, LogPi = model.train_step(batch)
    #obs, acts, mask, lengths = bc_set.next()
    #loss, kl = BC_train_step(obs, acts, BETA, mask, lengths, model.pi_optimizer, actor)
    #log_BC_metrics(summary_writer, loss, kl, collected_steps+j)

  # now collect epsiodes
  total_steps = steps_per_epoch * epochs
  steps_collected = 0
  epoch_ticker = 0

  sample_size = 20
  train_set = iter(dataset.shuffle(valid_len).batch(sample_size).repeat(EPOCHS))
  BC_set = iter(dataset.shuffle(valid_len).batch(BATCH_SIZE).repeat(EPOCHS))
  # sample a bunch of expert trajectories to try to imitate
  s_i, z, s_g, obs, acts, _, lengths = sample_expert_trajectory(train_set, encoder)
  for r in range(sample_size):
      trajectory = z[r] if latent else None
      episodes, steps = rollout_trajectories(n_steps=max_ep_len, z = trajectory, env=env, max_ep_len=lengths[r], actor=SAC.actor.get_stochastic_action,
                                             start_state=s_i[r], s_g = s_g[r], exp_name=exp_name, return_episode=True,goal_based=True,
                                             current_total_steps=int(steps_collected), summary_writer = summary_writer, lstm_actor = lstm_actor)

      steps_collected += steps
      [replay_buffer.store_hindsight_episode(episode, encoder = hindsight_encoder, latent = latent, baseline_actor = lstm_actor ) for episode in episodes]
      log_metrics(summary_writer, steps_collected, episodes, obs[r], acts[r], trajectory, lengths[r], encoder, latent = latent)


  # now update after
  update_models(SAC, replay_buffer, steps=steps_collected, batch_size=batch_size, bc_set = BC_set, actor = SAC.actor, collected_steps=steps_collected, summary_writer= summary_writer)

  # now act with our actor, and alternately collect data, then train.
  train_set = iter(dataset.shuffle(valid_len).batch(1).repeat(EPOCHS))


  # now act with our actor, and alternately collect data, then train.
  while steps_collected < total_steps:
    s_i, z, s_g, obs, acts, _, lengths = sample_expert_trajectory(train_set, encoder)
    trajectory = z[0] if latent else None
    # collect an episode
    episodes, steps = rollout_trajectories(n_steps=max_ep_len, z = trajectory, env=env, max_ep_len=lengths[0],actor=SAC.actor.get_stochastic_action,
                                           start_state = s_i[0], s_g= s_g[0],current_total_steps=int(steps_collected), exp_name=exp_name,
                                           return_episode=True, goal_based=True, summary_writer = summary_writer,lstm_actor = lstm_actor)
    steps_collected += steps
    # take than many training steps
    [replay_buffer.store_hindsight_episode(episode, encoder = hindsight_encoder, latent = latent, baseline_actor = lstm_actor) for episode in episodes]
    update_models(SAC, replay_buffer, steps = steps, batch_size = batch_size, bc_set = BC_set, actor = SAC.actor, collected_steps=steps_collected, summary_writer=summary_writer)
    log_metrics(summary_writer, steps_collected, episodes, obs[0], acts[0], trajectory,lengths[0], encoder, latent = latent)
    #log_metrics(summary_writer, steps_collected, episodes, obs[0], acts[0], lengths[0], z = None,encoder = hindsight_encoder)

    # if an epoch has elapsed, save and test.
    if steps_collected >= epoch_ticker:
        SAC.save_weights()
        for i in range(0,10):
          s_i, _, s_g, obs, acts, mu_z, lengths = sample_expert_trajectory(train_set, encoder)
          trajectory = mu_z[0] if latent else None
          episodes, steps = rollout_trajectories(n_steps=lengths[0], z = trajectory, env=test_env, max_ep_len=lengths[0],
                                                 actor=SAC.actor.get_deterministic_action,start_state = s_i[0], s_g=s_g[0],
                                                 current_total_steps=int(steps_collected), exp_name=exp_name,return_episode=True,
                                                 goal_based=True, train = False, render = True, summary_writer = summary_writer, lstm_actor = lstm_actor)

          log_metrics(summary_writer, steps_collected, episodes, obs[0], acts[0], trajectory,lengths[0], encoder, latent = latent)
        # Test the performance of the deterministic version of the agent.
        #rollout_trajectories(n_steps = max_ep_len*10,env = test_env, max_ep_len = max_ep_len, actor = SAC.actor.get_deterministic_action, summary_writer=summary_writer, current_total_steps = steps_collected, train = False, render = True, exp_name = exp_name, return_episode = True, goal_based = True)
        epoch_ticker += steps_per_epoch
        SAC.save_weights()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='pointMass-v0')
    parser.add_argument('--hid', type=int, default=128)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--max_ep_len', type=int, default=200)
    parser.add_argument('--exp_name', type=str, default='experiment_1')
    parser.add_argument('--load', type=bool, default=False)
    parser.add_argument('--render', type=bool, default=False)
    parser.add_argument('--strategy', type=str, default='future')

    args = parser.parse_args()

    experiment_name = 'HLIL_' + args.env + '_Hidden_' + str(args.hid) + 'l_' + str(args.l)

    training_loop(lambda: gym.make(args.env),
                  ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
                  gamma=args.gamma, seed=args.seed, epochs=args.epochs, load=args.load, exp_name=experiment_name,
                  max_ep_len=args.max_ep_len, render=args.render, strategy=args.strategy)




