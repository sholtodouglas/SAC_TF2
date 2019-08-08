import numpy as np
import tensorflow as tf
import time
import datetime
from tqdm import tqdm
from tensorflow.keras.layers import Dense, Flatten, Conv2D,Bidirectional, LSTM, Dropout
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
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
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as PathEffects
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import pandas as pd
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
extension = '/Z_learning_B005'

# @title Dataset
observations = np.load('collected_data/10000GAILpointMass-v0_Hidden_128l_2expert_obs_.npy').astype(
    'float32')[:, 0:OBS_GOAL_INDEX]  # Don't include the goal in the obs
actions = np.load('collected_data/10000GAILpointMass-v0_Hidden_128l_2expert_actions.npy').astype('float32')
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

        self.bi_lstm = Bidirectional(CuDNNLSTM(LAYER_SIZE, return_sequences=True), merge_mode=None)
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

encoder = TRAJECTORY_ENCODER_LSTM(LAYER_SIZE, LATENT_DIM, P_DROPOUT)
actor = ACTOR(LAYER_SIZE, ACT_DIM, P_DROPOUT)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
load_weights(extension)