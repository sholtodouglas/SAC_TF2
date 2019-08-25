import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Bidirectional, LSTM, Dropout
import tensorflow_probability as tfp

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions
tfb = tfp.bijectors
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
import numpy as np

drive_path = ''

class Dataloader():
    def __init__(self, observations, actions,  MIN_SEQ_LEN = 15, MAX_SEQ_LEN =60, train_test_split = 0.9,):

        self.OBS_DIM = observations.shape[1]
        self.ACT_DIM = actions.shape[1]
        self.MIN_SEQ_LEN = MIN_SEQ_LEN
        self.MAX_SEQ_LEN = MAX_SEQ_LEN
        self.train_test_split = train_test_split
        self.train_len = int(len(observations) * train_test_split)
        self.train_obs_subset = observations[:self.train_len, :]
        self.train_acts_subset = actions[:self.train_len, :]
        self.valid_obs_subset = observations[self.train_len:, :]
        self.valid_acts_subset = actions[self.train_len:, :]
        self.train_len = len(self.train_obs_subset) - MAX_SEQ_LEN
        self.valid_len = len(self.valid_obs_subset) - MAX_SEQ_LEN


    def data_generator(self,unnecessary_input, subset):

      if subset == b'Train':
            set_len = self.train_len
            obs_set = self.train_obs_subset
            act_set = self.train_acts_subset
      if subset == b'Valid':
            set_len = self.valid_len
            obs_set = self.valid_obs_subset
            act_set = self.valid_acts_subset

      for idx in range(0, set_len):
          # yield the observation randomly between min and max sequence length.
          length = np.random.randint(self.MIN_SEQ_LEN, (self.MAX_SEQ_LEN))

          if length % 2 != 0:
            length -= 1


          obs_padding = np.zeros((self.MAX_SEQ_LEN-length, self.OBS_DIM))



          padded_obs = np.concatenate((obs_set[idx:idx+length], obs_padding), axis = 0)

          act_padding = np.zeros((self.MAX_SEQ_LEN-length, self.ACT_DIM))
          padded_act = np.concatenate((act_set[idx:idx+length], act_padding), axis = 0)

          # ones to length of actions, zeros for the rest to mask out loss.
          mask = np.concatenate((np.ones((length, self.ACT_DIM)),act_padding), axis = 0 )

          if len(padded_obs) != self.MAX_SEQ_LEN:
            print(idx, length, len(padded_obs))

          if len(padded_act) != self.MAX_SEQ_LEN:
            print(idx, length, len(padded_act))


          yield (padded_obs, padded_act, mask, length)


class TRAJECTORY_ENCODER_LSTM(Model):
    def __init__(self, LAYER_SIZE, LATENT_DIM, P_DROPOUT):
        super(TRAJECTORY_ENCODER_LSTM, self).__init__()

        # Ensure all these arguments are defined so that it can use CuDNNRNN if the correct
        # hardware is available.
        self.bi_lstm = Bidirectional(
            LSTM(LAYER_SIZE, return_sequences=True, activation='tanh', recurrent_activation='sigmoid',
                 recurrent_dropout=0, use_bias=True), merge_mode=None)

        self.mu = Dense(LATENT_DIM)
        # self.log_std = Dense(LATENT_DIM)
        self.std = Dense(LATENT_DIM, activation='softplus')
        self.dropout1 = tf.keras.layers.Dropout(P_DROPOUT)
        self.frame_skip = 2
        self.ENCODE_ACTS = True

    def call(self, obs, acts, training=False):
        if self.ENCODE_ACTS:
            x = tf.concat([obs, acts], axis=2)  # concat observations and actions together.
        else:
            x = obs

        x = x[:, ::self.frame_skip, :]

        x = self.bi_lstm(x)
        x = self.dropout1(x, training=training)
        bottom = x[0][:, -1, :]  # Take the last element of the bottom row
        top = x[1][:, 0, :]  # Take the first elemetn of the top row cause Bidirectional, top row goes backward.
        x = tf.concat([bottom, top], axis=1)
        mu = self.mu(x)
        # log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (self.log_std(x) + 1)
        # s = tf.exp(log_std)
        s = self.std(x)
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


class PLANNER(Model):
    def __init__(self, LAYER_SIZE, LATENT_DIM, P_DROPOUT):
        super(PLANNER, self).__init__()

        self.d1 = Dense(LAYER_SIZE, activation='relu')
        self.d2 = Dense(LAYER_SIZE, activation='relu')
        self.mu = Dense(LATENT_DIM)
        self.scale = Dense(LATENT_DIM, activation='softplus')

        self.dropout1 = tf.keras.layers.Dropout(P_DROPOUT)
        self.dropout2 = tf.keras.layers.Dropout(P_DROPOUT)
        self.dropout3 = tf.keras.layers.Dropout(P_DROPOUT)

    def call(self, s_i, s_g, training=False):
        x = tf.concat([s_i, s_g], axis=1)
        x = self.d1(x)
        x = self.dropout1(x, training=training)
        x = self.d2(x)
        x = self.dropout2(x, training=training)
        mu = self.mu(x)
        s = self.scale(x)

        return mu, s


class VAE_Encoder(Model):
    def __init__(self, LATENT_DIM, MAX_SEQ_LEN, decimate_factor = 2):
        super(VAE_Encoder, self).__init__()
        self.flatten = Flatten()
        self.d1 = Dense(400, activation='relu')
        self.d2 = Dense(400, activation='relu')
        self.mu = Dense(LATENT_DIM)
        self.scale = Dense(LATENT_DIM, activation='softplus')
        self.seq_len  = MAX_SEQ_LEN//decimate_factor



    def call(self, x, acts=None, training=False):
        # check if input is too short, if so, zero pad?

        if len(x.shape) == 3:
            batch = x.shape[0]
            time = x.shape[1]
            obs = x.shape[2]
            if time < self.seq_len: #then pad

                padding = np.zeros((batch,self.seq_len-time,obs))
                x = np.concatenate((x,padding), axis= 1)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        mu = self.mu(x)
        s = self.scale(x)
        return mu, s


class VAE_Decoder(Model):
    def __init__(self, MAX_SEQ_LEN, ACHIEVED_GOAL_INDEX):
        super(VAE_Decoder, self).__init__()
        self.flatten = Flatten()
        self.d1 = Dense(400, activation='relu')
        self.d2 = Dense(400, activation='relu')
        self.out = Dense(MAX_SEQ_LEN * ACHIEVED_GOAL_INDEX)  # SEQ_LEN * 2 for x y coords.

    def call(self, x):
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        out = self.out(x)

        return out


def load_weights(extension, BATCH_SIZE, MAX_SEQ_LEN, OBS_DIM, ACT_DIM,ACHIEVED_GOAL_INDEX, LATENT_DIM, encoder, actor, planner = None):
    print('Loading in network weights...')
    # load some sample data to initialise the model
    #         load_set = iter(self.dataset.shuffle(self.TRAIN_LEN).batch(self.BATCH_SIZE))
    obs = tf.zeros((BATCH_SIZE, MAX_SEQ_LEN, OBS_DIM))
    acts = tf.zeros((BATCH_SIZE, MAX_SEQ_LEN, ACT_DIM))
    mask = acts
    lengths = tf.cast(tf.ones(BATCH_SIZE), tf.int32)

    s_i = obs[:, 0, :]
    s_g = obs[:, -1,:ACHIEVED_GOAL_INDEX]
    GOAL_DIM = s_g.shape[-1]
    _,_ = encoder(obs, acts, training=True)
    _,_ = planner(s_i, s_g)
    _,_,_,_,_ = actor(tf.zeros((BATCH_SIZE, MAX_SEQ_LEN, OBS_DIM+LATENT_DIM+GOAL_DIM)))


    print('Models Initalised')

    encoder.load_weights(extension + '/encoder.h5')
    actor.load_weights(extension + '/actor.h5')
    try:
        planner.load_weights(extension + '/planner.h5')
    except:
        print('No planner in these weights')
    print('Weights loaded.')
    return encoder, actor, planner



def save_weights(extension, encoder, actor, planner):
    try:
        os.mkdir(extension)
    except Exception as e:
        # print(e)
        pass

    # print(extension)
    actor.save_weights(extension + '/actor.h5')
    encoder.save_weights(extension + '/encoder.h5')
    planner.save_weights(extension + '/planner.h5')




def MLP_OBS_load_weights(extension, BATCH_SIZE, MAX_SEQ_LEN, OBS_DIM, ACT_DIM,ACHIEVED_GOAL_INDEX, LATENT_DIM, encoder, decoder):
    print('Loading in network weights...')
    # load some sample data to initialise the model
    #         load_set = iter(self.dataset.shuffle(self.TRAIN_LEN).batch(self.BATCH_SIZE))
    obs = tf.zeros((BATCH_SIZE, MAX_SEQ_LEN, OBS_DIM))
    acts = tf.zeros((BATCH_SIZE, MAX_SEQ_LEN, ACT_DIM))
    mask = acts
    lengths = tf.cast(tf.ones(BATCH_SIZE), tf.int32)
    s_i = obs[:, 0, :]
    s_g = obs[:, -1,:ACHIEVED_GOAL_INDEX]
    GOAL_DIM = s_g.shape[-1]
    _,_ = encoder(obs[:,::2,:])
    _ =  decoder(tf.zeros((BATCH_SIZE, LATENT_DIM)))
    print('Models Initalised')
    encoder.load_weights(extension + '/encoder.h5')
    decoder.load_weights(extension + '/decoder.h5')
    print('Weights loaded.')
    return encoder, decoder


