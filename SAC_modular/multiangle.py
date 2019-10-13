# coding: utf-8

# In[5]:


from tensorflow.keras.layers import Dense, Lambda, Conv2D, Flatten, LeakyReLU, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


import tensorflow as tf
import numpy as np

from huskarl.policy import EpsGreedy, Greedy
from huskarl.core import Agent, HkException
from huskarl import memory
from tensorflow.keras.models import Sequential
from itertools import count
from collections import namedtuple
from queue import Empty
from time import sleep
import multiprocessing as mp
import random
import numpy as np
import cloudpickle # For pickling lambda functions and more
import math
from huskarl.memory import Transition
from huskarl.core import HkException

import matplotlib.pyplot as plt
import gym
import ur5_RL
import huskarl as hk

from scipy import ndimage, misc
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

if tf.test.gpu_device_name() != '/device:GPU:0':
    print('WARNING: GPU device not found.')
else:
    print('SUCCESS: Found GPU: {}'.format(tf.test.gpu_device_name()))


def rotate_around_point_highperf(xy, degrees, origin=(0, 0)):
    """Rotate a point around a given point.

    I call this the "high performance" version since we're caching some
    values that are needed >1 time. It's less readable than the previous
    function but it's faster.
    """
    x, y = xy
    radians = degrees * math.pi / 180
    offset_x, offset_y = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = math.cos(radians)
    sin_rad = math.sin(radians)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y

    return qx, qy
# In[2]:
class FNC_Q(Model):
    def __init__(self):
        super(FNC_Q, self).__init__()
        # self.flatten = Flatten()
        self.c1 = Conv2D(filters=32, kernel_size=4, strides=2, padding='same', activation=tf.nn.leaky_relu)
        self.c2 = Conv2D(filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.leaky_relu)
        self.c3 = Conv2D(filters=128, kernel_size=4, strides=1, padding='same', activation=tf.nn.leaky_relu)
        self.c4 = Conv2D(filters=256, kernel_size=4, strides=1, padding='same', activation=tf.nn.leaky_relu)

        self.up1 = Conv2DTranspose(filters=32, kernel_size=4, strides=2, padding='same', activation=tf.nn.leaky_relu)
        self.up2 = Conv2DTranspose(filters=1, kernel_size=4, strides=2, padding='same', activation=tf.nn.leaky_relu)
        self.flatten = Flatten()

    def call(self, x):
        # x = self.flatten(x)
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)

        x = self.up1(x)
        x = self.up2(x)
        x = self.flatten(x)
        return x


class DQN(Agent):
    """Deep Q-Learning Network

    Base implementation:
        "Playing Atari with Deep Reinforcement Learning" (Mnih et al., 2013)

    Extensions:
        Multi-step returns: "Reinforcement Learning: An Introduction" 2nd ed. (Sutton & Barto, 2018)
        Double Q-Learning: "Deep Reinforcement Learning with Double Q-learning" (van Hasselt et al., 2015)
        Dueling Q-Network: "Dueling Network Architectures for Deep Reinforcement Learning" (Wang et al., 2016)
    """

    def __init__(self, model, optimizer=None, policy=None, test_policy=None,
                 memsize=10_000, target_update=3, gamma=0.6, batch_size=32, nsteps=1,
                 enable_double_dqn=True, enable_dueling_network=False, dueling_type='avg'):
        """
        TODO: Describe parameters
        """
        self.optimizer = Adam(lr=3e-3) if optimizer is None else optimizer

        self.policy = 0.1 if policy is None else policy  # eps greedy 0.1
        self.test_policy = 0 if test_policy is None else test_policy  # fully greedy

        self.memsize = memsize
        self.memory = memory.PrioritizedExperienceReplay(memsize, nsteps)

        self.target_update = target_update
        self.gamma = gamma
        self.batch_size = batch_size
        self.nsteps = nsteps
        self.training = True
        self.pixel_length = 128
        self.num_angles = 8

        # Extension options
        self.enable_double_dqn = enable_double_dqn
        self.enable_dueling_network = enable_dueling_network
        self.dueling_type = dueling_type

        self.model = model()

        # Clone model to use for delayed Q targets
        self.target_model = model()
        self.build_models()
        # set target weights to main weights
        for v_main, v_targ in zip(self.model.trainable_variables, self.target_model.trainable_variables):
            tf.compat.v1.assign(v_targ, v_main)

    def save(self, filename, overwrite=False):
        """Saves the model parameters to the specified file."""
        self.model.save_weights(filename, overwrite=overwrite)

    def build_models(self):
        self.model(np.zeros([1, self.pixel_length, self.pixel_length, 3]))
        self.target_model(np.zeros([1, self.pixel_length, self.pixel_length, 3]))

    def best_q_value(self, states):
        B, H, W, D = states.shape

        angle_range = tf.range(0, self.num_angles) * 180 / self.num_angles
        images = []
        for i in range(0, B):
            for a in range(0, self.num_angles):
                # negative here is critical!
                rotated_image = ndimage.rotate(states[i], -angle_range[a], reshape=False)
                images.append(rotated_image)
        images = np.array(images)

        q_values = self.model(images)
        resh = tf.reshape(q_values, [B, self.num_angles, -1])
        # reshape now is B, num angles, -1
        # we want per batch, the maximum
        values, best_pixels = tf.math.top_k(resh)
        values, best_angles = tf.math.top_k(tf.squeeze(values, axis = -1))
        best_pixels = tf.squeeze(best_pixels, axis = -1)
        best_angles = tf.squeeze(best_angles, axis = -1)

        if len(best_angles.shape) == 0:
            best_pixels = best_pixels[best_angles]
            return values, best_angles, best_pixels
        else:
            indices = tf.transpose(tf.stack([tf.range(0, B, 1), best_angles]))
            best_pixels = tf.gather_nd(best_pixels, indices)
            # note, we now still need to back calculate.
        return values, best_angles, best_pixels

    # this function computes the q values with a given model at the specified indices
    def specified_q_value(self, model, states, angles, pixel_indices):

        B, H, W, D = states.shape
        angle_range = tf.range(0, self.num_angles) * 180 / self.num_angles
        images = []
        for i in range(0, B):
            # negative here is critical!
            rotated_image = ndimage.rotate(states[i], -angle_range[angles[i]], reshape=False)
            images.append(rotated_image)
        images = np.array(images)

        q_values = model(images)
        reshape = tf.reshape(q_values, [B, -1])

        indices = tf.transpose(tf.stack([tf.range(0, B, 1), pixel_indices]))
        q_values = tf.gather_nd(reshape, indices)
        return q_values

    def act(self, state, instance=0):
        """Returns the action to be taken given a state."""

        q, angle, index = self.best_q_value(np.expand_dims(state, 0))
        angle, index = np.squeeze(angle.numpy()), np.squeeze(index.numpy())
        if self.training:
            if random.random() < self.policy:  # eps greedy it.
                print('episolon')
                angle = random.randrange(self.num_angles)
                index = random.randrange(self.pixel_length * self.pixel_length)

        # plt.imshow(state)
        # plt.imshow(np.reshape(qvals, [128,128]), alpha = 0.5, cmap = 'plasma')
        # plt.savefig('q_overlay')
        # # plt.show()
        # we know our original shape is 1,128,128,1
        world_range = 0.26 * 2
        mid = self.pixel_length / 2
        pixel_index = np.unravel_index(index, [self.pixel_length, self.pixel_length])
        angle_range = tf.range(0, self.num_angles) * 180 / self.num_angles
        angle_to_robot = angle_range[angle]
        pixel_index = np.array(rotate_around_point_highperf((pixel_index[0], pixel_index[1]), -angle_to_robot, (mid, mid))).astype(int)
        pixels = (np.array(pixel_index) - mid) / (self.pixel_length / world_range)
        # also need to put the pixel back in the original frames reference, as it will be in the rotated one
        # for some reason our rotation function needs it negative for equivalency. Meh.


        return np.array(list(pixels) + list([0]) + list(
            [angle_to_robot])), angle, index  # eventually have it from the depth map

    def push(self, transition, instance=0):
        """Stores the transition in memory."""
        self.memory.put(transition)

    def train(self, step):
        """Trains the agent for one step."""
        if len(self.memory) <= 1:
            return

        # Update target network
        if self.target_update >= 1 and step % self.target_update == 0:
            # Perform a hard update
            for v_main, v_targ in zip(self.model.trainable_variables, self.target_model.trainable_variables):
                tf.compat.v1.assign(v_targ, v_main)
        elif self.target_update < 1:
            # Perform a soft update
            for v_main, v_targ in zip(self.model.trainable_variables, self.target_model.trainable_variables):
                tf.compat.v1.assign(v_targ, self.target_update * v_targ + (1 - self.target_update) * v_main)

                # Train even when memory has fewer than the specified batch_size
        batch_size = min(len(self.memory), self.batch_size)

        # Sample batch_size traces from memory
        state_batch, action_batch, reward_batches, end_state_batch, not_done_mask = self.memory.get(batch_size)

        # Compute the value of the last next states
        target_qvals = np.zeros(batch_size)
        non_final_last_next_states = [es for es in end_state_batch if es is not None]

        if len(non_final_last_next_states) > 0:
            if self.enable_double_dqn:
                # "Deep Reinforcement Learning with Double Q-learning" (van Hasselt et al., 2015)
                # The online network predicts the actions while the target network is used to estimate the Q-values
                values, best_angles, best_pixels = self.best_q_value(np.array(non_final_last_next_states))
                # q_values = self.model.predict_on_batch(np.array(non_final_last_next_states))
                # actions = np.argmax(q_values, axis=1)
                # Estimate Q-values using the target network but select the values with the
                # highest Q-value wrt to the online model (as computed above).
                # target_q_values = self.target_model.predict_on_batch(np.array(non_final_last_next_states))
                selected_target_q_vals = self.specified_q_value(self.target_model, np.array(non_final_last_next_states),
                                                                best_angles, best_pixels)
                # selected_target_q_vals = target_q_values[range(len(target_q_values)), actions]
            else:
                # Use delayed target network to compute target Q-values
                selected_target_q_vals = self.target_model.predict_on_batch(np.array(non_final_last_next_states)).max(1)
            non_final_mask = list(map(lambda s: s is not None, end_state_batch))
            target_qvals[non_final_mask] = selected_target_q_vals

        # Compute n-step discounted return
        # If episode ended within any sampled nstep trace - zero out remaining rewards
        for n in reversed(range(self.nsteps)):
            rewards = np.array([b[n] for b in reward_batches])
            target_qvals *= np.array([t[n] for t in not_done_mask])
            target_qvals = rewards + (self.gamma * target_qvals)

        # Compile information needed by the custom loss function
        # If using PrioritizedExperienceReplay then we need to provide the trace indexes
        # to the loss function as well so we can update the priorities of the traces
        # Train model
        target_qvals = target_qvals.astype(np.float32)
        state_batch = np.array(state_batch).astype(np.float32)

        with tf.GradientTape() as tape:
            angles = np.array(action_batch)[:, 0]
            pixel_indices = np.array(action_batch)[:, 1]
            qvals = self.specified_q_value(self.model, state_batch, angles, pixel_indices)
            loss = tf.keras.losses.mse(qvals, target_qvals)
            print('loss', loss)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        if isinstance(self.memory, memory.PrioritizedExperienceReplay):
            td_error = np.abs((target_qvals - qvals).numpy())
            traces_idxs = self.memory.last_traces_idxs()
            self.memory.update_priorities(traces_idxs, td_error)


# Packet used to transmit experience from environment subprocesses to main process
# The first packet of every episode will have reward set to None
# The last packet of every episode will have state set to None
RewardState = namedtuple('RewardState', ['reward', 'state'])


class Simulation:
    """Simulates an agent interacting with one of multiple environments."""

    def __init__(self, create_env, agent, mapping=None):
        self.create_env = create_env
        self.agent = agent
        self.mapping = mapping

    def train(self, max_steps=100_000, instances=1, visualize=False, plot=None, max_subprocesses=0):
        """Trains the agent on the specified number of environment instances."""
        self.agent.training = True
        if max_subprocesses == 0:
            # Use single process implementation
            self._sp_train(max_steps, instances, visualize, plot)
        elif max_subprocesses is None or max_subprocesses > 0:
            # Use multiprocess implementation
            self._mp_train(max_steps, instances, visualize, plot, max_subprocesses)
        else:
            raise HkException(f"Invalid max_subprocesses setting: {max_subprocesses}")

    def _sp_train(self, max_steps, instances, visualize, plot):
        """Trains using a single process."""
        # Keep track of rewards per episode per instance
        episode_reward_sequences = [[] for i in range(instances)]
        episode_step_sequences = [[] for i in range(instances)]
        episode_rewards = [0] * instances

        # Create and initialize environment instances
        envs = [self.create_env() for i in range(instances)]
        envs[0].render(mode='human')
        states = [env.reset()['observation'][0] for env in envs]  # get the image

        for step in range(max_steps):
            for i in range(instances):
                if visualize: envs[i].render()
                action, angle_index, action_index = self.agent.act(states[i], i)

                next_state, reward, done, _ = envs[i].step(action)
                (next_image, next_depth) = next_state['observation']
                self.agent.push(
                    Transition(states[i], [angle_index, action_index], reward, None if done else next_image), i)
                episode_rewards[i] += reward
                if done:
                    episode_reward_sequences[i].append(episode_rewards[i])
                    episode_step_sequences[i].append(step)
                    episode_rewards[i] = 0
                    if plot: plot(episode_reward_sequences, episode_step_sequences)
                    (image, depth) = envs[i].reset()['observation']
                    states[i] = image
                else:
                    states[i] = next_image
            # Perform one step of the optimization
            self.agent.train(step)

        if plot: plot(episode_reward_sequences, episode_step_sequences, done=True)


# In[3]:


# Setup gym environment
create_env = lambda: gym.make('ur5_RL_lego-v0')
dummy_env = create_env()

model = FNC_Q
# Create Deep Q-Learning Network agent
# agent = DQN(model, actions=dummy_env.action_space.n, nsteps=3)

agent = DQN(model, nsteps=2)


def plot_rewards(episode_rewards, episode_steps, done=False):
    plt.clf()
    plt.xlabel('Step')
    plt.ylabel('Reward')
    for ed, steps in zip(episode_rewards, episode_steps):
        plt.plot(steps, ed)
    plt.show() if done else plt.pause(0.001)  # Pause a bit so that the graph is updated


# Create simulation, train and then test
sim = Simulation(create_env, agent)

sim.train(max_steps=3000, visualize=True, plot=plot_rewards)
agent.model.save_weights('agent_multi_angle.h5')
sim.test(max_steps=1000)

# In[4]:


sim.test(max_steps=1000)

# In[ ]:

