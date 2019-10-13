#!/usr/bin/env python
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

import numpy as np
import cloudpickle # For pickling lambda functions and more

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


# In[2]:

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

        self.policy = EpsGreedy(0.1) if policy is None else policy
        self.test_policy = Greedy() if test_policy is None else test_policy

        self.memsize = memsize
        self.memory = memory.PrioritizedExperienceReplay(memsize, nsteps)

        self.target_update = target_update
        self.gamma = gamma
        self.batch_size = batch_size
        self.nsteps = nsteps
        self.training = True

        # Extension options
        self.enable_double_dqn = enable_double_dqn
        self.enable_dueling_network = enable_dueling_network
        self.dueling_type = dueling_type

        self.model =model

        # Define loss function that computes the MSE between target Q-values and cumulative discounted rewards
        # If using PrioritizedExperienceReplay, the loss function also computes the TD error and updates the trace priorities
        def masked_q_loss(data, y_pred):
            """Computes the MSE between the Q-values of the actions that were taken and	the cumulative discounted
            rewards obtained after taking those actions. Updates trace priorities if using PrioritizedExperienceReplay.
            """
            action_batch, target_qvals = data[:, 0], data[:, 1]
            seq = tf.cast(tf.range(0, tf.shape(action_batch)[0]), tf.int32)
            action_idxs = tf.transpose(tf.stack([seq, tf.cast(action_batch, tf.int32)]))
            qvals = tf.gather_nd(y_pred, action_idxs)

            if isinstance(self.memory, memory.PrioritizedExperienceReplay):
                def update_priorities(_qvals, _target_qvals, _traces_idxs):
                    """Computes the TD error and updates memory priorities."""
                    td_error = np.abs((_target_qvals - _qvals).numpy())
                    _traces_idxs = (tf.cast(_traces_idxs, tf.int32)).numpy()
                    self.memory.update_priorities(_traces_idxs, td_error)
                    return _qvals
                qvals = tf.py_function(func=update_priorities, inp=[qvals, target_qvals, data[:,2]], Tout=tf.float32)
            return tf.keras.losses.mse(qvals, target_qvals)

        self.model.compile(optimizer=self.optimizer, loss=masked_q_loss)

        # Clone model to use for delayed Q targets
        self.target_model = tf.keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

    def save(self, filename, overwrite=False):
        """Saves the model parameters to the specified file."""
        self.model.save_weights(filename, overwrite=overwrite)

    def act(self, state, instance=0):
        """Returns the action to be taken given a state."""


        qvals = self.model.predict(np.array([state]))[0]
        # plt.imshow(state)
        # plt.imshow(np.reshape(qvals, [128,128]), alpha = 0.5, cmap = 'plasma')
        # plt.savefig('q_overlay')
        # # plt.show()
        # we know our original shape is 1,128,128,1
        world_range = 0.26 * 2
        pixel_range = 128
        mid = pixel_range / 2
        index = self.policy.act(qvals) if self.training else self.test_policy.act(qvals)
        pixel_index = np.unravel_index(index, [128,128])
        return np.array(list((np.array(pixel_index)-mid)/(pixel_range/world_range)) + list([0])),  index# eventually have it from the depth map

    def push(self, transition, instance=0):
        """Stores the transition in memory."""
        self.memory.put(transition)

    def train(self, step):
        """Trains the agent for one step."""
        if len(self.memory) == 0:
            return

        # Update target network
        if self.target_update >= 1 and step % self.target_update == 0:
            # Perform a hard update
            self.target_model.set_weights(self.model.get_weights())
        elif self.target_update < 1:
            # Perform a soft update
            mw = np.array(self.model.get_weights())
            tmw = np.array(self.target_model.get_weights())
            self.target_model.set_weights(self.target_update * mw + (1 - self.target_update) * tmw)

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

                q_values = self.model.predict_on_batch(np.array(non_final_last_next_states))
                actions = np.argmax(q_values, axis=1)
                # Estimate Q-values using the target network but select the values with the
                # highest Q-value wrt to the online model (as computed above).
                target_q_values = self.target_model.predict_on_batch(np.array(non_final_last_next_states))
                selected_target_q_vals = target_q_values[range(len(target_q_values)), actions]
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
        loss_data = [action_batch, target_qvals]

        # If using PrioritizedExperienceReplay then we need to provide the trace indexes
        # to the loss function as well so we can update the priorities of the traces
        if isinstance(self.memory, memory.PrioritizedExperienceReplay):
            loss_data.append(self.memory.last_traces_idxs())

        # Train model

        self.model.train_on_batch(np.array(state_batch), np.stack(loss_data).transpose())


# In[ ]:




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
        states = [env.reset()['observation'][0] for env in envs] # get the image

        for step in range(max_steps):
            for i in range(instances):
                if visualize: envs[i].render()
                action, action_index = self.agent.act(states[i], i)
                next_state, reward, done, _ = envs[i].step(action)
                (next_image, next_depth) = next_state['observation']
                self.agent.push(Transition(states[i], action_index, reward, None if done else next_image), i)
                episode_rewards[i] += reward
                if done:
                    episode_reward_sequences[i].append(episode_rewards[i])
                    episode_step_sequences[i].append(step)
                    episode_rewards[i] = 0
                    if plot: plot(episode_reward_sequences, episode_step_sequences)
                    (image, depth) =  envs[i].reset()['observation']
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

# Build a simple neural network with 3 fully connected layers as our model
# model = Sequential([
#     Dense(16, activation='relu', input_shape=dummy_env.observation_space.shape),
#     Dense(16, activation='relu'),
#     Dense(16, activation='relu'),
# ])

inputs = tf.keras.Input(shape=(128,128,3), name='img')
x = Conv2D(filters=32, kernel_size=4, strides=2, padding='same')(inputs)
x = LeakyReLU()(x)
x =  Conv2D(filters=64, kernel_size=4, strides=2, padding='same')(x)
x = LeakyReLU()(x)
x =  Conv2D(filters=128, kernel_size=4, strides=1, padding='same')(x)
x = LeakyReLU()(x)
x =  Conv2D(filters=256, kernel_size=4, strides=1, padding='same')(x)
x = LeakyReLU()(x)
x =  Conv2DTranspose(filters=32,kernel_size=4,strides=2,padding='same')(x)
x = LeakyReLU()(x)
outputs =  Conv2DTranspose(filters=1,kernel_size=4,strides=2,padding='same')(x)
outputs = Flatten()(outputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs, name='model')
print(model.summary())

# Create Deep Q-Learning Network agent
#agent = DQN(model, actions=dummy_env.action_space.n, nsteps=3)

agent = DQN(model, nsteps=2)

def plot_rewards(episode_rewards, episode_steps, done=False):
    plt.clf()
    plt.xlabel('Step')
    plt.ylabel('Reward')
    for ed, steps in zip(episode_rewards, episode_steps):
        plt.plot(steps, ed)
    plt.show() if done else plt.pause(0.001) # Pause a bit so that the graph is updated

# Create simulation, train and then test
sim = Simulation(create_env, agent)
model.save('convolutional_boi.h5')
sim.train(max_steps=3000, visualize=True, plot=plot_rewards)
model.save('convolutional_boi.h5')
sim.test(max_steps=1000)


# In[4]:


sim.test(max_steps=1000)


# In[ ]:




