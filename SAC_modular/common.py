

import tensorflow as tf
import numpy as np
import copy
import time


def assign_variables(net1, net2):
  for main, targ in zip(net1.trainable_variables, net2.trainable_variables):
    tf.compat.v1.assign(targ, main)

  # collects a n_steps steps for the replay buffer.


# collects a n_steps steps for the replay buffer.
# Arguments
# -- Replay Trajectory, if this is passed in a sequence of actions will be replayed to demonstrate that the environment is determinsitic if the same actions are applied as in a demo from the same s_i
# --Compare_states, this is the corresponding sequence of states, if this is passed in the corresponding state will be recorded in the replay buffer, so that reward can be computed
#   the direct euclidean distance between states acheived and demo states, to test the non discriminator parts of our algorithim work.
# collects a n_steps steps for the replay buffer.
# Arguments
# -- Replay Trajectory, if this is passed in a sequence of actions will be replayed to demonstrate that the environment is determinsitic if the same actions are applied as in a demo from the same s_i
# --Compare_states, this is the corresponding sequence of states, if this is passed in the corresponding state will be recorded in the replay buffer, so that reward can be computed
#   the direct euclidean distance between states acheived and demo states, to test the non discriminator parts of our algorithim work.
# --Return episode: Returns a list of transitions, either to convert to a trajectory for plotting/encoding, or for direct insertion into a HER buffer.
def episode_to_trajectory(episode, include_goal=False, flattened=False, representation_learning=False):
    # episode arrives as a list of o, a, r, o2, d
    # trajectory is two lists, one of o s, one of a s.
    observations = []
    actions = []
    for transition in episode:
        if representation_learning:
            o, a, r, o2, d, z = transition
        else:
            o, a, r, o2, d = transition
        if flattened:
            observations.append(o)
        else:
            if include_goal:
                observations.append(np.concatenate(o['observation'], o['desired_goal']))
            else:
                observations.append(o['observation'])
        actions.append(a)

    return np.array(observations), np.array(actions)


# collects a n_steps steps for the replay buffer.
# Arguments
# -- Replay Trajectory, if this is passed in a sequence of actions will be replayed to demonstrate that the environment is determinsitic if the same actions are applied as in a demo from the same s_i
# --Compare_states, this is the corresponding sequence of states, if this is passed in the corresponding state will be recorded in the replay buffer, so that reward can be computed
#   the direct euclidean distance between states acheived and demo states, to test the non discriminator parts of our algorithim work.
# lstm_actor is a secondary, lstm based actor which we don't train with RL, but can act as a guideline.
def rollout_trajectories(n_steps, env, max_ep_len=200, actor=None, replay_buffer=None, summary_writer=None,
                         current_total_steps=0,
                         render=False, train=True, exp_name=None, z=None, s_g=None, return_episode=False,
                         replay_trajectory=None,
                         compare_states=None, start_state=None, goal_based=False, lstm_actor=None,
                         only_use_baseline=False,
                         replay_obs=None, extra_info=None, end_on_reward = False):


    # reset the environment
    def set_init(o, env, extra_info):

        if extra_info is not None:

            env.initialize_start_pos(start_state, extra_info)
        else:
            env.initialize_start_pos(start_state)  # init vel to 0, but x and y to the desired pos.
        o['observation'] = start_state


        return o

    ###################  quick fix for the need for this to activate rendering pre env reset.  ###################
    ###################  MUST BE A BETTER WAY? Env realising it needs to change pybullet client?  ###################
    if 'reacher' in exp_name or 'point' in exp_name or 'robot' in exp_name or 'UR5' in exp_name:
        pybullet = True
    else:
        pybullet = False

    if pybullet:
        if render:
            # have to do it beforehand to connect up the pybullet GUI
            env.render(mode='human')

    ###################  ###################  ###################  ###################  ###################

    if z is not None and s_g is not None:
        z_learning = True
    else:
        z_learning = False

    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    if start_state is not None:
        o = set_init(o, env, extra_info)
    if s_g is not None:
        env.reset_goal_pos(s_g)

    # if we want to store expert actions

    if return_episode:
        episode_buffer = []
        episode = []

    if lstm_actor is not None:
        past_state = [None]

    for t in range(n_steps):

        if actor == 'random':
            a = env.action_space.sample()
        elif replay_trajectory is not None:  # replay a trajectory that we've fed in so that we can make sure this is properly deterministic and compare it to our estimated action based trajectory/

            a = replay_trajectory[t]
        elif replay_obs is not None:

            a, past_state = lstm_actor(np.concatenate([replay_obs[t], z, o['desired_goal']], axis=0),
                                       past_state=past_state)
        elif z_learning:
            if lstm_actor is not None:
                # print(o['observation'],z,o['desired_goal'])
                a_base, past_state = lstm_actor(np.concatenate([o['observation'], z, o['desired_goal']], axis=0),
                                                past_state=past_state)

                if only_use_baseline:
                    a = a_base
                else:
                    o['baseline_action'] =a_base # extend the observation by the lstm_actor's act.
                    a = actor(np.concatenate([o['observation'], z, o['desired_goal'], a_base], axis=0))
            else:
                a = actor(np.concatenate([o['observation'], z, o['desired_goal']], axis=0))
        elif goal_based:

            a = actor(np.concatenate([o['observation'], o['desired_goal']], axis=0))

        else:

            a = actor(o)
        # Step the env
        # if not train:
        #     print(list(a.numpy()),',')
        if lstm_actor is not None and only_use_baseline is False:
            o2, r, d, info = env.step(
                a + a_base)  # final action is the sum of the baseline, and the adjustment by our RL learnt actor.
            o2['baseline_action'] = a_base
        else:

            o2, r, d, info = env.step(a)


        #       if z_learning: # need to include z and s_g in the obs for the replay buffer
        #         o2 = np.concatenate([o2['observation'],z,o2['desired_goal']], axis = 0)

        if render:
            env.render(mode='human')

        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d
        
        # Store experience to replay buffer # dont use a replay buffer with HER, because we need to take entire episodes and do some computation so that happens after.
        if replay_buffer:
            #         if compare_states != None:
            #           # if we want to use the direct corresponding expert state for our reward function
            #           if t < (n_steps-1):
            #             exp_state = compare_states[t+1,:]
            #           else:
            #             exp_state = s_g
            #           exp_state = np.concatenate([exp_state,z,s_g], axis = 0)
            #           replay_buffer.store(o, a, r, o2, d, exp_state)
            #         else:
            replay_buffer.store(o, a, r, o2, d)

        if return_episode:
            if z_learning:
                episode.append([o, a, r, o2, d, z])
            else:
                episode.append([o, np.array(a), r, o2, d])  # add the full transition to the episode.

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!

        o = o2
        # if either we've ended an episdoe, collected all the steps or have reached max ep len and
        # thus need to log ep reward and reset
        if end_on_reward:
            if r >= 0:
                d = 1

        if d or (ep_len == int(max_ep_len)) or (t == int((n_steps - 1))):
            if return_episode:
                episode_buffer.append(episode)
                episode = []
            if summary_writer:
                with summary_writer.as_default():
                    if train:
                        print('Frame: ', t + current_total_steps, ' Return: ', ep_ret)
                        tf.summary.scalar('Episode_return', ep_ret, step=t + current_total_steps)
                    else:
                        print('Test Frame: ', t + current_total_steps, ' Return: ', ep_ret)
                        tf.summary.scalar('Test_Episode_return', ep_ret, step=t + current_total_steps)
                        tf.summary.scalar('Test Success', info['is_success'], step = t+current_total_steps)
            # reset the env if there are still steps to collect
            if t < n_steps - 1:
                o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

            # else it will be the end of the loop and of the function.
    if return_episode:
        return {'episodes': episode_buffer, 'n_steps': n_steps}
    return n_steps


# used in expert collection, and with conversion of episodes to HER
# used in expert collection, and with conversion of episodes to HER
#extra info is for resetting determinsiticly.
def episode_to_trajectory(episode, flattened = False, representation_learning = False, include_extra_info= False):
  # episode arrives as a list of o, a, r, o2, d
  # trajectory is two lists, one of o s, one of a s.
  observations = []
  actions = []
  if include_extra_info:
    extra_info= []
  for transition in episode:
    if representation_learning:
      o, a, r, o2, d, z = transition
    else:
      o, a, r, o2, d = transition

    if flattened:
      observations.append(o)
    else:
      observations.append(o['observation'])

    if include_extra_info:
        extra_info.append(o['extra_info'])

    actions.append(a)
  if include_extra_info:
    return np.array(observations), np.array(actions), np.array(extra_info)
  return np.array(observations), np.array(actions)
