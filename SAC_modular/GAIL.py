import numpy as np
import tensorflow as tf
import gym
import time
import pybullet
import reach2D
from SAC import *
from train import *

expert_obs = np.load('collected_data/expert_obs_Pendulum-v0_Hidden_32l_21000.npy')
expert_acts = np.load('collected_data/expert_actions_Pendulum-v0_Hidden_32l_21000.npy')

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


def discriminator_train_step(batch, expert_batch,  discriminator, discriminator_optimizer):
    batch_obs, batch_acts = batch['obs1'], batch['acts']
    batch_expert_obs, batch_expert_acts = expert_batch['obs'], expert_batch['acts']
    with tf.GradientTape() as tape:
        # We'd like to maximise the log probability of the expert actions, and minmise log prob of generated actions.
        expert_probs = discriminator(batch_expert_obs,batch_expert_acts)
        # in ML, we take gradient to minimise, therefore minimise negative log probability 
        expert_loss = -tf.math.log(expert_probs)
        agent_probs = discriminator(batch_obs,batch_acts)
        # i.e, minimise -log(1-prob_generated_is_true)
        agent_loss = -(tf.math.log(1-agent_probs))
        # and thus, the reward our SAC agent gets will be -(tf.math.log(1-agent_probs)), it is trying to maximise this, 
        # discriminator is trying to mimise it.
        loss = tf.reduce_sum(expert_loss + agent_loss)
        expert_accuracy = tf.reduce_mean(tf.cast(expert_probs > 0.5, tf.float32))
        agent_accuracy  = tf.reduce_mean(tf.cast(agent_probs < 0.5, tf.float32))


        
    gradients = tape.gradient(loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))
    return loss.numpy(), expert_accuracy.numpy(), agent_accuracy.numpy()


def sample_expert_transitions(batch_size):
    idxs = np.random.randint(0, len(expert_obs), size=batch_size)
    return {'obs':expert_obs[idxs], 'acts':expert_acts[idxs]}


def training_loop(env_fn,  ac_kwargs=dict(), seed=0, 
        steps_per_epoch=5000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=5000, 
        max_ep_len=1000, save_freq=1, load = False, exp_name = "Experiment_1", render = False, discrim_req_acc = 0.7):

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

    def update_models(model, replay_buffer, steps, batch_size):
        agent_accuracy = 0
        # until the discriminator is trained to sufficiently distinguish correct transitions.
        print('Updating Discriminator')
        while agent_accuracy < discrim_req_acc or expert_acurracy < discrim_req_acc:
            batch = replay_buffer.sample_batch(batch_size)
            expert_batch = sample_expert_transitions(batch_size)
            _,expert_acurracy,agent_accuracy = discriminator_train_step(batch, expert_batch, discriminator, discriminator_optimizer)
            print(expert_acurracy, agent_accuracy)

        # now update SAC
        print('Updating Policy')
        for j in range(steps):
            batch = replay_buffer.sample_batch(batch_size)
            batch_obs, batch_acts = batch['obs1'], batch['acts']
            agent_probs = discriminator(batch_obs,batch_acts)
            agent_reward = -(tf.math.log(1-agent_probs)).numpy().squeeze().astype('float32')
            # use GAIL reward instead of environment reward
            batch['rews'] = agent_reward

            LossPi, LossQ1, LossQ2, LossV, Q1Vals, Q2Vals, VVals, LogPi = model.train_step(batch)


    # now collect epsiodes
    total_steps = steps_per_epoch * epochs
    steps_collected = 0

    # collect some initial random steps to initialise
    random_steps = 5000
    steps_collected  += rollout_trajectories(n_steps = random_steps,env = env, max_ep_len = max_ep_len, actor = 'random', replay_buffer = replay_buffer, summary_writer = summary_writer)


    update_models(SAC, replay_buffer, steps = random_steps, batch_size = batch_size)

    # now act with our actor, and alternately collect data, then train.
    while steps_collected < total_steps:
    # collect an episode
        steps_collected  += rollout_trajectories(n_steps = max_ep_len,env = env, max_ep_len = max_ep_len, actor = SAC.get_action, replay_buffer = replay_buffer, summary_writer=summary_writer, current_total_steps = steps_collected)
        # take than many training steps
        update_models(SAC, replay_buffer, steps = max_ep_len, batch_size = batch_size)

        # if an epoch has elapsed, save and test.
        if steps_collected  > 0 and steps_collected  % steps_per_epoch == 0:
            #SAC.save_weights()
            # Test the performance of the deterministic version of the agent.
            rollout_trajectories(n_steps = max_ep_len*10,env = test_env, max_ep_len = max_ep_len, actor = SAC.get_deterministic_action, summary_writer=summary_writer, current_total_steps = steps_collected, train = False, render = True)


# MODIFIABLE VARIBALES TODO PROPERLY PUT THIS IN A CLASS
#ENV_NAME='reacher2D-v0'
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Pendulum-v0')
    parser.add_argument('--hid', type=int, default=32)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--max_ep_len', type=int, default=200)
    parser.add_argument('--exp_name', type=str, default='experiment_1')
    parser.add_argument('--load', type=bool, default=False)
    parser.add_argument('--render', type=bool, default=False)
    parser.add_argument('--discrim_req_acc', type = float, default = 0.7)
    args = parser.parse_args()

    experiment_name = args.env+'_Hidden_'+str(args.hid)+'l_'+str(args.l)

    training_loop(lambda : gym.make(args.env), 
      ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
      gamma=args.gamma, seed=args.seed, epochs=args.epochs, load = args.load, exp_name = experiment_name, max_ep_len = args.max_ep_len, render = args.render, discrim_req_acc = args.discrim_req_acc)


