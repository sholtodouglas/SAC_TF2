import numpy as np
import tensorflow as tf
import gym
import time
import pybullet
import reach2D
from SAC import *
from common import *
from gym import wrappers
# expert_obs = np.load('collected_data/expert_obs_Pendulum-v0_Hidden_32l_21000.npy')
# expert_acts = np.load('collected_data/expert_actions_Pendulum-v0_Hidden_32l_21000.npy')

# expert_obs= np.load('collected_data/expert_obs_reacher2D-v0_Hidden_128l_25000.npy').astype('float32')
# expert_acts = np.load('collected_data/expert_actions_reacher2D-v0_Hidden_128l_25000.npy').astype('float32')

# expert_obs= np.load('collected_data/10000no_reset_vel_pointMass-v0_Hidden_128l_2expert_obs_.npy').astype('float32')
# expert_acts= np.load('collected_data/10000no_reset_vel_pointMass-v0_Hidden_128l_2expert_actions.npy').astype('float32')


expert_obs= np.load('collected_data/11000seg_pointMass-v0_Hidden_128l_2expert_obs_.npy').astype('float32')
expert_acts= np.load('collected_data/11000seg_pointMass-v0_Hidden_128l_2expert_actions.npy').astype('float32')



# a replay buffer where we can purge non informative transitions?
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
                    done=self.done_buf[idxs],
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


def discriminator_train_step(batch, expert_batch,  discriminator, discriminator_optimizer, replay_buffer, batch_size = 100, discrim_req_acc = 0.7):
    batch_obs, batch_acts = batch['obs1'], batch['acts']
    batch_expert_obs, batch_expert_acts = expert_batch['obs'], expert_batch['acts']
    
    with tf.GradientTape() as tape:
        # We'd like to maximise the log probability of the expert actions, and minmise log prob of generated actions.
        expert_probs = discriminator(batch_expert_obs,batch_expert_acts)
        # in ML, we take gradient to minimise, therefore minimise negative log probability 
        # the +1e8 is for numerical stability, as log(0) = inf
        expert_loss = -tf.math.log(expert_probs+1e-8)
        agent_probs = discriminator(batch_obs,batch_acts)
        
        # purge useless transitions

        to_purge  = np.where(agent_probs.numpy().squeeze() < 0.2)[0]
        if len(to_purge) > 0:
            idxs_to_purge = batch['idxs'][to_purge]
            print("Purging: ", idxs_to_purge, "with values", agent_probs.numpy()[to_purge])
            replay_buffer.purge(idxs_to_purge, batch_size)
        # i.e, minimise -log(1-prob_generated_is_true)
        agent_loss = -(tf.math.log(1-agent_probs+1e-8))
        # and thus, the reward our SAC agent gets will be -(tf.math.log(1-agent_probs)), it is trying to maximise this, 
        # discriminator is trying to mimise it.
        loss = tf.reduce_sum(expert_loss) + tf.reduce_sum(agent_loss)
        expert_accuracy = tf.reduce_mean(tf.cast(expert_probs > 0.5, tf.float32))
        agent_accuracy  = tf.reduce_mean(tf.cast(agent_probs < 0.5, tf.float32))
        


        
    gradients = tape.gradient(loss, discriminator.trainable_variables)
    
    if agent_accuracy < discrim_req_acc or expert_accuracy < discrim_req_acc:
        print('Updating Discriminator')
        discriminator_optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))

    return loss.numpy(), expert_accuracy.numpy(), agent_accuracy.numpy()


def sample_expert_transitions(batch_size):
    idxs = np.random.randint(0, len(expert_obs), size=batch_size)
    return {'obs':expert_obs[idxs], 'acts':expert_acts[idxs]}

# Behavioural clone this mf.
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




def GAIL(env_fn,  ac_kwargs=dict(), seed=0, 
        steps_per_epoch=5000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=5000, 
        max_ep_len=1000, save_freq=1, load = False, exp_name = "Experiment_1", render = False, discrim_req_acc = 0.7, BC = False):

    tf.random.set_seed(seed)
    np.random.seed(seed)
    env, test_env = env_fn(), env_fn()
    env = wrappers.FlattenDictWrapper(env, dict_keys=['observation', 'desired_goal'])
    test_env  = wrappers.FlattenDictWrapper(test_env , dict_keys=['observation', 'desired_goal'])
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

    GAIL(lambda : gym.make(args.env), 
      ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
      gamma=args.gamma, seed=args.seed, epochs=args.epochs, load = args.load, exp_name = experiment_name, max_ep_len = args.max_ep_len, render = args.render, discrim_req_acc = args.discrim_req_acc, BC = args.BC)


