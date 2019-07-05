import numpy as np
import tensorflow as tf
import gym
import time

from core import *


# End Core, begin SAC.
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
                    done=self.done_buf[idxs])
"""

Soft Actor-Critic

(With slight variations that bring it closer to TD3)

"""

def sac(env_fn, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=5000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=5000, 
        max_ep_len=1000, logger_kwargs=dict(), save_freq=1, load = False, exp_name = 'Experiment_1'):
    """

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: A function which takes in placeholder symbols 
            for state, ``x_ph``, and action, ``a_ph``, and returns the main 
            outputs from the agent's Tensorflow computation graph:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``mu``       (batch, act_dim)  | Computes mean actions from policy
                                           | given states.
            ``pi``       (batch, act_dim)  | Samples actions from policy given 
                                           | states.
            ``logp_pi``  (batch,)          | Gives log probability, according to
                                           | the policy, of the action sampled by
                                           | ``pi``. Critical: must be differentiable
                                           | with respect to policy parameters all
                                           | the way through action sampling.
            ``q1``       (batch,)          | Gives one estimate of Q* for 
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q2``       (batch,)          | Gives another estimate of Q* for 
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q1_pi``    (batch,)          | Gives the composition of ``q1`` and 
                                           | ``pi`` for states in ``x_ph``: 
                                           | q1(x, pi(x)).
            ``q2_pi``    (batch,)          | Gives the composition of ``q2`` and 
                                           | ``pi`` for states in ``x_ph``: 
                                           | q2(x, pi(x)).
            ``v``        (batch,)          | Gives the value estimate for states
                                           | in ``x_ph``. 
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic 
            function you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to 
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    """

    Soft Actor-Critic

    (With slight variations that bring it closer to TD3)

    """


    # logger = EpochLogger(**logger_kwargs)

    tf.random.set_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
        # Get Env dimensions
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space
    print(ac_kwargs)
    hidden_sizes = ac_kwargs['hidden_sizes']
    activation = 'relu'
    # Creater actor network
    actor = mlp_gaussian_policy(act_dim, act_limit, hidden_sizes, activation, None)
    # Create two q functions
    q_func1 = mlp(hidden_sizes+[1], activation, None)
    q_func2 = mlp(hidden_sizes+[1], activation, None)
    # create value and target value functions
    value = mlp(hidden_sizes+[1], activation, None)
    value_targ = mlp(hidden_sizes+[1], activation, None)

    models = {'actor':actor, 'q1':q_func1, 'q2':q_func2, 'value':value}
    # build them - TODO convert this into a function which loads prev variables or builds
    build_models(actor, q_func1, q_func2, value, value_targ, batch_size, obs_dim, act_dim)

    if load:
      
        load_weights(models, exp_name)

    pi_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    value_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables
    var_counts = tuple(count_vars(model) for model in 
                       [actor, q_func1, q_func2, value])
    print(('\nNumber of parameters: \t pi: %d, \t' + \
           'q1: %d, \t q2: %d, \t v: %d')%var_counts)

    @tf.function
    def train_step(batch):
      with tf.GradientTape() as actor_tape, tf.GradientTape() as value_tape:
        x = batch['obs1']
        x2 = batch['obs2']
        a = batch['acts']
        r = batch['rews']
        d = batch['done']
        
        mu, pi, logp_pi = actor(x)
        
        q1 = tf.squeeze(q_func1(tf.concat([x,a], axis=-1)))
        q1_pi = tf.squeeze(q_func1(tf.concat([x,pi], axis=-1)))
        q2 = tf.squeeze(q_func2(tf.concat([x,a], axis=-1)))
        q2_pi = tf.squeeze(q_func2(tf.concat([x,pi], axis=-1)))
        v = tf.squeeze(value(x))
        v_targ = tf.squeeze(value_targ(x2))

        # Min Double-Q:
        min_q_pi = tf.minimum(q1_pi, q2_pi)

        # Targets for Q and V regression
        q_backup = tf.stop_gradient(r + gamma*(1-d)*v_targ)
        v_backup = tf.stop_gradient(min_q_pi - alpha * logp_pi)

        # Soft actor-critic losses
        pi_loss = tf.reduce_mean(input_tensor=alpha * logp_pi - q1_pi)
        q1_loss = 0.5 * tf.reduce_mean(input_tensor=(q_backup - q1)**2)
        q2_loss = 0.5 * tf.reduce_mean(input_tensor=(q_backup - q2)**2)
        v_loss = 0.5 * tf.reduce_mean(input_tensor=(v_backup - v)**2)
        value_loss = q1_loss + q2_loss + v_loss

      # Policy train step
      # (has to be separate from value train step, because q1_pi appears in pi_loss)
      pi_gradients = actor_tape.gradient(pi_loss, actor.trainable_variables)
      pi_optimizer.apply_gradients(zip(pi_gradients, actor.trainable_variables))
      
      # Value train step
      value_variables = q_func1.trainable_variables + q_func2.trainable_variables + value.trainable_variables
      value_gradients = value_tape.gradient(value_loss, value_variables)
      #One notable byproduct of eager execution is that tf.control_dependencies() is no longer required, as all lines of code execute in order (within a tf.function, code with side effects execute in the order written).
      # Therefore, should no longer need controldependencies here. 
      value_optimizer.apply_gradients(zip(value_gradients, value_variables))
      
      # Polyak averaging for target variables
      for v_main, v_targ in zip(value.trainable_variables, value_targ.trainable_variables):
        tf.compat.v1.assign(v_targ, polyak*v_targ + (1-polyak)*v_main) 
        
      return pi_loss, q1_loss, q2_loss, v_loss, q1, q2, v, logp_pi


    def get_action(o, deterministic=False):
        mu, pi, logp_pi = actor(o.reshape(1,-1))
        return  mu[0] if deterministic else pi[0]

    def test_agent(n=10):
        for j in range(n):
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time 
                o, r, d, _ = test_env.step(get_action(o, True))
                ep_ret += r
                ep_len += 1
            with train_summary_writer.as_default():
              tf.summary.scalar('test_episode_return', ep_ret, step=t)


    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    total_steps = steps_per_epoch * epochs

    train_log_dir = 'logs/gradient_tape/' + str(start_time) + '/stochastic'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        """
        Until start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards, 
        use the learned policy. 
        """
        if t > start_steps:
            a = get_action(o)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        if d or (ep_len == max_ep_len):
            """
            Perform all SAC updates at the end of the trajectory.
            This is a slight difference from the SAC specified in the
            original paper.
            """
            for j in range(ep_len):
                batch = replay_buffer.sample_batch(batch_size)
                
                LossPi, LossQ1, LossQ2, LossV, Q1Vals, Q2Vals, VVals, LogPi = train_step(batch)

                # logger.store(LossPi=LossPi, LossQ1=LossQ1, LossQ2=LossQ2,
                #              LossV=LossV, Q1Vals=Q1Vals, Q2Vals=Q2Vals,
                #              VVals=VVals, LogPi=LogPi)

            # logger.store(EpRet=ep_ret, EpLen=ep_len)
            with train_summary_writer.as_default():
              print('Frame: ', t, '\nEp Ret: ', ep_ret, '\nLossPi: ', LossPi.numpy(), '\nLossQ1: ',LossQ1.numpy(), '\nLossQ2: ', LossQ2.numpy(), '\nLossV: ', LossV.numpy(), '\nAvg Q1Vals: ', np.mean(Q1Vals.numpy()), '\nAvg Q2Vals: ', np.mean(Q2Vals.numpy()), '\nAvg VVals: ', np.mean(VVals.numpy()), '\nAvg LogPi: ',np.mean(LogPi.numpy()))
              tf.summary.scalar('episode_return', ep_ret, step=t)
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0


        # End of epoch wrap-up
        if t > 0 and t % steps_per_epoch == 0:
            epoch = t // steps_per_epoch

            # # Save model
            if (epoch % save_freq == 0) or (epoch == epochs-1):
                save_weights(models, exp_name)

            # Test the performance of the deterministic version of the agent.
            test_agent()
            print('-----------------Epoch ', epoch,'-----------------')
            print('Frame: ', t, '\nEp Ret: ', ep_ret, '\nLossPi: ', LossPi.numpy(), '\nLossQ1: ',LossQ1.numpy(), '\nLossQ2: ', LossQ2.numpy(), '\nLossV: ', LossV.numpy(), '\nAvg Q1Vals: ', np.mean(Q1Vals.numpy()), '\nAvg Q2Vals: ', np.mean(Q2Vals.numpy()), '\nAvg VVals: ', np.mean(VVals.numpy()), '\nAvg LogPi: ',np.mean(LogPi.numpy()))
            print('----------------------------------------')

            # Log info about epoch
            # logger.log_tabular('Epoch', epoch)
            # logger.log_tabular('EpRet', with_min_and_max=True)
            # logger.log_tabular('TestEpRet', with_min_and_max=True)
            # logger.log_tabular('EpLen', average_only=True)
            # logger.log_tabular('TestEpLen', average_only=True)
            # logger.log_tabular('TotalEnvInteracts', t)
            # logger.log_tabular('Q1Vals', with_min_and_max=True) 
            # logger.log_tabular('Q2Vals', with_min_and_max=True) 
            # logger.log_tabular('VVals', with_min_and_max=True) 
            # logger.log_tabular('LogPi', with_min_and_max=True)
            # logger.log_tabular('LossPi', average_only=True)
            # logger.log_tabular('LossQ1', average_only=True)
            # logger.log_tabular('LossQ2', average_only=True)
            # logger.log_tabular('LossV', average_only=True)
            # logger.log_tabular('Time', time.time()-start_time)
            # logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Pendulum-v0')
    parser.add_argument('--hid', type=int, default=32)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='experiment_1')
    parser.add_argument('--load', type=bool, default=False)
    args = parser.parse_args()

    # from logx import setup_logger_kwargs
    # logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    sac(lambda : gym.make(args.env), 
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
        gamma=args.gamma, seed=args.seed, epochs=args.epochs, load = args.load, exp_name = args.exp_name)
