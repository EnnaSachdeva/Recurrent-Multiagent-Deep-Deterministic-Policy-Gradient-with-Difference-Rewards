import argparse
import torch
import time
import os
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.maddpg import MADDPG
from copy import copy

USE_CUDA = False

def make_parallel_env(env_id, n_rollout_threads, seed, discrete_action):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=discrete_action)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])


def run(config):
    # Make directory to store the results
    model_dir = Path('./models')/config.env_id/config.model_name
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)

    # initialize tensorboard summary writer
    logger = SummaryWriter(str(log_dir))

    # use provided seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # IDK how helpful this is
    if not USE_CUDA:
        torch.set_num_threads(config.n_training_threads)

    env = make_parallel_env(config.env_id, config.n_rollout_threads, config.seed,
                            config.discrete_action)

    maddpg = MADDPG.init_from_env(env, agent_alg=config.agent_alg,
                                  adversary_alg=config.adversary_alg,
                                  tau=config.tau,
                                  lr=config.lr,
                                  hidden_dim=config.hidden_dim,
                                  )
    if not rnn:    # TODO: this might break. code might not be modular (yet). Code works with RNN
        replay_buffer = ReplayBuffer(config.buffer_length, maddpg.nagents,
                                     [obsp.shape[0] for obsp in env.observation_space],
                                     [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                      for acsp in env.action_space])
    else:
        # replay buffer obs space size is increased
        rnn_replay_buffer = ReplayBuffer(config.buffer_length, maddpg.nagents,
                                     [obsp.shape[0]*history_steps for obsp in env.observation_space],
                                     [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                      for acsp in env.action_space])

        # This is just to store the global rewards and not for updating the policies
        g_storage_buffer = ReplayBuffer(config.buffer_length, maddpg.nagents,
                                        [obsp.shape[0]*history_steps for obsp in env.observation_space],
                                        [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                         for acsp in env.action_space])

    t = 0
    #####################################################################################################
    #                                       START EPISODES                                              #
    #####################################################################################################
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        print("Episodes %i-%i of %i" % (ep_i + 1,
                                        ep_i + 1 + config.n_rollout_threads,
                                        config.n_episodes))

        # List of Observations for each of the agents
        # E.g., For simple_spread, shape is {1,3,18}
        obs = env.reset()

        # For RNN history buffer. I know this is not modular.
        obs_tminus_0 = copy(obs)
        obs_tminus_1 = copy(obs)
        obs_tminus_2 = copy(obs)
        obs_tminus_3 = copy(obs)
        obs_tminus_4 = copy(obs)
        obs_tminus_5 = copy(obs)

        # # for 3 time-steps
        # obs_history = np.empty([1,3,54])
        # next_obs_history = np.empty([1,3,54])

        # For 6 time-steps (18*3 = 54)
        obs_history = np.empty([1,3,108])
        next_obs_history = np.empty([1,3,108])

        maddpg.prep_rollouts(device='cpu')

        # Exploration percentage remaining. IDK if this is a standard way of doing it however.
        explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
        maddpg.scale_noise(config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
        maddpg.reset_noise()

        ##################################################################################################
        #                                       START TIME-STEPS                                         #
        ##################################################################################################

        for et_i in range(config.episode_length):

            # Populate current history
            for a in range(3):  # env.nagents
                obs_history[0][a][:] = np.concatenate((obs_tminus_0[0][a][:], obs_tminus_1[0][a][:], obs_tminus_2[0][a][:],
                                                      obs_tminus_3[0][a][:], obs_tminus_4[0][a][:], obs_tminus_5[0][a][:]))
                # Now, temp has history of 6 timesteps for each agent

            if not rnn:    # TODO: This might break. Code works with RNN. !RNN not tested.
                # rearrange observations to be per agent, and convert to torch Variable
                torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                      requires_grad=False)
                             for i in range(maddpg.nagents)]

                # get actions (from learning algorithm) as torch Variables. For simple_spread this is discrete[5]
                torch_agent_actions = maddpg.step(torch_obs, explore=True)

            else:
                # rearrange histories to be per agent, and convert to torch Variable
                rnn_torch_obs = [Variable(torch.Tensor(np.vstack(obs_history[:, i])),
                                      requires_grad=False)
                             for i in range(maddpg.nagents)]
                # TODO: for RNN, actions should condition on history (DONE)
                torch_agent_actions = maddpg.step(rnn_torch_obs, explore=True)


            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]    # print(torch_agent_actions[0].data)
            # rearrange actions to be per environment. For single thread, it wont really matter.
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            next_obs, rewards, dones, infos = env.step(actions)

            ############### WHICH REWARD TO USE ##############
            # the rewards now contain global as well as difference rewards
            # Keep the global for logging, and difference for updates

            use_diff_reward = False    #TODO: THIS IS THE TYPE OF REWARD YOU USE

            # DIFFERENCE REWARDS
            d_rewards = []
            for n in range(maddpg.nagents):
                d_rewards.append([rewards[0][n][1]])
            d_rewards = [d_rewards]
            d_rewards = np.array(d_rewards)

            # GLOBAL REWARDS
            g_rewards = []
            for n in range(maddpg.nagents):
                g_rewards.append([rewards[0][n][0]])
            g_rewards = [g_rewards]
            g_rewards = np.array(g_rewards)

            # replace "reward" with the reward that you want to use
            if use_diff_reward:
                rewards = d_rewards
            else:
                rewards = g_rewards

            # Create history for next state
            '''
            history is [t, t-1, t-2]
            history[0] is because [0] is for one thread
            '''
            for a in range(3):      # env.nagents
                next_obs_history[0][a][:] = np.concatenate((next_obs[0][a][:], obs_tminus_0[0][a][:], obs_tminus_1[0][a][:],
                                                            obs_tminus_2[0][a][:], obs_tminus_3[0][a][:], obs_tminus_4[0][a][:]))
                    # Now, next_obs_history has history of 6 timesteps for each agent the next state

            # for RNN, replay buffer needs to store for e.g., states=[obs_t-2, obs_t-1, obs_t]
            if not rnn:
                replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
                obs = next_obs
            else:
                # Buffer used for updates
                rnn_replay_buffer.push(obs_history, agent_actions, rewards, next_obs_history, dones)
                # push global rewards into g_replay_buffer
                g_storage_buffer.push(obs_history, agent_actions, g_rewards, next_obs_history, dones)

            # Update histories
            obs_tminus_5 = copy(obs_tminus_4)
            obs_tminus_4 = copy(obs_tminus_3)
            obs_tminus_3 = copy(obs_tminus_2)

            obs_tminus_2 = copy(obs_tminus_1)
            obs_tminus_1 = copy(obs_tminus_0)
            obs_tminus_0 = copy(next_obs)

            t += config.n_rollout_threads
            if (len(rnn_replay_buffer) >= config.batch_size and
                (t % config.steps_per_update) < config.n_rollout_threads):
                if USE_CUDA:
                    maddpg.prep_training(device='gpu')
                else:
                    maddpg.prep_training(device='cpu')
                for u_i in range(config.n_rollout_threads):
                    for a_i in range(maddpg.nagents):
                        sample = rnn_replay_buffer.sample(config.batch_size,
                                                      to_gpu=USE_CUDA)
                        maddpg.update(sample, a_i, logger=logger)
                    maddpg.update_all_targets()
                maddpg.prep_rollouts(device='cpu')
        # For plotting, use global reward achieved using difference rewards
        ep_rews = g_storage_buffer.get_average_rewards(
            config.episode_length * config.n_rollout_threads)
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)

        if ep_i % config.save_interval < config.n_rollout_threads:
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            maddpg.save(run_dir / 'model.pt')

    maddpg.save(run_dir / 'model.pt')
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()

    print()


def parse_arguments():
    global parser
    parser = argparse.ArgumentParser(description='Multiagent Deep Deterministic Policy Gradient')
    parser.add_argument("--env_id", default='simple_spread', help="Name of environment")
    parser.add_argument("--model_name", default='Exp',
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("--seed",
                        default=1, type=int,
                        help="Random seed")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=15000, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for model training")
    parser.add_argument("--n_exploration_eps", default=10000, type=int)
    parser.add_argument("--init_noise_scale", default=0.2, type=float)
    parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--save_interval", default=500, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--agent_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--adversary_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--discrete_action",
                        action='store_false')



if __name__=="__main__":
    '''
    Having rollout_threads more than 1 will create multiple environments I reckon.
    So it can run multiple instances of the environment in parallel. This will make it run "Asynchronously".
    Basically, this should allow the algorithm to run without using replay buffer (what paper was it..?)
    However, I don't know if we can get rid of replay buffer.
    '''
    parse_arguments()
    config = parser.parse_args()
    rnn = True    # also need to set this inside MADDPG class. Make args later.
    if rnn:
        history_steps = 6

    config.discrete_action = True
    run(config)
    print("Done")

    # TODO: NOTE: if there is mismatch, change discrete_action in environment.
