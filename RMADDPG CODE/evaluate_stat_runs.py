import argparse
import torch
import time
import imageio
import numpy as np
from pathlib import Path
from torch.autograd import Variable
from utils.make_env import make_env
from algorithms.maddpg import MADDPG
from copy import copy
import pickle as pkl


def run(config):
    original_model_path = (Path('./models') / config.env_id / config.model_name /
                  ('run%i' % config.run_num))
    # if config.incremental is not None:
    #     model_path = model_path / 'incremental' / ('model_ep%i.pt' %
    #                                                config.incremental)
    # else:
    #     model_path = model_path / 'model.pt'
    #
    # print(model_path)

    ###########################################################################
    #                      FORCE MODEL PATH                                   #
    ###########################################################################
    model_path_list = []
    rrange = [1, 1001, 2001, 3001, 4001, 5001, 6001, 7001,
              8001, 9001]

    # FOR EACH MODEL, DO STATISTICAL RUNS
    # for r in rrange:
    #     model_path = model_path / 'incremental' / ('model_ep%i.pt' % r)

    ######################  SAVING STAT RUNS FOR EACH MODEL ###################
    stat_run_all_models = []

    for r in rrange:
        model_path = original_model_path / 'incremental' / ('model_ep%i.pt' % r)
        if config.save_gifs:
            gif_path = model_path.parent / 'gifs'
            gif_path.mkdir(exist_ok=True)

        maddpg = MADDPG.init_from_save(model_path)
        env = make_env(config.env_id, discrete_action=maddpg.discrete_action)
        maddpg.prep_rollouts(device='cpu')
        ifi = 1 / config.fps  # inter-frame interval

        #####################################################################################################
        #                             CONFIGURATION FOR STATISTICAL RUNS (EPISODES)
        #####################################################################################################
        #####################################################################################################
        #                                       START EPISODES                                              #
        #####################################################################################################
        stat_return_list = []
        for ep_i in range(config.n_episodes):  # number of stat runs
            print("Episode %i of %i" % (ep_i + 1, config.n_episodes))
            obs = env.reset()
            # For RNN history buffer
            obs_tminus_0 = copy(obs)
            obs_tminus_1 = copy(obs)
            obs_tminus_2 = copy(obs)
            obs_tminus_3 = copy(obs)
            obs_tminus_4 = copy(obs)
            obs_tminus_5 = copy(obs)

            # TODO: obs_history shape different from main.py, so parameterize it based on "obs"
            # It is different because main.py can run multiple threads, so has an extra dimension
            obs_history = np.empty([3,108])
            next_obs_history = np.empty([3,108])

            if config.save_gifs:
                frames = []
                frames.append(env.render('rgb_array')[0])
            #env.render('human')

            ##################################################################################################
            #                                       START TIME-STEPS                                         #
            ##################################################################################################
            episode_reward = 0
            for t_i in range(config.episode_length):

                # Populate current history for RNN
                for a in range(3):  # env.nagents
                        #obs_history[a][:] = np.concatenate((obs_tminus_0[a][:], obs_tminus_1[a][:], obs_tminus_2[a][:]))
                        obs_history[a][:] = np.concatenate(
                            (obs_tminus_0[a][:], obs_tminus_1[a][:], obs_tminus_2[a][:],
                             obs_tminus_3[a][:], obs_tminus_4[a][:], obs_tminus_5[a][:]))
                        # Now, temp has history of 6 timesteps for each agent

                calc_start = time.time()

                # rearrange observations to be per agent, and convert to torch Variable
                rnn_torch_obs = [Variable(torch.Tensor(obs_history[i]).view(1, -1),
                                      requires_grad=False)
                             for i in range(maddpg.nagents)]
                # get actions as torch Variables
                torch_actions = maddpg.step(rnn_torch_obs, explore=False)
                # convert actions to numpy arrays
                actions = [ac.data.numpy().flatten() for ac in torch_actions]
                next_obs, rewards, dones, infos = env.step(actions)

                # get the global reward
                episode_reward += rewards[0][0]

                # Update histories
                obs_tminus_5 = copy(obs_tminus_4)
                obs_tminus_4 = copy(obs_tminus_3)
                obs_tminus_3 = copy(obs_tminus_2)
                obs_tminus_2 = copy(obs_tminus_1)
                obs_tminus_1 = copy(obs_tminus_0)
                obs_tminus_0 = copy(next_obs)
                # --------------------------------------#

                if config.save_gifs:
                    frames.append(env.render('rgb_array')[0])
                calc_end = time.time()
                elapsed = calc_end - calc_start
                if elapsed < ifi:
                    time.sleep(ifi - elapsed)
                #env.render('human')
                # end of an episode

            if config.save_gifs:
                gif_num = 0
                while (gif_path / ('%i_%i.gif' % (gif_num, ep_i))).exists():
                    gif_num += 1
                imageio.mimsave(str(gif_path / ('%i_%i.gif' % (gif_num, ep_i))),
                                frames, duration=ifi)
            # end of episodes (one stat-run)
            stat_return_list.append(episode_reward / config.episode_length)
        # end of model
        stat_run_all_models.append(stat_return_list)
        env.close()

    pickling_on = open(str(original_model_path)+"/stat_runs", "wb")
    pkl.dump(stat_run_all_models, pickling_on)
    pickling_on.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="simple_spread", help="Name of environment")
    parser.add_argument("--model_name", default="Exp",
                        help="Name of model")
    parser.add_argument("--run_num", default=62, type=int)
    parser.add_argument("--save_gifs", action="store_true",
                        help="Saves gif of each episode into model directory")
    parser.add_argument("--incremental", default=None, type=int,
                        help="Load incremental policy from given episode " +
                             "rather than final policy")
    parser.add_argument("--n_episodes", default=15, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--fps", default=30, type=int)

    config = parser.parse_args()

    # FORCE SETTINGS
    config.save_gifs = False
    # config.incremental = False

    run(config)
