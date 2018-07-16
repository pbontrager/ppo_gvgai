#!/usr/bin/env python3
#import gym
#import gym_gvgai
from gym_env import *

import os
import sys
import uuid
import argparse
from baselines import logger
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.ppo2 import ppo2
from baselines.ppo2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy, MlpPolicy
import multiprocessing
import tensorflow as tf

#Search file for #Philip to find todos
def make_path(f):
    return os.makedirs(f, exist_ok=True)

# def train(env_id, num_timesteps, seed, policy):

#     ncpu = multiprocessing.cpu_count()
#     if sys.platform == 'darwin': ncpu //= 2
#     config = tf.ConfigProto(allow_soft_placement=True,
#                             intra_op_parallelism_threads=ncpu,
#                             inter_op_parallelism_threads=ncpu)
#     config.gpu_options.allow_growth = True #pylint: disable=E1101
#     tf.Session(config=config).__enter__()

#     env = VecFrameStack(make_gvgai_env(env_id, 8, seed), 4)
#     policy = {'cnn' : CnnPolicy, 'lstm' : LstmPolicy, 'lnlstm' : LnLstmPolicy, 'mlp': MlpPolicy}[policy]
#     ppo2.learn(policy=policy, env=env, nsteps=128, nminibatches=4,
#         lam=0.95, gamma=0.99, noptepochs=4, log_interval=1,
#         ent_coef=.01,
#         lr=lambda f : f * 2.5e-4,
#         cliprange=lambda f : f * 0.1,
#         total_timesteps=int(num_timesteps * 1.1))

# def main():
#     parser = atari_arg_parser()
#     parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm', 'mlp'], default='cnn')
#     args = parser.parse_args()
#     logger.configure()
#     print("Loading Zelda")
#     train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
#         policy=args.policy)

def train(env_id, model, num_envs, num_timesteps, lrschedule, save_interval, seed):

    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    sess = tf.Session(config=config).__enter__()
   
    print("Starting experiment")

    # Level selector
    #level_path = './results/' + experiment_name + '/levels/' + experiment_id + '/'
    #level_selector = LevelSelector.get_selector(args.selector, args.game, level_path)

    # Make gym environment
    #env = make_gvgai_env(env_id=env_id,
    #                     num_env=args.num_envs,
    #                     seed=args.seed,
    #                     level_selector=level_selector)
    env = VecFrameStack(make_gvgai_env(env_id, num_envs, seed), 4)

    # Select model
    policy = {'cnn' : CnnPolicy, 'lstm' : LstmPolicy, 'lnlstm' : LnLstmPolicy, 'mlp': MlpPolicy}[model]

    #Philip: how to resume?, lrschedule is not used yet
    ppo2.learn(policy=policy, env=env, nsteps=128, nminibatches=4,
        lam=0.95, gamma=0.99, noptepochs=4, log_interval=1,
        save_interval=save_interval,
        ent_coef=.01,
        lr=lambda f : f * 2.5e-4,
        cliprange=lambda f : f * 0.1,
        total_timesteps=int(num_timesteps * 1.1))

    #Verify there are no features here that I still want
    #learn(policy=policy,
    #      env=env,
    #      experiment_name=experiment_name,
    #      experiment_id=experiment_id,
    #      seed=args.seed,
    #      total_timesteps=args.num_timesteps,
    #      lrschedule=args.lrschedule,
    #      frame_skip=False,
    #      save_interval=args.save_interval,
    #      level_selector=level_selector,
    #      render=args.render)

    env.close()

    print("Experiment DONE")

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm'], default='cnn')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='constant')
    parser.add_argument('--num-envs', help='Number of environments/workers to run in parallel (default=12)', type=int, default=8)
    parser.add_argument('--num-timesteps', help='Number of timesteps to train the model', type=int, default=int(1e6))
    parser.add_argument('--game', help='Game name (default=zelda)', default='zelda')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--save-interval', help='Model saving interval in steps', type=int, default=int(1e5))
    parser.add_argument('--level', help='Level (integer) to train on', type=int, default=0)
    parser.add_argument('--resume', help='The experiment id to resume', default=None)
    #parser.add_argument('--repetitions', help='Number of repetitions to run sequentially (default=1)', type=int, default=1)
    #parser.add_argument('--selector', help='Level selector to use in training - will ignore the level argument if set (default: None)',
    #                    choices=[None] + LevelSelector.available, default=None)
    parser.add_argument('--render', action='store_true', default=False,
                        help='Render screen (default: False)')
    args = parser.parse_args()

    # Gym environment name
    env_id = "gvgai-" + args.game + "-lvl" + str(args.level) + "-v0"

    experiment_path = "./results"
    make_path(experiment_path)

    # Experiment name
    experiment_name = args.game + "-lvl-" + str(args.level)
    experiment_path += "/" + experiment_name
    make_path(experiment_path)

    # Unique id for experiment
    if args.resume is None:
        experiment_id = str(uuid.uuid1())
    else:
        experiment_id = args.resume
    experiment_path += "/" + experiment_id
    make_path(experiment_path)

    logger.configure(dir=experiment_path)

    train(env_id=env_id, model=args.policy, num_envs=args.num_envs, 
        num_timesteps=args.num_timesteps, lrschedule=args.lrschedule, 
        save_interval=args.save_interval, seed=args.seed)

    

if __name__ == '__main__':
    #brew upgrade cmake
    main()
