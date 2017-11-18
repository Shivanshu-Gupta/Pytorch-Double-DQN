import sys
import yaml
import argparse
import gym
from utils.gym import get_env, get_wrapper_by_name
from agent import DQNCNNAgent

import torch

parser = argparse.ArgumentParser(description='DQN')
parser.add_argument('--config', default='params.yaml', metavar='PATH')
parser.add_argument('--save_dir', default='gym-results/', metavar='PATH')
parser.add_argument('--modelpath', default='', metavar='PATH')
args = parser.parse_args()


def main(env, max_timesteps, config):
    def stopping_criterion(env):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= max_timesteps
    agent = DQNCNNAgent(env, config)
    if config['mode'] == 'train':
        agent.train(stopping_criterion)
        agent.save()
    if config['mode'] == 'eval':
        agent.eval(10)


if __name__ == '__main__':
    config = yaml.load(open(args.config))
    config['save_dir'] = args.save_dir
    if config.modelpath:
        config['modelpath'] = args.modelpath

    if 'use_gpu' in config:
        config['use_gpu'] = config['use_gpu'] and torch.cuda.is_available()
    else:
        config['use_gpu'] = torch.cuda.is_available()
    # Get Atari games.
    benchmark = gym.benchmark_spec('Atari40M')
    task = benchmark.tasks[config['env_id']]

    # Run training
    seed = config['seed']
    env = get_env(task, seed, args.save_dir)

    main(env, task.max_timesteps, config)
