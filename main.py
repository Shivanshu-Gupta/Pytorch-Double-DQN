import sys
import yaml
import gym
from utils.gym import get_env, get_wrapper_by_name
from agent import DQNCNNAgent

import torch


def main(env, max_timesteps, args):
    def stopping_criterion(env):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= max_timesteps
    agent = DQNCNNAgent(env, args)
    if args['mode'] == 'train':
        agent.train(stopping_criterion)
        agent.save()
    if args['mode'] == 'eval':
        agent.eval(10)


if __name__ == '__main__':
    assert(len(sys.argv) >= 3)
    param_file = sys.argv[1]
    save_dir = sys.argv[2]
    args = yaml.load(open(param_file))
    args['save_dir'] = save_dir
    if len(sys.argv) > 3:
        args['modelpath'] = sys.argv[3]

    if 'use_gpu' in args:
        args['use_gpu'] = args['use_gpu'] and torch.cuda.is_available()
    else:
        args['use_gpu'] = torch.cuda.is_available()
    # Get Atari games.
    benchmark = gym.benchmark_spec('Atari40M')
    task = benchmark.tasks[args['env_id']]

    # Run training
    seed = args['seed']
    env = get_env(task, seed, save_dir)

    main(env, task.max_timesteps, args)
