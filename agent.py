import os
import gym
import random
import pickle
import numpy as np
from copy import deepcopy
from itertools import count
from model import DQNCNNModel
from utils.schedule import LinearSchedule

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from utils.gym import get_wrapper_by_name
from utils.memory import Transition, ReplayMemory
from IPython.core.debugger import Tracer; debug_here = Tracer()

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


class DQNCNNAgent(object):
    def __init__(self, env, args):
        assert type(env.action_space) == gym.spaces.Discrete
        assert type(env.observation_space) == gym.spaces.Box
        assert len(env.observation_space.shape) == 3

        self.args = args['agent_params']
        self.env = env
        self.n_actions = env.action_space.n
        self.hist_len = self.args['hist_len']
        img_h, img_w, img_c = env.observation_space.shape
        self.state_size = [self.hist_len * img_c, img_h, img_w]

        # setup controller model and replay memory
        self.setup_controller(args)
        self.memory = ReplayMemory(self.args['replay_memory'], state_size=self.state_size, hist_len=self.hist_len)

        self.epsilon = LinearSchedule(
            schedule_timesteps=self.args['epsilon']['end_t'],
            initial_p=self.args['epsilon']['start'],
            final_p=self.args['epsilon']['end'])
        self.gamma = self.args['gamma']   # discount rate

        self.optimizer = optim.RMSprop(self.Qnet.parameters(), **self.args['optim_params'])

        self.stats = {
            "mean_episode_rewards": [],
            "best_mean_episode_rewards": []
        }
        self.save_dir = args['save_dir']

    def setup_controller(self, args):
        self.Qnet = DQNCNNModel(input_size=self.state_size[0], output_size=self.n_actions)
        if 'modelpath' in args:
            self.load(args['modelpath'])
        self.target_Qnet = deepcopy(self.Qnet)
        if args['use_gpu']:
            self.Qnet = self.Qnet.cuda()
            self.target_Qnet = self.target_Qnet.cuda()
        print(self.Qnet)

    def act(self, state, epsilon):
        # select epsilon-greedy action
        if random.random() <= epsilon:
            return random.randrange(self.n_actions)
        else:
            state = Variable(torch.from_numpy(state).type(Tensor) / 255, volatile=True).unsqueeze(0)
            act_values = self.Qnet(state)   # B*n_actions
            _, action = act_values.data.max(1)
            return action[0]

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        batch = self.memory.sample(batch_size)
        state_batch = Variable(batch.state / 255)
        action_batch = Variable(batch.action)
        reward_batch = Variable(batch.reward)
        non_final = LongTensor([i for i, done in enumerate(batch.done) if not done])
        non_final_mask = 1 - batch.done
        # To prevent backprop through the target action values, set volatile=False (also sets requires_grad=False)
        non_final_next_states = Variable(batch.next_state.index_select(0, non_final) / 255, volatile=True)

        # Compute Q(s_t, a)
        Q_state_action = self.Qnet(state_batch).gather(1, action_batch)

        # Double DQN - Compute V(s_{t+1}) for all next states.
        V_next_state = Variable(torch.zeros(batch_size).type(Tensor))
        if self.args['double_dqn']:
            _, next_state_actions = self.Qnet(non_final_next_states).max(1, keepdim=True)
            V_next_state[non_final_mask] = self.target_Qnet(non_final_next_states).gather(1, next_state_actions)
        else:
            V_next_state[non_final_mask] = self.target_Qnet(non_final_next_states).max(1)[0]

        # Remove Volatile as it sets all variables computed from them volatile.
        # The Variable will just have requires_grad=False.
        V_next_state.volatile = False

        # Compute the target Q values
        target_Q_state_action = reward_batch + (self.gamma * V_next_state)

        # Compute loss
        loss = F.smooth_l1_loss(Q_state_action, target_Q_state_action)
        # td_error = target_Q_state_action - Q_state_action
        # clipped_error = td_error.clamp(-1, 1)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.Qnet.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def train(self, stopping_criterion):
        mean_episode_reward = -float('nan')
        best_mean_episode_reward = -float('inf')
        batch_size = 32
        num_updates = 0
        screen = self.env.reset()
        state = np.zeros(self.state_size, dtype=np.uint8)
        state[-1] = screen.transpose(2, 0, 1)
        for t in count():
            if stopping_criterion is not None and stopping_criterion(self.env):
                break
            # debug_here()
            state = np.vstack([state[1:], screen.transpose(2, 0, 1)])
            if t > self.args['replay_start']:
                action = self.act(state, self.epsilon.value(t))
            else:
                action = self.env.action_space.sample()
            next_screen, reward, done, _ = self.env.step(action)
            next_state = np.vstack([state[1:], next_screen.transpose(2, 0, 1)])
            self.memory.push(Transition(state, action, reward, next_state, done))
            if done:
                next_screen = self.env.reset()
                state = np.zeros(self.state_size, dtype=np.uint8)
                state[-1] = next_screen.transpose(2, 0, 1)
            else:
                state = next_state

            if t > self.args['replay_start'] and t % self.args['replay_freq'] == 0:
                self.replay(batch_size=batch_size)
                num_updates += 1
                if num_updates % self.args['target_update_freq'] == 0:
                    self.target_Qnet.load_state_dict(self.Qnet.state_dict())

            if t > self.args['replay_start']:
                if t % self.args['log_freq'] == 0:
                    episode_rewards = get_wrapper_by_name(self.env, "Monitor").get_episode_rewards()
                    if len(episode_rewards) > 0:
                        mean_episode_reward = np.mean(episode_rewards[-100:])
                    if len(episode_rewards) > 100:
                        best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)

                    self.stats["mean_episode_rewards"].append(mean_episode_reward)
                    self.stats["best_mean_episode_rewards"].append(best_mean_episode_reward)

                    print("Timestep %d" % (t,))
                    print("mean reward (100 episodes) %f" % mean_episode_reward)
                    print("best mean reward %f" % best_mean_episode_reward)
                    print("episodes %d" % len(episode_rewards))
                    print("exploration %f" % self.epsilon.value(t))
                if t % self.args['save_freq'] == 0:
                    self.save()

    def eval(self, n_episodes):
        for i_ep in range(n_episodes):
            state = np.zeros(self.state_size, dtype=np.uint8)
            screen = self.env.reset()
            state[-1] = screen.transpose(2, 0, 1)
            for t in count():
                self.env.render()
                action = self.act(state, 0)
                next_screen, reward, done, _ = self.env.step(action)
                if done:
                    break
                else:
                    state = np.vstack([state[1:], next_screen.transpose(2, 0, 1)])

    def load(self, modelpath):
        state_dict = torch.load(modelpath, map_location=lambda storage, loc: storage)
        self.Qnet.load_state_dict(state_dict)
        print("Loaded model from {}".format(modelpath))

    def save(self):
        # Save Q Network weights
        modelpath = os.path.join(self.save_dir, 'QNetwork.pth.tar')
        torch.save(self.Qnet.state_dict(), modelpath)
        print("Saved to model to {}".format(modelpath))

        # Dump statistics to pickle
        statspath = os.path.join(self.save_dir, 'stats.pkl')
        with open(statspath, 'wb') as f:
            pickle.dump(self.stats, f)
        print("Saved to statistics to {}".format(statspath))
