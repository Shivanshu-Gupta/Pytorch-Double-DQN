import random
import numpy as np
from collections import namedtuple
import torch

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayMemory(object):
    def __init__(self, capacity, state_size=[4, 84, 84], hist_len=4):
        self.capacity = capacity
        self.states = np.empty([capacity] + state_size, dtype=np.uint8)
        self.actions = np.empty([capacity, 1], dtype=np.uint8)
        self.rewards = np.empty([capacity], dtype=np.float32)
        self.next_states = np.empty([capacity] + state_size, dtype=np.uint8)
        self.done = np.empty([capacity], dtype=np.uint8)
        self.hist_len = hist_len
        self.position = 0
        self.size = 0

    def push(self, transition):
        """Saves a transition."""
        self.states[self.position] = transition.state
        self.actions[self.position] = transition.action
        self.rewards[self.position] = transition.reward
        if transition.next_state is not None:
            self.next_states[self.position] = transition.next_state
        self.done[self.position] = transition.done
        self.position = (self.position + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def sample(self, batch_size):
        indices = random.sample(range(self.size), batch_size)
        batch = Transition(
            torch.from_numpy(self.states[indices]).type(Tensor),
            torch.from_numpy(self.actions[indices]).type(LongTensor),
            torch.from_numpy(self.rewards[indices]).type(Tensor),
            torch.from_numpy(self.next_states[indices]).type(Tensor),
            torch.from_numpy(self.done[indices]).type(ByteTensor))
        return batch

    def __len__(self):
        return self.size