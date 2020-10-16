import random
from collections import namedtuple
import numpy as np

# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

Transition = namedtuple(
    'Transition', ('state', 'action', 'mask', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    # def sample(self, batch_size):
    #     return random.sample(self.memory, batch_size)
    
    def sample(self, batch_size):
        avail_indice = np.array(range(len(self.memory)))
        if batch_size > 0:
            indice = np.random.choice(avail_indice, batch_size)
        else:
            indice = avail_indice
        assert len(indice) > 0, 'No available indice can be sampled.'
        sample_data = [self.memory[idx] for idx in indice]
        return sample_data, indice

    def shuffle(self):
        random.shuffle(self.memory)

    def __len__(self):
        return len(self.memory)
