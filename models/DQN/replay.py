import random
from collections import deque


class ReplayMemory(object):

    def __init__(self, capacity, transition):
        """Initializes memory object and the transition tuple"""
        self.memory = deque([], maxlen=capacity)
        self.Transition = transition

    def push(self, *args):
        """Saving a transition to the memory"""
        self.memory.append(self.Transition(*args))
    
    def sample(self, batch_size):
        """Sampling random transitions from memory given batch size"""
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
