import random
import numpy as np
import torch
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity: int, device: torch.device):
        self.buffer: deque = deque(maxlen=capacity) # used deque data structure to remove older experiences not in capactity
        self.device = device
    
    def __len__(self)->int:
        return len(self.buffer)
    
    def push(self, state: tuple, action, reward: float, next_state: tuple, done: bool):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)

        s, a, r, s2, d = zip(*batch) # unpack into seperate tuples

        s  = torch.as_tensor(np.stack(s),  device=self.device)
        a  = torch.as_tensor(a,            device=self.device, dtype=torch.int64)
        r  = torch.as_tensor(r,            device=self.device, dtype=torch.float32)
        s2 = torch.as_tensor(np.stack(s2), device=self.device)
        d  = torch.as_tensor(d,            device=self.device, dtype=torch.float32)

        return s, a, r, s2, d