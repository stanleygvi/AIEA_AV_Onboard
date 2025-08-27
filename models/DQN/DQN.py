import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DQN(nn.module):
    def __init__(self, num_actions:int):
        super().__init__()
        self.features = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size = 8, stride = 4),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size = 4, stride = 2),
                nn.ReLU(),
            )
        with torch.no_grad():
            x = torch.zeros(1, 4, 84, 84)
            n_flat = self.features(x).view(1, -1).size(1)

        self.head = nn.Sequential(
            nn.Linear(n_flat, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_actions),
        )
    def forward(self, x):
        if x.dtype == torch.uint8:
            x = x.float().div_(255.0)
        z = self.features(x)
        z = z.view(z.size(0), -1)
        return self.head(z)

