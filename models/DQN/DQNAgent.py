import math
import matplotlib
import matplotlib.pyplot as plt

from DQN import DQN
from replay import ReplayMemory
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F




Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class DQNAgent():
    """DQN Agent to use in run.py. Based on original DQN paper and https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html"""
    def __init__(self, env, batch_size = 128, gamma = 0.99, eps_start = 0.9, eps_end = 0.01, eps_decay = 2500, tau = 0.005, lr = 3e-4):
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.EPS_START = eps_start
        self.EPS_END = eps_end
        self.TAU = tau
        self.LR = lr
        self.env = env
        self.memory = ReplayMemory(10000, Transition)
        self.n_actions = env.action_space.n
        self.state, self.info = env.reset()
        self.n_observations = len(state)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
        self.policy_net = DQN(self.n_actions).to(self.device)
        self.target_net = DQN(self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.parameters(), lr = self.LR, amsgrad = True)
        self.steps_done = 0
        self.episode_durations = []

    def select_action(self, state):
        """
        Selects actions based on epsilon-greedy policy.
        The probability of choosing random action starts at EPS_START and ends at EPS_END
        """
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) *  math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > self.eps_threshold:
            with torch.no_grad():
                # Selects the highest predicted Q-Value and returns in [1,1] tensor
                return policy_net(state).max(1).indices.view(1,1)
        else:
            return torch.tensor([[env.action_space.sample()]], device=self.device, dtype=torch.long)
    
    def plot_durations(self, show_result=False):
        """
        Helper function for plotting duration of episodes and averages of last 100.
        """
        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())
        
        def optimize_model(self):
            """
            Performs single step of optimizaton using the replay memory.
            """

            if len(self.memory) < self.BATCH_SIZE:
                return
            
            transitions = self.memory.sample(self.BATCH_SIZE)
            # Convert list of transitions to a Transition of lists
            # Using a Transition of lists makes it easier to process entire batches at once instead of one sample at a time
            # batch looks like:
                # batch.state = (s1, s2, s3, ...)
                # batch.action = (a1, a2, a3, ...)
                # batch.next_state = (s1’, s2’, s3’, ...)
                # batch.reward = (r1, r2, r3, ...)
            batch = Transition(*zip(*transitions))
            
            # Builds Boolean mask which marks which next starts are valid (not terminal) 
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                        batch.next_state)),
                              device=device, dtype=torch.bool)
            
            # Combines all valid states into a tensor
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)
            
            # Current state values
            state_action_values = policy_net(state_batch).gather(1, action_batch)

            # Next state values
            next_state_values = torch.zeros(BATCH_SIZE, device=device)
            with torch.no_grad():
                next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
            
            # Compute expected return target using Bellman update
            expected_state_action_values = (next_state_values * GAMMA) + reward_batch
            
            # Compute loss
            criterion = nn.SmoothL1Loss()
            loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

            # Backpropogate loss
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)

            # Update network parameters with gradient step
            optimizer.step()
        
        def learn(self, total_timesteps):
            if torch.cuda.is_available() or torch.backends.mps.is_available():
                num_episodes = total_timesteps
            else:
                num_episodes = 50
            
            for i in range(num_episodes):
                state, info = self.env.reset()
                state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                for t in count():
                    action = self.select_action(state)
                    observation, reward, terminated, truncated, _ = self.env.step(action.item())
                    reward = torch.tensor([reward], device=device)
                    done = terminated or truncated
                    if terminated:
                        next_state = None
                    else:
                        next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                    self.memory.push(state, action, next, reward)
                    state = next_state
                    self.optimize_model()
                    target_net_state_dict = self.target_net.state_dict()
                    policy_net_state_dict = self.policy_net.state_dict()
                    for key in policy_net_state_dict:
                        self.target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                    self.target_net.load_state_dict(target_net_state_dict)
                    if done:
                        self.episode_durations.append(t + 1)
                        self.plot_durations()
                        break
            print('Complete')
            self.plot_durations(show_result=True)
            plt.ioff()
            plt.show()














