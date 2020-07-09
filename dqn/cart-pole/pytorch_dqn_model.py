# %%
import torch
from torch import nn
from torch.optim import RMSprop, Adam
import numpy as np
from collections import deque
import random
import torch.nn.functional as F


def copy_params(source, target):
    for source_param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.copy_(source_param.data)


class DQN(nn.Module):
    def __init__(self, n_actions, n_states):
        super().__init__()
        self.n_actions = n_actions
        self.n_states = n_states
        self.model = nn.Sequential(
            nn.Linear(self.n_states, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, self.n_actions)
        )
        self.optimizer = Adam(lr=1e-4, params=self.parameters())

    def forward(self, x):
        return self.model(x)


class Memory():
    def __init__(self, mem_size=100):
        self.memory = []
        self.mem_size = mem_size

    def save(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.mem_size:
            del self.memory[0]

    def sample(self, batch_size):
        size = min(len(self.memory), batch_size)
        return random.sample(self.memory, size)

    def __len__(self):
        return len(self.memory)


class Agent():
    def __init__(self,
                 mem_size,
                 n_states,
                 n_actions,
                 epsilon=1,
                 min_eps=0.01,
                 eps_decay=0.999,
                 batch_size=512,
                 gamma=0.99):
        self.memory = Memory(mem_size=mem_size)
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.min_eps = min_eps
        self.eps_decay = eps_decay
        self.net = DQN(n_actions, n_states)
        self.net_pred = DQN(n_actions, n_states).eval()
        self.batch_size = batch_size
        self.gamma = gamma

    def choose_action(self, state):
        if np.random.uniform() <= self.epsilon:  # random sample
            action = random.choice(np.arange(self.n_actions))
        else:
            with torch.no_grad():
                logits = self.net(torch.FloatTensor(state).unsqueeze(0))
                action = logits.argmax().item()
        return action

    def save_to_memory(self, transition):
        self.memory.save(transition)

    def learn(self):
        if len(self.memory) < 0:
            return

        transition = self.memory.sample(self.batch_size)
        _states, _actions, _rewards, _next_states, _done, _has_next = zip(*transition)

        states = torch.FloatTensor(_states)
        actions = torch.LongTensor(_actions).unsqueeze(1)
        rewards = torch.FloatTensor(_rewards).unsqueeze(1)
        next_states = torch.FloatTensor(_next_states)
        dones = torch.FloatTensor(_done).unsqueeze(1)
        has_next = torch.FloatTensor(_has_next).unsqueeze(1)

        current_Q = self.net(states).gather(1, actions)
        # max_next_Q = self.net(next_states).detach().max(1)[0].unsqueeze(1)
        with torch.no_grad():
            max_next_Q = self.net_pred(next_states).detach().max(1)[0].unsqueeze(1)

        targets = rewards + (self.gamma * max_next_Q * has_next)
        loss = F.mse_loss(current_Q, targets)

        self.net.optimizer.zero_grad()
        loss.backward()
        self.net.optimizer.step()

    def decay_epsilon(self):
        self.epsilon = self.epsilon * self.eps_decay if self.epsilon > self.min_eps else self.min_eps

    def save_checkpoint(self):
        torch.save(self.net.state_dict(), 'models/net.pt')
        torch.save(self.net_pred.state_dict(), 'models/pred.pt')

    def load_checkpoint(self):
        self.net.load_state_dict(torch.load('models/net.pt'))
        self.net_pred.load_state_dict(torch.load('models/pred.pt'))
