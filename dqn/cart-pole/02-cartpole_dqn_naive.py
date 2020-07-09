import gym
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LinearDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, input_dims):
        super(LinearDeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

    def forward(self, state):
        layer1 = F.relu(self.fc1(state))
        actions = self.fc2(layer1)
        return actions


class Agent():
    def __init__(self, input_dims, n_actions, lr, gamma=0.99, eps=1.0, eps_dec=1e-5, eps_min=0.01):
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.action_space = [i for i in range(self.n_actions)]
        self.Q = LinearDeepQNetwork(lr, n_actions, input_dims)

    def choose_action(self, observation):
        if np.random.random() > self.eps:
            state = torch.tensor(observation, dtype=torch.float)
            actions = self.Q.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def decrement_eps(self):
        self.eps = self.eps - self.eps_dec if self.eps > self.eps_min else self.eps_min

    def learn(self, state, action, reward, state_):
        self.Q.optimizer.zero_grad()
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action)
        reward = torch.tensor(reward)
        state_ = torch.tensor(state_, dtype=torch.float)

        q_pred = self.Q.forward(state)[action]
        q_next = self.Q.forward(state_).max()
        q_target = reward + self.gamma * q_next

        loss = self.Q.loss(q_target, q_pred)
        loss.backward()
        self.Q.optimizer.step()
        self.decrement_eps()


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    n_games = 10000
    scores = []
    eps_history = []

    agent = Agent(lr=0.0001, input_dims=env.observation_space.shape,
                  n_actions=env.action_space.n)

    for i in range(n_games):
        score = 0
        done = False
        obs = env.reset()

        while not done:
            action = agent.choose_action(obs)
            obs_, reward, done, info = env.step(action)
            score += reward
            agent.learn(obs, action, reward, obs_)
            obs = obs_
        scores.append(score)
        eps_history.append(agent.eps)

        if i % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f'episode {i} score {score}, avg score {avg_score} eps {agent.eps:.4f}')


torch.tensor([0, 1])
