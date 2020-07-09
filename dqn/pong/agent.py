import numpy as np
import torch
from network import DeepQNetwork
from memory import ReplayBuffer


class DQNAgent():
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims, mem_size, batch_size=32,
                 eps_min=0.1, eps_dec=1e-5, tau=1000, env_name='Pong', chkpt_dir='models/'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.tau = tau
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)
        self.q_eval = DeepQNetwork(lr, n_actions, f'{env_name}_q_eval.pth', input_dims, chkpt_dir)
        self.q_next = DeepQNetwork(lr, n_actions, f'{env_name}_q_next.pth', input_dims, chkpt_dir).eval()

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = torch.tensor([observation], dtype=torch.float).to(self.q_eval.device)
            action = self.q_eval.forward(state).argmax().item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        state, action, reward, states_, done = self.memory.sample_buffer(self.batch_size)
        states = torch.tensor(state).to(self.q_eval.device)
        rewards = torch.tensor(reward).to(self.q_eval.device)
        dones = torch.tensor(done).to(self.q_eval.device)
        actions = torch.tensor(action).to(self.q_eval.device)
        states_s = torch.tensor(states_).to(self.q_eval.device)
        return states, actions, rewards, states_s, dones

    def update_target_network(self):
        if self.learn_step_counter % self.tau == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_eps(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        if self.batch_size > self.memory.mem_cntr:
            return

        states, actions, rewards, states_, dones = self.sample_memory()
        indices = np.arange(self.batch_size)

        q_pred = self.q_eval.forward(states)[indices, actions]  # select the values only for actions the agent have taken actions 0 or 1
        with torch.no_grad():
            q_next = self.q_next.forward(states_).max(dim=1)[0]
        q_next[dones] = 0.0  # for terminal states, there's no other state ahead, so reward = 0.
        q_target = rewards + self.gamma * q_next

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        self.q_eval.optimizer.zero_grad()
        loss.backward()
        self.q_eval.optimizer.step()

        self.learn_step_counter += 1
        self.update_target_network()  # decide to update or not the weights of q_next
        self.decrement_eps()
