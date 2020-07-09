import numpy as np
import pickle


class Agent():
    def __init__(self, lr, gamma, n_actions, n_states, eps_start, eps_min, eps_dec):
        self.lr = lr
        self.gamma = gamma
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = eps_start
        self.eps_end = eps_min
        self.eps_dec = eps_dec
        self.checkpoint = 'q_table.pickle'
        self.Q = {}
        self.init_Q()

    def init_Q(self):
        for state in range(self.n_states):
            for action in range(self.n_actions):
                self.Q[(state, action)] = 0.0

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice(np.arange(self.n_actions))
        else:
            actions = np.array([self.Q[(state, a)] for a in range(self.n_actions)])
            action = np.argmax(actions)
        return action

    def decrement_epsilon(self):
        self.epsilon = self.epsilon * self.eps_dec if self.epsilon > self.eps_end else self.eps_end

    def learn(self, state, action, reward, state_):
        # get all possible actions given the new state_ the agent is now
        actions = np.array([self.Q[(state_, a)] for a in range(self.n_actions)])  # all actions given t

        # select the action that get the highest value
        a_max = np.argmax(actions)

        # calculate the discounted reward
        discounted_reward = reward + self.gamma * self.Q[(state_, a_max)] - self.Q[(state, action)]

        # update the previous state value with the new value
        self.Q[(state, action)] += self.lr * discounted_reward

        self.decrement_epsilon()

    def save_checkpoint(self):
        with open(self.checkpoint, 'wb') as f:
            pickle.dump(self.Q, f, pickle.HIGHEST_PROTOCOL)
            f.close()

    def load_checkpoint(self):
        with open(self.checkpoint, 'rb') as f:
            self.Q = pickle.load(f)
            f.close()
