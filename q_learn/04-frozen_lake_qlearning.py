import gym
import matplotlib.pyplot as plt
import numpy as np
from q_learning_agent import Agent
import pickle

if __name__ == '__main__':
    env = gym.make('FrozenLake-v0', is_slippery=True)
    agent = Agent(lr=0.001, gamma=0.9, eps_start=1.0, eps_min=0.01, eps_dec=0.9999995, n_actions=4, n_states=16)
    agent.load_checkpoint()
    scores = []
    wins = []
    N_GAMES = 500000

    for i in range(N_GAMES):
        done = False
        obs = env.reset()
        score = 0
        while not done:
            action = agent.choose_action(obs)
            obs_, reward, done, info = env.step(action)  # obs_ -> new observation
            agent.learn(obs, action, reward, obs_)
            score += reward
            obs = obs_
        scores.append(score)
        if i % 100 == 0:
            mean_score = np.mean(scores[-100:])
            wins.append(mean_score)
            if i % 1000 == 0:
                print(f'episode {i} mean_score {mean_score:.2f}% epsilon {agent.epsilon:.3f}')
    agent.save_checkpoint()
    plt.plot(wins)
    plt.show()
    env.close()
