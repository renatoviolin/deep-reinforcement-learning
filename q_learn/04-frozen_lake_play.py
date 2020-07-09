import gym
import matplotlib.pyplot as plt
import numpy as np
from q_learning_agent import Agent
import pprint
pp = pprint.PrettyPrinter()

if __name__ == '__main__':
    env = gym.make('FrozenLake-v0', is_slippery=True)
    agent = Agent(lr=0.001, gamma=0.9, eps_start=0.0, eps_min=0.01, eps_dec=0.9999995, n_actions=4, n_states=16)
    agent.load_checkpoint()
    print('loading checkpoint')
    pp.pprint(agent.Q)
    input('press any key to continue')

    wins = 0
    N_GAMES = 100000
    for i in range(N_GAMES):
        done = False
        obs = env.reset()
        while not done:
            action = agent.choose_action(obs)
            obs_, reward, done, info = env.step(action)  # obs_ -> new observation
            wins += reward
            obs = obs_
            if i % 1000 == 0:
                print(f'played games..: {i}')
                print(f'number of wins: {wins:.0f}')

        # print just the end state to see if win or loose
        env.render()
        # input()
    env.close()
