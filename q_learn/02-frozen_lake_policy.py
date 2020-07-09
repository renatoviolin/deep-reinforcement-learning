import gym
import time
import random
import numpy as np

# SFFF       (S: starting point, safe)
# FHFH       (F: frozen surface, safe)
# FFFH       (H: hole, fall to your doom)
# HFFG

env = gym.make('FrozenLake-v0', is_slippery=True)
total_wins = 0
scores = []
N_GAMES = 1000
policy = {
    0: 1,  # key-> position in grid (F)
    1: 2,  # val -> action
    2: 1,
    3: 0,
    4: 1,
    6: 1,
    8: 2,
    9: 1,
    10: 1,
    13: 2,
    14: 2
}

for i in range(N_GAMES):
    done = False
    obs = env.reset()
    score = 0
    while not done:
        action = policy[obs]
        obs, reward, done, info = env.step(action)
        total_wins += reward

env.close()

print(f'play games: {N_GAMES}')
print(f'win games.: {total_wins:.0f}')
