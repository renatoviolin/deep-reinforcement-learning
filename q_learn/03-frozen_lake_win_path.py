import gym
import time
import random
import numpy as np

# SFFF       (S: starting point, safe)
# FHFH       (F: frozen surface, safe)
# FFFH       (H: hole, fall to your doom)
# HFFG

env = gym.make('FrozenLake-v0', is_slippery=False)
total_wins = 0
scores = []
N_GAMES = 1000
win_path = {
    0: 2,
    1: 2,
    2: 1,
    6: 1,
    10: 1,
    14: 2
}

for i in range(1000):
    done = False
    obs = env.reset()
    score = 0
    for a in win_path.keys():
        # env.render()
        action = win_path[a]
        obs, reward, done, info = env.step(action)
        score += reward
        if done:
            break
    total_wins += reward

env.close()

print(f'play games: {N_GAMES}')
print(f'win games.: {total_wins:.0f}')
