import gym
import time
import random
import numpy as np

env = gym.make('FrozenLake-v0')
wins = 0
scores = []
N_GAMES = 1000

for i in range(N_GAMES):
    done = False
    observation = env.reset()
    score = 0
    while not done:
        action = random.randint(0, 3)
        observation, reward, done, info = env.step(action)
        score += reward  # 1 when achieve the goal, 0 otherwise

    wins += reward
    scores.append(score)

env.close()

print(f'mean score..: {np.mean(scores)}')
print(f'win games..: {wins}')
