import gym
import numpy as np
import pickle
from pytorch_dqn_model import *

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


env = gym.make('CartPole-v1')
N_GAMES = 100000
n_actions = env.action_space.n
n_states = 4

env.action_space.seed(RANDOM_SEED)
env.seed(RANDOM_SEED)

agent = Agent(mem_size=100000, n_states=n_states, n_actions=n_actions, epsilon=0.0)
agent.load_checkpoint()
agent.net.eval()

if __name__ == '__main__':
    for i in range(N_GAMES):
        done = False
        state = env.reset()
        score = 0
        while True:
            env.render()
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            score += 1
            
            next_state[0] /= 2.5
            next_state[1] /= 2.5
            next_state[2] /= 0.2
            next_state[3] /= 2.5
 
            state = next_state

            if done:
                print(f'episode {i} score {score:.2f} epsilon {agent.epsilon:.3f}')
                break

    env.close()
