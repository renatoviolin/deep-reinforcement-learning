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

agent = Agent(mem_size=100000, n_states=n_states, n_actions=n_actions)

if __name__ == '__main__':
    try:
        for i in range(N_GAMES):
            done = False
            state = env.reset()
            score = 0
            steps = 0
            while True:
                env.render()

                action = agent.choose_action(state)
                next_state, reward, done, info = env.step(action)
                steps += 1

                # normalize inputs
                next_state[0] /= 2.5
                next_state[1] /= 2.5
                next_state[2] /= 0.2
                next_state[3] /= 2.5

                has_next = True
                if done:
                    has_next = False
                score += reward

                agent.save_to_memory((state, action, reward, next_state, done, has_next))
                state = next_state
                agent.learn()
                agent.decay_epsilon()

                if done:
                    print(f'episode {i} steps {steps} score {score:.2f} epsilon {agent.epsilon:.3f} mem {len(agent.memory)}')
                    break

            if i % 10 == 0:
                print('copying params')
                copy_params(agent.net, agent.net_pred)
                agent.save_checkpoint()
    except KeyboardInterrupt:
        print('saving checkpoint')
        agent.save_checkpoint()
        env.close()
