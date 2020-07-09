import numpy as np
from agent import DQNAgent
from utils import make_env
import time

if __name__ == "__main__":
    env = make_env('PongNoFrameskip-v4')
    best_score = -np.inf
    n_games = 10
    agent = DQNAgent(gamma=0.99, epsilon=0.4, lr=1e-4, n_actions=env.action_space.n, input_dims=(env.observation_space.shape),
                     mem_size=50000, batch_size=32, eps_min=0.1, eps_dec=1e-5, tau=1000, env_name='PongNoFrameskip-v4', chkpt_dir='models/')

    agent.load_models()
    n_steps = 0
    scores, eps_history = [], []

    for i in range(n_games):
        done = False
        state = env.reset()
        score = 0

        while not done:
            env.render()
            action = agent.choose_action(state)
            state_, reward, done, info = env.step(action)
            score += reward
            state = state_
            n_steps += 1
            time.sleep(.03)
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print(f'episode {i} score {score} avg_score {avg_score:.1f} eps {agent.epsilon:.4f} steps {n_steps}')
