import torch
import numpy as np
from agent import DQNAgent
import itertools
from vizdoom import DoomGame, Mode, ScreenFormat, ScreenResolution
from skimage.transform import resize

frame_repeat = 12
resolution = (1, 84, 84)
config_file_path = "basic.cfg"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def initialize_vizdoom(config):
    game = DoomGame()
    game.load_config(config)
    game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.GRAY8)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.init()
    return game


def preprocess(img):
    return torch.from_numpy(resize(img, resolution[1:]).astype(np.float32)).unsqueeze(0)


def game_state(game):
    return preprocess(game.get_state().screen_buffer)


game = initialize_vizdoom(config_file_path)
n = game.get_available_buttons_size()
agent = DQNAgent(gamma=0.99, epsilon=1.0, lr=1e-4, n_actions=8, input_dims=resolution,
                 mem_size=30000, batch_size=64, eps_min=0.1, eps_dec=5e-5, tau=1000, env_name='doom_2_net', chkpt_dir='models/')
n_steps = 0


if __name__ == "__main__":
    best_score = -np.inf
    n_games = 20000  # = 20 epochs * 2000 iterations per epoch

    for i in range(n_games):
        done = False
        state = game_state(game)  # (84,84)
        total_score = 0
        episodes_finished = 0
        scores = []
        while not done:
            action = agent.choose_action(state)
            reward = game.make_action(agent.action_to_game[action], frame_repeat)

            if game.is_episode_finished():
                done, state_ = True, None
                total_score = game.get_total_reward()
                game.new_episode()
                episodes_finished += 1
            else:
                state_ = game_state(game)
            agent.store_transition(state, action, reward, state_, int(done))
            agent.learn()
            state = state_
            n_steps += 1

        scores.append(total_score)

        if i % 100 == 0:
            avg_score = np.mean(scores)
            print(f'episode {i} avg_score {avg_score} eps {agent.epsilon:.4f} steps {n_steps}')

            if avg_score > best_score:
                agent.save_models()
                best_score = avg_score
