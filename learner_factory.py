from enum import Enum
import torch
import gymnasium as gym
from deepqlearning.qlearner import QLearner
from networks.dqn_fully_connected import DQN
from environments.env_wrappers import TorchWrapper


class EnvironmentTag(Enum):
    ACROBOT = "ACROBOT"
    MOUNTAINCAR = "MOUNTAINCAR"
    CARTPOLE = "CARTPOLE"


def create_learner(env_tag: EnvironmentTag, model_parameter_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if env_tag == EnvironmentTag.ACROBOT:
        env     = TorchWrapper(gym.make("Acrobot-v1", render_mode="human"), device)
        model   = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
        params = {
            'replay_buffer_size': 10000,
            'batch_size': 128,
            'eps_start': 0.9,
            'eps_end': 0.05,
            'eps_decay': 1000,
            'learning_rate': 0.0001,
            'gamma': 0.99,
            'tau': 0.005
        }
    elif env_tag == EnvironmentTag.MOUNTAINCAR:
        env     = TorchWrapper(gym.make("MountainCar-v0", render_mode="human"), device)
        model   = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
        params = {
            'replay_buffer_size': 10000,
            'batch_size': 128,
            'eps_start': 0.9,
            'eps_end': 0.05,
            'eps_decay': 5000,
            'learning_rate': 0.0001,
            'gamma': 0.99,
            'tau': 0.005
        }
    elif env_tag == EnvironmentTag.CARTPOLE:
        env     = TorchWrapper(gym.make("CartPole-v1", render_mode="human"), device)
        model   = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
        params = {
            'replay_buffer_size': 10000,
            'batch_size': 128,
            'eps_start': 0.9,
            'eps_end': 0.05,
            'eps_decay': 1000,
            'learning_rate': 0.0001,
            'gamma': 0.99,
            'tau': 0.005
        }
    else:
        raise ValueError("Invalid environment tag")
    
    if model_parameter_path:
        model.load_state_dict(torch.load(model_parameter_path))
    return QLearner(env, model, device, **params)