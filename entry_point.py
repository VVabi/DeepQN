import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepqlearning.qlearner import QLearner
from networks.dqn_fully_connected import DQN

class TorchWrapper(gym.ObservationWrapper):
    def __init__(self, env, device):
        super(TorchWrapper, self).__init__(env)
        self.env                = env
        self.device = device

    def observation(self, observation):
        return torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = TorchWrapper(gym.make("Acrobot-v1", render_mode="human"), device)

model = DQN(env.observation_space.shape[0], env.action_space.n).to(device)

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

learner = QLearner(env, model, device, **params)

learner.train(60, 1000)

learner.test(10)