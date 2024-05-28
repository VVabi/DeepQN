import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepqlearning.qlearner import QLearner

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    


class TorchWrapper(gym.ObservationWrapper):
    def __init__(self, env, device):
        super(TorchWrapper, self).__init__(env)
        self.env                = env
        self.device = device

    def observation(self, observation):
        return torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = TorchWrapper(gym.make("CartPole-v1", render_mode="human"), device)

model = DQN(env.observation_space.shape[0], env.action_space.n).to(device)


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

learner = QLearner(env, model, device, **params)

learner.train(1000, 1000)

learner.test(10)