import torch
import gymnasium as gym

class TorchWrapper(gym.ObservationWrapper):
    def __init__(self, env, device):
        super(TorchWrapper, self).__init__(env)
        self.env                = env
        self.device = device

    def observation(self, observation):
        return torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)