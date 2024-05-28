from collections import deque
import torch
import gymnasium as gym

class TorchWrapper(gym.ObservationWrapper):
    def __init__(self, env, device):
        super(TorchWrapper, self).__init__(env)
        self.env                = env
        self.device = device

    def observation(self, observation):
        return torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
    

class StackframeWrapper(gym.ObservationWrapper):
    def __init__(self, env, device):
        super().__init__(env)
        self.frame_memory = deque([], maxlen=4)
        self.device = device
        for _ind in range(4):
            self.frame_memory.append(torch.zeros(96, 96, 3).to(device))
    def observation(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        self.frame_memory.append(obs)
        concat = torch.cat(list(self.frame_memory), dim=2)
        return concat.transpose(0, 2).unsqueeze(0)