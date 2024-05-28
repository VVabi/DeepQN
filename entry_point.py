import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepqlearning.qlearner import QLearner
from networks.dqn_fully_connected import DQN
from environments.env_wrappers import TorchWrapper
from learner_factory import EnvironmentTag, create_learner


learner = create_learner(EnvironmentTag.RACINGCAR, "racingcar_model_825.459789352119.pth")
#learner.train(500, 900)

average_reward = learner.test(10)

#learner.save_model(f"racingcar_model_{average_reward}.pth")