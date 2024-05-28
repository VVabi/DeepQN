import torch


class TorchTrainWrapper():
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion


    def train_step(self, prediction, labels):
        loss = self.criterion(prediction, labels)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
        self.optimizer.step()