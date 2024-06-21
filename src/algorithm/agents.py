from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from src.model.gnn import SwapGNN, CriticSwapGNN

class AbstractAgent(ABC):
    def __init__(self, actor: nn.Module, critic: nn.Mod):
        self.actor = actor
        self.critic = critic

    @abstractmethod
    def choose_action(self, state):
        pass

    @abstractmethod
    def learn(self, experiences):
        pass

class PPOAgent(AbstractAgent):
    pass