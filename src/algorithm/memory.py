from torch import Tensor
import torch
import numpy as np
from typing import Union, Sequence, Any
from torch_geometric.data import Data, Batch
from typing import Dict, Tuple
import copy

class OnPolicyMemory:
    def __init__(self, trajectory_size: int,  batch_size: int):
        self.memory = {
            "states": [],
            "values": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "probs": []
        }
        self.graph_state_memory = []
        self.batch_size: int = batch_size
        self.trajectory_size: int = trajectory_size
        self.current_traj_step = 0

    def store(self, state: Data, val: float, action: Tuple[Tuple[int, int], Tuple[int, int]],
              reward: float, done: bool, prob: float) -> None:
        state = copy.deepcopy(state)
        self.memory['states'].append(state)
        self.memory['actions'].append(action)
        self.memory["values"].append(val.detach())
        self.memory['rewards'].append(reward)
        self.memory['dones'].append(done)
        self.memory['probs'].append(prob.detach())
        self.current_traj_step += 1


    def _batch_list(self):
        return [self.graph_state_memory[i:i + self.batch_size] for i in range(self.current_traj_step)]

    def get_batch(self, start_index: int = None) -> Dict[str, torch.Tensor]:
        if start_index is not None:
            batch_indices = torch.arange(start_index, min(start_index + self.batch_size, len(self.memory['states'])))
        else:
            batch_indices: Tensor = torch.randint(0, len(self.memory['states']), (self.batch_size,))

        batch = {
            "states": [self.memory['states'][i] for i in batch_indices],
            "actions": [self.memory['actions'][i] for i in batch_indices],
            "rewards": [self.memory['rewards'][i] for i in batch_indices],
            "values": [self.memory['values'][i] for i in batch_indices],
            "dones": [self.memory['dones'][i] for i in batch_indices],
            "probs": [self.memory['probs'][i] for i in batch_indices]
        }

        return batch

    def get_full_trajectory(self):
        return self.memory

    def get_experiences(self):
        for i in range(0, len(self.memory['states']), self.batch_size):
            yield self.get_batch(i)

    def clear_memory(self) -> None:
        self.memory: Dict[str, list] = {
            "states": [],
            "actions": [],
            "rewards": [],
            "values": [],
            "dones": [],
            "probs": []
        }
