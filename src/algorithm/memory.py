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


class ExperienceReplayBuffer:
    def __init__(self, max_size: int, batch_size: int, normalize_latencies: bool = True):
        self.max_size = max_size
        self.batch_size = batch_size

        self.memory = {
            "states": [],
            "states_": [],
            "actions": [],
            "rewards": [],
            "dones": [],
        }
        self.current_step = 0
        self.should_normalize_latencies = normalize_latencies
        self.latency_avg_tracker = 0
        self.latency_std_tracker = 0


    def normalize_latencies(self, latencies, mean, std):
        return (latencies - mean) / std

    def denormalize_latencies(self, latencies, mean, std):
        return latencies * std + mean

    def add(self,
            state: Data,
            next_state: Data,
            action: Tuple[Tuple[int, int], Tuple[int, int]],
            reward: float, done: bool) -> None:
        state = copy.deepcopy(state)
        next_state = copy.deepcopy(next_state)

        if self.should_normalize_latencies:
            if isinstance(state, list):
                self.latency_avg_tracker += torch.mean(torch.cat((state[-1].latency, next_state[-1].latency))).item()
                self.latency_std_tracker += torch.std(torch.cat((state[-1].latency, next_state[-1].latency))).item()
                mean_latency = self.latency_avg_tracker / (self.current_step + 1)
                std_latency = self.latency_std_tracker / (self.current_step + 1)

                for s, ns in zip(state, next_state):
                        s.latency = self.normalize_latencies(s.latency, mean_latency, std_latency)
                        ns.latency = self.normalize_latencies(ns.latency, mean_latency, std_latency)
            else:
                self.latency_avg_tracker += torch.mean(torch.cat((state.latency, next_state.latency))).item()
                self.latency_std_tracker += torch.std(torch.cat((state.latency, next_state.latency))).item()
                mean_latency = self.latency_avg_tracker / (self.current_step + 1)
                std_latency = self.latency_std_tracker / (self.current_step + 1)
                state.latency = self.normalize_latencies(state.latency, mean_latency, std_latency)
                next_state.latency = self.normalize_latencies(next_state.latency, mean_latency, std_latency)

        if self.current_step > self.max_size:
            insert_ind = self.current_step % self.max_size
            self.memory["states"].insert(insert_ind, state)
            self.memory['actions'].insert(insert_ind, action)
            self.memory["states_"].insert(insert_ind, next_state)
            self.memory['rewards'].insert(insert_ind, reward)
            self.memory['dones'].insert(insert_ind, done)
        else:
            self.memory['states'].append(state)
            self.memory['actions'].append(action)
            self.memory["states_"].append(next_state)
            self.memory['rewards'].append(reward)
            self.memory['dones'].append(done)

        self.current_step += 1

    def sample(self):
        indices = np.random.choice(np.arange(self.current_step), self.batch_size, replace=False)
        states = [self.memory["states"][i] for i in indices]
        states_ = [self.memory["states_"][i] for i in indices]
        rewards = [self.memory["rewards"][i] for i in indices]
        actions = [self.memory["actions"][i] for i in indices]
        dones = [self.memory["dones"][i] for i in indices]

        return states, states_, rewards, actions, dones, indices

    def clear_memory(self):
        self.memory = {
            "states": [],
            "states_": [],
            "actions": [],
            "rewards": [],
            "dones": [],
        }
        self.current_step = 0


class ExpertKnowledgeReplayBuffer(ExperienceReplayBuffer):
    def fill_with_expert_knowledge(self, expert_knowledge: Sequence[Dict[str, Any]]):
        for exp in expert_knowledge:
            self.add(exp["state"], exp["next_state"], exp["action"], exp["reward"], exp["done"])