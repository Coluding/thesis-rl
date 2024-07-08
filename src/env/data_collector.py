import pickle
import networkx as nx
import matplotlib.pyplot as plt
from typing import Tuple, Union, List
import random
from dataclasses import dataclass

@dataclass
class StateActionPair:
    state: nx.Graph
    action: Union[Tuple[str, str], Tuple[Tuple[str, str], Tuple[str, str]]]


@dataclass
class StateActionData:
    state_action_pairs: List[StateActionPair] = None

    def __post_init__(self):
        if self.state_action_pairs is None:
            self.state_action_pairs = []

    def add_pair(self, state: nx.Graph, action: Union[Tuple[str, str], Tuple[Tuple[str, str], Tuple[str, str]]]):
        self.state_action_pairs.append(StateActionPair(state, action))

    def save(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename: str):
        with open(filename, 'rb') as f:
            return pickle.load(f)


def render_and_get_input(env, data: StateActionData):
    while True:
        env.visualize_2d_world()

        action_type = input("Enter action type (active/passive/both): ").strip().lower()
        if action_type not in ["active", "passive", "both"]:
            print("Invalid action type. Try again.")
            continue

        if action_type == "active":
            remove_dc = input(f"Select active node to remove from {list(env.active_replicas)}: ").strip()
            add_dc = input(f"Select inactive node to add from {list(env.available_active_replicas)}: ").strip()
            action = (remove_dc, add_dc)
        elif action_type == "passive":
            remove_dc = input(f"Select passive node to remove from {list(env.passive_replicas)}: ").strip()
            add_dc = input(f"Select inactive node to add from {list(env.available_passive_replicas)}: ").strip()
            action = (remove_dc, add_dc)
        elif action_type == "both":
            active_remove = input(f"Select active node to remove from {list(env.active_replicas)}: ").strip()
            active_add = input(f"Select inactive node to add from {list(env.available_active_replicas)}: ").strip()
            passive_remove = input(f"Select passive node to remove from {list(env.passive_replicas)}: ").strip()
            passive_add = input(f"Select inactive node to add from {list(env.available_passive_replicas)}: ").strip()
            action = ((active_remove, active_add), (passive_remove, passive_add))

        state = env.graph.copy()
        data.add_pair(state, action)

        env.step(action)
        env.visualize()

        cont = input("Continue? (y/n): ").strip().lower()
        if cont == 'n':
            break

    return data


