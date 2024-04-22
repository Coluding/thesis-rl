from typing import Union, List, Tuple, Dict

import gym
import networkx as nx
import numpy as np
from dataclasses import dataclass
import torch
from gym import Env
from gym import spaces
from gym.core import ActType, ObsType
from tensordict import TensorDictBase
from torchrl.envs import EnvBase, GymLikeEnv, GymEnv
from torch_geometric.utils.convert import from_networkx
from abc import ABC, abstractmethod
import torch_geometric

from src.env.conversion_utils import convert_interval_result_to_graph

@dataclass
class RSMEnvConfig:
    """
    Configuration for the RSMEnv environment
    """
    device: str
    dtype: torch.dtype
    batch_size: int
    n_actions: int
    reward_threshold: int
    num_nodes: int
    feature_dim_node: int
    run_type_checks: bool = False
    allow_done_after_reset: bool = False
    feature_dim_edge: int = 1


@dataclass
class IntervalResult:
    IntervalLatencyMap: Dict[int, Dict[int, int]]

class SimulationBase(ABC):
    """
    Base class for simulation. This will be used as the interface for the simulation environment that interacts with
    the Java simulation environment using jpype
    """
    def __init__(self):
        pass

    @abstractmethod
    def runInterval(self) -> IntervalResult:
        """
        Runs a measurement interval with the current placement configuration
        """
        pass

    @abstractmethod
    def setPlacement(self, action: int) -> None:
        """
        Take a step in the simulation environment
        """
        pass


    @abstractmethod
    def reset(self) -> Union[Dict[int, Dict[int, int]], List[int]]:
        """
        Reset the simulation environment and return the initial state and a list of all possible locations of the nodes
        """
        pass


class RSMEnv(Env):
    """
    A network container for OpenAI GYM
    """

    def __init__(self,
                 config: RSMEnvConfig,
                 simulation: SimulationBase):

        self.config = config
        self.simulation = simulation

        self.node_space = spaces.Box(low=0, high=np.inf, shape=(config.feature_dim_node,))
        self.edge_space = spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)

        self.observation_space = spaces.Graph(node_space=self.node_space, edge_space=self.edge_space)

        self.graph = nx.Graph()
        initial_latencies, all_locations = self.simulation.reset()
        self.graph.add_nodes_from(all_locations)

        self.action_space = spaces.Discrete(config.n_actions)
        self.state = None


    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:

        self.simulation.setPlacement(action)
        interval_result = self.simulation.runInterval()
        self.graph = convert_interval_result_to_graph(interval_result)

        # Calculate reward (dummy implementation)
        reward = self.compute_reward()
        done = False
        info = {}

        return self.graph, reward, done, info

    def compute_reward(self) -> float:
        """
        Compute the reward based on the current state
        """

        return np.sum((self.graph.edges.data("weight")))



class TorchGraphObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env: RSMEnv):
        super().__init__(env)

    def observation(self, observation) -> torch_geometric.data.Data:
        return from_networkx(observation)


if __name__ == "__main__":
    config = RSMEnvConfig(device="cuda", dtype=torch.float32, batch_size=32, n_actions=4, reward_threshold=100, feature_dim_node=128)
    env = RSMEnv(config)
    envg = GymEnv("Pendulum-v1")
    reset = env.reset()