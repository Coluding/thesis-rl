from typing import Union, List, Tuple, Dict
import matplotlib.pyplot as plt
import gym
import networkx as nx
import numpy as np
from dataclasses import dataclass
import torch
from gym import Env
from gym import spaces
from gym.core import ActType, ObsType
from torch_geometric.utils.convert import from_networkx
from abc import ABC, abstractmethod
import torch_geometric






@dataclass
class RSMEnvConfig:
    """
    Configuration for the RSMEnv environment
    """
    device: str
    dtype: torch.dtype
    n_actions: int
    num_nodes: int
    feature_dim_node: int
    feature_dim_edge: int = 1


@dataclass
class IntervalResult:
    IntervalLatencyMap: Dict[int, Dict[int, int]]


class SimulationBase(ABC):
    """
    Base class for simulation. This will be used as the interface for the simulation environment that interacts with
    the Java simulation environment using jpype
    """

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
    def reset(self) -> Dict[int, Dict[int, int]]:
        """
        Reset the simulation environment and return the initial state and a list of all possible locations of the nodes
        """
        pass


class RSMEnv(Env):
    """
    A network container for OpenAI GYM for the environment of a Replicated State Machine system
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
        initial_latencies = self.simulation.reset()
        self.graph = adjust_networkx_graph(self.graph, initial_latencies)

        self.action_space = spaces.Discrete(config.n_actions)
        self.state = None

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:

        self.simulation.setPlacement(action)
        interval_result = self.simulation.runInterval()
        self.graph = adjust_networkx_graph(self.graph, interval_result)

        # Calculate reward (dummy implementation)
        reward = self.compute_reward()
        done = False
        truncated = False
        info = {}

        return self.graph, reward, done, truncated, info

    def compute_reward(self) -> float:
        """
        Compute the reward based on the current state
        """

        return np.random.random()

    def render(self, mode='human'):
        """
        Render the graph using matplotlib.
        """
        if mode == 'human':
            plt.figure(figsize=(8, 6))
            pos = nx.spring_layout(self.graph)  # positions for all nodes

            # nodes
            nx.draw_networkx_nodes(self.graph, pos, node_size=700)

            # edges
            nx.draw_networkx_edges(self.graph, pos, width=6)

            # node labels
            nx.draw_networkx_labels(self.graph, pos, font_size=20, font_family='sans-serif')

            # edge weight labels
            edge_labels = nx.get_edge_attributes(self.graph, 'weight')
            nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)

            plt.axis('off')  # Turn off the axis
            plt.show()  # Display the plot

class TorchGraphObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env: RSMEnv):
        super().__init__(env)

    def observation(self, observation) -> torch_geometric.data.Data:
        return from_networkx(observation)


def adjust_networkx_graph(G: nx.Graph, latency_map: IntervalResult):
    """
    Adjust the networkx graph with the latency map
    :param G:
    :param latency_map:
    :return:
    """
    for src_id, edges in latency_map.IntervalLatencyMap.items():
        for dest_id, latency in edges.items():
            G.add_edge(src_id, dest_id, weight=latency)

    return G

