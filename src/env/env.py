from typing import Union, List, Tuple, Dict
import matplotlib.pyplot as plt
import gym
import networkx as nx
import numpy as np
from enum import Enum
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


class NodeState(Enum):
    """
    Enum for the state of a node in the network
    """
    ACTIVE = 1
    PASSIVE = 2
    OFF = 3


@dataclass
class IntervalResult:
    IntervalLatencyMap: Dict[int, Dict[int, int]]
    IntervalConfiguration: Dict[NodeState, List[int]]


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
    def reset(self) -> IntervalResult:
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
        initial_latencies: IntervalResult = self.simulation.reset()
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

    def render(self, mode='human',
               color_mapping: Dict[NodeState, str] = {0: 'gray', 1: 'blue', 2: 'orange'}):
        """
        Render the graph using matplotlib.
        """

        if mode == 'human':
            plt.figure(figsize=(8, 6))
            pos = nx.spring_layout(self.graph)

            # Fetch the state for each node and determine its outline color
            node_outline_colors = [color_mapping[self.graph.nodes[node].get('state', 2)] for node in
                                   self.graph.nodes()]

            # Inner color of nodes (can be uniform or different)
            node_inner_color = 'white'  # All nodes have a white inner color

            # Draw nodes with inner color
            nx.draw_networkx_nodes(self.graph, pos, node_color=node_inner_color, node_size=700)

            # Draw only the borders with a specific outline color
            nx.draw_networkx_nodes(self.graph, pos, node_color=node_outline_colors, node_size=700,
                                   edgecolors=node_outline_colors, linewidths=2)

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

    def observation(self, observation: nx.Graph) -> torch_geometric.data.Data:
        return from_networkx(observation)


def create_networkx_graph(interval_result: IntervalResult) -> nx.Graph:
    """
    Create a NetworkX graph from the IntervalResult object
    :param interval_result:
    :return:
    """
    # Create a new directed graph
    graph = nx.Graph()

    # Add nodes with attributes based on their features
    for feature, nodes in interval_result.IntervalConfiguration.items():
        for node in nodes:
            graph.add_node(node, state=feature)

    # Add edges based on the latency map
    for node, connections in interval_result.IntervalLatencyMap.items():
        for connected_node, latency in connections.items():
            graph.add_edge(node, connected_node, weight=latency)

    return graph


def adjust_networkx_graph(G: nx.Graph, interval_result) -> nx.Graph:
    """
    Adjust the networkx graph with the latency map and update node states from IntervalConfiguration.

    :param G: An existing NetworkX graph
    :param interval_result: An instance of IntervalResult with attributes IntervalLatencyMap and IntervalConfiguration
    :return: The modified NetworkX graph
    """
    # Update edges with new latencies
    for src_id, edges in interval_result.IntervalLatencyMap.items():
        for dest_id, latency in edges.items():
            # This will create the edge if it doesn't exist or update the weight if it does
            G.add_edge(src_id, dest_id, weight=latency)

    # Update node states
    node_states = {}
    state_map = {k: i for i, k in enumerate(interval_result.IntervalConfiguration.keys())}
    for state, nodes in interval_result.IntervalConfiguration.items():
        for node in nodes:
            if not G.has_node(node):
                G.add_node(node)
            node_states[node] = state_map[state]

    # Apply the state attributes to nodes in the graph
    nx.set_node_attributes(G, node_states, 'state')

    return G
