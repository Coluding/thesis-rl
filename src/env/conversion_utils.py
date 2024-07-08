from torch_geometric.data import Data
from typing import Dict
import networkx as nx


def create_networkx_graph(latency_map: Dict[int, Dict[int, int]]):
    """
    Create a networkx graph from the latency map
    :param latency_map:
    :return:
    """
    G = nx.Graph()
    for src_id, edges in latency_map.items():
        for dest_id, latency in edges.items():
            G.add_edge(src_id, dest_id, weight=latency)

    return G



