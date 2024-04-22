from env import IntervalResult
from torch_geometric.data import Data
import networkx as nx


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

    for node_id, requests in latency_map.ClientRequests.items():
        if node_id in G.nodes:  # Ensure the node already exists in the graph
            G.nodes[node_id]['requests'] = requests
        else:
            G.add_node(node_id, requests=requests)

    return G