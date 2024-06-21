import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def custom_layout(G, clusters, clients):
    pos = {}
    cluster_count = len(clusters)
    spacing = 1.0 / (cluster_count + 1)

    # Position clusters horizontally with some vertical variation
    for i, cluster in enumerate(clusters):
        x = (i + 1) * spacing
        y_spacing = 1.0 / (len(cluster) + 1)
        for j, node in enumerate(cluster):
            pos[node] = (x, (j + 1) * y_spacing)

    # Position clients randomly
    for client in clients:
        pos[client] = (np.random.rand(), np.random.rand())

    return pos


# Example of creating a graph
G = nx.Graph()

# Define clusters and clients
clusters = [
    ['c1n1', 'c1n2', 'c1n3'],
    ['c2n1', 'c2n2', 'c2n3'],
    ['c3n1', 'c3n2', 'c3n3']
]

clients = ['client1', 'client2', 'client3']

# Add nodes and edges
for cluster in clusters:
    G.add_nodes_from(cluster)
    for i in range(len(cluster) - 1):
        G.add_edge(cluster[i], cluster[i + 1], latency=np.random.rand())

G.add_nodes_from(clients)
for client in clients:
    # Connect clients to random nodes in random clusters
    ind = np.random.choice(np.arange(len(clusters)))
    cluster = clusters[ind]
    node = np.random.choice(cluster)
    G.add_edge(client, node, latency=np.random.rand())

# Get custom layout
pos = custom_layout(G, clusters, clients)

# Plot the graph
plt.figure(figsize=(12, 8))
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, edge_color='gray')
nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): f"{d['latency']:.2f}" for u, v, d in G.edges(data=True)})
plt.title('Data Center Network Graph')
plt.show()
