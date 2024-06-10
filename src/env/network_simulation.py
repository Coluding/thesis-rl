import time

import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple


class NetworkEnvironment:
    def __init__(self, num_centers, clusters, num_clients, cluster_region_start = "asia",
                 total_requests_per_interval=10000, k=3, p=1, client_start_region: str | dict = 'asia'):
        self.graph = nx.Graph()
        self.data_centers = []
        self.clients = [f'client_{i}' for i in range(num_clients)]
        self.client_regions = {f'client_{i}': client_start_region for i in range(num_clients)} if isinstance(
            client_start_region, str) else client_start_region
        self.client_weights = np.ones(num_clients)
        self.internal_latencies = {}
        self.cluster_region_start = cluster_region_start
        self.active_replicas = set()
        self.passive_replicas = set()
        self.clusters = clusters
        self.num_centers = num_centers
        self.total_requests_per_interval = total_requests_per_interval
        self.time_of_day = 0  # Internal time counter
        self.k = k  # Number of active data centers
        self.p = p # Number of passive data centers

        self._initialize_data_centers()
        self._initialize_internal_latencies()
        self.update_active_dc_latencies(self.active_replicas)

        # Add data centers and clients to the graph
        self.graph.add_nodes_from(self.data_centers + self.clients)
        self._initialize_edges()

    def _initialize_data_centers(self):
        locations = ['europe', 'asia', 'usa']
        for loc, count in zip(locations, self.clusters):
            if loc == self.cluster_region_start:
                active_inds = np.random.choice(count, self.k, replace = False)
                passive_inds = np.random.choice([i for i in range(count) if i not in active_inds], self.p, replace = False)
            for i in range(count):
                self.data_centers.append(f'{loc}_dc_{i + 1}')
                if loc == self.cluster_region_start and i in active_inds:
                    self.active_replicas.add(f'{loc}_dc_{i + 1}')
                elif loc == self.cluster_region_start and i in passive_inds:
                    self.passive_replicas.add(f'{loc}_dc_{i + 1}')

    def _initialize_internal_latencies(self):
        intra_cluster_latency = (10, 100)
        inter_cluster_latency = (100, 500)

        for i in range(len(self.data_centers)):
            for j in range(i + 1, len(self.data_centers)):
                dc1 = self.data_centers[i]
                dc2 = self.data_centers[j]
                if dc1.split('_')[0] == dc2.split('_')[0]:
                    latency = random.randint(*intra_cluster_latency)
                else:
                    latency = random.randint(*inter_cluster_latency)
                self.internal_latencies[(dc1, dc2)] = latency
                self.internal_latencies[(dc2, dc1)] = latency

    def _initialize_edges(self):
        for client in self.clients:
            for dc in self.active_replicas:
                latency = random.randint(10, 100)
                self.graph.add_edge(client, dc, latency=latency)

    def reconfigure_active_nodes(self, new_active_list):
        self.active_replicas = set(new_active_list)
        self.update_client_latencies()
        self.update_active_dc_latencies(new_active_list)

    def reconfigure_passive_nodes(self, new_passive_list):
        self.passive_replicas = set(new_passive_list)
        self.update_active_dc_latencies(self.active_replicas)

    def update_active_dc_latencies(self, active_list):
        for dc1 in active_list:
            for dc2 in active_list:
               if dc1 != dc2:
                    latency = self.internal_latencies[(dc1, dc2)]
                    self.graph.add_edge(dc1, dc2, latency=latency)

            for passive_dc in self.passive_replicas:
                if dc1 != passive_dc:
                    latency = self.internal_latencies[(dc1, passive_dc)]
                    self.graph.add_edge(dc1, passive_dc, latency=latency)

    def get_latency(self, node1, node2):
        try:
            return self.graph[node1][node2]['latency']
        except KeyError:
            return float('inf')

    def aggregate_latency(self):
        total_latency = 0
        count = 0

        for client in self.clients:
            for active in self.active_replicas:
                total_latency += self.get_latency(client, active)
                count += 1

        active_replicas_list = list(self.active_replicas)
        for i in range(len(active_replicas_list)):
            for j in range(i + 1, len(active_replicas_list)):
                total_latency += self.get_latency(active_replicas_list[i], active_replicas_list[j])
                count += 1

        return total_latency / count if count != 0 else 0

    def distribute_requests(self):
        requests = np.random.dirichlet(self.client_weights, 1)[0] * self.total_requests_per_interval
        request_distribution = dict(zip(self.clients, requests.astype(int)))
        return request_distribution

    def _compute_time_distance_of_intervals(self, x: float, timepoint: float) -> float:
        dist = min(abs(x - timepoint), abs(24 + x - timepoint))

        if dist >= 10:
            dist *= 4
            return dist
        if dist >= 8:
            dist *= 3
            return dist

        if dist == 0:
            return dist + 1e-6

        return dist

    def simulate_client_movements(self):
        peak_times = {
            'europe': (8, 16),
            'asia': (0, 8),
            'usa': (16, 24)
        }

        for i, client in enumerate(self.clients):
            dist_europe = self._compute_time_distance_of_intervals(
                self.time_of_day, (peak_times['europe'][0]) / 2 + (peak_times['europe'][1]) / 2)
            dist_asia = self._compute_time_distance_of_intervals(
                self.time_of_day, (peak_times['asia'][0]) / 2 + (peak_times['asia'][1]) / 2)
            dist_usa = self._compute_time_distance_of_intervals(
                self.time_of_day, (peak_times['usa'][0]) / 2 + (peak_times['usa'][1]) / 2)

            prob_europe = (1 / dist_europe) / (1/dist_europe + 1/dist_asia + 1/dist_usa)
            prob_asia = (1 / dist_asia) / (1/dist_europe + 1/dist_asia + 1/dist_usa)
            prob_usa = (1 / dist_usa) / (1/dist_europe + 1/dist_asia + 1/dist_usa)

            region = np.random.choice(['europe', 'asia', 'usa'], p=[prob_europe, prob_asia, prob_usa])
            # TODO: adjust client weights based on region
            self._move_client(client, region)

    def _move_client(self, client, region):
        for dc in self.active_replicas:
            if region in dc:
                latency = random.randint(10, 100)
            else:
                latency = random.randint(100, 500)
            self.graph[client][dc]['latency'] = latency
            self.client_regions[client] = region

    def update_client_latencies(self):
        self.graph.remove_edges_from([(client, dc) for client in self.clients for dc in self.data_centers])

        for client, region in self.client_regions.items():
            for dc in self.active_replicas:
                if region in dc:
                    latency = random.randint(10, 100)
                else:
                    latency = random.randint(100, 500)

                self.graph.add_edge(client, dc, latency=latency)

    def fluctuate_latencies(self):
        for key in self.internal_latencies:
            fluctuation = random.uniform(-0.01, 0.01) * self.internal_latencies[key]
            self.internal_latencies[key] += fluctuation

    def step(self):
        self.simulate_client_movements()
        request_distribution = self.distribute_requests()

        print(f"Client regions: {self.client_regions}")
        print(f"Active replicas: {self.active_replicas}")
        total_latency = 0
        for client, num_requests in request_distribution.items():
            for active_replica in self.active_replicas:
                latency = self.get_latency(client, active_replica)
                total_latency += latency * num_requests
                print(f"Client: {client}, DC: {active_replica}, Latency: {latency}, Requests: {num_requests}")
        avg_latency = self.aggregate_latency()
        print(f"Average latency: {avg_latency}")
        reward = -avg_latency

        self.time_of_day = (self.time_of_day + 1) % 24
        self.fluctuate_latencies()

        state = self.get_external_state()
        done = self.time_of_day == 0

        return state, reward, done

    def reset(self):
        self.time_of_day = 0
        self.update_client_latencies()
        return self.get_external_state()

    def get_external_state(self):
        client_locs = [self.client_regions[client] for client in self.clients]
        return client_locs, list(self.active_replicas)

    def get_internal_state(self):
        return self.internal_latencies

    def get_client_locations(self):
        client_locations = {}
        for client in self.clients:
            for dc in self.active_replicas:
                if self.get_latency(client, dc) < float('inf'):
                    client_locations[client] = dc
                    break
        return client_locations

    def visualize(self):
        pos = {}
        colors = []
        labels = {}

        for node in self.graph.nodes:
            if 'europe' in node:
                pos[node] = (random.uniform(0, 1), random.uniform(0, 1))
                if node in self.active_replicas:
                    colors.append('darkblue')
                elif node in self.passive_replicas:
                    colors.append('blue')
                else:
                    colors.append('lightblue')

            elif 'asia' in node:
                pos[node] = (random.uniform(1, 2), random.uniform(1, 2))
                if node in self.active_replicas:
                    colors.append('darkred')
                elif node in self.passive_replicas:
                    colors.append('red')
                else:
                    colors.append('lightcoral')
            elif 'usa' in node:
                pos[node] = (random.uniform(2, 3), random.uniform(2, 3))
                if node in self.active_replicas:
                    colors.append('darkgreen')
                elif node in self.passive_replicas:
                    colors.append('green')
                else:
                    colors.append('lightgreen')
            elif 'client' in node:
                pos[node] = (random.uniform(0, 3), random.uniform(0, 3))
                colors.append('yellow')

            labels[node] = node

        edge_labels = {(u, v): f"{d['latency']}" for u, v, d in self.graph.edges(data=True)}

        plt.figure(figsize=(15, 10))
        nx.draw(self.graph, pos, node_color=colors, with_labels=True, labels=labels, node_size=3000, font_size=10,
                font_color='white')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_color='black')
        plt.title(f"Time of day: {self.time_of_day}")
        plt.show()


# Example usage
num_centers = 10
clusters = [3, 4, 3]  # Europe, Asia, USA
num_clients = 5

env = NetworkEnvironment(num_centers=num_centers, clusters=clusters, num_clients=num_clients)

# Example usage
num_centers = 15
clusters = [5, 5, 5]  # Europe, Asia, USA
num_clients = 5

env = NetworkEnvironment(num_centers=num_centers, clusters=clusters, num_clients=num_clients)

# Simulate for 24 hours
for i in range(24):
    print(f"Time of day: {i}")
    env.step()
    env.visualize()
    time.sleep(0.5)

    if i == 6:
        env.reconfigure_active_nodes(['asia_dc_1', 'asia_dc_2', 'europe_dc_1'])
        env.reconfigure_passive_nodes(['europe_dc_2'])
        print("Reconfiguring active nodes")
        print("-------------------------------------------------------------------------")

    if i == 12:
        env.reconfigure_active_nodes(['europe_dc_1', 'europe_dc_2', 'europe_dc_3'])
        env.reconfigure_passive_nodes(['europe_dc_4'])
        print("Reconfiguring active nodes")
        print("-------------------------------------------------------------------------")

    elif i == 20:
        env.reconfigure_active_nodes(['usa_dc_1', 'usa_dc_2', 'usa_dc_3'])
        env.reconfigure_passive_nodes(['usa_dc_4'])
        print("Reconfiguring active nodes")
        print("-------------------------------------------------------------------------")

