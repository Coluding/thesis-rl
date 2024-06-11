import time
from dataclasses import dataclass
import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Union, Optional
from enum import Enum


class ActionType(Enum):
    PASSIVE = 1
    ACTIVE = 2
    BOTH = 3

@dataclass
class Penalty:
    ReconfigurationPenalty: float = 100
    PassiveCost: float = 10
    ActiveCost: float = 10
    WrongActiveAfterPassivePenalty: float = 100
    WrongActivePenalty: float = 100
    WrongNumberPassivePenalty: float = 100

@dataclass
class PenaltyWeights:
    LatencyPenalty: float = 1
    ReconfigurationPenalty: float = 1
    PassiveCost: float = 1
    ActiveCost: float = 1
    WrongActivePenalty: float = 1
    WrongActiveAfterPassivePenalty: float = 1
    WrongNumberPassivePenalty: float = 1

class NetworkEnvironment:
    def __init__(self, num_centers, clusters, num_clients, penalty: Optional[Penalty] = Penalty,
                 penalty_weights: Optional[PenaltyWeights] = PenaltyWeights,
                 cluster_region_start = "asia",
                 action_type: ActionType = ActionType.BOTH,
                 total_requests_per_interval=10000, k=3, p=1, client_start_region: str | dict = 'asia'):
        self.graph = nx.Graph()
        self.action_type = action_type
        self.data_centers = []
        self.penalty = penalty
        self.penalty_weights = penalty_weights
        self.clients = [f'client_{i}' for i in range(num_clients)]
        self.client_regions = {f'client_{i}': client_start_region for i in range(num_clients)} if isinstance(
            client_start_region, str) else client_start_region
        self.client_weights = np.ones(num_clients)
        self.internal_latencies = {}
        self.cluster_region_start = cluster_region_start
        self.active_replicas = set()
        self.active_history = set()
        self.passive_replicas = set()
        self.passive_history = set()
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

        self.penalty_state = 0

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

    def reconfigure_active_nodes(self):
        self.update_client_latencies()
        self.update_active_dc_latencies(self.active_replicas)

    def reconfigure_passive_nodes(self):
        self.update_active_dc_latencies(self.active_replicas)

    def convert_action(self, action: Union[Tuple[str, str], Tuple[Tuple[str, str], Tuple[str, str]]]):
        """
        Convert the action to the appropriate format based on the action type
        """
        if self.action_type == ActionType.PASSIVE:
            if action[0] not in self.passive_replicas:
                raise ValueError("The suggested location to be replaed is currently not in the passive data centers")
            valid = self.review_passive_action(action)
            if valid:
                self.passive_replicas.remove(action[0])
                self.passive_replicas.add(action[1])
                self._review_reconfiguration(action)
            self.reconfigure_passive_nodes()
        elif self.action_type == ActionType.ACTIVE:
            if action[0] not in self.active_replicas:
                raise ValueError("The suggested location to be replaced is currently not in the active data centers")
            valid = self._review_active_action(action)
            if valid:
                self.active_replicas.remove(action[0])
                self.active_replicas.add(action[1])
                self._review_reconfiguration(action)
            self.reconfigure_active_nodes()
        elif self.action_type == ActionType.BOTH:
            if len(action) != 2:
                raise ValueError("Invalid action format. It must be a tuple of two tuples.")
            valid = self._review_active_action(action[0])
            if valid:
                self.active_replicas.remove(action[0][0])
                self.active_replicas.add(action[0][1])
                self._review_active_action(action[0])
            valid = self.review_passive_action(action[1])
            if valid:
                self.passive_replicas.remove(action[1][0])
                self.passive_replicas.add(action[1][1])
                self._review_reconfiguration(action[1])
            self.reconfigure_active_nodes()
            self.reconfigure_passive_nodes()
        else:
            raise ValueError("Invalid action type")

        self.active_history = self.active_history.union(self.active_replicas)
        self.passive_history = self.passive_history.union(self.passive_replicas)


    def review_passive_action(self, action: Tuple[str, str]) -> bool:
        """
        Review the passive action to check whether the number of passive nodes is not exceeded. If so, the internal
        penalty state is triggered.
        """
        valid = True
        if action[1] in self.passive_replicas:
            self.penalty_state += (self.penalty_weights.WrongActivePenalty * self.penalty.WrongActivePenalty)
            valid = False

        return valid

    def _review_active_action(self, action: Tuple[str, str]) -> bool:
        """
        Review the active action to check wether a previous non-passive node was made active. If so, the internal
        penalty state is triggered.
        """
        valid = True
        if action[1] not in self.passive_history:
            self.penalty_state += (self.penalty_weights.WrongActiveAfterPassivePenalty *
                                   self.penalty.WrongActiveAfterPassivePenalty)

        if action[1] in self.active_replicas:
            self.penalty_state += (self.penalty_weights.WrongActivePenalty * self.penalty.WrongActivePenalty)
            valid = False

        return valid

    def _review_reconfiguration(self, action: Tuple[str, str]):
        """
        Review the reconfiguration to chck whether new nodes werwe selected. If so, the internal penalty state is
        triggered.
        """
        if action[0] != action[1]:
            self.penalty_state += (self.penalty_weights.ReconfigurationPenalty * self.penalty.ReconfigurationPenalty)

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
                latency = random.randint(150, 750)
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

    def step(self, action: Optional[Union[Tuple[str, str], Tuple[Tuple[str, str], Tuple[str, str]]] ] = None):
        self.simulate_client_movements()
        request_distribution = self.distribute_requests()
        #print(f"Client distribution: {request_distribution}")

        if action is not None:
            self.convert_action(action)

        print(f"Client regions: {self.client_regions}")
        print(f"Active replicas: {self.active_replicas}")
        total_latency = 0
        for client, num_requests in request_distribution.items():
            for active_replica in self.active_replicas:
                latency = self.get_latency(client, active_replica)
                total_latency += latency * num_requests
                #print(f"Client: {client}, DC: {active_replica}, Latency: {latency}, Requests: {num_requests}")
        avg_latency = self.aggregate_latency()
        print(f"Average latency: {avg_latency}")

        reward = self.penalty_weights.LatencyPenalty * (-avg_latency) - self.penalty_state
        print(f"Reward: {reward}")

        self.time_of_day = (self.time_of_day + 1) % 24
        self.fluctuate_latencies()

        state = self.get_external_state()
        #done = self.time_of_day == 0
        done = False

        self.penalty_state = 0

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
num_centers = 15
clusters = [5, 5, 5]  # Europe, Asia, USA
num_clients = 20

env = NetworkEnvironment(num_centers=num_centers, clusters=clusters, num_clients=num_clients)
intial_dcs = list(env.active_replicas)
print(f"Initial active data centers: {intial_dcs}")
initial_passive_dcs = list(env.passive_replicas)
print(f"Initial passive data centers: {initial_passive_dcs}")

# Simulate for 24 hours
for i in range(24):
    print(f"Time of day: {i}")
    #env.visualize()
    #time.sleep(0.5)

    if i == 6:
        print("Reconfiguring active nodes")
        print("-------------------------------------------------------------------------")
        env.step(((intial_dcs[0], 'europe_dc_2'), (initial_passive_dcs[0], 'europe_dc_1')))
        print("-------------------------------------------------------------------------")

    if i == 12:
        print("Reconfiguring active nodes")
        print("-------------------------------------------------------------------------")
        env.step(((intial_dcs[1], 'europe_dc_1'), (list(env.passive_replicas)[0], 'europe_dc_2')))
        print("-------------------------------------------------------------------------")

    elif i == 20:
        print("Reconfiguring active nodes")
        print("-------------------------------------------------------------------------")
        env.step(((intial_dcs[2], 'usa_dc_1'), ('europe_dc_2', 'usa_dc_3')))
        print("-------------------------------------------------------------------------")
    else:
        env.step()
