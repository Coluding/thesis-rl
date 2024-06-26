import time
from dataclasses import dataclass
import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Union, Optional
from enum import Enum
import plotly.graph_objects as go  # Import Plotly for visualization
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from abc import ABC, abstractmethod

#TODO: Different reward for active and passive nodes

class ActionType(Enum):
    PASSIVE = 1
    ACTIVE = 2
    BOTH = 3


@dataclass
class Penalty:
    ReconfigurationPenalty: float = 50
    PassiveCost: float = 10
    ActiveCost: float = 10
    WrongActiveAfterPassivePenalty: float = 100
    WrongActivePenalty: float = 100
    WrongPassivePenalty: float = 100
    WrongNumberPassivePenalty: float = 100
    ActiveNodeAlsoPassivePenalty: float = 500
    NewEdgesDiscoveredReward: float = 300

@dataclass
class PenaltyWeights:
    LatencyPenalty: float = 1
    ReconfigurationPenalty: float = 1
    PassiveCost: float = 1
    ActiveCost: float = 1
    WrongActivePenalty: float = 1
    WrongPassivePenalty: float = 1
    WrongActiveAfterPassivePenalty: float = 1
    WrongNumberPassivePenalty: float = 1
    ActiveNodeAlsoPassivePenalty: float = 1
    ReExplorationPassiveWeight: float = 1
    EdgeDifferenceWeight: float = 1
    NewEdgesDiscoveredReward: float = 1


class Regions(Enum):
    ASIA = "asia"
    EUROPE = "europe"
    USA = "usa"
    AFRICA = "africa"
    EU_WEST = "eu_west"
    EU_EAST = "eu_east"
    US_WEST = "us_west"
    US_EAST = "us_east"
    ASIA_EAST = "asia_east"
    ASIA_WEST = "asia_west"
    US_SOUTH = "us_south"


@dataclass
class RegionWithDistanceCalc:
    region: Regions
    coordinates: Tuple[float, float, float, float]
    peak_times: Tuple[float, float]
    intra_region_latencies: Tuple[float, float]


AsiaRegion = RegionWithDistanceCalc(Regions.ASIA, (60, 120, 10, 50), (0, 8), (10, 70))
EuropeRegion = RegionWithDistanceCalc(Regions.EUROPE, (-10, 10, 40, 60), (8, 16), (10, 40))
USRegion = RegionWithDistanceCalc(Regions.USA, (-130, -80, 20, 50), (16, 24), (20, 60))
AfricaRegion = RegionWithDistanceCalc(Regions.AFRICA, (-20, 50, -40, 40), (12, 18), (30, 70))
USSOUTHRegion = RegionWithDistanceCalc(Regions.US_SOUTH, (-130, -80, 10, 20), (16, 24), (15, 45))

latencies = {
    (Regions.ASIA, Regions.EUROPE): (200, 260),
    (Regions.ASIA, Regions.USA): (270, 330),
    (Regions.ASIA, Regions.AFRICA): (220, 280),
    (Regions.ASIA, Regions.EU_WEST): (190, 230),
    (Regions.ASIA, Regions.EU_EAST): (200, 240),
    (Regions.ASIA, Regions.US_WEST): (260, 320),
    (Regions.ASIA, Regions.US_EAST): (290, 350),
    (Regions.ASIA, Regions.ASIA_EAST): (60, 100),
    (Regions.ASIA, Regions.ASIA_WEST): (80, 120),
    (Regions.ASIA, Regions.US_SOUTH): (280, 340),

    (Regions.EUROPE, Regions.USA): (100, 140),
    (Regions.EUROPE, Regions.AFRICA): (150, 210),
    (Regions.EUROPE, Regions.EU_WEST): (30, 50),
    (Regions.EUROPE, Regions.EU_EAST): (40, 60),
    (Regions.EUROPE, Regions.US_WEST): (140, 180),
    (Regions.EUROPE, Regions.US_EAST): (110, 150),
    (Regions.EUROPE, Regions.ASIA_EAST): (200, 260),
    (Regions.EUROPE, Regions.ASIA_WEST): (170, 230),
    (Regions.EUROPE, Regions.US_SOUTH): (120, 160),

    (Regions.USA, Regions.AFRICA): (190, 250),
    (Regions.USA, Regions.EU_WEST): (90, 130),
    (Regions.USA, Regions.EU_EAST): (100, 140),
    (Regions.USA, Regions.US_WEST): (50, 90),
    (Regions.USA, Regions.US_EAST): (20, 40),
    (Regions.USA, Regions.ASIA_EAST): (270, 330),
    (Regions.USA, Regions.ASIA_WEST): (260, 320),
    (Regions.USA, Regions.US_SOUTH): (30, 70),

    (Regions.AFRICA, Regions.EU_WEST): (130, 190),
    (Regions.AFRICA, Regions.EU_EAST): (140, 200),
    (Regions.AFRICA, Regions.US_WEST): (200, 260),
    (Regions.AFRICA, Regions.US_EAST): (180, 240),
    (Regions.AFRICA, Regions.ASIA_EAST): (220, 280),
    (Regions.AFRICA, Regions.ASIA_WEST): (210, 270),
    (Regions.AFRICA, Regions.US_SOUTH): (190, 250),

    (Regions.EU_WEST, Regions.EU_EAST): (10, 30),
    (Regions.EU_WEST, Regions.US_WEST): (130, 170),
    (Regions.EU_WEST, Regions.US_EAST): (100, 140),
    (Regions.EU_WEST, Regions.ASIA_EAST): (190, 250),
    (Regions.EU_WEST, Regions.ASIA_WEST): (180, 240),
    (Regions.EU_WEST, Regions.US_SOUTH): (110, 150),

    (Regions.EU_EAST, Regions.US_WEST): (140, 180),
    (Regions.EU_EAST, Regions.US_EAST): (110, 150),
    (Regions.EU_EAST, Regions.ASIA_EAST): (200, 260),
    (Regions.EU_EAST, Regions.ASIA_WEST): (190, 250),
    (Regions.EU_EAST, Regions.US_SOUTH): (120, 160),

    (Regions.US_WEST, Regions.US_EAST): (50, 90),
    (Regions.US_WEST, Regions.ASIA_EAST): (250, 310),
    (Regions.US_WEST, Regions.ASIA_WEST): (240, 300),
    (Regions.US_WEST, Regions.US_SOUTH): (40, 80),

    (Regions.US_EAST, Regions.ASIA_EAST): (280, 340),
    (Regions.US_EAST, Regions.ASIA_WEST): (270, 330),
    (Regions.US_EAST, Regions.US_SOUTH): (30, 70),

    (Regions.ASIA_EAST, Regions.ASIA_WEST): (60, 100),
    (Regions.ASIA_EAST, Regions.US_SOUTH): (280, 340),

    (Regions.ASIA_WEST, Regions.US_SOUTH): (270, 330),
}


class NetworkEnvironment:
    def __init__(self,
                 num_centers: int, clusters: List[int],
                 num_clients: int, penalty: Optional[Penalty] = Penalty,
                 region_objects: List[RegionWithDistanceCalc] = (AsiaRegion, EuropeRegion, USRegion),
                 penalty_weights: Optional[PenaltyWeights] = PenaltyWeights,
                 cluster_region_start=Regions.ASIA,
                 action_type: ActionType = ActionType.BOTH,
                 period_length=5000,
                 total_requests_per_interval=10000, k=3, p=1, client_start_region: Union[Regions, dict] = Regions.ASIA):
        self.global_step = 1
        self.graph = nx.Graph()
        if len(clusters) != len(region_objects):
            raise ValueError("The number of clusters must match the number of region objects")
        self.region_objects = region_objects
        self.action_type = action_type
        self.data_centers = []
        self.penalty = penalty
        self.penalty_weights = penalty_weights
        self.clients = [f'c_{i}' for i in range(num_clients)]
        self.client_regions = {f'c_{i}': client_start_region.value for i in range(num_clients)} if isinstance(
            client_start_region, Enum) else client_start_region
        self.client_weights = np.ones(num_clients)
        self.type_mapping = {
            "active": 2,
            "passive": 1,
            "inactive": 0,
            "client": 3
        }
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
        self.p = p  # Number of passive data centers
        self.str_to_region = None
        self._initialize_data_centers()

        self.dc_to_int = {dc: i for i, dc in enumerate(self.data_centers)}
        self.int_to_dc = {i: dc for i, dc in enumerate(self.data_centers)}
        self.client_to_int = {client: i for i, client in enumerate(self.clients)}
        self.int_to_client = {i: client for i, client in enumerate(self.clients)}

        self._initialize_internal_latencies()
        self.update_active_dc_latencies(self.active_replicas)

        # Add data centers and clients to the graph
        self.graph.add_nodes_from(self.data_centers + self.clients)
        #self._initialize_edges()
        self._initialize_clients()

        self._active_penalty_state = 0
        self._passive_penalty_state = 0
        self.positions = {}
        self._generate_dc_positions()
        self._generate_client_positions()  # New attribute for positions

        self.available_active_replicas = set(self.data_centers).difference(set(self.active_replicas))
        self.available_passive_replicas = set(self.data_centers).difference(set(self.passive_replicas))

        self.active_penalty_tracker = []
        self.passive_penalty_tracker = []
        self.current_request_distribution = None
        self.current_latency = 0

        self._add_mask_to_graph()

        self.period_length = period_length
        self._total_number_of_edges = self._compute_total_number_of_potential_edges_between_dcs()

    def _compute_total_number_of_potential_edges_between_dcs(self):
        total = 0
        for i in range(len(self.data_centers)):
            for j in range(i + 1, len(self.data_centers)):
                total += 1
        return total

    def _initialize_data_centers(self):
        for loc, count in zip(self.region_objects, self.clusters):
            if loc.region == self.cluster_region_start:
                active_inds = np.random.choice(count, self.k, replace=False)
                passive_inds = np.random.choice([i for i in range(count) if i not in active_inds], self.p,
                                                replace=False)
            for i in range(count):
                dc_name = f'{loc.region.value}_dc_{i + 1}'
                self.data_centers.append(dc_name)
                self.graph.add_node(dc_name, type=self.type_mapping['inactive'])
                if loc.region == self.cluster_region_start and i in active_inds:
                    self.active_replicas.add(dc_name)
                    self.graph.nodes[dc_name]['type'] = self.type_mapping['active']
                elif loc.region == self.cluster_region_start and i in passive_inds:
                    self.passive_replicas.add(dc_name)
                    self.graph.nodes[dc_name]['type'] = self.type_mapping['passive']

                self.graph.nodes[dc_name]['requests'] = 0
                self.graph.nodes[dc_name]['update_step'] = self.global_step

        self.str_to_region = {loc.region.value: loc for loc in self.region_objects}

    def _add_mask_to_graph(self):
        for node in self.graph.nodes:
            if node in self.available_active_replicas:
                self.graph.nodes[node]['active_mask'] = 0
            else:
                self.graph.nodes[node]['active_mask'] = -np.inf

            if node in self.available_passive_replicas:
                self.graph.nodes[node]['passive_mask'] = 0
            else:
                self.graph.nodes[node]['passive_mask'] = -np.inf

    def _initialize_internal_latencies(self):
        intra_cluster_latency = (10, 100)
        inter_cluster_latency = (100, 500)

        for i in range(len(self.data_centers)):
            for j in range(i + 1, len(self.data_centers)):
                dc1 = self.data_centers[i]
                dc2 = self.data_centers[j]
                if dc1.split('_')[0] == dc2.split('_')[0]:
                    region_calc = self.str_to_region[dc1.split('_')[0]]
                    latency = random.randint(*region_calc.intra_region_latencies)
                else:
                    region_calc_1 = self.str_to_region[dc1.split('_')[0]]
                    region_calc_2 = self.str_to_region[dc2.split('_')[0]]
                    comb1 = (region_calc_1.region, region_calc_2.region)
                    comb2 = (region_calc_2.region, region_calc_1.region)
                    if comb1 in latencies:
                        latency = random.randint(*latencies[comb1])
                    elif comb2 in latencies:
                        latency = random.randint(*latencies[comb2])
                    else:
                        raise ValueError(f"Latency between {region_calc_1.region} and {region_calc_2.region} not found")
                self.internal_latencies[(dc1, dc2)] = latency
                self.internal_latencies[(dc2, dc1)] = latency

    def _initialize_clients(self):
        for c in self.clients:
            self.graph.add_node(c, type=self.type_mapping['client'])
            self.graph.nodes[c]['requests'] = 0
            self.graph.nodes[c]['update_step'] = self.global_step

    def _initialize_edges(self):
        for client in self.clients:
            for dc in self.active_replicas:
                latency = random.randint(10, 100)
                self.graph.add_edge(client, dc, latency=latency)
            self.graph.nodes[client]['type'] = self.type_mapping['client']
            self.graph.nodes[client]['requests'] = 0

    def _generate_dc_positions(self):
        positions = {}
        for dc in self.data_centers:
            for loc in self.region_objects:
                if loc.region.value in dc:
                    positions[dc] = (random.uniform(loc.coordinates[0], loc.coordinates[1]),
                                     random.uniform(loc.coordinates[2], loc.coordinates[3]))
        self.positions = positions

    def _generate_client_positions(self):
        for client, region in self.client_regions.items():
            for loc in self.region_objects:
                if loc.region.value in region:
                    self.positions[client] = (random.uniform(loc.coordinates[0], loc.coordinates[1]),
                                              random.uniform(loc.coordinates[2], loc.coordinates[3]))

    def reconfigure_active_nodes(self):
        self.update_client_latencies()
        self.update_active_dc_latencies(self.active_replicas)

    def reconfigure_passive_nodes(self):
        self.update_active_dc_latencies(self.active_replicas)

    def update_available_replicas(self):
        self.available_active_replicas = set(self.data_centers).difference(set(self.active_replicas))
        self.available_passive_replicas = set(self.data_centers).difference(set(self.passive_replicas))

    def do_action(self, action: Union[Tuple[str, str], Tuple[Tuple[str, str], Tuple[str, str]]]):
        """
        Convert the action to the appropriate format based on the action type
        """

        if self.action_type == ActionType.PASSIVE:
            if action[0] not in self.passive_replicas:
                raise ValueError("The suggested location to be replaced is currently not in the passive data centers")
            if isinstance(action[0][0], int):
                action = [action[0], action[1]]
                action[0] = [self.int_to_dc[action[0]], self.int_to_dc[action[1]]]
            valid = self.review_passive_action(action)
            if valid:
                self.passive_replicas.remove(action[0])
                self.graph.nodes[action[0]]['type'] = 'inactive'
                self.passive_replicas.add(action[1])
                self.graph.nodes[action[1]]['type'] = 'passive'
                self.graph.nodes[action[1]]["update_step"] = self.global_step
                self._review_reconfiguration(action, active=False)
                self.update_available_replicas()
            self.reconfigure_passive_nodes()
        elif self.action_type == ActionType.ACTIVE:
            if isinstance(action[0][0], int):
                action = [action[0], action[1]]
                action[0] = [self.int_to_dc[action[0]], self.int_to_dc[action[1]]]
            if action[0] not in self.active_replicas:
                raise ValueError("The suggested location to be replaced is currently not in the active data centers")
            valid = self._review_active_action(action)
            if valid:
                self.active_replicas.remove(action[0])
                self.graph.nodes[action[0]]['type'] = self.type_mapping['inactive']
                self.active_replicas.add(action[1])
                self.graph.nodes[action[1]]['type'] = self.type_mapping['active']
                self.graph.nodes[action[1]]["update_step"] = self.global_step
                self._review_reconfiguration(action, active=True)
                self.update_available_replicas()
            self.reconfigure_active_nodes()
        elif self.action_type == ActionType.BOTH:
            if len(action) != 2:
                raise ValueError("Invalid action format. It must be a tuple of two tuples.")
            if isinstance(action[0][0], int):
                action = [action[0], action[1]]
                action[0] = [self.int_to_dc[action[0][0]], self.int_to_dc[action[0][1]]]
                action[1] = [self.int_to_dc[action[1][0]], self.int_to_dc[action[1][1]]]
            valid = self._review_active_action(action[0])
            if valid:
                active_action_remove = action[0][0]
                active_action_add = action[0][1]
                self.active_replicas.remove(active_action_remove)
                self.graph.nodes[active_action_remove]['type'] = self.type_mapping['inactive']
                self.active_replicas.add(active_action_add)
                self.graph.nodes[active_action_add]['type'] = self.type_mapping['active']
                self.graph.nodes[active_action_add]["update_step"] = self.global_step
                self._review_active_action((active_action_remove, active_action_add))
                self._review_reconfiguration((active_action_remove, active_action_add), active=True)
            valid = self.review_passive_action(action[1])
            if valid:
                passive_action_remove = action[1][0]
                self.passive_replicas.remove(passive_action_remove)
                self.graph.nodes[passive_action_remove]['type'] = self.type_mapping['inactive']
                passive_action_add = action[1][1]
                self.passive_replicas.add(passive_action_add)
                self.graph.nodes[action[1][1]]['type'] = self.type_mapping['passive']
                self.graph.nodes[action[1][1]]["update_step"] = self.global_step
                self._review_reconfiguration(passive_action_add, active=False)
            self.reconfigure_active_nodes()
            self.reconfigure_passive_nodes()
        else:
            raise ValueError("Invalid action type")

        self.active_history = self.active_history.union(self.active_replicas)
        self.passive_history = self.passive_history.union(self.passive_replicas)
        self.update_available_replicas()

    def review_passive_action(self, action: Tuple[str, str]) -> bool:
        """
        Review the passive action to check whether the number of passive nodes is not exceeded. If so, the internal
        penalty state is triggered.
        """
        valid = True
        if action[1] in self.passive_replicas:
            self._passive_penalty_state -= (self.penalty_weights.WrongPassivePenalty * self.penalty.WrongPassivePenalty)
            self.passive_penalty_tracker.append("WrongPassivePenalty")
            valid = False

        if action[1] in self.active_replicas:
            self._passive_penalty_state -= (self.penalty_weights.ActiveNodeAlsoPassivePenalty *
                                   self.penalty.ActiveNodeAlsoPassivePenalty)
            self.passive_penalty_tracker.append("ActiveNodeAlsoPassivePenalty")
            valid = False

        old_passive, new_passive = action
        new_node = self.graph.nodes[new_passive]

        waiting_time_to_reexplore = self.global_step - new_node['update_step']
        edges_difference = self._total_number_of_edges - len(self.graph.edges)

        # give extra reward if node was selected with no previous edges
        bonus_new_edges = 0
        if self.graph.degree(new_passive) == 0:
            bonus_new_edges = self.penalty.NewEdgesDiscoveredReward

        self._passive_penalty_state += (self.penalty_weights.ReExplorationPassiveWeight * waiting_time_to_reexplore)
        self._passive_penalty_state += (self.penalty_weights.EdgeDifferenceWeight * edges_difference)
        self._passive_penalty_state += (self.penalty_weights.NewEdgesDiscoveredReward * bonus_new_edges)


        return valid

    def _review_active_action(self, action: Tuple[str, str]) -> bool:
        """
        Review the active action to check whether a previous non-passive node was made active. If so, the internal
        penalty state is triggered.
        """
        valid = True
        # Here we make sure that only a node that was previously passive should be made active
        # This is not forced, i.e. the agent can opt to nelect this step. However, when done, there will be a penalty
        # for the action. Important: This only concerns the passive nodes in the current interval. Hence, a node that
        # was passive once in the history of the system but is not passive in the current interval does not count.
        if action[1] not in self.passive_replicas:
            self._active_penalty_state += (self.penalty_weights.WrongActiveAfterPassivePenalty *
                                   self.penalty.WrongActiveAfterPassivePenalty)
            self.active_penalty_tracker.append("WrongActiveAfterPassivePenalty")

        # This is to make sure that the number of active nodes does not exceed the limit.
        if action[1] in self.active_replicas and action[0] != action[1]:
            self._active_penalty_state += (self.penalty_weights.WrongActivePenalty * self.penalty.WrongActivePenalty)
            self.active_penalty_tracker.append("WrongActivePenalty")
            valid = False

        return valid

    def _review_reconfiguration(self, action: Tuple[str, str], active: bool):
        """
        Review the reconfiguration to check whether new nodes were selected. If so, the internal penalty state is
        triggered.
        """
        if action[0] != action[1]:
            if active:
                tracker = self.active_penalty_tracker
                self._active_penalty_state += (self.penalty_weights.ReconfigurationPenalty *
                                               self.penalty.ReconfigurationPenalty)
            else:
                tracker = self.passive_penalty_tracker
                self._passive_penalty_state -= (self.penalty_weights.ReconfigurationPenalty *
                                                self.penalty.ReconfigurationPenalty)

            tracker.append("Reconfiguration")

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

    def _aggregate_latency(self):
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

        self.current_latency = total_latency / count if count != 0 else 0

    def _distribute_requests(self):
        requests = np.random.dirichlet(self.client_weights, 1)[0] * self.total_requests_per_interval
        request_distribution = dict(zip(self.clients, requests.astype(int)))
        for client, num_requests in request_distribution.items():
            self.graph.nodes[client]['requests'] = num_requests
        self.current_request_distribution = request_distribution


    def _compute_time_distance_of_intervals(self, x: float, timepoint: float) -> float:
        dist = min(abs(x - timepoint), abs(24 + x - timepoint))
        if dist >= 10:
            dist *= 8
            return dist
        if dist >= 8:
            dist *= 6
            return dist
        if dist >= 6:
            dist *= 2
            return dist
        if dist == 0:
            return dist + 1e-6
        return dist

    def simulate_client_movements(self):
        dists = {}
        for i, client in enumerate(self.clients):
            for loc in self.region_objects:
                peak_times = loc.peak_times
                dists[loc.region.value] = self._compute_time_distance_of_intervals(self.time_of_day,
                                                                                   (peak_times[0] + peak_times[1]) / 2)
            probs = [1 / dists[region] / sum(1 / dist for dist in dists.values()) for region in dists]
            region = np.random.choice(self.region_objects, p=probs)
            self._move_client(client, region)

    def _move_client(self, client, region):
        for dc in self.active_replicas:
            if region.region.value in dc:
                latency = random.randint(10, 100)
            else:
                latency = random.randint(150, 750)
            try:
                self.graph[client][dc]['latency'] = latency
            except KeyError:
                self.graph.add_edge(client, dc, latency=latency)

            self.client_regions[client] = region.region.value

    def update_client_latencies(self):
        self.graph.remove_edges_from([(client, dc) for client in self.clients for dc in self.data_centers])

        for client, region in self.client_regions.items():
            for dc in self.active_replicas:
                if region in dc:
                    latency = random.randint(10, 100)
                else:
                    latency = random.randint(100, 750)

                self.graph.add_edge(client, dc, latency=latency)

    def fluctuate_latencies(self):
        for key in self.internal_latencies:
            fluctuation = random.uniform(-0.01, 0.01) * self.internal_latencies[key]
            self.internal_latencies[key] += fluctuation

    def step(self, action: Optional[Union[Tuple[str, str], Tuple[Tuple[str, str], Tuple[str, str]]]] = None):
        self._active_penalty_state = 0
        self._passive_penalty_state = 0
        self.penalty_tracker = []
        self.simulate_client_movements()
        self._distribute_requests()

        if action is not None:
            self.do_action(action)

        total_latency = 0
        for client, num_requests in self.current_request_distribution.items():
            for active_replica in self.active_replicas:
                latency = self.get_latency(client, active_replica)
                total_latency += latency * num_requests
        self._aggregate_latency()

        reward = self._get_active_and_passive_reward()

        self.time_of_day = (self.time_of_day + 1) % 24
        self.fluctuate_latencies()

        state = self.graph

        done = False
        if self.global_step >= self.period_length:
            done = True
            self.global_step = 0

        self._add_mask_to_graph()

        self.global_step += 1

        return state, reward, done

    def _get_active_and_passive_reward(self):
        active_reward = self.penalty_weights.LatencyPenalty * (-self.current_latency) - self._active_penalty_state
        passive_reward = self._passive_penalty_state

        return active_reward, passive_reward



    def reset(self):
        self.time_of_day = 0
        self.update_client_latencies()
        self.update_client_latencies()
        return self.get_external_state()

    def get_external_state(self):
        client_locs = [self.client_regions[client] for client in self.clients]
        return client_locs, list(self.active_replicas)

    def get_internal_state(self):
        return self.internal_latencies

    def convert_int_to_dc(self, action):
        return [self.int_to_dc[dc] for dc in action]

    def no_op_active(self):
        remove_dc = random.choice(list(self.active_replicas))
        add_dc = remove_dc

        return remove_dc, add_dc

    def no_op_passive(self):
        remove_dc = random.choice(list(self.passive_replicas))
        add_dc = remove_dc

        return remove_dc, add_dc

    def get_client_locations(self):
        client_locations = {}
        for client in self.clients:
            for dc in self.active_replicas:
                if self.get_latency(client, dc) < float('inf'):
                    client_locations[client] = dc
                    break
        return client_locations

    def visualize(self, return_fig=False):
        pos = {}
        colors = []
        labels = {}
        plt.clf()
        for node in self.graph.nodes:
            for loc in self.region_objects:
                if loc.region.value in node:
                    pos[node] = (random.uniform(loc.coordinates[0], loc.coordinates[1]),
                                 random.uniform(loc.coordinates[2], loc.coordinates[3]))
                    if node in self.active_replicas:
                        colors.append('green')
                    elif node in self.passive_replicas:
                        colors.append('blue')
                    else:
                        colors.append('gray')
                elif 'c_' in node[:2]:
                    pos[node] = (random.uniform(-10, 10), random.uniform(-10, 10))
                    colors.append('yellow')

            labels[node] = node

        edge_labels = {(u, v): f"{d['latency']}" for u, v, d in self.graph.edges(data=True)}
        fig, ax = plt.subplots(figsize=(16, 8))
        #plt.figure(figsize=(15, 10))
        nx.draw(self.graph, pos, node_color=colors, with_labels=True, labels=labels, node_size=3000, font_size=10,
                font_color='white', ax=ax)
        #nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_color='black')
        plt.title(f"Time of day: {self.time_of_day}")
        if return_fig:
            return fig

        plt.show()

    def visualize_3d(self, return_fig=False):
        data = []
        edges = []
        edges_with_weights = []
        nodes_without_text = []
        self._generate_dc_positions()
        self._generate_client_positions()

        # Add nodes
        for node, pos in self.positions.items():
            if 'c_' in node[:2]:
                color = 'yellow'
                text = f"{node} (Requests: {self.graph.nodes[node]['requests']})"
            elif node in self.active_replicas:
                color = 'green'
                text = f"{node} (Active)"
            elif node in self.passive_replicas:
                color = 'blue'
                text = f"{node} (Passive)"
            else:
                color = 'gray'
                text = f"{node} (Inactive)"
            node_trace = go.Scattergeo(
                lon=[pos[0]],
                lat=[pos[1]],
                mode='markers+text',
                marker=dict(color=color, size=15),
                name=node
            )
            data.append(node_trace)

            node_without_text_trace = go.Scattergeo(
                lon=[pos[0]],
                lat=[pos[1]],
                mode='markers',
                marker=dict(color=color, size=15),
                name=node
            )
            nodes_without_text.append(node_without_text_trace)

        # Add edges
        for u, v, d in self.graph.edges(data=True):
            edge_trace = go.Scattergeo(
                lon=[self.positions[u][0], self.positions[v][0]],
                lat=[self.positions[u][1], self.positions[v][1]],
                mode='lines',
                line=dict(width=1, color='gray'),
                opacity=0.5,
                showlegend=False
            )
            edges.append(edge_trace)

            edge_with_weight_trace = go.Scattergeo(
                lon=[self.positions[u][0], self.positions[v][0]],
                lat=[self.positions[u][1], self.positions[v][1]],
                mode='lines+text',
                text=[f"{d['latency']}", ''],
                textposition='middle right',
                line=dict(width=1, color='gray'),
                opacity=0.5,
                showlegend=False
            )
            edges_with_weights.append(edge_with_weight_trace)

        fig = go.Figure(data=data)

        # Create dropdown menus
        edge_buttons = [
            dict(label="Show Edges",
                 method="update",
                 args=[{"visible": [True] * len(data) + [True] * len(edges) + [False] * len(edges_with_weights)},
                       {"title": "Network Visualization with Edges"}]),
            dict(label="Hide Edges",
                 method="update",
                 args=[{"visible": [True] * len(data) + [False] * len(edges) + [False] * len(edges_with_weights)},
                       {"title": "Network Visualization without Edges"}])
        ]

        client_buttons = [
            dict(label="Show Clients",
                 method="update",
                 args=[{"visible": [True] * len(data) + [True] * len(edges) + [False] * len(edges_with_weights)},
                       {"title": f"Network Visualization with Clients at Time of Day: {self.time_of_day}:00"}]),
            dict(label="Hide Clients",
                 method="update",
                 args=[{"visible": [node['marker']['color'] != 'yellow' for node in data] + [False] * len(edges) + [
                     False] * len(edges_with_weights)},
                       {"title": "Network Visualization without Clients"}])
        ]

        text_buttons = [
            dict(label="Show Node Text",
                 method="update",
                 args=[{"visible": [True] * len(data) + [True] * len(edges) + [False] * len(edges_with_weights)},
                       {"title": "Network Visualization with Node Text"}]),
            dict(label="Hide Node Text",
                 method="update",
                 args=[{"visible": [False] * len(data) + [False] * len(edges) + [False] * len(edges_with_weights) + [
                     True] * len(nodes_without_text)},
                       {"title": "Network Visualization without Node Text"}])
        ]

        fig.update_layout(
            title=f"Network Visualization at Time of Day: {self.time_of_day}",
            showlegend=False,
            geo=dict(
                projection=dict(type='orthographic'),
                showland=True,
                landcolor='rgb(243, 243, 243)',
                countrycolor='rgb(204, 204, 204)',
            ),
            font=dict(
                family="Courier New, monospace",
                size=12,  # Set the font size here
                color="Black"
            ),
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=edge_buttons,
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.7,
                    xanchor="left",
                    y=1.15,
                    yanchor="top"
                ),
                dict(
                    type="buttons",
                    direction="left",
                    buttons=client_buttons,
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.7,
                    xanchor="left",
                    y=1.05,
                    yanchor="top"
                ),
                dict(
                    type="buttons",
                    direction="left",
                    buttons=text_buttons,
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.7,
                    xanchor="left",
                    y=0.95,
                    yanchor="top"
                )
            ]
        )

        # Add edge traces to the figure
        for trace in edges:
            fig.add_trace(trace)
        for trace in edges_with_weights:
            fig.add_trace(trace)
        for trace in nodes_without_text:
            fig.add_trace(trace)

        if return_fig:
            return fig

        fig.show()

    def visualize_2d_world(self, return_fig=False):
        self._generate_client_positions()
        fig, ax = plt.subplots(figsize=(16, 8), subplot_kw={'projection': ccrs.PlateCarree()})

        # Set extent to zoom in on the world map
        ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())

        # Add map features
        ax.add_feature(cfeature.LAND, edgecolor='black')
        ax.add_feature(cfeature.BORDERS, linestyle=':')

        # Initialize legend labels
        legend_labels = set()

        # Plot data centers
        for dc, pos in self.positions.items():
            label = None
            if 'c_' in dc[:2]:
                color = 'yellow'
                label = 'Client'
            elif dc in self.active_replicas:
                color = 'green'
                label = 'Active DC'
            elif dc in self.passive_replicas:
                color = 'blue'
                label = 'Passive DC'
            else:
                color = 'gray'
                label = 'Inactive DC'

            if label not in legend_labels:
                ax.scatter(pos[0], pos[1], color=color, s=150, label=label,
                           transform=ccrs.PlateCarree())  # Increased size
                legend_labels.add(label)
            else:
                ax.scatter(pos[0], pos[1], color=color, s=150, transform=ccrs.PlateCarree())  # Increased size

            #ax.text(pos[0], pos[1], dc, fontsize=12, ha='right', transform=ccrs.PlateCarree())  # Larger font size

        # Plot edges
        for u, v, data in self.graph.edges(data=True):
            pos_u = self.positions[u]
            pos_v = self.positions[v]
            ax.plot([pos_u[0], pos_v[0]], [pos_u[1], pos_v[1]], color='gray', linewidth=0.5, alpha=0.7,
                    transform=ccrs.PlateCarree())  # More transparent

        prefix = "0" if self.time_of_day < 10 else ""
        plt.title(f"Network Visualization at Time of Day: {prefix}{self.time_of_day}:00"
                  f" with average latency of {self.current_latency:.2f} and \n"
                  f"active reconfiguration costs of {self._active_penalty_state:.2f} \n"
                  f"and penalties of {self.active_penalty_tracker}", fontsize=14)  # Larger title font
        plt.legend(fontsize=12)  # Larger legend font

        if return_fig:
            return fig

        plt.show()


def main():
    # Example usage
    num_centers = 15
    clusters = [5, 5, 5]  # Europe, Asia, USA
    num_clients = 30

    env = NetworkEnvironment(num_centers=num_centers, clusters=clusters, num_clients=num_clients)
    intial_dcs = list(env.active_replicas)
    print(f"Initial active data centers: {intial_dcs}")
    initial_passive_dcs = list(env.passive_replicas)
    print(f"Initial passive data centers: {initial_passive_dcs}")

    # Simulate for 24 hours
    for i in range(24):
        print(f"Time of day: {i}")
        print(env.client_regions)
        print(env.active_replicas)
        print(env.passive_replicas)
        print(env.current_request_distribution)
        print("............................................................................")
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

            env.visualize_2d_world()


if __name__ == "__main__":
    main()
