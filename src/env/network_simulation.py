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


class Regions(Enum):
    ASIA = "asia"
    EUROPE = "europe"
    USA = "usa"


class NetworkEnvironment:
    def __init__(self,
                 num_centers: int, clusters: List[int],
                 num_clients: int, penalty: Optional[Penalty] = Penalty,
                 penalty_weights: Optional[PenaltyWeights] = PenaltyWeights,
                 cluster_region_start=Regions.ASIA,
                 action_type: ActionType = ActionType.BOTH,
                 total_requests_per_interval=10000, k=3, p=1, client_start_region: Union[Regions, dict] = Regions.ASIA):
        self.graph = nx.Graph()
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

        self.penalty_state = 0
        self.positions = {}
        self._generate_dc_positions()
        self._generate_client_positions()  # New attribute for positions

        self.available_active_replicas = set(self.data_centers).difference(set(self.active_replicas))
        self.available_passive_replicas = set(self.data_centers).difference(set(self.passive_replicas))

        self.current_request_distribution = None
        self.current_latency = 0

    def _initialize_data_centers(self):
        locations = ['europe', 'asia', 'usa']
        for loc, count in zip(locations, self.clusters):
            if loc == self.cluster_region_start.value:
                active_inds = np.random.choice(count, self.k, replace=False)
                passive_inds = np.random.choice([i for i in range(count) if i not in active_inds], self.p, replace=False)
            for i in range(count):
                dc_name = f'{loc}_dc_{i + 1}'
                self.data_centers.append(dc_name)
                self.graph.add_node(dc_name, type=self.type_mapping['inactive'])
                if loc == self.cluster_region_start.value and i in active_inds:
                    self.active_replicas.add(dc_name)
                    self.graph.nodes[dc_name]['type'] = self.type_mapping['active']
                elif loc == self.cluster_region_start.value and i in passive_inds:
                    self.passive_replicas.add(dc_name)
                    self.graph.nodes[dc_name]['type'] = self.type_mapping['passive']

                self.graph.nodes[dc_name]['requests'] = 0

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

    def _initialize_clients(self):
        for c in self.clients:
            self.graph.add_node(c, type=self.type_mapping['client'])
            self.graph.nodes[c]['requests'] = 0

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
            if 'europe' in dc:
                positions[dc] = (random.uniform(-10, 10), random.uniform(40, 60))  # Europe coordinates
            elif 'asia' in dc:
                positions[dc] = (random.uniform(60, 140), random.uniform(10, 50))  # Asia coordinates
            elif 'usa' in dc:
                positions[dc] = (random.uniform(-130, -60), random.uniform(20, 50))  # USA coordinates
        self.positions = positions

    def _generate_client_positions(self):
        for client, region in self.client_regions.items():
            if region == 'europe':
                self.positions[client] = (random.uniform(-10, 10), random.uniform(40, 60))  # Close to Europe data centers
            elif region == 'asia':
                self.positions[client] = (random.uniform(60, 140), random.uniform(10, 50))  # Close to Asia data centers
            elif region == 'usa':
                self.positions[client] = (random.uniform(-130, -60), random.uniform(20, 50))  # Close to USA data centers
            else:
                self.positions[client] = (random.uniform(-180, 180), random.uniform(-90, 90))  # Random global coordinates

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
            valid = self.review_passive_action(action)
            if valid:
                self.passive_replicas.remove(action[0])
                self.graph.nodes[action[0]]['type'] = 'inactive'
                self.passive_replicas.add(action[1])
                self.graph.nodes[action[1]]['type'] = 'passive'
                self._review_reconfiguration(action)
                self.update_available_replicas()
            self.reconfigure_passive_nodes()
        elif self.action_type == ActionType.ACTIVE:
            if action[0] not in self.active_replicas:
                raise ValueError("The suggested location to be replaced is currently not in the active data centers")
            valid = self._review_active_action(action)
            if valid:
                self.active_replicas.remove(action[0])
                self.graph.nodes[action[0]]['type'] = self.type_mapping['inactive']
                self.active_replicas.add(action[1])
                self.graph.nodes[action[1]]['type'] = self.type_mapping['active']
                self._review_reconfiguration(action)
                self.update_available_replicas()
            self.reconfigure_active_nodes()
        elif self.action_type == ActionType.BOTH:
            if len(action) != 2:
                raise ValueError("Invalid action format. It must be a tuple of two tuples.")
            valid = self._review_active_action(action[0])
            if valid:
                self.active_replicas.remove(action[0][0])
                self.graph.nodes[action[0][0]]['type'] = self.type_mapping['inactive']
                self.active_replicas.add(action[0][1])
                self.graph.nodes[action[0][1]]['type'] = self.type_mapping['active']
                self._review_active_action(action[0])
            valid = self.review_passive_action(action[1])
            if valid:
                self.passive_replicas.remove(action[1][0])
                self.graph.nodes[action[1][0]]['type'] = self.type_mapping['inactive']
                self.passive_replicas.add(action[1][1])
                self.graph.nodes[action[1][1]]['type'] = self.type_mapping['passive']
                self._review_reconfiguration(action[1])
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
            self.penalty_state += (self.penalty_weights.WrongActivePenalty * self.penalty.WrongActivePenalty)
            valid = False

        return valid

    def _review_active_action(self, action: Tuple[str, str]) -> bool:
        """
        Review the active action to check whether a previous non-passive node was made active. If so, the internal
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
        Review the reconfiguration to check whether new nodes were selected. If so, the internal penalty state is
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

        self.current_latency =  total_latency / count if count != 0 else 0

    def _distribute_requests(self):
        requests = np.random.dirichlet(self.client_weights, 1)[0] * self.total_requests_per_interval
        request_distribution = dict(zip(self.clients, requests.astype(int)))
        for client, num_requests in request_distribution.items():
            self.graph.nodes[client]['requests'] = num_requests
        self.current_request_distribution =  request_distribution

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

            prob_europe = (1 / dist_europe) / (1 / dist_europe + 1 / dist_asia + 1 / dist_usa)
            prob_asia = (1 / dist_asia) / (1 / dist_europe + 1 / dist_asia + 1 / dist_usa)
            prob_usa = (1 / dist_usa) / (1 / dist_europe + 1 / dist_asia + 1 / dist_usa)

            region = np.random.choice(['europe', 'asia', 'usa'], p=[prob_europe, prob_asia, prob_usa])
            self._move_client(client, region)

    def _move_client(self, client, region):
        for dc in self.active_replicas:
            if region in dc:
                latency = random.randint(10, 100)
            else:
                latency = random.randint(150, 750)
            try:
                self.graph[client][dc]['latency'] = latency
            except KeyError:
                self.graph.add_edge(client, dc, latency=latency)

            self.client_regions[client] = region

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
        self.penalty_state = 0
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

        reward = self.penalty_weights.LatencyPenalty * (-self.current_latency) - self.penalty_state

        self.time_of_day = (self.time_of_day + 1) % 24
        self.fluctuate_latencies()

        state = self.graph
        done = False

        # add action mask here
        for node in state.nodes:
            if node in self.available_active_replicas:
                state.nodes[node]['active_mask'] = 0
            else:
                state.nodes[node]['active_mask'] = -np.inf

            if node in self.available_passive_replicas:
                state.nodes[node]['passive_mask'] = 0
            else:
                state.nodes[node]['passive_mask'] = -np.inf

        return state, reward, done

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
            elif 'c_' in node[:2]:
                pos[node] = (random.uniform(0, 3), random.uniform(0, 3))
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
                text=text,
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
                       {"title": f"Network Visualization with Clients at Time of Day: {self.time_of_day}"}]),
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

        plt.title(f"Network Visualization at Time of Day: {self.time_of_day} "
                  f"with average latency of {self.current_latency:.4f} and \n"
                  f"reconfiguration costs of {self.penalty_state:.4f}", fontsize=14)  # Larger title font
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