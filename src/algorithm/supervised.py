from typing import Union, Tuple
import torch.nn as nn
import torch
from itertools import chain
from torch import Tensor
from torch_geometric.data import Data, Batch, Dataset
from torch_geometric.utils import from_networkx
from torch.optim import Adam
from tqdm import tqdm
from src.model.gnn import CriticSwapGNN, TransformerGNN
from src.env.network_simulation import NetworkEnvironment, PenaltyWeights

from cpprb import ReplayBuffer
replay = ReplayBuffer(buffer_size=1000000)

#TODO: only show edges of active nodes

class TorchGraphWrapper:
    def __init__(self,
                 device = "cuda" if torch.cuda.is_available() else "cpu",
            ):
        self.device = device
    def __call__(self, sample):
        observation, reward = sample
        torch_graph = from_networkx(observation)

        if "active_mask" in torch_graph:
            torch_graph.passive_mask = torch_graph.passive_mask.to(self.device)
            torch_graph.active_mask = torch_graph.active_mask.to(self.device)

        torch_graph.type = torch_graph.type.to(torch.float32).to(self.device)
        torch_graph.requests = torch_graph.requests.to(torch.float32).to(self.device)
        torch_graph.edge_index = torch_graph.edge_index.to(self.device)
        torch_graph.update_step = torch_graph.update_step.to(self.device)
        torch_graph.latency = torch_graph.latency.to(self.device)
        torch_graph.target = torch.tensor(reward[0]).to(self.device)

        return torch_graph

def construct_supervised_samples():
    penalty_weights = PenaltyWeights(
        LatencyPenalty=1,
        ReconfigurationPenalty=0.001,
        PassiveCost=0.001,
        ActiveCost=0.001,
        WrongActivePenalty=0.001,
        WrongPassivePenalty=0.001,
        WrongActiveAfterPassivePenalty=0.001,
        WrongNumberPassivePenalty=0.001,
        ActiveNodeAlsoPassivePenalty=0.001,
        ReExplorationPassiveWeight=1,
        EdgeDifferenceWeight=1,
        NewEdgesDiscoveredReward=1,
    )
    env = NetworkEnvironment(num_centers=3, k=1, clusters=[2,2,2], num_clients=20, penalty_weights=penalty_weights,
                             period_length=1000000)

    obs_wrapper = TorchGraphWrapper(device="cpu")

    samples = []
    for i in tqdm(range(1000000)):
        action = env.random_action()
        action = ((env.dc_to_int[action[0][0]], env.dc_to_int[action[0][1]]),
                  (env.dc_to_int[action[1][0]], env.dc_to_int[action[1][1]]),)

        observation, reward, done = env.step(action)
        observation = obs_wrapper((observation, reward))
        replay.add(observation)
        samples.append(observation)
    return samples



class SupervisedDataset(Dataset):
    def __init__(self, samples):
        super().__init__()
        self.samples = samples

    def normalize(self, mean, std, feature="latency"):
        for sample in self.samples:
            sample[feature] = (sample[feature] - mean) / std

    def __len__(self):
        return len(self.samples)

    def shuffle(
        self,
        return_perm: bool = False,
    ) -> Union['Dataset', Tuple['Dataset', Tensor]]:
        perm = torch.randperm(len(self))
        if return_perm:
            return self[perm], perm
        return self[perm]

    def shuffle_inplace(self):
        perm = torch.randperm(len(self))
        self.samples = [self.samples[i] for i in perm]


    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return Batch.from_data_list(self.samples[idx])
        if isinstance(idx, torch.Tensor):
            return Batch.from_data_list([self.samples[i] for i in idx])
        return self.samples[idx]


class DataLoader:
    def __init__(self, batch_size, dataset, shuffle=True):
        self.batch_size = batch_size
        self.dataset = dataset
        self.current_index = 0

        if shuffle:
            self._reshuffle()
    def __iter__(self):
        return self


    def __len__(self):
        return len(self.dataset) // self.batch_size

    def _reshuffle(self):
        self.dataset.shuffle_inplace()
        self.current_index = 0

    def __next__(self):
        if self.current_index >= len(self.dataset):
            self._reshuffle()
            raise StopIteration
        batch = self.dataset[self.current_index:self.current_index+self.batch_size]
        if isinstance(batch, list):
            batch = Batch.from_data_list(batch)

        self.current_index += self.batch_size
        return batch


def construct_samples_and_train(traj_length: int = 50000,
                                batch_size:int = 64,
                                num_trajectories: int = 1000,
                                num_epochs: int = 20,
                                learning_rate: float = 3e-4):
    # Initialize the model, optimizer, and loss function
    device = "cpu"
    model = CriticSwapGNN(device=device,
                          feature_dim_node=16,
                          hidden_channels=64,
                          num_heads=2,
                          activation=nn.LeakyReLU,
                          num_gat_layers=4,
                          num_mlp_layers=4,
                          fc_hidden_dim=64)

    model = TransformerGNN(n_layers=4,
                           feature_size=2,
                           n_heads=3,
                           embedding_size=64,
                           dropout_rate=0.2,
                           top_k_ratio=0.5,
                           dense_neurons=256,
                           device=device
                           )

    #model.load_state_dict(torch.load("critic_gcnn.pth"))

    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss(reduction="mean")
    penalty_weights = PenaltyWeights(
        LatencyPenalty=1,
        ReconfigurationPenalty=0.001,
        PassiveCost=0.001,
        ActiveCost=0.001,
        WrongActivePenalty=0.001,
        WrongPassivePenalty=0.001,
        WrongActiveAfterPassivePenalty=0.001,
        WrongNumberPassivePenalty=0.001,
        ActiveNodeAlsoPassivePenalty=0.001,
        ReExplorationPassiveWeight=1,
        EdgeDifferenceWeight=1,
        NewEdgesDiscoveredReward=1,
    )
    env = NetworkEnvironment(num_centers=3, k=3, clusters=[5,5, 5],
                             num_clients=50,
                             penalty_weights=penalty_weights,
                             period_length=1000000)

    obs_wrapper = TorchGraphWrapper()
    best_loss = torch.inf

    for n in range(num_trajectories):
        trajectory = []
        for i in tqdm(range(traj_length)):
            action = env.random_action()
            action = ((env.dc_to_int[action[0][0]], env.dc_to_int[action[0][1]]),
                      (env.dc_to_int[action[1][0]], env.dc_to_int[action[1][1]]),)

            observation, reward, done = env.step(action)
            reward = (-reward[0], reward[1])
            observation = obs_wrapper((observation, reward))
            replay.add(observation)
            trajectory.append(observation)

        targets = torch.stack([x.target for x in trajectory])
        latencies = chain.from_iterable([x.latency.tolist() for x in trajectory])
        latencies = torch.tensor(list(latencies))
        dataset = SupervisedDataset(trajectory)
        dataset.normalize(latencies.mean(), latencies.std())
        dataset.normalize(targets.mean(), targets.std(), feature="target")
        data_loader = DataLoader(batch_size, dataset)
        # Set the device
        device = "cpu"
        # Training loop
        num_epochs = num_epochs
        for epoch in range(num_epochs):
            loss = train_gnn(model, data_loader, optimizer, criterion, device)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")

            if loss < best_loss:
                torch.save(model.state_dict(), "critic_gcnn.pth")
                best_loss = loss
                print(f"Model saved with loss: {loss:.4f}")

def load_dataset():
    return torch.load("env_dataset_samples.pt")

def train_gnn(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    iterator = tqdm(data_loader)
    for data in iterator:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data, data.batch)
        loss = criterion(output.squeeze(), data.target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)


def train(model, criterion, optimizer, dataset, num_epochs: int = 10, batch_size:int = 32) -> None:
    data_loader = DataLoader(batch_size, dataset)
    # Set the device
    device = "cpu"
    # Training loop
    num_epochs = num_epochs
    for epoch in range(num_epochs):
        loss = train_gnn(model, data_loader, optimizer, criterion, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")

        if loss < best_loss:
            torch.save(model.state_dict(), "critic_gcnn.pth")
            best_loss = loss
            print(f"Model saved with loss: {loss:.4f}")

def save_data():
    samples = construct_supervised_samples()
    dataset = SupervisedDataset(samples)
    # dataloader = DataLoader(8, dataset)
    # save samples
    torch.save(dataset, "env_dataset_samples.pt")

#TODO: include more extreme latencies, the current distribution is peaking strongly at values around 0-300
if __name__ == "__main__":
    construct_samples_and_train(traj_length=5000,
                                batch_size=128,
                                num_trajectories=100000,
                                num_epochs=20,
                                learning_rate=0.00001)