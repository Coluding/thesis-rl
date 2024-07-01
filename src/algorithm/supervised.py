import torch
from torch_geometric.data import Data, Batch, Dataset
from torch_geometric.utils import from_networkx
from torch.optim import Adam
from tqdm import tqdm
from src.model.gnn import CriticSwapGNN
from src.env.network_simulation import NetworkEnvironment, PenaltyWeights



class TorchGraphWrapper:
    def __init__(self, device = "cuda" if torch.cuda.is_available() else "cpu"):
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

    obs_wrapper = TorchGraphWrapper()

    samples = []
    for i in tqdm(range(1000000)):
        action = env.random_action()
        action = ((env.dc_to_int[action[0][0]], env.dc_to_int[action[0][1]]),
                  (env.dc_to_int[action[1][0]], env.dc_to_int[action[1][1]]),)

        observation, reward, done = env.step(action)
        observation = obs_wrapper((observation, reward))
        samples.append(observation)
    return samples



class SupervisedDataset(Dataset):
    def __init__(self, samples):
        super().__init__()
        self.samples = samples

    def __len__(self):
        return len(self.samples)

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

    def _reshuffle(self):
        self.dataset = self.dataset[torch.randperm(len(self.dataset))]
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


def construct_samples_and_train(traj_length: int = 5000, batch_size:int = 64, num_trajectories: int = 1000):
    # Initialize the model, optimizer, and loss function
    model = CriticSwapGNN()
    optimizer = Adam(model.parameters(), lr=3e-4)
    loss = torch.nn.MSELoss()
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
    env = NetworkEnvironment(num_centers=3, k=1, clusters=[2, 2, 2], num_clients=20, penalty_weights=penalty_weights,
                             period_length=1000000)

    obs_wrapper = TorchGraphWrapper()

    for _ in range(num_trajectories):
        trajectory = []
        for i in tqdm(range(traj_length)):
            action = env.random_action()
            action = ((env.dc_to_int[action[0][0]], env.dc_to_int[action[0][1]]),
                      (env.dc_to_int[action[1][0]], env.dc_to_int[action[1][1]]),)

            observation, reward, done = env.step(action)
            observation = obs_wrapper((observation, reward))
            trajectory.append(observation)

        dataset = SupervisedDataset(trajectory)
        train(model, loss, optimizer, dataset, num_epochs=100, batch_size=batch_size)

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
    data_loader = DataLoader(32, dataset)
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_loss = float("inf")
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


if __name__ == "__main__":
    construct_samples_and_train()