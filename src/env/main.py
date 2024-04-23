import networkx as nx
import matplotlib.pyplot as plt
import torch

from src.env.java import JavaClassManager, jvm_context, SimulationWrapper
from src.env.env import IntervalResult, RSMEnvConfig, RSMEnv, TorchGraphObservationWrapper
from src.model.gnn import CriticGNN


def main():
    critic = CriticGNN(3, 48, 24, 128, num_nodes=10)
    with jvm_context(classpath=['../../java/untitled/out/production/simulation']):
        dummy = SimulationWrapper("MockSimulation", 10, 4, 1)
        config = RSMEnvConfig(device="cuda", dtype=torch.float32, num_nodes=10, n_actions=4, feature_dim_node=1)
        env = RSMEnv(config, dummy)
        env = TorchGraphObservationWrapper(env, one_hot=True)
        env.render("human")
        observation, reward, done, trunc, info = env.step(1)
        critic(observation)
        env.render("human")
        env.step(2)
        env.render("human")





if __name__ == '__main__':
    main()