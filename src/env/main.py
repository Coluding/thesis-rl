import networkx as nx
import matplotlib.pyplot as plt
import torch

from src.env.java import JavaClassManager, jvm_context, SimulationWrapper
from src.env.env import IntervalResult, RSMEnvConfig, RSMEnv, TorchGraphObservationWrapper


def main():

    with jvm_context(classpath=['../../java/untitled/out/production/simulation']):
        dummy = SimulationWrapper("MockSimulation", 7)
        nested_map = dummy.runInterval()
        config = RSMEnvConfig(device="cuda", dtype=torch.float32, num_nodes=10, n_actions=4, feature_dim_node=1)
        env = RSMEnv(config, dummy)
        env = TorchGraphObservationWrapper(env)
        ret = env.step(1)
        env.render("human")
        env.step(2)
        env.render("human")





if __name__ == '__main__':
    main()