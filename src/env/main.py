import networkx as nx
import matplotlib.pyplot as plt
import torch
from gym.wrappers.monitoring.video_recorder import VideoRecorder
#from src.env.java import JavaClassManager, jvm_context, SimulationWrapper
from src.env.env import (IntervalResult,
                         RSMEnvConfig,
                         RSMEnv,
                         TorchGraphObservationWrapper,
                         TorchGraphNetworkxWrapper,
                         NetworkEnvGym,
                         CustomNetworkConfig)
from src.model.gnn import CriticGCNN, ActorGCNN, SwapGNN

def use_java():
    critic = CriticGCNN(4, 48, 24, 128, num_nodes=15)
    actor = ActorGCNN(4, 4, 24, 12, 128, num_nodes=15)
    swap = SwapGNN(4, 4, 64, 128, num_nodes=15)
    with jvm_context(classpath=['../../java/untitled/out/production/simulation']):
        dummy = SimulationWrapper("MockSimulation", 15, 4, 1, 2)
        config = RSMEnvConfig(device="cuda", dtype=torch.float16, num_nodes=15, n_actions=4, feature_dim_node=1,
                              use_prev_latencies=True, render_mode="human", fps=2)
        env = RSMEnv(config, dummy)
        env = TorchGraphObservationWrapper(env, one_hot=True)
        video = VideoRecorder(env, f"test.mp4", )
        env.render()
        observation, reward, done, trunc, info = env.step(1)
        swap(observation)
        for i in range(10):
            try:
                env.step(i)
                env.render()
                video.capture_frame()
            except:
                continue

        video.close()

def use_custom():
    import time
    swap_active = SwapGNN(4, 4, 64, 128, num_nodes=15, for_active=True)
    swap_passive = SwapGNN(4, 4, 64, 128, num_nodes=15, for_active=False)
    config = CustomNetworkConfig(num_centers=3, clusters=[5, 5, 5], num_clients=20, render_mode="2d")
    env = NetworkEnvGym(config)
    env = TorchGraphNetworkxWrapper(env, one_hot=False)
    state = env.reset()
    done = False
    while not done:
        action = env.sample_action()  # Sample a random action
        state, reward, done, _, _ = env.step(action)
        swap_active(state)
        time.sleep(2)  # Add delay to simulate real-time steps
    # wrapped_env.run()
    env.close()

def main():
    JAVA = False
    if JAVA:
        use_java()
    else:
        use_custom()


if __name__ == '__main__':
    main()