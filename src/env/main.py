import networkx as nx
import matplotlib.pyplot as plt
import torch
from gym.wrappers.monitoring.video_recorder import VideoRecorder


from src.env.java import JavaClassManager, jvm_context, SimulationWrapper
from src.env.env import IntervalResult, RSMEnvConfig, RSMEnv, TorchGraphObservationWrapper
from src.model.gnn import CriticGCNN, ActorGCNN, SwapGNN



def main():
    critic = CriticGCNN(4, 48, 24, 128, num_nodes=15)
    actor = ActorGCNN(4, 4, 24, 12, 128, num_nodes=15)
    swap = SwapGNN(4, 4, 64, 128, num_nodes=15)
    with jvm_context(classpath=['../../java/untitled/out/production/simulation']):
        dummy = SimulationWrapper("MockSimulation", 15, 4, 1,2)
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


if __name__ == '__main__':
    main()