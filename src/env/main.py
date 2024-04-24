import networkx as nx
import matplotlib.pyplot as plt
import torch
from gym.wrappers.monitoring.video_recorder import VideoRecorder


from src.env.java import JavaClassManager, jvm_context, SimulationWrapper
from src.env.env import IntervalResult, RSMEnvConfig, RSMEnv, TorchGraphObservationWrapper
from src.model.gnn import CriticGCNN, ActorGCNN


def main():
    critic = CriticGCNN(3, 48, 24, 128, num_nodes=10)
    actor = ActorGCNN(4, 1, 24, 12, 128, num_nodes=10)
    with jvm_context(classpath=['../../java/untitled/out/production/simulation']):
        dummy = SimulationWrapper("MockSimulation", 10, 4, 1)
        config = RSMEnvConfig(device="cuda", dtype=torch.float32, num_nodes=10, n_actions=4, feature_dim_node=1,
                              use_prev_latencies=False, render_mode="rgb_array", fps=3)
        env = RSMEnv(config, dummy)
        env = TorchGraphObservationWrapper(env, one_hot=True)
        video = VideoRecorder(env, f"test_overwrite_latencies.mp4", )
        env.render("rgb_array")
        observation, reward, done, trunc, info = env.step(1)
        #critic(observation)
        for i in range(10):
            env.step(i)
            env.render("rgb_array")
            video.capture_frame()

        video.close()






if __name__ == '__main__':
    main()