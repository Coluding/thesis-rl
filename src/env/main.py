import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
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
    clusters = [5, 5, 5]
    config = CustomNetworkConfig(num_centers=3, clusters=clusters, num_clients=20, render_mode="human",
                                 render_type="2d")
    swap_active = SwapGNN(4, 4, 64, 128, num_nodes=15,
                          for_active=True, num_locations=sum(clusters))
    swap_passive = SwapGNN(4, 4, 64, 128, num_nodes=15,
                           for_active=False, num_locations=sum(clusters))

    env = NetworkEnvGym(config)
    env = TorchGraphNetworkxWrapper(env, one_hot=False)
    video = VideoRecorder(env, f"test.mp4", )
    state = env.reset()
    state, reward, done, _, _ = env.step(None)
    done = False
    i = 0
    europes = {"europe_dc_1", "europe_dc_2", "europe_dc_3"}
    while not done:
        if env.env.env.time_of_day == 11:
            remove_dc = np.random.choice(list(env.env.env.active_replicas))
            diff = europes.difference(env.env.env.active_replicas)
            add_dc = np.random.choice(list(diff), 1)[0]
            remove_passive = np.random.choice(list(env.env.env.passive_replicas))
            add_dc_pa = "asia_dc_1"
            action = ((remove_dc, add_dc), (remove_passive, add_dc_pa))
        elif env.env.env.time_of_day == 12:
            remove_dc = np.random.choice(list(env.env.env.active_replicas))
            diff = europes.difference(env.env.env.active_replicas)
            add_dc = np.random.choice(list(diff), 1)[0]
            remove_passive = np.random.choice(list(env.env.env.passive_replicas))
            add_dc_pa = "asia_dc_1"
            action = ((remove_dc, add_dc), (remove_passive, add_dc_pa))
        elif env.env.env.time_of_day == 10:
            remove_dc = np.random.choice(list(env.env.env.active_replicas))
            diff = europes.difference(env.env.env.active_replicas)
            add_dc = np.random.choice(list(diff), 1)[0]
            remove_passive = np.random.choice(list(env.env.env.passive_replicas))
            add_dc_pa = "asia_dc_1"
            action = ((remove_dc, add_dc), (remove_passive, add_dc_pa))
        else:
            action_active = swap_active(state)
            action_passive = swap_passive(state)
            action = ((action_active[1][0].item(), action_active[1][1].item()),
                      (action_passive[1][0].item(), action_passive[1][1].item()))
        print(f"Time: {i}")
        print(f"Active action: {action_active[1]}")
        print(f"Passive action: {action_passive[1]}")
        print(env.env.env.active_replicas)
        print(env.env.env.passive_replicas)
        print(".....................................")
        # Sample a random action
        state, reward, done, _, _ = env.step(action)
        #env.render()
        video.capture_frame()
        print(f"Reward: {reward}")

        time.sleep(1)
        if i == 25:
            break
        i += 1
    video.close()
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