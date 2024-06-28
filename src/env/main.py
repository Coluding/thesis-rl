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
                         CustomNetworkConfig,
                         StackStatesTemporal)

from src.algorithm.agents import SwapPPOAgentConfigActionTypeSingle, SwapPPOAgentConfigActionTypeBoth, PPOAgentActionTypeBoth
from src.model.gnn import CriticGCNN, ActorGCNN, SwapGNN, CriticSwapGNN

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
    device = "cpu"
    config = CustomNetworkConfig(num_centers=3, clusters=clusters, num_clients=20, render_mode="human",
                                 render_type="2d", device=device)
    swap_active = SwapGNN(4, 4, 64, 128, num_nodes=15,
                          for_active=True, num_locations=sum(clusters), device=device)
    swap_passive = SwapGNN(4, 4, 64, 128, num_nodes=15,
                           for_active=False, num_locations=sum(clusters), device=device)

    critic_active = CriticSwapGNN(4, 4, 64, 128, num_nodes=15,
                                  for_active=True, num_locations=sum(clusters), device=device)

    critic_passive = CriticSwapGNN(4, 4, 64, 128, num_nodes=15,
                                   for_active=False, num_locations=sum(clusters), device=device)

    ppo_config = SwapPPOAgentConfigActionTypeBoth(swap_active, critic_active, swap_passive, critic_passive,
                                                  batch_size=8, chkpt_dir="../algorithm/logs/models/tmp/ppo")
    agent = PPOAgentActionTypeBoth(ppo_config)
    agent.load_models()
    env = NetworkEnvGym(config)
    env = TorchGraphNetworkxWrapper(env, one_hot=False)
    env_extended = StackStatesTemporal(env, 4)
    video = VideoRecorder(env, f"test.mp4", )
    state, _ = env.reset()
    action = (env.env.env.no_op_active(), env.env.env.no_op_passive())
    state, reward, done, _, _ = env.step(action)
    done = False
    i = 0
    europes = {"europe_dc_1", "europe_dc_2", "europe_dc_3"}
    while not done:
        if env.env.env.time_of_day == 10 or env.env.env.time_of_day == 11 or env.env.env.time_of_day == 12:
            remove_dc = np.random.choice(list(env.env.env.active_replicas))
            diff = europes.difference(env.env.env.active_replicas)
            add_dc = np.random.choice(list(diff), 1)[0]
            remove_passive = np.random.choice(list(env.env.env.passive_replicas))
            add_dc_pa = "asia_dc_1"
            action_active = (env.env.env.dc_to_int[remove_dc], env.env.env.dc_to_int[add_dc])
            action_passive = (env.env.env.dc_to_int[remove_passive], env.env.env.dc_to_int[add_dc_pa])
            action = (action_active, action_passive)
            baseline_active = critic_active(state)
            baseline_passive = critic_passive(state)
            log_prob_active = torch.tensor([-0.1, -0.1])
            log_prob_passive = torch.tensor([-0.1, -0.1])
        else:
            active_out = swap_active(state)
            action_active = active_out.actions
            log_prob_active = active_out.log_probs
            passive_out = swap_passive(state)
            action_passive = passive_out.actions
            log_prob_passive = passive_out.log_probs
            baseline_active = critic_active(state)
            baseline_passive = critic_passive(state)
            action = ((action_active[0].item(), action_active[1].item()),
                      (action_passive[0].item(), action_passive[1].item()))
        print(f"Time: {env.env.env.time_of_day}")
        print(f"Active action: {action_active[1]}")
        print(f"Passive action: {action_passive[1]}")
        print(env.env.env.active_replicas)
        print(env.env.env.passive_replicas)
        print(".....................................")
        # Sample a random action
        state, reward, done, _, _ = env.step(action)
        #state, reward, done, _, _ = env_extended.step(action)
        agent.add_experience(state, torch.stack([baseline_active, baseline_passive]), action, reward, done,
                             torch.stack([log_prob_active, log_prob_passive]))
        env.render()
        video.capture_frame()
        print(f"Reward: {reward}")
        if i == 10:
            agent.learn()
        time.sleep(4)
        if i == 25:
            break
        i += 1
    #video.close()
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