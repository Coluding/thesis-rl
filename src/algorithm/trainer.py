import os

import gym
import torch
from enum import Enum
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter
from typing import Optional
import logging
from tqdm import tqdm

from src.model.gnn import CriticGCNN, ActorGCNN, SwapGNN, CriticSwapGNN, TransformerSwapGNN, TransformerGNN, QNetworkSwapGNN
from src.algorithm.agents import SwapPPOAgentConfigActionTypeSingle, SwapPPOAgentConfigActionTypeBoth, PPOAgentActionTypeBoth, AbstractAgent, DQGNAgent, DQNConfig
from src.env.env import RSMEnvConfig, RSMEnv, TorchGraphObservationWrapper, TorchGraphNetworkxWrapper, NetworkEnvGym, CustomNetworkConfig, StackStatesTemporal
from src.env.network_simulation import *

class TrainingAlgorithms(Enum):
    PPO = 1
    DDQN = 2

logging.basicConfig(
    filename='training.log',  # Specify the file to log to
    filemode='a',        # Mode 'a' for append, 'w' for overwrite
    level=logging.DEBUG,  # Set the minimum logging level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # Define the log message format
)

logging = logging.getLogger(__name__)


@dataclass
class TrainingParams:
    train_steps: int = 1000
    num_train_runs: int = 1000
    batch_size: int = 8
    gae_lambda: Optional[float] = 0.95
    clip_eps: Optional[float] = 0.2
    eps: Optional[float] = 0.2
    eps_min: Optional[float] = 0.01
    eps_decay: Optional[float] = 0.99
    gamma: Optional[float] = 0.99
    n_epochs: Optional[int] = 2
    replace_rate: Optional[int] = 10
    save_steps: Optional[int] = 100
    evaluation_period_in_steps: Optional[int] = 5000
    evaluation_episodes: Optional[int] = 10
    evaluation_render: Optional[bool] = False


@dataclass
class RepStateMachineTrainerConfig:
    agent: AbstractAgent
    env: gym.Env
    training_params: TrainingParams
    training_algorithm: TrainingAlgorithms
    day_periods: Optional[int] = 100
    log_dir: Optional[str] = "logs/"
    tensorboard_log_dir: Optional[str] = None


class RepStateMachineTrainerRL:
    def __init__(self, config: RepStateMachineTrainerConfig):
        self.agent = config.agent
        self.env = config.env
        self.training_algorithm = config.training_algorithm
        self.tensorboard_log_dir = config.tensorboard_log_dir
        self.log_dir = config.log_dir
        self.training_params = config.training_params
        self.day_periods = config.day_periods
        logging.info(f"Initialized RepStateMachineTrainerRL with {config}")

        self.summary_writer = self.agent.summary_writer

        self._create_dirs()

        self.global_step = 0

    def _create_dirs(self):
        os.makedirs(self.log_dir + "/models", exist_ok=True)
        os.makedirs(self.log_dir + "/logs", exist_ok=True)
        os.makedirs(self.log_dir + "/videos", exist_ok=True)
        os.makedirs(self.log_dir + "/tensorboard", exist_ok=True)
        os.makedirs(self.log_dir + "/plots", exist_ok=True)
        os.makedirs(self.log_dir + "/models/" + self.agent.chkpt_dir, exist_ok=True)
        self.agent.chkpt_dir = self.log_dir + "/models/" + self.agent.chkpt_dir
        logging.info(f"Created directories for models, logs, videos, tensorboard and plots  in {self.log_dir}")

    def save_models(self):
        self.agent.save_models()
        logging.info("Models saved")

    def load_models(self):
        self.agent.load_models()
        logging.info("Models loaded")

    def train_ppo(self):
        logging.info("Training started")
        period_active_rewards, period_passive_rewards, step_active_rewards, step_passive_rewards, losses = [], [], [], [], []
        best_reward = -float('inf')

        length_training = self.training_params.num_train_runs * self.training_params.train_steps
        train_iterator = tqdm(range(length_training), desc="Training PPO", total=length_training)
        while True:
            state, _ = self.env.reset()
            action = (self.env.inner_env.no_op_active(), self.env.inner_env.no_op_passive())
            state, reward, done, _, _ = self.env.step(action)
            done = False
            total_active_reward = 0
            total_passive_reward = 0
            while not done:
                period_steps = 0
                agent_selection_active, agent_selection_passive = self.agent.choose_action(state)
                action_active = agent_selection_active[0]
                action_passive = agent_selection_passive[0]
                action = ((action_active[0].item(), action_active[1].item()),
                          (action_passive[0].item(), action_passive[1].item()))
                log_probs_active = agent_selection_active[1]
                log_probs_passive = agent_selection_passive[1]
                log_probs = torch.stack([log_probs_active, log_probs_passive])
                value = torch.stack([agent_selection_active[2], agent_selection_passive[2]])
                next_state, reward, done, _, _ = self.env.step(action)
                self.agent.add_experience(state, value, action, reward, done, log_probs)
                state = next_state
                total_active_reward += reward[0]
                total_passive_reward += reward[1]

                if self.global_step % self.training_params.train_steps == 0:
                    if self.agent.get_buffer_size() < self.training_params.batch_size:
                        logging.info(f"Step {self.global_step}: Memory buffer not filled yet")
                    else:
                        logging.info(f"Step {self.global_step}: Training")
                        self.agent.learn()
                        loss = self.agent.get_average_loss()
                        if self.summary_writer is not None:
                            self.summary_writer.add_scalar("active actor loss", loss["actor_active"], self.global_step)
                            self.summary_writer.add_scalar("passive actor loss", loss["actor_passive"], self.global_step)
                            self.summary_writer.add_scalar("active critic loss", loss["critic_active"], self.global_step)
                            self.summary_writer.add_scalar("passive critic loss", loss["critic_passive"], self.global_step)
                        losses.append(loss)
                        if total_active_reward > best_reward:
                            best_reward = total_active_reward
                            self.save_models()
                            logging.info(f"Step {self.global_step}: New best reward: {best_reward:.4f}")

                        #log the losses
                        logging.info(f"Step {self.global_step}: Losses: Active Actor: {loss['actor_active']:.4f},"
                                    f" Passive Actor: {loss['actor_passive']:.4f}, Active Critic: {loss['critic_active']:.4f},"
                                    f" Passive Critic: {loss['critic_passive']:.4f}")

                        logging.info(f"Step {self.global_step}: Rewards: Active: {total_active_reward / self.training_params.train_steps:.4f},"
                                    f" Passive: {total_passive_reward / self.training_params.train_steps:.4f}")
                        self.agent.clear_memory()

                self.global_step += 1
                period_steps += 1
                train_iterator.update(1)
                if self.global_step % self.training_params.evaluation_period_in_steps == 0:
                    self.evaluate()

                step_active_rewards.append(reward[0])
                step_passive_rewards.append(reward[1])

                self.summary_writer.add_scalar("Active Reward/Step", reward[0], self.global_step) if self.summary_writer is not None else None
                self.summary_writer.add_scalar("Passive Reward/Step", reward[1], self.global_step) if self.summary_writer is not None else None

            # TODO: maybe include early stop of env period if penalty is too high

            # rewards will be tracked for every period
            period_active_rewards.append(total_active_reward)
            period_passive_rewards.append(total_passive_reward)
            logging.info(f"Period finished after {period_steps} steps")
            self.summary_writer.add_scalar("Active Total Reward/Period", total_active_reward, self.global_step) if self.summary_writer is not None else None
            self.summary_writer.add_scalar("Passive Total Reward/Period", total_passive_reward, self.global_step) if self.summary_writer is not None else None
            # log rolling average of rewards
            active_roll_avg = sum(step_active_rewards[-1000:]) / 1000
            passive_roll_avg = sum(step_passive_rewards[-1000:]) / 1000
            logging.info(f"Step {self.global_step}: Active Rolling Average Reward: {active_roll_avg:.4f}")
            logging.info(f"Step {self.global_step}: Passive Rolling Average Reward: {passive_roll_avg:.4f}")
            self.summary_writer.add_scalar("Active Rolling Average Reward", active_roll_avg, self.global_step) if self.summary_writer is not None else None
            self.summary_writer.add_scalar("Passive Rolling Average Reward", passive_roll_avg, self.global_step) if self.summary_writer is not None else None
            logging.info(f"Step {self.global_step}: Active Reward: {total_active_reward:.4f}")
            logging.info(f"Step {self.global_step}: Passive Reward: {total_passive_reward:.4f}")

    def train_ddqn(self):
        logging.info("Training started")
        period_active_rewards, step_active_rewards, losses = [], [], []
        best_reward = -float('inf')

        length_training = self.training_params.num_train_runs * self.training_params.train_steps
        train_iterator = tqdm(range(length_training), desc="Training DDQN", total=length_training)
        while True:
            state, _ = self.env.reset()
            action = (self.env.inner_env.no_op_active(), self.env.inner_env.no_op_passive())
            state, reward, done, _, _ = self.env.step(action)
            done = False
            total_active_reward = 0

            while not done:
                period_steps = 0
                num_locs = len(self.env.base_env.dc_to_int)
                action = self.agent.choose_action(state, num_locs=num_locs)
                remove_action = int(np.random.choice(torch.where(state.passive_mask[:num_locs] == -np.inf)[0].cpu()))
                add_action = int(np.random.choice(torch.where(state.passive_mask[:num_locs] != -np.inf)[0].cpu()))
                passive_random_action = (remove_action, add_action)
                action_for_env = (action, passive_random_action)
                next_state, reward, done, _, _ = self.env.step(action_for_env)
                self.agent.add_experience(state, next_state, action, reward[0], done)
                state = next_state
                total_active_reward += reward[0]

                if self.global_step % self.training_params.train_steps == 0:
                    if self.agent.get_buffer_size() < self.agent.memory.batch_size:
                        logging.info(f"Step {self.global_step}: Memory buffer not filled yet")
                    else:
                        logging.info(f"Step {self.global_step}: Training")
                        loss, epsilon = self.agent.learn()

                        if self.summary_writer is not None:
                            self.summary_writer.add_scalar("target  loss", loss, self.global_step)
                        losses.append(loss)

                        if total_active_reward > best_reward:
                            best_reward = total_active_reward
                            self.save_models()
                            logging.info(f"Step {self.global_step}: New best reward: {best_reward:.4f}")


                        logging.info(
                            f"Step {self.global_step}: Rewards: Active: "
                            f"{total_active_reward / self.training_params.train_steps:.4f}"
                        )


                self.global_step += 1
                period_steps += 1
                train_iterator.update(1)
                if self.global_step % self.training_params.evaluation_period_in_steps == 0:
                    self.evaluate()

                step_active_rewards.append(reward[0])

                self.summary_writer.add_scalar("Reward/Step", reward[0],
                                               self.global_step) if self.summary_writer is not None else None

            # TODO: maybe include early stop of env period if penalty is too high

            # rewards will be tracked for every period
            period_active_rewards.append(total_active_reward)
            logging.info(f"Period finished after {period_steps} steps")
            self.summary_writer.add_scalar("Total Reward/Period", total_active_reward,
                                           self.global_step) if self.summary_writer is not None else None

            # log rolling average of rewards
            active_roll_avg = sum(step_active_rewards[-1000:]) / 1000
            logging.info(f"Step {self.global_step}: Rolling Average Reward: {active_roll_avg:.4f}")
            self.summary_writer.add_scalar(" Rolling Average Reward", active_roll_avg,
                                           self.global_step) if self.summary_writer is not None else None
            logging.info(f"Step {self.global_step}: Reward: {total_active_reward:.4f}")

    def train(self):
        if self.training_algorithm == TrainingAlgorithms.PPO:
            self.train_ppo()
        elif self.training_algorithm == TrainingAlgorithms.DDQN:
            self.train_ddqn()
        else:
            raise NotImplementedError(f"Training algorithm {self.training_algorithm} not implemented")

    def evaluate(self):
        print("evaluation not implemented yet")

    def save_model(self):
        pass

    def load_model(self):
        pass

def main():
    device = "cuda"
    clusters = [5, 5, 5]

    penalty_weights = PenaltyWeights(
        LatencyPenalty = 1,
        ReconfigurationPenalty = 0.001,
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
    num_active = 3
    assert num_active <= clusters[0]
    assert num_active <= clusters[1]
    assert num_active <= clusters[2]
    config = CustomNetworkConfig(num_centers=3, clusters=clusters, num_clients=20, render_mode="human",
                                 render_type="2d", device=device, penalty_weights=penalty_weights, num_active=3,
                                 display_all_latencies_from_start=True
                                 )

    # num gat_layers should be max 2 since the longest path is 2 and mor elayers would not add more information ?????
    swap_active = TransformerSwapGNN(n_layers=4,
                                     feature_size=2,
                                     n_heads=3,
                                     embedding_size=64,
                                     dropout_rate=0.2,
                                     top_k_ratio=0.5,
                                     dense_neurons=256,
                                     device=device,
                                     for_active=True
                                     )

    swap_passive = TransformerSwapGNN(n_layers=4,
                                     feature_size=2,
                                     n_heads=3,
                                     embedding_size=64,
                                     dropout_rate=0.2,
                                     top_k_ratio=0.5,
                                     dense_neurons=256,
                                     device=device,
                                     for_active=False
                           )

    critic_active = TransformerGNN(n_layers=4,
                           feature_size=2,
                           n_heads=3,
                           embedding_size=64,
                           dropout_rate=0.2,
                           top_k_ratio=0.5,
                           dense_neurons=256,
                                   device=device
                           )

    critic_passive = TransformerGNN(n_layers=4,
                           feature_size=2,
                           n_heads=3,
                           embedding_size=64,
                           dropout_rate=0.2,
                           top_k_ratio=0.5,
                           dense_neurons=256,
                            device=device
                           )

    ppo_config = SwapPPOAgentConfigActionTypeBoth(swap_active, critic_active, swap_passive, critic_passive,
                                                  batch_size=32, n_epochs=2,
                                                  summary_writer=SummaryWriter(log_dir="logs/tensorboard"),
                                                  lr=0.005)
    agent = PPOAgentActionTypeBoth(ppo_config)

    env = NetworkEnvGym(config)
    env = TorchGraphNetworkxWrapper(env, one_hot=False)

    training_args = TrainingParams(train_steps=3000)

    trainer_config = RepStateMachineTrainerConfig(agent=agent, env=env, training_params=training_args, day_periods=1000,
                                                  training_algorithm=TrainingAlgorithms.PPO)

    trainer = RepStateMachineTrainerRL(trainer_config)
    #trainer.load_model()

    trainer.train()


def dqn():
    device = "cuda"
    clusters = [5, 5, 5]

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
    num_active = 3
    assert num_active <= clusters[0]
    assert num_active <= clusters[1]
    assert num_active <= clusters[2]
    config = CustomNetworkConfig(num_centers=3, clusters=clusters, num_clients=20, render_mode="human",
                                 render_type="2d", device=device, penalty_weights=penalty_weights, num_active=3,
                                 display_all_latencies_from_start=True
                                 )

    q_eval = QNetworkSwapGNN(device=device,
                             feature_size=3,
                             embedding_size=64,
                             n_heads=3,
                             dropout_rate=0.2,
                             n_layers=3,
                             dense_neurons=256,
                             edge_dim=1,
                             num_locations=sum(clusters),
                             for_active=True,
                             lr=0.0001,
                             optimizer=torch.optim.AdamW,
                             reduce_action_space=False
                             )

    q_target_1 = QNetworkSwapGNN(device=device,
                                 feature_size=3,
                                 embedding_size=64,
                                 n_heads=3,
                                 dropout_rate=0.2,
                                 n_layers=3,
                                 dense_neurons=256,
                                 edge_dim=1,
                                 num_locations=sum(clusters),
                                 for_active=True,
                                 lr=0.0001,
                                 optimizer=torch.optim.AdamW,
                                 reduce_action_space=False
                                 )

    q_target_2 = QNetworkSwapGNN(device=device,
                                 feature_size=3,
                                 embedding_size=64,
                                 n_heads=3,
                                 dropout_rate=0.2,
                                 n_layers=3,
                                 dense_neurons=256,
                                 edge_dim=1,
                                 num_locations=sum(clusters),
                                 for_active=True,
                                 lr=0.0001,
                                 optimizer=torch.optim.AdamW,
                                 reduce_action_space=False
                                 )

    ddqn_config = DQNConfig(q_eval, q_target_1, q_target_2, batch_size=256, replace_target=1000)

    agent = DQGNAgent(ddqn_config)

    env = NetworkEnvGym(config)
    env = TorchGraphNetworkxWrapper(env, one_hot=False)

    training_args = TrainingParams(train_steps=2, num_train_runs=1000000)

    trainer_config = RepStateMachineTrainerConfig(agent=agent, env=env, training_params=training_args, day_periods=1000,
                                                  training_algorithm=TrainingAlgorithms.DDQN)

    trainer = RepStateMachineTrainerRL(trainer_config)

    trainer.train()


if __name__ == '__main__':
    dqn()