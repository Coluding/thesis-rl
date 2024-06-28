import copy
from abc import ABC, abstractmethod
import torch
from dataclasses import dataclass
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Tuple, Optional
from tensordict import TensorDict
import torch_geometric
from tqdm import tqdm
from typing import List
import logging

from src.model.gnn import SwapGNN, CriticSwapGNN
from src.algorithm.memory import OnPolicyMemory

logging.basicConfig(
    filename='training.log',  # Specify the file to log to
    filemode='a',        # Mode 'a' for append, 'w' for overwrite
    level=logging.DEBUG,  # Set the minimum logging level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # Define the log message format
)

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

class AbstractAgent(ABC):
    def __init__(self, config):
        self.chkpt_dir = None

    @abstractmethod
    def choose_action(self, state):
        pass

    @abstractmethod
    def learn(self):
        pass

    @abstractmethod
    def save_models(self):
        pass

    @abstractmethod
    def load_models(self):
        pass

    @abstractmethod
    def add_experience(self, state, val, action, reward, done, prob):
        pass

    @abstractmethod
    def get_average_loss(self):
        pass

    @abstractmethod
    def get_buffer_size(self):
        pass

    @abstractmethod
    def clear_memory(self):
        pass

@dataclass
class SwapPPOAgentConfigActionTypeSingle:
    actor: SwapGNN
    critic: CriticSwapGNN
    gae_lambda: Optional[float] = 0.95
    clip_eps: Optional[float] = 0.2
    eps: Optional[float] = 0.2
    eps_min: Optional[float] = 0.01
    eps_decay: Optional[float] = 0.99
    gamma: Optional[float] = 0.99
    batch_size: Optional[int] = 64
    n_epochs: Optional[int] = 10
    replace_rate: Optional[int] = 10
    chkpt_dir: Optional[str] = "tmp/ppo"
    summary_writer: Optional[SummaryWriter] = None


@dataclass
class SwapPPOAgentConfigActionTypeBoth:
    active_actor: SwapGNN
    active_critic: CriticSwapGNN
    passive_actor: SwapGNN
    passive_critic: CriticSwapGNN
    optimizer: Optional[nn.Module] = torch.optim.Adam
    lr: Optional[float] = 0.001
    entropy_reg: Optional[float] = 0.01
    gae_lambda: Optional[float] = 0.95
    clip_eps: Optional[float] = 0.2
    eps: Optional[float] = 0.2
    eps_min: Optional[float] = 0.01
    eps_decay: Optional[float] = 0.99
    gamma: Optional[float] = 0.99
    batch_size: Optional[int] = 64
    trajectory_size: Optional[int] = 1000
    n_epochs: Optional[int] = 10
    replace_rate: Optional[int] = 10
    chkpt_dir: Optional[str] = "tmp/ppo"
    summary_writer: Optional[SummaryWriter] = None


class PPOAgentActionTypeBoth(AbstractAgent):

    def __init__(self, config: SwapPPOAgentConfigActionTypeSingle | SwapPPOAgentConfigActionTypeBoth):
        super().__init__(config)
        self.swap_actor_active = config.active_actor
        self.swap_critic_active = config.active_critic
        self.swap_actor_passive = config.passive_actor
        self.swap_critic_passive = config.passive_critic
        self.gae_lambda = config.gae_lambda
        self.clip_eps = config.clip_eps
        self.eps = config.eps
        self.eps_min = config.eps_min
        self.eps_decay = config.eps_decay
        self.gamma = config.gamma
        self.batch_size = config.batch_size
        self.n_epochs = config.n_epochs
        self.replace_rate = config.replace_rate
        self.chkpt_dir = config.chkpt_dir
        self.summary_writer = config.summary_writer
        self.trajectory_size = config.trajectory_size
        self.entropy_reg = config.entropy_reg
        self.memory = OnPolicyMemory(self.trajectory_size, self.batch_size)


    def _form_remove_and_add_mask(self, mask: torch.Tensor):
        remove_mask = mask.clone()
        remove_mask[:self.swap_actor_active.num_locations][mask[:self.swap_actor_active.num_locations] == 0] = -np.inf
        remove_mask[:self.swap_actor_active.num_locations][mask[:self.swap_actor_active.num_locations] == -np.inf] = 0

        return mask, remove_mask

    def _extract_possible_actions_from_mask(self, mask: torch.Tensor):
        valid_actions = torch.where(mask != -np.inf)[0]
        return valid_actions

    def choose_action_eps(self, state):
        if np.random.random() > self.eps:
            output = self.swap_actor(state)
            action = output.actions
        else:
            if self.active_action:
                mask = state.active_mask
            else:
                mask = state.passive_mask

            add_mask, remove_mask = self._form_remove_and_add_mask(mask)
            valid_add_actions = self._extract_possible_actions_from_mask(add_mask)
            valid_remove_actions = self._extract_possible_actions_from_mask(remove_mask)

            add_action = np.random.choice(valid_add_actions)
            remove_action = np.random.choice(valid_remove_actions)

            action = (remove_action, add_action)

    def choose_action_sampled_active(self, state) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.swap_actor_active(state, state.batch)
        action = output.actions
        log_probs = output.log_probs

        value = self.swap_critic_active(state, state.batch)

        return action, log_probs, value

    def choose_action_sampled_passive(self, state) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.swap_actor_passive(state)
        action = output.actions
        log_probs = output.log_probs

        value = self.swap_critic_passive(state)

        return action, log_probs, value

    # TODO: add type hinting for experiences
    def add_experience(self, state, val, action, reward, done, prob):
        self.memory.store(state, val, action, reward, done, prob)

    def _compute_gae_both(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor) -> torch.FloatTensor:
        advantages = torch.zeros_like(rewards, dtype=torch.float)

        last_advantage = 0
        for t in reversed(range(len(rewards))):
            terminal = 1 - dones[t].int()
            next_value = values[t + 1] if t + 1 < len(rewards) else 0
            delta = terminal * (rewards[t] + self.gamma * next_value) - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * terminal * last_advantage
            last_advantage = advantages[t]

        return advantages

    def _compute_entropy(self, log_probs: torch.Tensor):
        return -torch.sum(log_probs.exp() * log_probs)

    def learn(self):
        self.loss ={
            "actor_active": 0,
            "actor_passive": 0,
            "critic_active": 0,
            "critic_passive": 0
        }
        self.steps = 0
        # Iterate through each epoch
        for epoch in range(self.n_epochs):
            logger.info(f"Epoch: {epoch}")  # Log the current epoch

            # Retrieve the full trajectory of experiences from memory
            full_trajectory = self.memory.get_full_trajectory()

            # Convert the states to a tensor batch suitable for processing
            states_tensor = torch_geometric.data.Batch.from_data_list(full_trajectory["states"])

            # Convert other trajectory components to tensors and move them to the appropriate device (GPU/CPU)
            rewards = torch.tensor(full_trajectory["rewards"]).to(self.swap_actor_active.device)
            dones = torch.tensor(full_trajectory["dones"]).to(self.swap_actor_active.device)
            actions = torch.tensor(full_trajectory["actions"]).to(self.swap_actor_active.device)
            values = torch.stack(full_trajectory["values"]).squeeze().to(self.swap_actor_active.device)
            old_probs = torch.stack(full_trajectory["probs"]).to(self.swap_actor_active.device)

            # Compute advantages using Generalized Advantage Estimation (GAE) for active and passive actions
            # I am transferring them to cpu to break the computation graph in order to backpropagate
            advantages_active_pre = self._compute_gae_both(rewards[:, 0], values[:, 0], dones).cpu().numpy()
            advantages_passive_pre = self._compute_gae_both(rewards[:, 1], values[:, 1], dones).cpu().numpy()

            # Split the trajectory into mini-batches for training
            batches = [torch.arange(i, min(i + self.batch_size, len(full_trajectory["states"]))) for i in
                       range(0, len(full_trajectory["states"]), self.batch_size)]
            batch_iterator = tqdm(batches, desc="PPO Training", unit="batch")

            # Iterate through each batch
            for batch in batch_iterator:
                # Prepare batched tensors for the current batch
                advantages_active = torch.tensor(advantages_active_pre[batch]).to(self.swap_actor_active.device)
                advantages_passive = torch.tensor(advantages_passive_pre[batch]).to(self.swap_actor_active.device)
                states_batched = torch_geometric.data.Batch.from_data_list(states_tensor[batch])
                rewards_batched = rewards[batch]
                dones_batched = dones[batch]
                actions_batched = actions[batch]
                values_batched = values[batch]
                old_probs_batched = old_probs[batch]

                # Compute value outputs for the active and passive critics
                per_graph_value_output_active: List[torch.Tensor] = self.swap_critic_active(states_batched,
                                                                                            states_batched.batch)
                per_graph_value_output_passive: List[torch.Tensor] = self.swap_critic_passive(states_batched,
                                                                                              states_batched.batch)

                # Compute the returns for active and passive actions
                returns_active = advantages_active + per_graph_value_output_active.squeeze()
                returns_passive = advantages_passive + per_graph_value_output_passive.squeeze()

                # Get actor network outputs for the active and passive actors
                actor_output_active = self.swap_actor_active(states_batched, states_batched.batch)
                actor_output_passive = self.swap_actor_passive(states_batched, states_batched.batch)

                # Extract the log probabilities of the new actions from the actor outputs
                log_prob_new_active = actor_output_active.log_probs
                log_prob_new_passive = actor_output_passive.log_probs

                # Compute the joint log probabilities for the active and passive actions
                joint_log_prob_active_actions = log_prob_new_active.sum(dim=1)
                joint_log_prob_passive_actions = log_prob_new_passive.sum(dim=1)

                # Compute the old joint log probabilities for the active and passive actions
                joint_log_prob_old_active_actions = old_probs_batched[:, 0].sum(dim=1)
                joint_log_prob_old_passive_actions = old_probs_batched[:, 1].sum(dim=1)

                # Compute the ratios of the new to old probabilities
                ratio_active = (joint_log_prob_active_actions - joint_log_prob_old_active_actions).exp()
                ratio_passive = (joint_log_prob_passive_actions - joint_log_prob_old_passive_actions).exp()

                # Clip the ratios to stabilize training
                clipped_ratio_active = torch.clamp(ratio_active, 1 - self.clip_eps, 1 + self.clip_eps)
                clipped_ratio_passive = torch.clamp(ratio_passive, 1 - self.clip_eps, 1 + self.clip_eps)

                # Compute the surrogate loss objectives
                objective_active = torch.min(ratio_active * advantages_active,
                                             clipped_ratio_active * advantages_active)
                objective_passive = torch.min(ratio_passive * advantages_passive,
                                              clipped_ratio_passive * advantages_passive)

                # Add entropy regularization to the objectives to encourage exploration
                objective_active_regularized = objective_active + self.entropy_reg * self._compute_entropy(log_prob_new_active)
                objective_passive_regularized = objective_passive + self.entropy_reg * self._compute_entropy(log_prob_new_passive)

                # Compute the actor losses as the negative mean of the objectives
                actor_active_loss = -objective_active_regularized.mean()
                actor_passive_loss = -objective_passive_regularized.mean()

                # Compute the critic losses as the mean squared error between returns and value outputs
                critic_active_loss = torch.mean((returns_active - per_graph_value_output_active.squeeze()) ** 2)
                critic_passive_loss = torch.mean((returns_passive - per_graph_value_output_passive.squeeze()) ** 2)

                # Zero the gradients for all the optimizers
                self.swap_actor_active.optimizer.zero_grad()
                self.swap_actor_passive.optimizer.zero_grad()
                self.swap_critic_active.optimizer.zero_grad()
                self.swap_critic_passive.optimizer.zero_grad()

                # Backpropagate the total loss (maybe do retain_graph=True) for every single loss
                actor_active_loss.backward(retain_graph=True)
                actor_passive_loss.backward(retain_graph=True)
                critic_active_loss.backward(retain_graph=True)
                critic_passive_loss.backward()

                # Update the network parameters
                self.swap_actor_active.optimizer.step()
                self.swap_actor_passive.optimizer.step()
                self.swap_critic_active.optimizer.step()
                self.swap_critic_passive.optimizer.step()

                # Log the losses
                self.loss["actor_active"] += actor_active_loss.item()
                self.loss["actor_passive"] += actor_passive_loss.item()
                self.loss["critic_active"] += critic_active_loss.item()
                self.loss["critic_passive"] += critic_passive_loss.item()
                self.steps += 1


    def clear_memory(self):
        self.memory.clear_memory()

    def get_average_loss(self):
        return {k: v / self.steps for k, v in self.loss.items()}

    def save_models(self):
        torch.save(self.swap_actor_active.state_dict(), self.chkpt_dir + "/swap_actor_active.pth")
        torch.save(self.swap_actor_passive.state_dict(), self.chkpt_dir + "/swap_actor_passive.pth")
        torch.save(self.swap_critic_active.state_dict(), self.chkpt_dir + "/swap_critic_active.pth")
        torch.save(self.swap_critic_passive.state_dict(), self.chkpt_dir + "/swap_critic_passive.pth")

    def load_models(self):
        self.swap_actor_active.load_state_dict(torch.load(self.chkpt_dir + "/swap_actor_active.pth"))
        self.swap_actor_passive.load_state_dict(torch.load(self.chkpt_dir + "/swap_actor_passive.pth"))
        self.swap_critic_active.load_state_dict(torch.load(self.chkpt_dir + "/swap_critic_active.pth"))
        self.swap_critic_passive.load_state_dict(torch.load(self.chkpt_dir + "/swap_critic_passive.pth"))

    def choose_action(self, state: torch_geometric.data.Batch | torch_geometric.data.Data) \
            -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        active_action = self.choose_action_sampled_active(state)
        passive_action = self.choose_action_sampled_passive(state)

        return active_action, passive_action

    def get_buffer_size(self):
        return len(self.memory.memory["states"])