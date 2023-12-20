import gymnasium as gym

import numpy as np
import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from gymnasium.spaces import Box
from skrl.agents.torch.ppo import PPO_RNN as PPO
from skrl.agents.torch.rpo import RPO_RNN as RPO
import itertools
from torch.distributions import Normal


import torch
import torch.nn as nn
import torch.nn.functional as F
from skrl.resources.schedulers.torch import KLAdaptiveLR
from torchmetrics.regression.explained_variance import ExplainedVariance
from skrl.memories.torch import Memory
import gymnasium
from typing import Any, Dict, Optional, Tuple, Union

class MetrixedPPO(PPO):

    def __init__(self,
                 models: Dict[str, Model],
                 memory: Optional[Union[Memory, Tuple[Memory]]] = None,
                 observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 cfg: Optional[dict] = None) -> None:
        self._state_dependent_std = cfg.get("state_dependent_std", False)
        super().__init__(models=models,
                         memory=memory,
                         observation_space=observation_space,
                         action_space=action_space,
                         device=device,
                         cfg=cfg)
        
        """
        if self.policy is not None and self.value is not None:
            if self.policy is self.value:
                self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self._learning_rate)
            else:
                self.optimizer = torch.optim.Adam(itertools.chain(self.policy.parameters(), self.value.parameters()),
                                                  lr=self._learning_rate)
            if self._learning_rate_scheduler is not None:
                self.scheduler = self._learning_rate_scheduler(self.optimizer, **self.cfg["learning_rate_scheduler_kwargs"])

            self.checkpoint_modules["optimizer"] = self.optimizer
        """

    def init(self, trainer_cfg: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the agent
        """
        super(PPO, self).init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")
        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(name="states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            if self._state_dependent_std:
                self.memory.create_tensor(name="mean_actions", size=self.action_space, dtype=torch.float32)
                self.memory.create_tensor(name="std_actions", size=self.action_space, dtype=torch.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="truncated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="log_prob", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="values", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="returns", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="advantages", size=1, dtype=torch.float32)
            # tensors sampled during training
            if self._state_dependent_std:
                self._tensors_names = ["states", "actions", "mean_actions", "std_actions", "terminated", "truncated", "log_prob", "values", "returns", "advantages"]
            else:
                self._tensors_names = ["states", "actions", "terminated", "truncated", "log_prob", "values", "returns", "advantages"]

        # RNN specifications
        self._rnn = False  # flag to indicate whether RNN is available
        self._rnn_tensors_names = []  # used for sampling during training
        self._rnn_final_states = {"policy": [], "value": []}
        self._rnn_initial_states = {"policy": [], "value": []}
        self._rnn_sequence_length = self.policy.get_specification().get("rnn", {}).get("sequence_length", 1)

        # policy
        for i, size in enumerate(self.policy.get_specification().get("rnn", {}).get("sizes", [])):
            self._rnn = True
            # create tensors in memory
            if self.memory is not None:
                self.memory.create_tensor(name=f"rnn_policy_{i}", size=(size[0], size[2]), dtype=torch.float32, keep_dimensions=True)
                self._rnn_tensors_names.append(f"rnn_policy_{i}")
            # default RNN states
            self._rnn_initial_states["policy"].append(torch.zeros(size, dtype=torch.float32, device=self.device))

        # value
        if self.value is not None:
            if self.policy is self.value:
                self._rnn_initial_states["value"] = self._rnn_initial_states["policy"]
            else:
                for i, size in enumerate(self.value.get_specification().get("rnn", {}).get("sizes", [])):
                    self._rnn = True
                    # create tensors in memory
                    if self.memory is not None:
                        self.memory.create_tensor(name=f"rnn_value_{i}", size=(size[0], size[2]), dtype=torch.float32, keep_dimensions=True)
                        self._rnn_tensors_names.append(f"rnn_value_{i}")
                    # default RNN states
                    self._rnn_initial_states["value"].append(torch.zeros(size, dtype=torch.float32, device=self.device))

        # create temporary variables needed for storage and computation
        self._current_log_prob = None
        self._current_next_states = None

    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        def compute_gae(rewards: torch.Tensor,
                        terminated: torch.Tensor,
                        truncated: torch.Tensor,
                        values: torch.Tensor,
                        next_values: torch.Tensor,
                        discount_factor: float = 0.99,
                        lambda_coefficient: float = 0.95) -> torch.Tensor:
            """Compute the Generalized Advantage Estimator (GAE)

            :param rewards: Rewards obtained by the agent
            :type rewards: torch.Tensor
            :param dones: Signals to indicate that episodes have ended
            :type dones: torch.Tensor
            :param values: Values obtained by the agent
            :type values: torch.Tensor
            :param next_values: Next values obtained by the agent
            :type next_values: torch.Tensor
            :param discount_factor: Discount factor
            :type discount_factor: float
            :param lambda_coefficient: Lambda coefficient
            :type lambda_coefficient: float

            :return: Generalized Advantage Estimator
            :rtype: torch.Tensor
            """
            advantage = 0
            advantages = torch.zeros_like(rewards)
            not_terminated = terminated.logical_not()
            not_truncated = truncated.logical_not()
            memory_size = rewards.shape[0]

            # advantages computation
            for i in reversed(range(memory_size)):
                next_values = values[i + 1] if i < memory_size - 1 else last_values
                advantage = rewards[i] - values[i] + discount_factor * not_terminated[i] * not_truncated[i] * (next_values + lambda_coefficient * advantage)
                advantages[i] = advantage
            # returns computation
            returns = advantages + values
            # normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            return returns, advantages

        
        
        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0
        cumulative_clipped_gradient = 0
        cumulative_kl_divergence = 0
        cumulative_explained_variance = 0
        
         # compute returns and advantages
        with torch.no_grad():
            self.value.train(False)
            rnn = {"rnn": self._rnn_initial_states["value"]} if self._rnn else {}
            last_values, _, _ = self.value.act({"states": self._state_preprocessor(self._current_next_states.float()), **rnn}, role="value")
            self.value.train(True)
        last_values = self._value_preprocessor(last_values, inverse=True)

        values = self.memory.get_tensor_by_name("values")
        returns, advantages = compute_gae(rewards=self.memory.get_tensor_by_name("rewards"),
                                        terminated=self.memory.get_tensor_by_name("terminated"),
                                        truncated=self.memory.get_tensor_by_name("truncated"),
                                        values=values,
                                        next_values=last_values,
                                        discount_factor=self._discount_factor,
                                        lambda_coefficient=self._lambda)
        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)

        # sample mini-batches from memory
        sampled_batches = self.memory.sample_all(names=self._tensors_names, mini_batches=self._mini_batches, sequence_length=self._rnn_sequence_length)

        rnn_policy, rnn_value = {}, {}
        if self._rnn:
            sampled_rnn_batches = self.memory.sample_all(names=self._rnn_tensors_names, mini_batches=self._mini_batches, sequence_length=self._rnn_sequence_length)
        # learning epochs
        for epoch in range(self._learning_epochs):
            kl_divergences = []
            
            #states = self.memory.get_tensor_by_name("states")
            #print(states)
            #_, current_log_prob, _ = self.policy.act({"states": self._state_preprocessor(states), **rnn}, role="policy")
            #self.memory.set_tensor_by_name("log_probs", current_log_prob)
            
            # mini-batches loop
            for i, sample_batch in enumerate(sampled_batches):
                if self._state_dependent_std:
                    sampled_states, sampled_actions, sampled_mean_actions, sampled_std_actions, sampled_terminated, sampled_truncated, sampled_log_prob, sampled_values, sampled_returns, sampled_advantages = sample_batch
                else:
                    sampled_states, sampled_actions, sampled_terminated, sampled_truncated, sampled_log_prob, sampled_values, sampled_returns, sampled_advantages = sample_batch
                d_cum_grad = None
                if self._rnn:
                    if self.policy is self.value:
                        rnn_policy = {"rnn": [s.transpose(0, 1) for s in sampled_rnn_batches[i]], "terminated": sampled_terminated, "truncated": sampled_truncated}
                        rnn_value = rnn_policy
                    else:
                        rnn_policy = {"rnn": [s.transpose(0, 1) for s, n in zip(sampled_rnn_batches[i], self._rnn_tensors_names) if "policy" in n], "terminated": sampled_terminated, "truncated": sampled_truncated}
                        rnn_value = {"rnn": [s.transpose(0, 1) for s, n in zip(sampled_rnn_batches[i], self._rnn_tensors_names) if "value" in n], "terminated": sampled_terminated, "truncated": sampled_truncated}

                sampled_states = self._state_preprocessor(sampled_states, train=not epoch)

                _, next_log_prob, outputs = self.policy.act({"states": sampled_states, "taken_actions": sampled_actions, **rnn_policy}, role="policy")

                # compute approximate KL divergence
                with torch.no_grad():
                    if self._state_dependent_std:
                        mean_actions_sample = outputs["mean_actions"]
                        std_actions_sample = outputs["log_std"]
                        mean_actions_target = sampled_mean_actions
                        std_actions_target = sampled_std_actions
                        norm_input = Normal(mean_actions_sample, std_actions_sample.exp() + 1e-6)
                        norm_target = Normal(mean_actions_target, std_actions_target.exp() + 1e-6)
                        kl_divergence = torch.distributions.kl_divergence(norm_input, norm_target).mean()
                    else:
                        ratio = next_log_prob - sampled_log_prob
                        kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                    kl_divergences.append(kl_divergence)

                # early stopping with KL divergence
                if self._kl_threshold and kl_divergence > self._kl_threshold:
                    break

                # compute entropy loss
                if self._entropy_loss_scale:
                    entropy_loss = -self._entropy_loss_scale * self.policy.get_entropy(role="policy").mean()
                else:
                    entropy_loss = 0

                # compute policy loss
                ratio = torch.exp(next_log_prob - sampled_log_prob)
                surrogate = sampled_advantages * ratio
                surrogate_clipped = sampled_advantages * torch.clip(ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip)

                policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                # compute value loss
                predicted_values, _, _ = self.value.act({"states": sampled_states, **rnn_value}, role="value")

                if self._clip_predicted_values:
                    predicted_values = sampled_values + torch.clip(predicted_values - sampled_values,
                                                                   min=-self._value_clip,
                                                                   max=self._value_clip)
                value_loss = self._value_loss_scale * F.mse_loss(sampled_returns, predicted_values)
                with torch.no_grad():
                    cumulative_explained_variance += ExplainedVariance().forward(predicted_values, sampled_returns)

                # optimization step
                self.optimizer.zero_grad()
                (policy_loss + entropy_loss + value_loss).backward()
                
                if self._grad_norm_clip > 0:
                    if self.policy is self.value:
                        gradient_clipped = nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)
                    else:
                        gradient_clipped = nn.utils.clip_grad_norm_(itertools.chain(self.policy.parameters(), self.value.parameters()), self._grad_norm_clip)
                    gradient_clipped = torch.sqrt(torch.square(gradient_clipped).mean())
                    cumulative_clipped_gradient += gradient_clipped
                self.optimizer.step()
                if d_cum_grad is None:
                    d_cum_grad = dict()
                for m, v in dict(self.policy.named_modules()).items():
                    if (("net." in m) or ("net_before." in m) or ("action" in m)) and type(v) == nn.Linear:
                        if "action" in m:
                            layer_id = "Final"
                        else:
                            layer_id = int(int(m.split(".")[1])//2) + 1
                        key = f"Policy Layer {layer_id}"
                        if key not in d_cum_grad:
                            d_cum_grad [key] = 0.0
                        d_cum_grad[key] += torch.sqrt(torch.square(v.weight.grad).mean())
                    elif "rnn" in m:
                        for i, w in enumerate(v.all_weights):
                            for j, w2 in enumerate(w):
    
                                key = f"Policy Recurrent Layer {i+1}, Parameter set ({j+1})"
                                if key not in d_cum_grad:
                                    d_cum_grad [key] = 0.0
                                d_cum_grad[key] += torch.sqrt(torch.square(w2.grad).mean())
                for m, v in dict(self.value.named_modules()).items():
                    if (("net." in m) or ("net_before." in m) or ("value" in m)) and type(v) == nn.Linear:
                        if "value" in m:
                            layer_id = "Final"
                        else:
                            layer_id = int(int(m.split(".")[1])//2) + 1
                        key = f"Value Layer {layer_id}"
                        if key not in d_cum_grad:
                            d_cum_grad [key] = 0.0
                        d_cum_grad[key] += torch.sqrt(torch.square(v.weight.grad).mean())
                    elif "rnn" in m:
                        for i, w in enumerate(v.all_weights):
                            for j, w2 in enumerate(w):
                                key = f"Value Recurrent Layer {i+1}, Parameter set ({j+1})"
                                if key not in d_cum_grad:
                                    d_cum_grad [key] = 0.0
                                d_cum_grad[key] += torch.sqrt(torch.square(w2.grad).mean())
                # update cumulative losses
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self._entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()

            # update learning rate
            kl_divergence = torch.tensor(kl_divergences).mean()
            cumulative_kl_divergence += kl_divergence
            if self._learning_rate_scheduler:
                if isinstance(self.scheduler, KLAdaptiveLR):
                    self.scheduler.step(torch.tensor(kl_divergences).mean())
                else:
                    self.scheduler.step()

        # record data
        for k,v in d_cum_grad.items():
            self.track_data(f"Gradients / {k}", v / (self._learning_epochs * self._mini_batches))
        self.track_data("Learning / Explained Variance", cumulative_explained_variance / (self._learning_epochs * self._mini_batches))
        self.track_data("Gradients / Overall Clipped Gradient", cumulative_clipped_gradient / (self._learning_epochs * self._mini_batches))
        self.track_data("Learning / KL divergence", cumulative_kl_divergence / (self._learning_epochs * self._mini_batches))
        self.track_data("Loss / Policy loss", cumulative_policy_loss / (self._learning_epochs * self._mini_batches))
        self.track_data("Loss / Value loss", cumulative_value_loss / (self._learning_epochs * self._mini_batches))
        if self._entropy_loss_scale:
            self.track_data("Loss / Entropy loss", cumulative_entropy_loss / (self._learning_epochs * self._mini_batches))

        self.track_data("Learning / Policy Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())

        if self._learning_rate_scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])

    def record_transition(self,
                          states: torch.Tensor,
                          actions: torch.Tensor,
                          rewards: torch.Tensor,
                          next_states: torch.Tensor,
                          terminated: torch.Tensor,
                          truncated: torch.Tensor,
                          infos: Any,
                          timestep: int,
                          timesteps: int) -> None:
        """Record an environment transition in memory

        :param states: Observations/states of the environment used to make the decision
        :type states: torch.Tensor
        :param actions: Actions taken by the agent
        :type actions: torch.Tensor
        :param rewards: Instant rewards achieved by the current actions
        :type rewards: torch.Tensor
        :param next_states: Next observations/states of the environment
        :type next_states: torch.Tensor
        :param terminated: Signals to indicate that episodes have terminated
        :type terminated: torch.Tensor
        :param truncated: Signals to indicate that episodes have been truncated
        :type truncated: torch.Tensor
        :param infos: Additional information about the environment
        :type infos: Any type supported by the environment
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        super(PPO, self).record_transition(states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps)

        if self.memory is not None:
            self._current_next_states = next_states

            # reward shaping
            if self._rewards_shaper is not None:
                rewards = self._rewards_shaper(rewards, timestep, timesteps)

            # compute values
            rnn = {"rnn": self._rnn_initial_states["value"]} if self._rnn else {}
            values, _, outputs = self.value.act({"states": self._state_preprocessor(states), **rnn}, role="value")
            if self._state_dependent_std:
                _, _, outputs_pol = self.policy.act({"states": self._state_preprocessor(states), **rnn}, role="policy")
                mean_actions = outputs_pol["mean_actions"]
                std_actions = outputs_pol["log_std"]
            values = self._value_preprocessor(values, inverse=True)

            # time-limit (truncation) boostrapping
            if self._time_limit_bootstrap:
                rewards += self._discount_factor * values * truncated

            # package RNN states
            rnn_states = {}
            if self._rnn:
                rnn_states.update({f"rnn_policy_{i}": s.transpose(0, 1) for i, s in enumerate(self._rnn_initial_states["policy"])})
                if self.policy is not self.value:
                    rnn_states.update({f"rnn_value_{i}": s.transpose(0, 1) for i, s in enumerate(self._rnn_initial_states["value"])})

            supp_states = dict(rnn_states)
            if self._state_dependent_std:
                supp_states["std_actions"] = std_actions
                supp_states["mean_actions"] = mean_actions

            # storage transition in memory
            self.memory.add_samples(states=states, actions=actions, rewards=rewards, next_states=next_states,
                                    terminated=terminated, truncated=truncated, log_prob=self._current_log_prob, values=values, **supp_states)
            for memory in self.secondary_memories:
                memory.add_samples(states=states, actions=actions, rewards=rewards, next_states=next_states,
                                   terminated=terminated, truncated=truncated, log_prob=self._current_log_prob, values=values, **supp_states)

        # update RNN states
        if self._rnn:
            self._rnn_final_states["value"] = self._rnn_final_states["policy"] if self.policy is self.value else outputs.get("rnn", [])

            # reset states if the episodes have ended
            truncated_episodes = truncated.nonzero(as_tuple=False)
            terminated_episodes = terminated.nonzero(as_tuple=False)
            if truncated_episodes.numel() + terminated_episodes.numel():
                for rnn_state in self._rnn_final_states["policy"]:
                    if truncated_episodes.numel():
                        rnn_state[:, truncated_episodes[:, 0]] = 0
                    if terminated_episodes.numel():
                        rnn_state[:, terminated_episodes[:, 0]] = 0
                    
                if self.policy is not self.value:
                    for rnn_state in self._rnn_final_states["value"]:
                        if truncated_episodes.numel():
                            rnn_state[:, truncated_episodes[:, 0]] = 0
                        if terminated_episodes.numel():
                            rnn_state[:, terminated_episodes[:, 0]] = 0

            self._rnn_initial_states = self._rnn_final_states