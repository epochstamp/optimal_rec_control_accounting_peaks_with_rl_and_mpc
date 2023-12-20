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


import torch
import torch.nn as nn
import torch.nn.functional as F
from skrl.resources.schedulers.torch import KLAdaptiveLR
from torchmetrics.regression.explained_variance import ExplainedVariance
from skrl.memories.torch import Memory
import gymnasium
from typing import Any, Dict, Optional, Tuple, Union, List

class MetrixedBC(PPO):

    def __init__(self,
                 models: Dict[str, Model],
                 expert_actions: List[float],
                 memory: Optional[Union[Memory, Tuple[Memory]]] = None,
                 observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 cfg: Optional[dict] = None,
                 trainable_rnn_states: bool = False) -> None:
        super().__init__(models=models,
                         memory=memory,
                         observation_space=observation_space,
                         action_space=action_space,
                         device=device,
                         cfg=cfg)
        self._expert_actions = torch.FloatTensor(np.asarray(expert_actions))
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self._learning_rate)
        self._trainable_rnn_states = trainable_rnn_states
        

    def init(self, trainer_cfg: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the agent
        """
        super(PPO, self).init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")

        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(name="states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="truncated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="log_prob", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="expert_actions", size=self.action_space, dtype=torch.float32)
            # tensors sampled during training
            self._tensors_names = ["states", "actions", "terminated", "truncated", "log_prob", "expert_actions"]

        # RNN specifications
        self._rnn = False  # flag to indicate whether RNN is available
        self._rnn_tensors_names = []  # used for sampling during training
        self._rnn_final_states = {"policy": []}
        self._rnn_initial_states = {"policy": []}
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

        # create temporary variables needed for storage and computation
        self._current_log_prob = None
        self._current_next_states = None

    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        """Process the environment's states to make a decision (actions) using the main policy

        :param states: Environment's states
        :type states: torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :return: Actions
        :rtype: torch.Tensor
        """
        rnn = {"rnn": self._rnn_initial_states["policy"]} if self._rnn else {}

        # sample random actions
        # TODO: fix for stochasticity, rnn and log_prob
        if timestep < self._random_timesteps:
            return self.policy.random_act({"states": self._state_preprocessor(states), **rnn}, role="policy")
        
        # sample stochastic actions
        pol_input = {"states": self._state_preprocessor(states), **rnn}
        actions, log_prob, outputs = self.policy.act({"states": self._state_preprocessor(states), **rnn}, role="policy")
        current_t = timestep % self._expert_actions.shape[0]
        if self.training:
            
            actions, log_prob, outputs = self.policy.act({"states": self._state_preprocessor(states),"overriding_actions": self._expert_actions[current_t, :].repeat(*actions.shape), **rnn}, role="policy")
        self._current_log_prob = log_prob

        if self._rnn:
            self._rnn_final_states["policy"] = outputs.get("rnn", [])

        return actions, log_prob, outputs


    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        

        

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0
        cumulative_clipped_gradient = 0
        cumulative_kl_divergence = 0
        cumulative_explained_variance = 0
        kl_divergences = []
        # learning epochs
        for epoch in range(self._learning_epochs):
            
            # sample mini-batches from memory
            sampled_batches = self.memory.sample_all(names=self._tensors_names, mini_batches=self._mini_batches, sequence_length=self._rnn_sequence_length)

            rnn_policy, rnn_value = {}, {}
            if self._rnn:
                sampled_rnn_batches = self.memory.sample_all(names=self._rnn_tensors_names, mini_batches=self._mini_batches, sequence_length=self._rnn_sequence_length)
            #states = self.memory.get_tensor_by_name("states")
            #print(states)
            #_, current_log_prob, _ = self.policy.act({"states": self._state_preprocessor(states), **rnn}, role="policy")
            #self.memory.set_tensor_by_name("log_probs", current_log_prob)
            
            # mini-batches loop
            for i, (sampled_states, sampled_actions, sampled_terminated, sampled_truncated, sampled_log_prob, expert_actions) in enumerate(sampled_batches):
                d_cum_grad = None
                if self._rnn:
                    if self.policy is self.value:
                        rnn_policy = {"rnn": [s.transpose(0, 1) for s in sampled_rnn_batches[i]], "terminated": sampled_terminated, "truncated": sampled_truncated}
                    else:
                        rnn_policy = {"rnn": [s.transpose(0, 1) for s, n in zip(sampled_rnn_batches[i], self._rnn_tensors_names) if "policy" in n], "terminated": sampled_terminated, "truncated": sampled_truncated}
                sampled_states = self._state_preprocessor(sampled_states, train=not epoch)

                _, expert_log_prob, expert_outputs = self.policy.act({"states": sampled_states, "taken_actions": expert_actions, **rnn_policy}, role="policy")
                with torch.no_grad():
                    next_actions, next_log_prob, policy_outputs = self.policy.act({"states": sampled_states, "taken_actions": sampled_actions, **rnn_policy}, role="policy")
                    

                # compute approximate KL divergence
                with torch.no_grad():
                    ratio = next_log_prob - expert_log_prob
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
                sampled_advantages = 1.0
                ratio = next_log_prob - expert_log_prob
                #kl_divergence = -expert_log_prob#F.mse_loss(torch.exp(ratio), torch.ones_like(ratio))#(policy_outputs["log_std"]**2 + (policy_outputs["mean_actions"] - expert_outputs["mean_actions"])**2)/(2*expert_outputs["log_std"]**2)#((torch.exp(ratio) - 1) - ratio)
                #loss_function = 
                #surrogate_clipped = loss_function#sampled_advantages * torch.clip(ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip)

                policy_loss = -(expert_log_prob.median())

                # compute value loss
                """
                predicted_values, _, _ = self.value.act({"states": sampled_states, **rnn_value}, role="value")

                if self._clip_predicted_values:
                    predicted_values = sampled_values + torch.clip(predicted_values - sampled_values,
                                                                   min=-self._value_clip,
                                                                   max=self._value_clip)
                value_loss = self._value_loss_scale * torch.sqrt(F.mse_loss(sampled_returns, predicted_values) + 1e-6)
                with torch.no_grad():
                    cumulative_explained_variance += ExplainedVariance().forward(predicted_values, sampled_returns)
                """
                # optimization step
                self.optimizer.zero_grad()
                (policy_loss).backward()
                
                if self._grad_norm_clip > 0:
                    gradient_clipped = nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)
                    cumulative_clipped_gradient += torch.sqrt(torch.square(gradient_clipped).mean())
                self.optimizer.step()
                if d_cum_grad is None:
                    d_cum_grad = dict()
                for m, v in dict(self.policy.named_modules()).items():
                    if "net." in m and type(v) == nn.Linear:
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
                # update cumulative losses
                cumulative_policy_loss += policy_loss.item()
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
            self.track_data(f"Gradients: {k}", v / (self._learning_epochs * self._mini_batches))
        self.track_data("Explained Variance", cumulative_explained_variance / (self._learning_epochs * self._mini_batches))
        self.track_data("Gradients: Overall Clipped Gradient", cumulative_clipped_gradient / (self._learning_epochs * self._mini_batches))
        self.track_data("Loss / KL divergence", cumulative_kl_divergence / (self._learning_epochs * self._mini_batches))
        self.track_data("Loss / Policy loss", cumulative_policy_loss / (self._learning_epochs * self._mini_batches))
        self.track_data("Loss / Value loss", cumulative_value_loss / (self._learning_epochs * self._mini_batches))
        if self._entropy_loss_scale:
            self.track_data("Loss / Entropy loss", cumulative_entropy_loss / (self._learning_epochs * self._mini_batches))

        self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())

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

            # package RNN states
            rnn_states = {}
            if self._rnn:
                rnn_states.update({f"rnn_policy_{i}": s.transpose(0, 1) for i, s in enumerate(self._rnn_initial_states["policy"])})
            if "current_t" not in infos:
                current_t = infos["final_info"][0]["current_t"]
            else:
                current_t = infos["current_t"]
            current_t -= 1
            
            # storage transition in memory
            self.memory.add_samples(states=states, actions=actions, rewards=rewards, next_states=next_states,
                                    terminated=terminated, truncated=truncated, log_prob=self._current_log_prob, expert_actions=self._expert_actions[current_t, :].repeat(*actions.shape), **rnn_states)
            for memory in self.secondary_memories:
                memory.add_samples(states=states, actions=actions, rewards=rewards, next_states=next_states,
                                   terminated=terminated, truncated=truncated, log_prob=self._current_log_prob, expert_actions=self._expert_actions[current_t, :].repeat(*actions.shape), **rnn_states)

        # update RNN states
        if self._rnn:
            # reset states if the episodes have ended
            truncated_episodes = truncated.nonzero(as_tuple=False)
            terminated_episodes = terminated.nonzero(as_tuple=False)
            if truncated_episodes.numel() + terminated_episodes.numel():
                for rnn_state in self._rnn_final_states["policy"]:
                    if truncated_episodes.numel():
                        rnn_state[:, truncated_episodes[:, 0]] = 0
                    if terminated_episodes.numel():
                        rnn_state[:, terminated_episodes[:, 0]] = 0
                #print(current_t, self._very_initial_rnn_states["policy"])

            self._rnn_initial_states = self._rnn_final_states
            