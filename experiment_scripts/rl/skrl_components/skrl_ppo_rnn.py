import gymnasium as gym

import numpy as np
import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from gymnasium.spaces import Box, Tuple, Dict

from typing import Any, Mapping, Tuple, Union
from torch.distributions import Normal
from .skrl_truncated_normal import TruncatedNormal
import math

class RNNLayer():

    def create_rnn(self, input_size, hidden_size, num_layers, sequence_length=1, num_envs=1):
        self._rnn_layer = self._create_rnn(input_size, hidden_size, num_layers)
        self._num_layers = num_layers
        self._hidden_size = hidden_size
        self._sequence_length = sequence_length
        self._num_envs=num_envs
        return self._rnn_layer
    
    def _create_rnn(self, input_size, hidden_size, num_layers):
        raise NotImplementedError()
    
    def forward_rnn(self, inputs, is_training=False):
        states = inputs["states"]
        terminated = inputs.get("terminated", None)
        truncated = inputs.get("truncated", None)
        hidden_states = inputs["rnn"][0]
        
        # training
        if is_training:
            rnn_input = states.view(-1, self._sequence_length, states.shape[-1])  # (N, L, Hin): N=batch_size, L=sequence_length
            hidden_states = hidden_states.view(self._num_layers, -1, self._sequence_length, hidden_states.shape[-1])  # (D * num_layers, N, L, Hout)
            # get the hidden states corresponding to the initial sequence
            hidden_states = hidden_states[:,:,0,:].contiguous()  # (D * num_layers, N, Hout)
            if terminated is not None or truncated is not None:
                
                if terminated is None:
                    done = truncated
                elif truncated is None:
                    done = terminated
                else:
                    done = torch.logical_or(terminated, truncated)
                # reset the RNN state in the middle of a sequence
                if done.any():
                    rnn_outputs = []
                    done = done.view(-1, self._sequence_length)
                    indexes = [0] + (done[:,:-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() + [self._sequence_length]
                    for i in range(len(indexes) - 1):
                        i0, i1 = indexes[i], indexes[i + 1]
                        rnn_output, hidden_states = self._forward_rnn(rnn_input[:,i0:i1,:], hidden_states)
                        hidden_states[:, (done[:,i1-1]), :] = 0
                        rnn_outputs.append(rnn_output)

                    rnn_output = torch.cat(rnn_outputs, dim=1)
                else:
                    rnn_output, hidden_states = self._forward_rnn(rnn_input, hidden_states)
            # no need to reset the RNN state in the sequence
            else:
                rnn_output, hidden_states = self._forward_rnn(rnn_input, hidden_states)
        # rollout
        else:
            rnn_input = states.view(-1, 1, states.shape[-1])  # (N, L, Hin): N=num_envs, L=1
            rnn_output, hidden_states = self._forward_rnn(rnn_input, hidden_states)
        return rnn_output, hidden_states

    def _forward_rnn(self, rnn_input, hidden_states):
        return self._rnn_layer(rnn_input, hidden_states)
    
    def postprocess_rnn_output(self, rnn_output, rnn_state, hook):
        raise NotImplementedError()

    def create_specification(self):
        raise NotImplementedError()
    
class LSTMLayer(RNNLayer):
    def _create_rnn(self, input_size, hidden_size, num_layers):
        return nn.LSTM(input_size=input_size,
                      hidden_size=hidden_size,
                      num_layers=num_layers,
                      batch_first=True)
    
    def postprocess_rnn_output(self, rnn_output, rnn_state):
        # flatten the RNN output
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)  # (N, L, D ∗ Hout) -> (N * L, D ∗ Hout)

        # Pendulum-v1 action_space is -2 to 2
        return rnn_output, {"rnn": [rnn_state[0], rnn_state[1]]}
    
    def create_specification(self):
        # batch size (N) is the number of envs
        return {"rnn": {"sequence_length": self._sequence_length,
                        "sizes": [(self._num_layers, self._num_envs, self._hidden_size),    # hidden states (D ∗ num_layers, N, Hout)
                                  (self._num_layers, self._num_envs, self._hidden_size)]}}  # cell states   (D ∗ num_layers, N, Hcell)
    
class GRULayer(RNNLayer):
    def _create_rnn(self, input_size, hidden_size, num_layers):
        return nn.GRU(input_size=input_size,
                      hidden_size=hidden_size,
                      num_layers=num_layers,
                      batch_first=True)
                      
    
    
        return rnn_output, hidden_states
    
    def postprocess_rnn_output(self, rnn_output, rnn_state):
        # flatten the RNN output
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)  # (N, L, D ∗ Hout) -> (N * L, D ∗ Hout)

        # Pendulum-v1 action_space is -2 to 2
        return rnn_output, {"rnn": [rnn_state]}
    
    def create_specification(self):
        return {"rnn": {"sequence_length": self._sequence_length,
                        "sizes": [(self._num_layers, self._num_envs, self._hidden_size)]}}  # hidden states (D ∗ num_layers, N, Hout)
    

d_rnn = {
    "gru": GRULayer,
    "lstm": LSTMLayer
}

class CReLU(nn.Module):
    def __init__(self):
        super(CReLU, self).__init__()
    def forward(self, x):
        return torch.cat((nn.functional.relu6(x), nn.functional.relu6(-x)), 1)
    
class LoggyReLU(nn.Module):
    def __init__(self):
        super(LoggyReLU, self).__init__()
    def forward(self, x):
        x = torch.where(x > 1, 0.5*((torch.log(x))+1), torch.where(x < -1, -0.5*((torch.log(-x))-1), x))
        return x
    
class LLeakyReLU(nn.Module):
    def __init__(self):
        super(LLeakyReLU, self).__init__()
    def forward(self, x):
        x = torch.where(x > 1, 1e-6 * x + 1, torch.where(x < -1, 1e-6 * x - 1, x))
        return x
    
class LTanH(nn.Module):
    def __init__(self):
        super(LTanH, self).__init__()
    def forward(self, x):
        return torch.where(torch.abs(x) > 0.5, nn.functional.tanh(x), x)

activations = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "elu": nn.ELU,
    "sigmoid": nn.Sigmoid,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "crelu": CReLU,
    "prelu": nn.PReLU
}

def xavier_uniform_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        num_layers = m.num_layers
        for layer in range(num_layers):
            for weight in m._all_weights[layer]:
                if "weight" in weight:
                    nn.init.xavier_uniform_(getattr(m, weight))
                if "bias" in weight:
                    nn.init.zeros_(getattr(m, weight))

initialisers = {
    "xavier_uniform": xavier_uniform_init
}

class GaussianMixMixin(GaussianMixin):

    def act(self,
            inputs: Mapping[str, Union[torch.Tensor, Any]],
            role: str = "") -> Tuple[torch.Tensor, Union[torch.Tensor, None], Mapping[str, Union[torch.Tensor, Any]]]:
        """Act stochastically in response to the state of the environment

        :param inputs: Model inputs. The most common keys are:

                       - ``"states"``: state of the environment used to make the decision
                       - ``"taken_actions"``: actions taken by the policy for the given states
        :type inputs: dict where the values are typically torch.Tensor
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        :return: Model output. The first component is the action to be taken by the agent.
                 The second component is the log of the probability density function.
                 The third component is a dictionary containing the mean actions ``"mean_actions"``
                 and extra output values
        :rtype: tuple of torch.Tensor, torch.Tensor or None, and dict

        Example::

            >>> # given a batch of sample states with shape (4096, 60)
            >>> actions, log_prob, outputs = model.act({"states": states})
            >>> print(actions.shape, log_prob.shape, outputs["mean_actions"].shape)
            torch.Size([4096, 8]) torch.Size([4096, 1]) torch.Size([4096, 8])
        """
        # map from states/observations to mean actions and log standard deviations
        mean_actions, log_std, outputs = self.compute(inputs, role)
        # clamp log standard deviations
        if self._g_clip_log_std[role] if role in self._g_clip_log_std else self._g_clip_log_std[""]:
            log_std = torch.clamp(log_std,
                                  self._g_log_std_min[role] if role in self._g_log_std_min else self._g_log_std_min[""],
                                  self._g_log_std_max[role] if role in self._g_log_std_max else self._g_log_std_max[""])

        self._g_log_std[role] = log_std
        self._g_num_samples[role] = mean_actions.shape[0]

        # distribution
        #print(torch.max(torch.abs(mean_actions)), log_std.exp())
        #print(log_std.exp() + 1)
        self._g_distribution[role] = Normal(mean_actions, log_std.exp() + 1e-6)

        # sample using the reparameterization trick (excepted for deterministic eval)
        if self._eval_mode and not self._explore_when_eval:
            actions = mean_actions
        elif not self._eval_mode and "overriding_actions" in inputs:
            actions = inputs["overriding_actions"]
            mean_actions = actions
        else:
            actions = self._g_distribution[role].rsample()

        # clip actions
        if self._g_clip_actions[role] if role in self._g_clip_actions else self._g_clip_actions[""]:
            actions = torch.clamp(actions, min=self.clip_actions_min, max=self.clip_actions_max)

        
            

        # log of the probability density function
        log_prob = self._g_distribution[role].log_prob(inputs.get("taken_actions", actions))
        if "taken_actions" in inputs:
            actions_for_log_prob_clip = inputs.get("taken_actions", None)
        else:
            actions_for_log_prob_clip = actions
        if torch.any(actions_for_log_prob_clip >= self.clip_actions_max):
            log_prob[actions_for_log_prob_clip >= self.clip_actions_max] = torch.log(1-self._g_distribution[role].cdf(self.clip_actions_max)[actions_for_log_prob_clip >= self.clip_actions_max])
        if torch.any(actions_for_log_prob_clip <= self.clip_actions_min):
            log_prob[actions_for_log_prob_clip <= self.clip_actions_min] = torch.log(self._g_distribution[role].cdf(self.clip_actions_min)[actions_for_log_prob_clip <= self.clip_actions_min])
        
        reduction = self._g_reduction[role] if role in self._g_reduction else self._g_reduction[""]
        if reduction is not None:
            log_prob = reduction(log_prob, dim=-1)
        if log_prob.dim() != actions.dim():
            log_prob = log_prob.unsqueeze(-1)

        outputs["mean_actions"] = mean_actions
        outputs["log_std"] = log_std
        return actions, log_prob, outputs


# define abstract approximator (stochastic and deterministic models) using mixins
class Approximator(Model):
    def __init__(self, observation_space, action_space, device,
                 num_envs=1, rnn_num_layers=1, rnn_hidden_size=64, sequence_length=128, rnn_layer:str="lstm",
                 net_hidden_size=64, net_num_layers=1, before_net_hidden_size=64, include_actions_in_input=False, before_net_num_layers=0, net_activation="relu", before_net_activation="relu", provide_hidden_state_at_last_layer=False, initializer="xavier_uniform", rnn_initializer="xavier_uniform"):
        self._separated_state_exogenous = type(observation_space) == Dict
        Model.__init__(self, observation_space, action_space, device)
        #GaussianMixMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        self._rnn_layer = d_rnn[rnn_layer]()
        self.num_envs = num_envs
        self.rnn_num_layers = rnn_num_layers
        self.rnn_hidden_size = rnn_hidden_size  # Hout
        self.sequence_length = sequence_length
        self._net_hidden_size=net_hidden_size
        self._net_num_layers=net_num_layers
        self._eval_mode = False
        self._provide_hidden_state_at_last_layer = provide_hidden_state_at_last_layer
        self._before_net_num_layers = before_net_num_layers
        self._before_net_hidden_size = before_net_hidden_size
        self._initializer = initialisers[initializer]
        self._include_actions_in_input = include_actions_in_input
        self._rnn_initializer = initialisers[rnn_initializer]
        if net_activation == "crelu":
            current_hidden_size_multiplier = 2
        else:
            current_hidden_size_multiplier = 1

        if before_net_activation == "crelu":
            current_hidden_size_multiplier_before = 2
        else:
            current_hidden_size_multiplier_before = 1
        net_activation = activations[net_activation]
        before_net_activation = activations[before_net_activation]

        self.net_before = None
        if self._before_net_num_layers > 0:
            if self._separated_state_exogenous:
                self._input_net_before_size = observation_space["states"].shape[0]
            else:
                self._input_net_before_size = self.num_observations
            before_net_layers = [nn.Linear(self._input_net_before_size, self._before_net_hidden_size), before_net_activation()]
            current_hidden_size_before = self._before_net_hidden_size*current_hidden_size_multiplier_before
            for _ in range(self._before_net_num_layers-1):
                before_net_layers += [nn.Linear(current_hidden_size_before, current_hidden_size_before), before_net_activation()]
                current_hidden_size_before = current_hidden_size_before*current_hidden_size_multiplier_before
            self.net_before = nn.Sequential(*before_net_layers)
            self.net_before.apply(self._initializer)

        if self._separated_state_exogenous:
            if self._before_net_num_layers == 0:
                self._input_rnn_size = current_hidden_size_before
            else:
                self._input_rnn_size = observation_space["exogenous"].shape[0]
            if provide_hidden_state_at_last_layer:
                self._input_net_size = observation_space["states"].shape[0]
            else:
                self._input_net_size = self.rnn_hidden_size + observation_space["states"].shape[0]
        else:
            if self._before_net_num_layers > 0:
                self._input_rnn_size = current_hidden_size_before
            else:
                self._input_rnn_size = self.num_observations
            self._input_net_size = self.rnn_hidden_size
        self._rnn_layer_obj = self._rnn_layer.create_rnn(self._input_rnn_size,
                                   self.rnn_hidden_size,
                                   self.rnn_num_layers,
                                   sequence_length=sequence_length,
                                   num_envs=num_envs)  # batch_first -> (batch, sequence, features)
        self._rnn_layer_obj.apply(self._rnn_initializer)

        
        self.net = None
        
        current_hidden_size = None
        if net_num_layers > 0:
            net_layers = [nn.Linear(self._input_net_size, self._net_hidden_size), net_activation()]
            
            current_hidden_size = self._net_hidden_size*current_hidden_size_multiplier
            for _ in range(self._net_num_layers-1):
                net_layers += [nn.Linear(current_hidden_size, current_hidden_size), net_activation()]
                current_hidden_size = current_hidden_size*current_hidden_size_multiplier
            #net_layers += [
            #    nn.Linear(current_hidden_size, 1)
            #]
            self.net = nn.Sequential(*net_layers)
            self.net.apply(self._initializer)
            
            
            

        

        if not self._provide_hidden_state_at_last_layer or not self._separated_state_exogenous:
            if current_hidden_size is None:
                current_hidden_size = rnn_hidden_size
            else:
                current_hidden_size = (self._before_net_hidden_size if self._before_net_num_layers > 0 else self._input_net_size)
        else:
            if current_hidden_size is None:
                current_hidden_size = rnn_hidden_size
            else:
                current_hidden_size = rnn_hidden_size + (self._before_net_hidden_size if self._before_net_num_layers > 0 else self._input_net_size)
        self._complete_output_network(current_hidden_size)

    def _complete_output_network(self, input_size):
        raise NotImplementedError()
    
    def _get_output_network(self, input_data):
        raise NotImplementedError()

    def get_specification(self):
        # batch size (N) is the number of envs
        return self._rnn_layer.create_specification()
    
    def change_num_envs(self, num_envs:int):
        self.num_envs = num_envs
        self._rnn_layer._num_envs = num_envs
    
    def set_eval_mode(self):
        self._eval_mode = True

    def set_train_mode(self):
        self._eval_mode = False

    def compute(self, inputs, role):



        if self._separated_state_exogenous:
            inputs_rnn = dict(inputs)
            inputs_rnn["states"] = inputs["states"][:, :self._input_rnn_size]
            rnn_output, new_hidden_states = self._rnn_layer.forward_rnn(inputs_rnn, is_training=self.training)
            rnn_output, new_hidden_states = self._rnn_layer.postprocess_rnn_output(rnn_output, new_hidden_states)
            if not self._provide_hidden_state_at_last_layer:
                if self.net_before is not None:
                    input_net_to_cat = self.net_before(inputs["states"][:, self._input_rnn_size:])
                else:
                    input_net_to_cat = inputs["states"][:, self._input_rnn_size:]
                net_input = torch.cat([rnn_output,  input_net_to_cat], axis=1)
            else:
                net_input = inputs["states"][:, self._input_rnn_size:]
        else:
            rnn_inputs = dict(inputs)
            if self.net_before is not None:
                rnn_inputs["states"] = self.net_before(rnn_inputs["states"])
            rnn_output, new_hidden_states = self._rnn_layer.forward_rnn(rnn_inputs, is_training=self.training)
            rnn_output, new_hidden_states = self._rnn_layer.postprocess_rnn_output(rnn_output, new_hidden_states)
            net_input = rnn_output
        if self.net is not None:
            hidden_out = self.net(net_input)
        else:
            hidden_out = net_input
        if self._provide_hidden_state_at_last_layer and self._separated_state_exogenous:
            hidden_out = torch.cat([hidden_out, rnn_output], axis=1)
        return self._get_output_network(hidden_out, new_hidden_states)
        

# define models (stochastic and deterministic models) using mixins
class Policy(GaussianMixMixin, Approximator):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum",
                 num_envs=1, rnn_num_layers=1, rnn_hidden_size=64, sequence_length=128, rnn_layer:str="lstm",
                 net_hidden_size=64, net_num_layers=1, before_net_hidden_size=64, before_net_num_layers=0, net_activation="relu", before_net_activation="relu", divide_last_layer_by=1.0, explore_when_eval=True, state_dependent_std=False, tanh_on_output_action=False, provide_hidden_state_at_last_layer=False, initializer="xavier_uniform", rnn_initializer="xavier_uniform", clip_distribution=True):
        self._separated_state_exogenous = type(observation_space) == Dict
        self._explore_when_eval = explore_when_eval
        self._state_dependent_std = state_dependent_std
        self._tanh_on_output_action=tanh_on_output_action
        self._divide_last_layer_by=divide_last_layer_by
        Approximator.__init__(
            self,
            observation_space,
            action_space,
            device,
            num_envs=num_envs,
            rnn_num_layers=rnn_num_layers,
            sequence_length=sequence_length,
            rnn_layer=rnn_layer,
            net_hidden_size=net_hidden_size,
            net_num_layers=net_num_layers,
            before_net_hidden_size=before_net_hidden_size,
            net_activation=net_activation,
            provide_hidden_state_at_last_layer=provide_hidden_state_at_last_layer,
            initializer=initializer,
            before_net_num_layers=before_net_num_layers,
            before_net_activation=before_net_activation,
            rnn_hidden_size=rnn_hidden_size,
            include_actions_in_input=False,
            rnn_initializer=rnn_initializer
        )
        self._clip_distribution = clip_distribution
        GaussianMixMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        

    def _complete_output_network(self, input_size):
        self.action = nn.Linear(input_size, (self.num_actions*(1 if not self._state_dependent_std else 2)))
        self._initializer(self.action)
        if self._divide_last_layer_by > 1.0:
            with torch.no_grad():
                self.action.weight /= self._divide_last_layer_by
                self.action.bias /= self._divide_last_layer_by
        if not self._state_dependent_std:
            self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))
    
    def _get_output_network(self, input_data, hidden_states):
        action = self.action(input_data)
        if not self._state_dependent_std:
            mean_actions = action
            log_std_actions = self.log_std_parameter
        else:
            mean_actions, log_std_actions = torch.split(action, self.num_actions, dim=1)
        if self._tanh_on_output_action:
            mean_actions = torch.tanh(mean_actions)
        else:
            pass
        return mean_actions, log_std_actions, hidden_states

class Value(DeterministicMixin, Approximator):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 num_envs=1, rnn_num_layers=1, rnn_hidden_size=64, sequence_length=128, rnn_layer:str="lstm",
                 net_hidden_size=64, net_num_layers=1, before_net_hidden_size=64, before_net_num_layers=0, include_actions_in_input=True, net_activation="relu", before_net_activation="relu", provide_hidden_state_at_last_layer=False, initializer="xavier_uniform", rnn_initializer="xavier_uniform"):
        Approximator.__init__(
            self,
            observation_space,
            action_space,
            device,
            num_envs=num_envs,
            rnn_num_layers=rnn_num_layers,
            sequence_length=sequence_length,
            rnn_layer=rnn_layer,
            net_hidden_size=net_hidden_size,
            net_num_layers=net_num_layers,
            before_net_hidden_size=before_net_hidden_size,
            net_activation=net_activation,
            provide_hidden_state_at_last_layer=provide_hidden_state_at_last_layer,
            initializer=initializer,
            before_net_num_layers=before_net_num_layers,
            before_net_activation=before_net_activation,
            include_actions_in_input=include_actions_in_input,
            rnn_hidden_size=rnn_hidden_size,
            rnn_initializer=rnn_initializer
            )
        DeterministicMixin.__init__(self, clip_actions)

    def _complete_output_network(self, input_size):
        self.critic = nn.Linear(input_size, 1)
        self._initializer(self.critic)
    
    def _get_output_network(self, input_data, hidden_states):
       value = self.critic(input_data)
       return value, hidden_states