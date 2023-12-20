from typing import Any
from ray.rllib.core.models.torch.heads import TorchMLPHead
from ray.rllib.core.models.torch.encoder import TorchStatefulActorCriticEncoder
from ray.rllib.core.models.configs import (
                ActorCriticEncoderConfig
            )
import torch as tc
from torch import nn

class TrainableInitialStateWrapper(object):
    
    
    def __init__(self, encoder:TorchStatefulActorCriticEncoder, type_initial_state:str = "trainable_gaussian"):
        self._wrapped_encoder = encoder
        self._type_initial_state = type_initial_state
        self._initial_parametrised_state = None

    def __getattr__(self, attr):
        if "_initial_parametrised_state" in attr:
            return 
        elif "_wrapped_encoder" in attr:
            return self._wrapped_encoder
        elif "_type_initial_state" in attr:
            return self._type_initial_state
        elif "get_initial_state" in attr:
            return self.get_initial_state()
        else:
            return getattr(self._wrapped_encoder, attr)
    
    def get_initial_state(self):
        if self._initial_parametrised_state is None:
            initial_state = self._wrapped_encoder.get_initial_state()
            keys_initial_state = list(initial_state.keys())
            lst_split = [initial_state[k].shape[0] for k in keys_initial_state]
            initial_state_whole = tc.vstack([initial_state[k] for k in keys_initial_state])
            if self._type_initial_state == "trainable_gaussian":
                initial_state_whole_mean = tc.zeros_like(initial_state_whole)
                initial_state_whole_std = tc.multiply(tc.ones_like(initial_state_whole), tc.sqrt(tc.FloatTensor([2.0])))
                parameter = tc.normal(initial_state_whole_mean, initial_state_whole_std)
                self._initial_parametrised_state_parameter = nn.Parameter(parameter)
                initial_parametrised_state_parameter_split = tc.split(self._initial_parametrised_state_parameter, lst_split, 0)
                self._initial_parametrised_state = {
                    key:initial_parametrised_state_parameter_split[i] for i,key in enumerate(keys_initial_state)
                }
            elif self._type_initial_state == "trainable_gaussian_stochastic":
                initial_state_whole_mean = tc.zeros_like(initial_state_whole)
                initial_state_whole_std = tc.multiply(tc.ones_like(initial_state_whole), tc.sqrt(tc.FloatTensor([2.0])))
                initial_parametrised_state_parameter_mean = tc.normal(initial_state_whole_mean, initial_state_whole_std)
                initial_parametrised_state_parameter_std = initial_state_whole_std
                initial_parametrised_state_parameter_split = tc.split(initial_parametrised_state_parameter_mean, lst_split, 0)
                initial_parametrised_state_parameter_split = tc.split(initial_parametrised_state_parameter_std, lst_split, 0)
                self._initial_parametrised_state = {
                    key:(initial_parametrised_state_parameter_mean[i], initial_parametrised_state_parameter_std[i]) for i,key in enumerate(keys_initial_state)
                }
            #print(initial_state_whole.shape)
            #print(tc.zeros_like(initial_state_whole).shape)
            #print(tc.ones_like(initial_state_whole).shape)
            #print((tc.ones_like(initial_state_whole)*tc.sqrt(tc.FloatTensor(2.0)).shape))
            
            
            #self._initial_parametrised_state = 
        if self._type_initial_state == "trainable_gaussian_stochastic":
            initial_parametrised_state = {
                    key:tc.unsqueeze(tc.distributions.Normal(param[0], tc.exp(param[1])+1e-6).sample(), axis=0) for key, param in self._initial_parametrised_state.items()
                }
        else:
            initial_parametrised_state = self._initial_parametrised_state
        return initial_parametrised_state
    
    def __call__(self, batch):
        return self._wrapped_encoder(batch)