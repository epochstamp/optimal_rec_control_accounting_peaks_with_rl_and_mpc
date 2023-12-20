import numpy as np

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from brc_pytorch.layers import BistableRecurrentCell, NeuromodulatedBistableRecurrentCell, MultiLayerBase
from brc_pytorch.layers import MultiLayerBase
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from gymnasium.spaces import Box, MultiDiscrete, Discrete
from torchviz import make_dot
import torch

class L(list):
    """
    A subclass of list that can accept additional attributes.
    Should be able to be used just like a regular list.

    The problem:
    a = [1, 2, 4, 8]
    a.x = "Hey!" # AttributeError: 'list' object has no attribute 'x'

    The solution:
    a = L(1, 2, 4, 8)
    a.x = "Hey!"
    print a       # [1, 2, 4, 8]
    print a.x     # "Hey!"
    print len(a)  # 4

    You can also do these:
    a = L( 1, 2, 4, 8 , x="Hey!" )                 # [1, 2, 4, 8]
    a = L( 1, 2, 4, 8 )( x="Hey!" )                # [1, 2, 4, 8]
    a = L( [1, 2, 4, 8] , x="Hey!" )               # [1, 2, 4, 8]
    a = L( {1, 2, 4, 8} , x="Hey!" )               # [1, 2, 4, 8]
    a = L( [2 ** b for b in range(4)] , x="Hey!" ) # [1, 2, 4, 8]
    a = L( (2 ** b for b in range(4)) , x="Hey!" ) # [1, 2, 4, 8]
    a = L( 2 ** b for b in range(4) )( x="Hey!" )  # [1, 2, 4, 8]
    a = L( 2 )                                     # [2]
    """
    def __new__(self, *args, **kwargs):
        return super(L, self).__new__(self, args, kwargs)

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and hasattr(args[0], '__iter__'):
            list.__init__(self, args[0])
        else:
            list.__init__(self, args)
        self.__dict__.update(kwargs)

    def __call__(self, **kwargs):
        self.__dict__.update(kwargs)
        return self

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

activations = {
    "tanh": nn.functional.tanh,
    "relu": nn.functional.relu,
    "relu6": nn.functional.relu6,
    "elu": nn.functional.elu,
    "sigmoid": nn.functional.sigmoid,
    "silu": nn.functional.silu,
    "gelu": nn.functional.gelu
}

activations_module = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "elu": nn.ELU,
    "sigmoid": nn.Sigmoid,
    "silu": nn.SiLU,
    "gelu": nn.GELU
}

def create_lstm(input_size, output_size, num_layers=1):
    return nn.LSTM(input_size, output_size, num_layers=num_layers, batch_first=True)

def get_lstm_initial_state(size, num_layers=1):
    if num_layers == 1:
        return [
            torch.zeros(size),
            torch.zeros(size)
        ]
    else:
        return [
            [torch.zeros(size),
            torch.zeros(size)] for _ in range(num_layers)
        ]

def forward_lstm(rnn, input, rnn_state):
    return rnn(
        input, [torch.unsqueeze(rnn_state[0], 0), torch.unsqueeze(rnn_state[1], 0)]
    )

def preprocess_lstm_actor_out(rnn_features, rnn_state):
    return rnn_features, [s.squeeze(0) for s in rnn_state]

def split_actor_critic_states_lstm(rnn_state):
    half = len(rnn_state)//2
    return rnn_state[:half], rnn_state[half:]

def create_gru(input_size, output_size, num_layers=1):
    return nn.GRU(input_size, output_size, num_layers=num_layers, batch_first=True)

def get_gru_initial_state(size, num_layers=1):
    return [torch.zeros(size) for _ in range(num_layers)]

def forward_gru(rnn, input, rnn_state):
    if rnn.num_layers == 1:
        return rnn(
            input, torch.unsqueeze(rnn_state[0], 0)
        )
    else:
        features, hidden_state = rnn(
            input, torch.stack(rnn_state)
        )
        return features, list(torch.unbind(hidden_state))

def split_actor_critic_states_gru(rnn_state):
    if len(rnn_state) == 1:
        return rnn_state, None
    else:
        return [rnn_state[0]], [rnn_state[1]]

def preprocess_gru_actor_out(rnn_features, rnn_state):
    return rnn_features, [rnn_state[0]]



def create_brc(input_size, output_size, cls=BistableRecurrentCell, num_layers=1):
    if num_layers == 1:
        brc = cls(input_size, output_size)
        brc.hidden_size=output_size
    else:
        brcs = [cls(input_size, output_size)]
        for _ in range(num_layers):
            brcs.append(cls(output_size, output_size))
        brc = MultiLayerBase(("BRC" if cls==BistableRecurrentCell else "nBRC"), brcs, output_size, batch_first=True)
        brc.hidden_size=output_size
    return brc

def create_nbrc(input_size, output_size, num_layers=1):
    return create_brc(input_size, output_size, cls=NeuromodulatedBistableRecurrentCell, num_layers=num_layers)

def get_brc_initial_state(size, num_layers=1):
    return [
        torch.zeros(size)
    ]

def forward_brc(rnn, input, rnn_state):
    features = rnn.forward(
        input, torch.unsqueeze(rnn_state[0], 1)
    )
    return features, [features]

def preprocess_brc_actor_out(rnn_features, rnn_state):
    return rnn_features, [s.squeeze(1) for s in rnn_state]

def split_actor_critic_states_brc(rnn_state):
    return (rnn_state[0], rnn_state[1]) if len(rnn_state) == 2 else (rnn_state[0], None)


def create_gru_nbrc(input_size, output_size, num_layers=1):
    rnns = L(
        nn.GRU(input_size, output_size//2, batch_first=True, device="cpu"),
        NeuromodulatedBistableRecurrentCell(input_size, output_size//2)
    )
    rnns.hidden_size=output_size
    rnns.num_layers=num_layers
    return rnns

def get_gru_nbrc_initial_state(size, num_layers=1):
    return [
        torch.zeros(size//2),
        torch.zeros(size//2)
    ]

def split_actor_critic_states_gru_nbrc(rnn_state):
    if len(rnn_state) == 2:
        return rnn_state, None
    else:
        half = len(rnn_state)//2
        return rnn_state[:half], rnn_state[half:]

def forward_gru_nbrc(rnn, input, rnn_state):
    features_gru, new_state_gru = rnn[0].forward(
        input, torch.unsqueeze(rnn_state[0], 0)
    )
    
    features_nbrc = rnn[1].forward(
        input, torch.unsqueeze(rnn_state[1], 1)
    )
    new_state_brc = features_nbrc
    features = torch.cat([features_gru, features_nbrc], dim=2)
    return features, [new_state_gru, new_state_brc]

def preprocess_gru_nbrc_actor_out(rnn_features, rnn_state):
    if len(rnn_state) == 2:
        return rnn_features, [rnn_state[0].squeeze(0), rnn_state[1].squeeze(1)]
    else:
        return rnn_features, [rnn_state[0].squeeze(0), rnn_state[1].squeeze(1), rnn_state[2].squeeze(0), rnn_state[3].squeeze(1)]


rnn_functions = {
    "lstm": (
        create_lstm, get_lstm_initial_state, split_actor_critic_states_lstm, forward_lstm, preprocess_lstm_actor_out
    ),
    "gru": (
        create_gru, get_gru_initial_state, split_actor_critic_states_gru, forward_gru, preprocess_gru_actor_out
    ),
    "brc": (
        create_brc, get_brc_initial_state, split_actor_critic_states_brc, forward_brc, preprocess_brc_actor_out
    ),
    "nbrc": (
        create_nbrc, get_brc_initial_state, split_actor_critic_states_brc, forward_brc, preprocess_brc_actor_out
    ),
    "nbrc_gru": (
        create_gru_nbrc, get_gru_nbrc_initial_state, split_actor_critic_states_gru_nbrc, forward_gru_nbrc, preprocess_gru_nbrc_actor_out
    ),

}

def orthogonal_init(layer):
    if type(layer) == nn.Linear:
        nn.init.orthogonal(layer.weight)
        nn.init.zeros(layer.bias)

def create_ortho_init(scale=1.0):
    def ortho_init(layer):
        if type(layer) in (nn.Linear, nn.LSTM, nn.GRU, BistableRecurrentCell, NeuromodulatedBistableRecurrentCell):
            lst_weights = []
            if type(layer) == nn.Linear:
                lst_weights = [layer.weight]
                lst_bias = [layer.bias]
            elif type(layer) in (nn.LSTM, nn.GRU):
                lst_weights = [
                    param for name_param, param in dict(layer.named_parameters()).items() if "weight" in name_param
                ]
                lst_bias = [
                    param for name_param, param in dict(layer.named_parameters()).items() if "bias" in name_param
                ]
            elif type(layer) in (BistableRecurrentCell, NeuromodulatedBistableRecurrentCell):
                lst_weights = [param for name_param, param in dict(layer.named_parameters()).items() if name_param not in ("bz", "br")]
                lst_bias = []
            for weight in lst_weights:
                shape = tuple(weight.shape)
                if len(shape) == 2:
                    flat_shape = shape
                elif len(shape) == 4: # assumes NHWC
                    flat_shape = (np.prod(shape[:-1]), shape[-1])
                else:
                    raise NotImplementedError
                a = np.random.normal(0.0, 1.0, flat_shape)
                u, _, v = np.linalg.svd(a, full_matrices=False)
                q = u if u.shape == flat_shape else v # pick the one with the correct shape
                q = q.reshape(shape)
                q = (scale * q[:shape[0], :shape[1]]).astype(np.float32)
                new_weights = torch.from_numpy(q)
                weight.data = new_weights.clone().detach().requires_grad_(True)
            for bias in lst_bias:
                nn.init.zeros_(bias)

    return ortho_init

class CustomRNNModel(TorchRNN, nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        hiddens_fc_actor=[64, 64],
        actor_activation="tanh",
        hiddens_fc_critic=[256, 256],
        critic_activation="relu",
        hiddens_fc_actor_exogenous=[],
        actor_activation_exogenous="tanh",
        hiddens_fc_critic_exogenous=[],
        critic_activation_exogenous="tanh",
        hiddens_fc_actor_after=[512],
        actor_activation_after="tanh",
        hiddens_fc_critic_after=[512],
        critic_activation_after="relu",
        rnn_type="lstm",
        rnn_state_size_actor=64,
        rnn_state_size_critic=64,
        rnn_n_layers_actor=1,
        rnn_n_layers_critic=1,
        separate_actor_critic_rnn=False,
        separate_actor_critic_mlp=True,
        layer_norm=False,
        ortho_init=False,
        **model_config_kwargs
    ):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        train_initial_state = model_config_kwargs.get("train_initial_state", False)
        self.obs_size = get_preprocessor(obs_space)(obs_space).size
        self._train_initial_state = train_initial_state
        self._create_rnn, self._get_rnn_initial_state, self._split_actor_critic_states_rnn, self._forward_rnn, self._preprocess_rnn_actor_out = rnn_functions[rnn_type]
        self._separate_actor_critic_rnn = separate_actor_critic_rnn
        self._separate_actor_critic_mlp = separate_actor_critic_mlp
        self._layer_norm = layer_norm
        self._ortho_init = ortho_init

        

        self.rnn_state_size_actor = model_config_kwargs.get("rnn_state_size_actor", rnn_state_size_actor)
        self.rnn_state_size_critic = model_config_kwargs.get("rnn_state_size_critic", rnn_state_size_critic)
        self.hiddens_fc_actor = model_config_kwargs.get("hiddens_fc_actor", hiddens_fc_actor)
        self.hiddens_fc_critic = model_config_kwargs.get("hiddens_fc_critic", hiddens_fc_critic)
        self._actor_activation = model_config_kwargs.get("actor_activation", actor_activation)
        self._critic_activation = model_config_kwargs.get("critic_activation", critic_activation)
        self.hiddens_fc_actor_exogenous = model_config_kwargs.get("hiddens_fc_actor", hiddens_fc_actor_exogenous)
        self.hiddens_fc_critic_exogenous = model_config_kwargs.get("hiddens_fc_critic", hiddens_fc_critic_exogenous)
        self._actor_activation_exogenous = model_config_kwargs.get("actor_activation", actor_activation_exogenous)
        self._critic_activation_exogenous = model_config_kwargs.get("critic_activation", critic_activation_exogenous)
        self.hiddens_fc_actor_after = model_config_kwargs.get("hiddens_fc_actor_after", hiddens_fc_actor_after)
        self.hiddens_fc_critic_after = model_config_kwargs.get("hiddens_fc_critic_after", hiddens_fc_critic_after)
        self._actor_activation_after = model_config_kwargs.get("actor_activation_after", actor_activation_after)
        self._critic_activation_after = model_config_kwargs.get("critic_activation_after", critic_activation_after)
        self._num_layers_rnn_actor = rnn_n_layers_actor
        self._num_layers_rnn_critic = rnn_n_layers_critic
        self._state_size = None
        self._exogenous_size = None
        if hasattr(obs_space, "original_space"):
            original_space = obs_space.original_space
            self._state_size = sum([
                (obs.shape[0] if type(obs) == Box else (obs.n if type(obs) == Discrete else (sum(obs.nvec)))) for obs in original_space[:-1]
            ])
            self._exogenous_size = sum([
                (obs.shape[0] if type(obs) == Box else (obs.n if type(obs) == Discrete else (sum(obs.nvec)))) for obs in original_space[-1:]
            ])
            self.fc_actor_state = None
            self.fc_critic_state = None
            self.fc_actor_exogenous = None
            self.fc_critic_exogenous = None

            #Prepare layers for state data
            if len(self.hiddens_fc_actor) > 0:
                self.fc_actor_state = [nn.Linear(self._state_size, self.hiddens_fc_actor[0])]
                if layer_norm:
                    self.fc_actor_state += [nn.LayerNorm(self.hiddens_fc_actor[0])]
                self.fc_actor_state += [activations_module[self._actor_activation]()]
                for i, _ in enumerate(self.hiddens_fc_actor):
                    
                    if i > 0:
                        self.fc_actor_state += [nn.Linear(self.hiddens_fc_actor[i-1], self.hiddens_fc_actor[i])]
                        if layer_norm:
                            self.fc_actor_state += [nn.LayerNorm(self.hiddens_fc_actor[i])]
                        self.fc_actor_state += [activations_module[self._actor_activation]()]
                    
                self.fc_actor_state = nn.Sequential(*self.fc_actor_state)
                if self._ortho_init:
                    self.fc_actor_state.apply(create_ortho_init(np.sqrt(2)))

            if len(self.hiddens_fc_critic) > 0 and self._separate_actor_critic_mlp:
                self.fc_critic_state = [nn.Linear(self._state_size, self.hiddens_fc_critic[0])]
                if layer_norm:
                    self.fc_critic_state += [nn.LayerNorm(self.hiddens_fc_critic[0])]
                self.fc_critic_state += [activations_module[self._critic_activation]()]
                for i, _ in enumerate(self.hiddens_fc_critic):
                    
                    if i > 0:
                        self.fc_critic_state += [nn.Linear(self.hiddens_fc_critic[i-1], self.hiddens_fc_critic[i])]
                        if layer_norm:
                            self.fc_critic_state += [nn.LayerNorm(self.hiddens_fc_critic[i])]
                        self.fc_critic_state += [activations_module[self._critic_activation]()]
                    
                self.fc_critic_state = nn.Sequential(*self.fc_critic_state)
                if self._ortho_init:
                    self.fc_critic_state.apply(create_ortho_init(np.sqrt(2)))

            #Prepare layers for exogenous data
            if len(self.hiddens_fc_actor_exogenous) > 0:
                self.fc_actor_exogenous = [nn.Linear(self._exogenous_size, self.hiddens_fc_actor_exogenous[0])]
                if layer_norm:
                    self.fc_actor_exogenous += [nn.LayerNorm(self.hiddens_fc_actor_exogenous[0])]
                self.fc_actor_exogenous += [activations_module[self._actor_activation_exogenous]()]
                for i, _ in enumerate(self.hiddens_fc_actor_exogenous):
                    if i > 0:
                        self.fc_actor_exogenous += [nn.Linear(self.hiddens_fc_actor_exogenous[i-1], self.hiddens_fc_actor_exogenous[i])]
                        if layer_norm:
                            self.fc_actor_exogenous += [nn.LayerNorm(self.hiddens_fc_actor_exogenous[i])]
                        self.fc_actor_exogenous += [activations_module[self._actor_activation_exogenous]()]
                        
                self.fc_actor_exogenous = nn.Sequential(*self.fc_actor_exogenous)
                if self._ortho_init:
                    self.fc_actor_exogenous.apply(create_ortho_init(np.sqrt(2)))
            if len(self.hiddens_fc_critic_exogenous) > 0 and self._separate_actor_critic_mlp:
                self.fc_critic_exogenous = [nn.Linear(self._exogenous_size, self.hiddens_fc_critic_exogenous[0])]
                if self._layer_norm:
                    self.fc_critic_exogenous += [nn.LayerNorm(self.hiddens_fc_critic_exogenous[0])]
                self.fc_critic_exogenous += [activations_module[self._critic_activation_exogenous]()]
                for i, _ in enumerate(self.hiddens_fc_critic_exogenous):
                    if i > 0:
                        self.fc_critic_exogenous += [nn.Linear(self.hiddens_fc_critic_exogenous[i-1], self.hiddens_fc_critic_exogenous[i])]
                        if layer_norm:
                            self.fc_critic_exogenous += [nn.LayerNorm(self.hiddens_fc_critic_exogenous[i])]
                        self.fc_critic_exogenous += [activations_module[self._critic_activation_exogenous]()]
                self.fc_critic_exogenous = nn.Sequential(*self.fc_critic_exogenous)
                if self._ortho_init:
                    self.fc_actor_exogenous.apply(create_ortho_init(np.sqrt(2)))
            
            if not self._separate_actor_critic_rnn:
                self.rnn_actor = self._create_rnn((self.hiddens_fc_actor_exogenous[-1] if self.fc_actor_exogenous is not None else self._exogenous_size), rnn_state_size_actor, num_layers=rnn_n_layers_actor)
                self.rnn_critic = None
            else:
                self.rnn_actor = self._create_rnn((self.hiddens_fc_actor_exogenous[-1] if self.fc_actor_exogenous is not None else self._exogenous_size), rnn_state_size_actor, num_layers=rnn_n_layers_actor)
                self.rnn_critic = self._create_rnn((self.hiddens_fc_critic_exogenous[-1] if self.fc_critic_exogenous is not None else self._exogenous_size), rnn_state_size_critic, num_layers=rnn_n_layers_critic)
            
            if layer_norm:
                self.rnn_actor_layer_norm = nn.LayerNorm(rnn_state_size_actor)
                self.rnn_critic_layer_norm = None
                if self.rnn_critic is not None:
                    self.rnn_critic_layer_norm = nn.LayerNorm(rnn_state_size_critic)
            #Init rnn actor weights
            if isinstance(self.rnn_actor, nn.Module):
                if type(self.rnn_actor) in (nn.LSTM, nn.GRU, NeuromodulatedBistableRecurrentCell):
                    if self._ortho_init:
                        self.rnn_actor.apply(create_ortho_init(1.0))
                        if self.rnn_critic is not None:
                            self.rnn_critic.apply(create_ortho_init(1.0))
            elif type(self.rnn_actor) in (list, tuple, L):
                for i, rnn_layer in enumerate(self.rnn_actor):
                    if type(rnn_layer) in (nn.LSTM, nn.GRU, NeuromodulatedBistableRecurrentCell):
                        if self._ortho_init:
                            rnn_layer.apply(create_ortho_init(1.0))
                            if self.rnn_critic is not None:
                                self.rnn_critic[i].apply(create_ortho_init(1.0))
            #Init rnn critic weights
            if self.rnn_critic is not None:
                if isinstance(self.rnn_critic, nn.Module):
                    if type(self.rnn_critic) in (nn.LSTM, nn.GRU, NeuromodulatedBistableRecurrentCell):
                        if self._ortho_init:
                            self.rnn_critic.apply(create_ortho_init(1.0))
                            if self.rnn_critic is not None:
                                self.rnn_critic.apply(create_ortho_init(1.0))
                elif type(self.rnn_critic) in (list, tuple, L):
                    for i, rnn_layer in enumerate(self.rnn_critic):
                        if type(rnn_layer) in (nn.LSTM, nn.GRU, NeuromodulatedBistableRecurrentCell):
                            if self._ortho_init:
                                rnn_layer.apply(create_ortho_init(1.0))
                                if self.rnn_critic is not None:
                                    self.rnn_critic[i].apply(create_ortho_init(1.0))
            #Merge state and rnn features to post layers
            self.fc_actor_after = None
            self.fc_critic_after = None
            features_size_actor = (self._state_size if self.fc_actor_state is None else self.hiddens_fc_actor[-1]) + rnn_state_size_actor
            features_size_critic = None
            if separate_actor_critic_rnn:
                features_size_critic = (self._state_size if self.fc_critic_state is None else self.hiddens_fc_critic[-1]) + rnn_state_size_critic
            if len(self.hiddens_fc_actor_after) > 0:
                self.fc_actor_after = [nn.Linear(features_size_actor, self.hiddens_fc_actor_after[0])]
                if layer_norm:
                    self.fc_actor_after += [nn.LayerNorm(self.hiddens_fc_actor_after[0])]
                self.fc_actor_after += [activations_module[self._actor_activation_after]()]
                for i, _ in enumerate(self.hiddens_fc_actor_after):
                    if i > 0:
                        self.fc_actor_after += [nn.Linear(self.hiddens_fc_actor_after[i-1], self.hiddens_fc_actor_after[i])]
                        #initializer_actor(self.fc_actor_after[-2].weight)
                        if layer_norm:
                            self.fc_actor_after += [nn.LayerNorm(self.hiddens_fc_actor_after[i])]
                        self.fc_actor_after += [activations_module[self._actor_activation_after]()]
                self.fc_actor_after = nn.Sequential(*self.fc_actor_after)
                if self._ortho_init:
                    self.fc_actor_after.apply(create_ortho_init(np.sqrt(2)))
            if len(self.hiddens_fc_critic_after) > 0 and self._separate_actor_critic_mlp:
                
                self.fc_critic_after = [nn.Linear(features_size_critic, self.hiddens_fc_critic_after[0])]
                if layer_norm:
                    self.fc_critic_after += [nn.LayerNorm(self.hiddens_fc_critic_after[0])]
                self.fc_critic_after += [activations_module[self._critic_activation_after]()]
                for i, _ in enumerate(self.hiddens_fc_critic_after):
                    if i > 0:
                        self.fc_critic_after += [nn.Linear(self.hiddens_fc_critic_after[i-1], self.hiddens_fc_critic_after[i])]
                        #initializer_critic(self.fc_critic_after[-2].weight)
                        if layer_norm:
                            self.fc_critic_after += [nn.LayerNorm(self.hiddens_fc_critic_after[i])]
                        self.fc_critic_after += [activations_module[self._critic_activation_after]()]
                self.fc_critic_after = nn.Sequential(*self.fc_critic_after)
                if self._ortho_init:
                    self.fc_critic_after.apply(create_ortho_init(np.sqrt(2)))
            if self.fc_actor_after is not None:
                self.action_branch = nn.Linear(self.hiddens_fc_actor_after[-1], num_outputs)
            else:
                self.action_branch = nn.Linear(features_size_actor, num_outputs)
            if self._ortho_init:
                self.action_branch.apply(create_ortho_init(1.0))
            if self.fc_critic_after is not None:
                self.critic_branch = nn.Linear(self.hiddens_fc_critic_after[-1], 1)
            else:
                if self._separate_actor_critic_rnn:
                    self.critic_branch = nn.Linear((features_size_critic if self.fc_actor_after is None else self.hiddens_fc_actor_after[-1]), 1)
                else:
                    self.critic_branch = nn.Linear((features_size_actor if self.fc_actor_after is None else self.hiddens_fc_actor_after[-1]), 1)
            if self._ortho_init:
                self.critic_branch.apply(create_ortho_init(1.0))
            # Holds the current "base" output (before logits layer).
            self._features_actor = None
            self._features_critic = None
            self._h = None
        else:
            self.fc_actor = None
            self.fc_critic = None
            if len(self.hiddens_fc_actor) > 0:
                self.fc_actor = [nn.Linear(self.obs_size, self.hiddens_fc_actor[0]), activations_module[self._actor_activation]()]
                for i, _ in enumerate(self.hiddens_fc_actor):
                    if i > 0:
                        self.fc_actor += [nn.Linear(self.hiddens_fc_actor[i-1], self.hiddens_fc_actor[i]), activations_module[self._actor_activation]()]
                self.fc_actor = nn.Sequential(*self.fc_actor)
                if self._ortho_init:
                    self.fc_actor.apply(create_ortho_init(np.sqrt(2)))
            if len(self.hiddens_fc_critic) > 0 and self._separate_actor_critic_mlp:
                self.fc_critic = [nn.Linear(self.obs_size, self.hiddens_fc_critic[0]), activations_module[self._critic_activation]()]
                for i, _ in enumerate(self.hiddens_fc_critic):
                    if i > 0:
                        self.fc_critic += [nn.Linear(self.hiddens_fc_critic[i-1], self.hiddens_fc_critic[i]), activations_module[self._critic_activation]()]
                self.fc_critic = nn.Sequential(*self.fc_critic)
                if self._ortho_init:
                    self.fc_critic.apply(create_ortho_init(np.sqrt(2)))
            
            if not self._separate_actor_critic_rnn:
                self.rnn_actor = self._create_rnn((self.hiddens_fc_actor[-1] if self.fc_actor is not None else self.obs_size), rnn_state_size_actor)
                self.rnn_critic = None
            else:
                self.rnn_actor = self._create_rnn((self.hiddens_fc_actor[-1] if self.fc_actor is not None else self.obs_size), rnn_state_size_actor)
                self.rnn_critic = self._create_rnn((self.hiddens_fc_critic[-1] if self.fc_critic is not None else self.obs_size), rnn_state_size_critic)
            if self._ortho_init:
                if isinstance(self.rnn_actor, nn.Module):
                    if type(self.rnn_actor) in (nn.LSTM, nn.GRU, NeuromodulatedBistableRecurrentCell):
                        self.rnn_actor.apply(create_ortho_init(1.0))
                        if self.rnn_critic is not None:
                            self.rnn_critic.apply(create_ortho_init(1.0))
                elif type(self.rnn_actor) in (list, tuple, L):
                    for i, rnn_layer in enumerate(self.rnn_actor):
                        if type(rnn_layer) in (nn.LSTM, nn.GRU, NeuromodulatedBistableRecurrentCell):
                            rnn_layer.apply(create_ortho_init(1.0))
                            if self.rnn_critic is not None:
                                self.rnn_critic[i].apply(create_ortho_init(1.0))
            self.fc_actor_after = None
            self.fc_critic_after = None
            if len(self.hiddens_fc_actor_after) > 0:
                self.fc_actor_after = [nn.Linear(rnn_state_size_actor, self.hiddens_fc_actor_after[0]), activations_module[self._actor_activation_after]()]
                for i, hidden_fc_actor_after in enumerate(self.hiddens_fc_actor_after):
                    if i > 0:
                        self.fc_actor_after += [nn.Linear(self.hiddens_fc_actor_after[i-1], self.hiddens_fc_actor_after[i]), activations_module[self._actor_activation_after]()]
                    #initializer_actor(self.fc_actor_after[-2].weight)
                self.fc_actor_after = nn.Sequential(*self.fc_actor_after)
                if self._ortho_init:
                    self.fc_actor_after.apply(create_ortho_init(np.sqrt(2)))
            if len(self.hiddens_fc_critic_after) > 0 and self._separate_actor_critic_mlp:
                self.fc_critic_after = [nn.Linear(rnn_state_size_critic, self.hiddens_fc_critic_after[0]), activations_module[self._critic_activation_after]()]
                for i, hidden_fc_critic_after in enumerate(self.hiddens_fc_critic_after):
                    if i > 0:
                        self.fc_critic_after += [nn.Linear(self.hiddens_fc_critic_after[i-1], self.hiddens_fc_critic_after[i]), activations_module[self._critic_activation_after]()]
                    #initializer_critic(self.fc_critic_after[-2].weight)
                self.fc_critic_after = nn.Sequential(*self.fc_critic_after)
                if self._ortho_init:
                    self.fc_critic_after.apply(create_ortho_init(np.sqrt(2)))
            if self.fc_actor_after is not None:
                self.action_branch = nn.Linear(self.hiddens_fc_actor_after[-1], num_outputs)
            else:
                self.action_branch = nn.Linear(rnn_state_size_actor, num_outputs)
            if self._ortho_init:
                self.action_branch.apply(create_ortho_init(1.0))
            if self.fc_critic_after is not None:
                self.critic_branch = nn.Linear(self.hiddens_fc_critic_after[-1], 1)
            else:
                if self._separate_actor_critic_rnn:
                    self.critic_branch = nn.Linear(rnn_state_size_critic, 1)
                else:
                    self.critic_branch = nn.Linear(rnn_state_size_actor, 1)
            if self._ortho_init:
                self.critic_branch.apply(create_ortho_init(1.0))
            # Holds the current "base" output (before logits layer).
            self._features_actor = None
            self._features_critic = None
            self._h = None

    @override(ModelV2)
    def get_initial_state(self):
        # TODO: (sven): Get rid of `get_initial_state` once Trajectory
        #  View API is supported across all of RLlib.
        # Place hidden states on same device as model.
        if self._separate_actor_critic_rnn or (self._state_size is not None and self.fc_critic_exogenous is not None) or (self._state_size is None and self.fc_critic is not None):
            if self._separate_actor_critic_rnn:
                initial_state = self._get_rnn_initial_state(self.rnn_actor.hidden_size, num_layers=self._num_layers_rnn_actor) + self._get_rnn_initial_state(self.rnn_critic.hidden_size, num_layers=self._num_layers_rnn_critic)
            else:
                initial_state = self._get_rnn_initial_state(self.rnn_actor.hidden_size, num_layers=self._num_layers_rnn_actor) + self._get_rnn_initial_state(self.rnn_actor.hidden_size, num_layers=self._num_layers_rnn_actor)
            self._rnn_state_size = len(initial_state)//2
        else:
            initial_state = self._get_rnn_initial_state(self.rnn_actor.hidden_size, num_layers=self._num_layers_rnn_actor)
            self._rnn_state_size = len(initial_state)
        return initial_state

    @override(ModelV2)
    def value_function(self):
        assert self._features_critic is not None, "must call forward() first"
        x = self._features_critic
        if self.fc_critic_after is not None:
            x = self.fc_critic_after(x)
        elif not self._separate_actor_critic_mlp and self.fc_actor_after is not None:
            x = self.fc_actor_after(x)
        critic_out = self.critic_branch(x)
        critic_out = critic_out.reshape([-1])
        return critic_out

    @override(TorchRNN)
    def forward_rnn(self, inputs, rnn_state, seq_lens):
        """Feeds `inputs` (B x T x ..) through the Gru Unit.

        Returns the resulting outputs as a sequence (B x T x ...).
        Values are stored in self._cur_value in simple (B) shape (where B
        contains both the B and T dims!).

        Returns:
            NN Outputs (B x T x ...) as sequence.
            The rnn_state batches as a List of two items (c- and h-states).
        
        if inputs.shape[0] != 32:
            import dill
            with open("./inputs.dat", "wb") as fileinput:
                dill.dump(inputs, fileinput)
            with open("./state.dat", "wb") as filestate:
                dill.dump(rnn_state, filestate)
            torch.save(self, "./model.m")
            exit()
        """
        

        if self._state_size is not None:
            inputs_state = inputs[..., :self._state_size]
            inputs_exogenous = inputs[..., self._state_size:]
            x_critic_state = inputs_state
            x_actor_state = inputs_state
            x_critic_exogenous = inputs_exogenous
            x_actor_exogenous = inputs_exogenous
            #Process first state and exogenous inputs
            
            if self.fc_actor_state is not None:
                x_actor_state = self.fc_actor_state(x_actor_state)
            if self.fc_critic_state is not None:
                x_critic_state = self.fc_critic_state(x_critic_state)
            else:
                x_critic_state = x_actor_state
            
            
            if self.fc_actor_exogenous is not None:
                x_actor_exogenous = self.fc_actor_exogenous(x_actor_exogenous)
            if self.fc_critic_exogenous is not None:
                x_critic_exogenous = self.fc_critic_exogenous(x_critic_exogenous)
            else:
                x_critic_exogenous = x_actor_exogenous

            #Plug processed exogenous data into rnn
            if self._separate_actor_critic_rnn or self.fc_critic_exogenous is not None:
                rnn_state_actor, rnn_state_critic = self._split_actor_critic_states_rnn(rnn_state)
            else:
                rnn_state_actor = rnn_state
            self._features_actor, new_rnn_state_actor = self._forward_rnn(
                self.rnn_actor, x_actor_exogenous, rnn_state_actor
            )
            if self._layer_norm:
                self._features_actor = self.rnn_actor_layer_norm(self._features_actor)
            if self.rnn_critic is not None:
                self._features_critic, new_rnn_state_critic = self._forward_rnn(
                    self.rnn_critic, x_critic_exogenous, rnn_state_critic
                )
                if self._layer_norm:
                    self._features_critic = self.rnn_critic_layer_norm(self._features_critic)
                self._features_critic = torch.cat([self._features_critic, x_critic_state], axis=2)
                new_rnn_state = new_rnn_state_actor + new_rnn_state_critic
            else:
                if self.fc_critic_exogenous is not None:
                    self._features_critic, new_rnn_state_critic = self._forward_rnn(
                        self.rnn_actor, x_critic_exogenous, rnn_state_critic
                    )
                    if self._layer_norm:
                        self._features_critic = self.rnn_actor_layer_norm(self._features_critic)
                    self._features_critic = torch.cat([self._features_critic, x_critic_state], axis=2)
                    new_rnn_state = new_rnn_state_actor + new_rnn_state_critic
                else:
                    self._features_critic = self._features_actor
                    if self._layer_norm:
                        self._features_critic = self.rnn_actor_layer_norm(self._features_critic)
                    self._features_critic = torch.cat([self._features_critic, x_critic_state], axis=2)
                    new_rnn_state = new_rnn_state_actor

            #Process layers after the rnn
            x_actor = torch.cat([self._features_actor, x_actor_state], axis=2)
            if self.fc_actor_after is not None:
                x_actor = self.fc_actor_after(x_actor)
            action_out = self.action_branch(x_actor)
        else:
            x_critic = inputs
            x_actor = inputs
            if self.fc_critic is not None:
                x_critic = self.fc_critic(x_critic)
            if self.fc_actor is not None:
                x_actor = self.fc_actor(x_actor)
            if self._separate_actor_critic_rnn or self.fc_critic is not None:
                rnn_state_actor, rnn_state_critic = self._split_actor_critic_states_rnn(rnn_state)
            else:
                rnn_state_actor = rnn_state
            self._features_actor, new_rnn_state_actor = self._forward_rnn(
                self.rnn_actor, x_actor, rnn_state_actor
            )
            if self.rnn_critic is not None:
                self._features_critic, new_rnn_state_critic = self._forward_rnn(
                    self.rnn_critic, x_critic, rnn_state_critic
                )
                new_rnn_state = new_rnn_state_actor + new_rnn_state_critic
            else:
                
                if self.fc_critic is not None:
                    self._features_critic, new_rnn_state_critic = self._forward_rnn(
                        self.rnn_actor, x_critic, rnn_state_critic
                    )
                    new_rnn_state = new_rnn_state_actor + new_rnn_state_critic

                else:
                    self._features_critic = self._features_actor
                    new_rnn_state = new_rnn_state_actor
            x_actor = self._features_actor
            if self.fc_actor_after is not None:
                x_actor = self.fc_actor_after(x_actor)
            action_out = self.action_branch(x_actor)
        self._action_out = action_out
        return self._preprocess_rnn_actor_out(action_out, new_rnn_state)

"""
class BRC_RNNModel(TorchRNN, nn.Module):
    
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        fc_sizes_actor=[64, 64],
        fc_sizes_critic=[256, 256],
        brc_state_size_actor=512,
        brc_state_size_critic=512,
        act_func_actor="tanh",
        act_func_critic="relu"
    ):
        #print(obs_space.original_space)
        #raise BaseException()
        #exit()
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self.obs_size = get_preprocessor(obs_space)(obs_space).size

        self.fc_sizes_actor = fc_sizes_actor
        self.fc_sizes_critic = fc_sizes_critic
        self.brc_state_size_actor = brc_state_size_actor
        self.brc_state_size_critic = brc_state_size_critic
        self.act_func_actor = activations[act_func_actor]
        self.act_func_critic = activations[act_func_critic]

        # Build the Module from fc + LSTM + 2xfc (action + value outs).
        self.fcs_actor = [nn.Linear(self.obs_size, self.fc_sizes_actor[0])]
        for i, _ in enumerate(fc_sizes_actor[1:]):
            self.fcs_actor.append(nn.Linear(self.fc_sizes_actor[i], self.fc_sizes_actor[i+1]))

        self.fcs_critic = [nn.Linear(self.obs_size, self.fc_sizes_critic[0])]
        for i, _ in enumerate(fc_sizes_critic[1:]):
            self.fcs_critic.append(nn.Linear(self.fc_sizes_critic[i], self.fc_sizes_critic[i+1]))

        self.brc_actor = BistableRecurrentCell(self.fc_sizes_actor[-1], self.brc_state_size_actor)
        self.brc_critic = BistableRecurrentCell(self.fc_sizes_critic[-1], self.brc_state_size_critic)
        self.action_branch = nn.Linear(self.brc_state_size_actor, num_outputs)
        self.value_branch = nn.Linear(self.brc_state_size_critic, 1)
        # Holds the current "base" output (before logits layer).
        self._features_actor = None
        self._features_critic = None

    @override(ModelV2)
    def get_initial_state(self):
        # TODO: (sven): Get rid of `get_initial_state` once Trajectory
        #  View API is supported across all of RLlib.
        # Place hidden states on same device as model.
        return [
            self.fcs_actor[-1].weight.new(1, self.brc_state_size_actor).zero_().squeeze(0),
            self.fcs_critic[-1].weight.new(1, self.brc_state_size_critic).zero_().squeeze(0)
        ]

    @override(ModelV2)
    def value_function(self):
        assert self._features_critic is not None, "must call forward() first"
        
        critic_out = self.value_branch(self._features_critic)
        critic_out = critic_out.reshape([-1])
        return critic_out

    @override(TorchRNN)
    def forward_rnn(self, inputs, rnn_state, seq_lens):

        x_actor = inputs
        for fc_actor in self.fcs_actor:
            x_actor = self.act_func_actor(fc_actor(x_actor))
        x_critic = inputs
        for fc_critic in self.fcs_critic:
            x_critic = self.act_func_critic(fc_critic(x_critic))
        self._features_actor = self.brc_actor.forward(
            x_actor, rnn_state[0].unsqueeze(1)
        )
        self._features_critic = self.brc_critic.forward(
            x_critic, rnn_state[1].unsqueeze(1)
        )
        action_out = self.action_branch(self._features_actor)
        return action_out, [self._features_actor.squeeze(1), self._features_critic.squeeze(1)]
"""