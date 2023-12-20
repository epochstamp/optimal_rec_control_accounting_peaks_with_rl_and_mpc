from ray.rllib.algorithms.dreamer.dreamer_model import DreamerModel
import torch

def register_classic_torch_rnn_model():
    from ray.rllib.models import ModelCatalog
    from ray.rllib.examples.models.rnn_model import TorchRNNModel
    ModelCatalog.register_custom_model(
        "my_model", TorchRNNModel
    )

def register_custom_torch_rnn_model():
    from ray.rllib.models import ModelCatalog
    from experiment_scripts.rl.rnn.custom_rnn_model import CustomRNNModel
    ModelCatalog.register_custom_model(
        "my_model", CustomRNNModel
    )


def register_custom_torch_dnc_model():
    from ray.rllib.models import ModelCatalog
    from ray.rllib.examples.models.neural_computer import DNCMemory
    ModelCatalog.register_custom_model(
        "my_model", DNCMemory
    )

def rec_2_custom_default():
    register_custom_torch_rnn_model()
    return {
        "custom_model": "my_model",
        "max_seq_len": 50,
        # Extra kwargs to be passed to your model's c'tor.
        "custom_model_config": {
            "hiddens_fc_actor": [64, 64],
            "hiddens_fc_critic": [64, 64],
            "actor_activation": "tanh",
            "critic_activation": "tanh"
        }     
    }

def rec_2_custom_trainable_initial_state():
    register_custom_torch_rnn_model()
    return {
        "custom_model": "my_model",
        "max_seq_len": 50,
        
        # Extra kwargs to be passed to your model's c'tor.
        "custom_model_config": {
            "train_initial_state": True
        }     
    }

def rec_2_custom_default_short_mem():
    register_custom_torch_rnn_model()
    return {
        "custom_model": "my_model",
        "max_seq_len": 25,
        # Extra kwargs to be passed to your model's c'tor.
        "custom_model_config": {

        }     
    }


def rec_2_default():
    return {
        "use_lstm": True,
        # Max seq len for training the LSTM, defaults to 20.
        "max_seq_len": 50,
        # Size of the LSTM cell.
        "lstm_cell_size": 256,
        # Number of hidden layers to be used.
        "fcnet_hiddens": [64, 64],
        # Whether to feed a_{t-1} to LSTM (one-hot encoded if discrete).
        "lstm_use_prev_action": False,
        # Whether to feed r_{t-1} to LSTM.
        "lstm_use_prev_reward": False,
        "is_custom": False,
        "fcnet_activation": "tanh",
        "vf_share_layers": True
    }

def rec_2_custom():
    return {
        "use_lstm": {
            "num_layers": 1,
            "rnn_type": "gru"
        },
        #"use_lstm": True,
        # Max seq len for training the LSTM, defaults to 20.
        "max_seq_len": 50,
        # Size of the LSTM cell.
        "lstm_cell_size": 256,
        # Number of hidden layers to be used.
        "post_fcnet_hiddens": {
            "vf": [64, 64],
            "vf_layer_norm": True,
            "pi": [64, 64],
            "pi_layer_norm": True
        },
        "post_fcnet_activation": {
            "vf": "tanh",
            "pi": "tanh"
        },
        "fcnet_hiddens": [],
        # Whether to feed a_{t-1} to LSTM (one-hot encoded if discrete).
        "lstm_use_prev_action": False,
        # Whether to feed r_{t-1} to LSTM.
        "lstm_use_prev_reward": False,
        "vf_share_layers": True,
        "is_custom": True
    }

def rec_2_default_separated_vf():
    return {
        **rec_2_default(),
        **{
            "vf_share_layers": False,
        }
    }
"""
def rec_2_default():
    return {
        "use_lstm": {
            "num_layers": 1,
            "rnn_type": "gru"
        },
        #"use_lstm": True,
        # Max seq len for training the LSTM, defaults to 20.
        "max_seq_len": 50,
        # Size of the LSTM cell.
        "lstm_cell_size": 256,
        # Number of hidden layers to be used.
        "post_fcnet_hiddens": {
            "vf": [64, 64],
            "pi": [64, 64]
        },
        "post_fcnet_activation": {
            "vf": "tanh",
            "pi": "tanh"
        },
        "fcnet_hiddens": [],
        # Whether to feed a_{t-1} to LSTM (one-hot encoded if discrete).
        "lstm_use_prev_action": False,
        # Whether to feed r_{t-1} to LSTM.
        "lstm_use_prev_reward": False,
        "vf_share_layers": True
    }
"""

def rec_2_default_short_mem():
    return {
        "use_lstm": True,
        # Max seq len for training the LSTM, defaults to 20.
        "max_seq_len": 25,
        # Size of the LSTM cell.
        "lstm_cell_size": 256,
        # Number of hidden layers to be used.
        "fcnet_hiddens": [64, 64],
        # Whether to feed a_{t-1} to LSTM (one-hot encoded if discrete).
        "lstm_use_prev_action": False,
        # Whether to feed r_{t-1} to LSTM.
        "lstm_use_prev_reward": False
    }

def rec_2_default_free_log_std():
    return {
        **rec_2_default(),
        **{
            "free_log_std": True
        }
    }

def rec_2_transformer_default():
    return {
        # Number of hidden layers to be used.
        "fcnet_hiddens": [64, 64],
        "fcnet_activation": "swish",
        # == Attention Nets (experimental: torch-version is untested) ==
        # Whether to use a GTrXL ("Gru transformer XL"; attention net) as the
        # wrapper Model around the default Model.
        "use_attention": True,
        # The number of transformer units within GTrXL.
        # A transformer unit in GTrXL consists of a) MultiHeadAttention module and
        # b) a position-wise MLP.
        "attention_num_transformer_units": 8,
        # The input and output size of each transformer unit.
        "attention_dim": 64,
        # The number of attention heads within the MultiHeadAttention units.
        "attention_num_heads": 8,
        # The dim of a single head (within the MultiHeadAttention units).
        "attention_head_dim": 64,
        # The memory sizes for inference and training.
        "attention_memory_inference": 25,
        "attention_memory_training": 25,
        # The output dim of the position-wise MLP.
        "attention_position_wise_mlp_dim": 64,
        # The initial bias values for the 2 GRU gates within a transformer unit.
        "attention_init_gru_gate_bias": 2.0,
        # Whether to feed a_{t-n:t-1} to GTrXL (one-hot encoded if discrete).
        "attention_use_n_prev_actions": 0,
        # Whether to feed r_{t-n:t-1} to GTrXL.
        "attention_use_n_prev_rewards": 0
    }

def rec_2_short_mem_relu():
    return {
        **rec_2_default_short_mem(),
        **{
            "fcnet_activation": "relu"
        }
    }

def rec_2_swish():
    return {
        **rec_2_default(),
        **{
            "fcnet_activation": "swish"
        }
    }

def rec_2_short_mem_swish():
    return {
        **rec_2_default_short_mem(),
        **{
            "fcnet_activation": "swish"
        }
    }

def rec_2_short_mem_elu():
    return {
        **rec_2_default_short_mem(),
        **{
            "fcnet_activation": "elu"
        }
    }

def rec_2_elu():
    return {
        **rec_2_default(),
        **{
            "fcnet_activation": "elu"
        }
    }

def rec_2_relu():
    return {
        **rec_2_default(),
        **{
            "fcnet_activation": "relu"
        }
    }

def rec_2_relu_free_log_std():
    return {
        **rec_2_relu(),
        **{
            "free_log_std": True
        }
    }

def rec_2_relu_tanh():
    return {
        **rec_2_relu(),
        **{
            "post_fcnet_hiddens": [64],
            "post_fcnet_activation": "tanh",
        }
    }

def rec_2_elu_separated_vf():
    return {
        **rec_2_elu(),
        **{
            "vf_share_layers": False
        }
    }

def rec_2_short_mem_elu_separated_vf():
    return {
        **rec_2_short_mem_elu(),
        **{
            "vf_share_layers": False
        }
    }

def rec_2_relu_separated_vf():
    return {
        **rec_2_relu(),
        **{
            "vf_share_layers": False
        }
    }

def rec_2_short_mem_relu_separated_vf():
    return {
        **rec_2_short_mem_relu(),
        **{
            "vf_share_layers": False
        }
    }

def rec_2_swish_separated_vf():
    return {
        **rec_2_swish(),
        **{
            "vf_share_layers": False
        }
    }

def rec_2_short_mem_swish_separated_vf():
    return {
        **rec_2_short_mem_swish(),
        **{
            "vf_share_layers": False
        }
    }

def rec_2_relu_tanh_separated_vf():
    return {
        **rec_2_relu_tanh(),
        **{
            "vf_share_layers": False
        }
    }

def rec_2_relu_separated_vf_and_free_log():
    return {
        **rec_2_relu_separated_vf(),
        **{
            "free_log_std": True,
        }
    }

def rec_2_default_short_mem_separated_vf():
    return {
        **rec_2_default_short_mem(),
        **{
            "vf_share_layers": False,
        }
    }

def rec_2_default_separated_vf_and_free_log():
    return {
        **rec_2_default_separated_vf(),
        **{
            "free_log_std": True,
        }
    }

def rec_2_custom_like_original():
    register_custom_torch_rnn_model()
    return {
        "custom_model": "my_model",
        "max_seq_len": 50,
        # Extra kwargs to be passed to your model's c'tor.
        "custom_model_config": {
            
            "hiddens_fc_actor": [64, 64],
            "hiddens_fc_critic": [64, 64],
            "actor_activation": "tanh",
            "critic_activation": "tanh",
            "hiddens_fc_actor_exogenous": [],
            "hiddens_fc_critic_exogenous": [],
             "actor_activation_exogenous": "tanh",
            "critic_activation_exogenous": "tanh",
            "hiddens_fc_actor_after": [],
            "hiddens_fc_critic_after": [],
            
            "actor_activation_after": "tanh",
            "critic_activation_after": "tanh",
            "rnn_state_size_actor": 256,
            "rnn_state_size_critic": 256,
            "rnn_type": "lstm",
            "separate_actor_critic_rnn": False,
            "separate_actor_critic_mlp": True
        
        }     
    }


def rec_28_default():
    return {
        "use_lstm": True,
        # Max seq len for training the LSTM, defaults to 20.
        "max_seq_len": 1440,
        # Size of the LSTM cell.
        "lstm_cell_size": 256,
        # Number of hidden layers to be used.
        "fcnet_hiddens": [128, 128],
        # Whether to feed a_{t-1} to LSTM (one-hot encoded if discrete).
        "lstm_use_prev_action": False,
        # Whether to feed r_{t-1} to LSTM.
        "lstm_use_prev_reward": False
    }

def rec_28_short_mem_default():
    return {
        "use_lstm": True,
        # Max seq len for training the LSTM, defaults to 20.
        "max_seq_len": 720,
        # Size of the LSTM cell.
        "lstm_cell_size": 256,
        # Number of hidden layers to be used.
        "fcnet_hiddens": [128, 128],
        # Whether to feed a_{t-1} to LSTM (one-hot encoded if discrete).
        "lstm_use_prev_action": False,
        # Whether to feed r_{t-1} to LSTM.
        "lstm_use_prev_reward": False
    }

def rec_28_default_separated_vf():
    return {
        **rec_28_default(),
        **{
            "vf_share_layers": False
        }
    }

def rec_28_short_mem_default_separated_vf():
    return {
        **rec_28_short_mem_default(),
        **{
            "vf_share_layers": False
        }
    }

def rec_28_swish():
    return {
        **rec_28_default(),
        **{
            "fcnet_activation": "swish"
        }
    }

def rec_28_short_mem_swish():
    return {
        **rec_28_short_mem_default(),
        **{
            "fcnet_activation": "swish"
        }
    }

def rec_28_swish_separated_vf():
    return {
        **rec_28_swish(),
        **{
            "vf_share_layers": False
        }
    }

def rec_28_short_mem_swish_separated_vf():
    return {
        **rec_28_short_mem_swish(),
        **{
            "vf_share_layers": False
        }
    }

def short_horizon_rec_7_from_rec_28_summer_end_default_short_mem():
    return {
        "use_lstm": True,
        # Max seq len for training the LSTM, defaults to 20.
        "max_seq_len": 120,
        # Size of the LSTM cell.
        "lstm_cell_size": 256,
        # Number of hidden layers to be used.
        "fcnet_hiddens": [64, 64],
        # Whether to feed a_{t-1} to LSTM (one-hot encoded if discrete).
        "lstm_use_prev_action": False,
        # Whether to feed r_{t-1} to LSTM.
        "lstm_use_prev_reward": False
    }

def short_horizon_rec_7_from_rec_28_summer_end_default():
    return {
        "use_lstm": {
            "num_layers": 1,
            "rnn_type": "gru"
        },
        #"use_lstm": True,
        # Max seq len for training the LSTM, defaults to 20.
        "max_seq_len": 360,
        # Size of the LSTM cell.
        "lstm_cell_size": 512,
        # Number of hidden layers to be used.
        "post_fcnet_hiddens": {
            "vf": [512, 512],
            "vf_layer_norm": True,
            "pi": [64, 64],
            "pi_layer_norm": True
        },
        "post_fcnet_activation": {
            "vf": "elu",
            "pi": "tanh"
        },
        "fcnet_hiddens": [],
        # Whether to feed a_{t-1} to LSTM (one-hot encoded if discrete).
        "lstm_use_prev_action": False,
        # Whether to feed r_{t-1} to LSTM.
        "lstm_use_prev_reward": False,
        "vf_share_layers": True,
        "is_custom": True
    }

def short_horizon_rec_7_from_rec_28_summer_end_default_gaussian_init_trainable_state():
    return {
        **short_horizon_rec_7_from_rec_28_summer_end_default(),
        **{
            "initial_state_type": "trainable_gaussian_stochastic",
            "use_lstm": {
                "num_layers": 1,
                "rnn_type": "lstm"
            }
        }
    }

def short_horizon_rec_7_from_rec_28_summer_end_default_separated_vf():
    return {
        **short_horizon_rec_7_from_rec_28_summer_end_default(),
        **{
            "vf_share_layers": False
        }
    }

def short_horizon_rec_2_from_rec_28_summer_end_default():
    return {
        "use_lstm": True,
        # Max seq len for training the LSTM, defaults to 20.
        "max_seq_len": 50,
        # Size of the LSTM cell.
        "lstm_cell_size": 256,
        # Number of hidden layers to be used.
        "fcnet_hiddens": [64, 64],
        # Whether to feed a_{t-1} to LSTM (one-hot encoded if discrete).
        "lstm_use_prev_action": False,
        # Whether to feed r_{t-1} to LSTM.
        "lstm_use_prev_reward": False
    }

def short_horizon_rec_2_from_rec_28_summer_end_default_separated_vf():
    return {
        **short_horizon_rec_2_from_rec_28_summer_end_default(),
        **{
            "vf_share_layers": False
        }
    }

def short_horizon_rec_7_from_rec_28_summer_end_default_separated_vf_prev_action():
    return {
        **short_horizon_rec_7_from_rec_28_summer_end_default_separated_vf(),
        **{
            "lstm_use_prev_action": True
        }
    }

def short_horizon_rec_7_from_rec_28_summer_end_default_bigger():
    return {
        "use_lstm": True,
        # Max seq len for training the LSTM, defaults to 20.
        "max_seq_len": 120,
        # Size of the LSTM cell.
        "lstm_cell_size": 1024,
        # Number of hidden layers to be used.
        "fcnet_hiddens": [1024, 1024],
        # Whether to feed a_{t-1} to LSTM (one-hot encoded if discrete).
        "lstm_use_prev_action": False,
        # Whether to feed r_{t-1} to LSTM.
        "lstm_use_prev_reward": False
    }

def short_horizon_rec_7_from_rec_28_summer_end_default_separated_vf_bigger():
    return {
        **short_horizon_rec_7_from_rec_28_summer_end_default(),
        **{
            "vf_share_layers": "False"
        }
    }

def short_horizon_rec_7_from_rec_28_summer_end_transformer_default():
    return {
        # Number of hidden layers to be used.
        "fcnet_hiddens": [64, 64],
        "fcnet_activation": "tanh",
        # == Attention Nets (experimental: torch-version is untested) ==
        # Whether to use a GTrXL ("Gru transformer XL"; attention net) as the
        # wrapper Model around the default Model.
        "use_attention": True,
        # The number of transformer units within GTrXL.
        # A transformer unit in GTrXL consists of a) MultiHeadAttention module and
        # b) a position-wise MLP.
        "attention_num_transformer_units": 2,
        # The input and output size of each transformer unit.
        "attention_dim": 64,
        # The number of attention heads within the MultiHeadAttention units.
        "attention_num_heads": 2,
        # The dim of a single head (within the MultiHeadAttention units).
        "attention_head_dim": 32,
        # The memory sizes for inference and training.
        "attention_memory_inference": 360,
        "attention_memory_training": 180,
        # The output dim of the position-wise MLP.
        "attention_position_wise_mlp_dim": 32,
        # The initial bias values for the 2 GRU gates within a transformer unit.
        "attention_init_gru_gate_bias": 2.0,
        # Whether to feed a_{t-n:t-1} to GTrXL (one-hot encoded if discrete).
        "attention_use_n_prev_actions": 2,
        # Whether to feed r_{t-n:t-1} to GTrXL.
        "attention_use_n_prev_rewards": 0,
        "vf_share_layers": False
    }

def very_short_horizon_rec_2_from_rec_28_summer_end_default():
    return {
        "use_lstm": True,
        # Max seq len for training the LSTM, defaults to 20.
        "max_seq_len": 45,
        # Size of the LSTM cell.
        "lstm_cell_size": 128,
        # Number of hidden layers to be used.
        "fcnet_hiddens": [64, 64],
        # Whether to feed a_{t-1} to LSTM (one-hot encoded if discrete).
        "lstm_use_prev_action": True,
        # Whether to feed r_{t-1} to LSTM.
        "lstm_use_prev_reward": False,
        "vf_share_layers": False
    }

def very_short_horizon_rec_7_from_rec_28_summer_end_default():
    return {
        "use_lstm": True,
        # Max seq len for training the LSTM, defaults to 20.
        "max_seq_len": 90,
        # Size of the LSTM cell.
        "lstm_cell_size": 256,
        # Number of hidden layers to be used.
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "swish",
        "post_fcnet_hiddens": [512, 512],
        "post_fcnet_activation": "swish",
        # Whether to feed a_{t-1} to LSTM (one-hot encoded if discrete).
        "lstm_use_prev_action": True,
        # Whether to feed r_{t-1} to LSTM.
        "lstm_use_prev_reward": False,
        "vf_share_layers": False,
        "_time_major": True
    }

def very_short_horizon_rec_7_from_rec_28_summer_end_custom_gru_nbrc():
    register_custom_torch_rnn_model()
    return {
        "custom_model": "my_model",
        "max_seq_len": 90,
        # Extra kwargs to be passed to your model's c'tor.
        "custom_model_config": {
            "hiddens_fc_actor": [512],
            "hiddens_fc_critic": [512],
            "actor_activation": "gelu",
            "critic_activation": "gelu",
            "hiddens_fc_actor_exogenous": [512],
            "hiddens_fc_critic_exogenous": [512],
            "actor_activation_exogenous": "gelu",
            "critic_activation_exogenous": "gelu",
            "hiddens_fc_actor_after": [512],
            "hiddens_fc_critic_after": [512],
            
            "actor_activation_after": "gelu",
            "critic_activation_after": "gelu",
            "rnn_state_size_actor": 256,
            "rnn_state_size_critic": 256,
            "rnn_type": "nbrc_gru",
            "separate_actor_critic_rnn": False,
            "separate_actor_critic_mlp": True,
            "ortho_init": False
        }     
    }

def very_short_horizon_rec_7_from_rec_28_summer_end_custom_gru():
    register_custom_torch_rnn_model()
    return {
        "custom_model": "my_model",
        "max_seq_len": 90,
        # Extra kwargs to be passed to your model's c'tor.
        "custom_model_config": {
            "hiddens_fc_actor": [256],
            "hiddens_fc_critic": [256],
            "actor_activation": "gelu",
            "critic_activation": "gelu",
            "hiddens_fc_actor_exogenous": [],
            "hiddens_fc_critic_exogenous": [],
            "actor_activation_exogenous": "tanh",
            "critic_activation_exogenous": "tanh",
            "hiddens_fc_actor_after": [512]*5,
            "hiddens_fc_critic_after": [512]*5,
            
            "actor_activation_after": "gelu",
            "critic_activation_after": "gelu",
            "rnn_state_size_actor": 512,
            "rnn_state_size_critic": 512,
            "rnn_type": "nbrc_gru",
            "separate_actor_critic_rnn": True,
            "separate_actor_critic_mlp": True,
            "layer_norm": True
        }     
    }

def short_horizon_rec_2_from_rec_28_summer_end_custom():
    register_custom_torch_rnn_model()
    return {
        "custom_model": "my_model",
        "max_seq_len": 360,
        # Extra kwargs to be passed to your model's c'tor.
        "custom_model_config": {
            
        }     
    }

def short_horizon_rec_7_from_rec_28_summer_end_custom():
    register_custom_torch_rnn_model()
    return {
        "custom_model": "my_model",
        "max_seq_len": 360,
        # Extra kwargs to be passed to your model's c'tor.
        "custom_model_config": {
            "hiddens_fc_actor": [64, 64],
            "hiddens_fc_critic": [256, 256],
            "actor_activation": "tanh",
            "critic_activation": "silu",
            "hiddens_fc_actor_exogenous": [256],
            "hiddens_fc_critic_exogenous": [256],
            "actor_activation_exogenous": "tanh",
            "critic_activation_exogenous": "silu",
            "hiddens_fc_actor_after": [256],
            "hiddens_fc_critic_after": [256],
            
            "actor_activation_after": "tanh",
            "critic_activation_after": "tanh",
            "rnn_state_size_actor": 512,
            "rnn_state_size_critic": 512,
        }     
    }

def short_horizon_rec_7_from_rec_28_summer_end_custom_gru():
    register_custom_torch_rnn_model()
    return {
        "custom_model": "my_model",
        "max_seq_len": 360,
        # Extra kwargs to be passed to your model's c'tor.
        "custom_model_config": {
            "hiddens_fc_actor": [256],
            "hiddens_fc_critic": [256],
            "actor_activation": "tanh",
            "critic_activation": "tanh",
            "hiddens_fc_actor_exogenous": [256],
            "hiddens_fc_critic_exogenous": [256],
            "actor_activation_exogenous": "tanh",
            "critic_activation_exogenous": "tanh",
            "hiddens_fc_actor_after": [256],
            "hiddens_fc_critic_after": [256],
            
            "actor_activation_after": "tanh",
            "critic_activation_after": "tanh",
            "rnn_state_size_actor": 512,
            "rnn_state_size_critic": 512,
            "rnn_type": "gru"
        }     
    }

def short_horizon_rec_7_from_rec_28_summer_end_custom_gru_nbrc():
    register_custom_torch_rnn_model()
    return {
        "custom_model": "my_model",
        "max_seq_len": 360,
        # Extra kwargs to be passed to your model's c'tor.
        "custom_model_config": {
            "hiddens_fc_actor": [64]*2,
            "hiddens_fc_critic": [256]*2,
            "actor_activation": "tanh",
            "critic_activation": "tanh",
            "hiddens_fc_actor_exogenous": [64]*2,
            "hiddens_fc_critic_exogenous": [256]*2,
            "actor_activation_exogenous": "tanh",
            "critic_activation_exogenous": "tanh",
            "hiddens_fc_actor_after": [],
            "hiddens_fc_critic_after": [],
            
            "actor_activation_after": "tanh",
            "critic_activation_after": "silu",
            "rnn_state_size_actor": 512,
            "rnn_state_size_critic": 512,
            "rnn_type": "nbrc_gru",
            "separate_actor_critic_rnn": True,
            "separate_actor_critic_mlp": True
        }     
    }

def short_horizon_rec_7_from_rec_28_summer_end_custom_large_gru():
    register_custom_torch_rnn_model()
    return {
        "custom_model": "my_model",
        "max_seq_len": 128,
        # Extra kwargs to be passed to your model's c'tor.
        "custom_model_config": {
            
            "hiddens_fc_actor": [64, 64]*2,
            "hiddens_fc_critic": [256, 256]*2,
            "actor_activation": "tanh",
            "critic_activation": "relu",
            "hiddens_fc_actor_exogenous": [64, 64]*2,
            "hiddens_fc_critic_exogenous": [256, 256]*2,
            "actor_activation_exogenous": "tanh",
            "critic_activation_exogenous": "relu",
            "hiddens_fc_actor_after": [64, 64]*2,
            "hiddens_fc_critic_after": [256, 256]*2,
            
            "actor_activation_after": "tanh",
            "critic_activation_after": "relu",
            "rnn_state_size_actor": 512,
            "rnn_state_size_critic": 512,
            "rnn_type": "gru",
            "rnn_n_layers_actor": 1,
            "rnn_n_layers_critic": 1,
            "separate_actor_critic_rnn": False,
            "separate_actor_critic_mlp": True
        
        }      
    }

def short_horizon_rec_7_from_rec_28_summer_end_custom_large_nbrc():
    register_custom_torch_rnn_model()
    return {
        "custom_model": "my_model",
        "max_seq_len": 512,
        # Extra kwargs to be passed to your model's c'tor.
        "custom_model_config": {
            
            "hiddens_fc_actor": [64, 64],
            "hiddens_fc_critic": [64, 64],
            "actor_activation": "tanh",
            "critic_activation": "tanh",
            "hiddens_fc_actor_exogenous": [64, 64],
            "hiddens_fc_critic_exogenous": [64, 64],
             "actor_activation_exogenous": "tanh",
            "critic_activation_exogenous": "tanh",
            "hiddens_fc_actor_after": [64, 64],
            "hiddens_fc_critic_after": [64, 64],
            
            "actor_activation_after": "tanh",
            "critic_activation_after": "tanh",
            "rnn_state_size_actor": 512,
            "rnn_state_size_critic": 512,
            "rnn_type": "nbrc",
            "rnn_n_layers_actor": 2,
            "rnn_n_layers_critic": 2,
            "separate_actor_critic_rnn": False,
            "separate_actor_critic_mlp": True
        
        }      
    }

def short_horizon_rec_7_from_rec_28_summer_end_custom_large_brc():
    register_custom_torch_rnn_model()
    return {
        "custom_model": "my_model",
        "max_seq_len": 512,
        # Extra kwargs to be passed to your model's c'tor.
        "custom_model_config": {
            
            "hiddens_fc_actor": [64, 64],
            "hiddens_fc_critic": [64, 64],
            "actor_activation": "tanh",
            "critic_activation": "tanh",
            "hiddens_fc_actor_exogenous": [64, 64],
            "hiddens_fc_critic_exogenous": [64, 64],
             "actor_activation_exogenous": "tanh",
            "critic_activation_exogenous": "tanh",
            "hiddens_fc_actor_after": [64, 64],
            "hiddens_fc_critic_after": [64, 64],
            
            "actor_activation_after": "tanh",
            "critic_activation_after": "tanh",
            "rnn_state_size_actor": 512,
            "rnn_state_size_critic": 512,
            "rnn_type": "brc",
            "rnn_n_layers_actor": 2,
            "rnn_n_layers_critic": 2,
            "separate_actor_critic_rnn": False,
            "separate_actor_critic_mlp": True
        
        }      
    }

def short_horizon_rec_7_from_rec_28_summer_end_custom_brc():
    register_custom_torch_rnn_model()
    return {
        "custom_model": "my_model",
        "max_seq_len": 360,
        # Extra kwargs to be passed to your model's c'tor.
        "custom_model_config": {
            "hiddens_fc_actor": [],
            "hiddens_fc_critic": [],
            "hiddens_fc_actor_after": [64, 64],
            "hiddens_fc_critic_after": [256, 256],
            "actor_activation_after": "tanh",
            "critic_activation_after": "tanh",
            "rnn_state_size_actor": 256,
            "rnn_state_size_critic": 256,
            "rnn_type": "brc"
        }     
    }

def short_horizon_rec_7_from_rec_28_summer_end_custom_nbrc():
    register_custom_torch_rnn_model()
    return {
        "custom_model": "my_model",
        "max_seq_len": 512,
        # Extra kwargs to be passed to your model's c'tor.
        "custom_model_config": {
            "hiddens_fc_actor": [],
            "hiddens_fc_critic": [],
            "hiddens_fc_actor_after": [64, 64],
            "hiddens_fc_critic_after": [256, 256],
            "actor_activation_after": "tanh",
            "critic_activation_after": "relu",
            "rnn_state_size_actor": 512,
            "rnn_state_size_critic": 512,
            "rnn_type": "nbrc"
        }     
    }

def short_horizon_rec_2_from_rec_28_summer_end_custom_brc():
    register_custom_torch_rnn_model()
    return {
        "custom_model": "my_model",
        "max_seq_len": 360,
        
        # Extra kwargs to be passed to your model's c'tor.
        "custom_model_config": {
            "rnn_type": "brc"
        }     
    }

def short_horizon_rec_2_from_rec_28_summer_end_custom_nbrc():
    register_custom_torch_rnn_model()
    return {
        "custom_model": "my_model",
        "max_seq_len": 360,
        
        # Extra kwargs to be passed to your model's c'tor.
        "custom_model_config": {
            "rnn_type": "nbrc"
        }     
    }

def short_horizon_rec_7_from_rec_28_summer_end_custom_dnc():
    register_custom_torch_dnc_model()
    return {
        "custom_model": "my_model",
        "max_seq_len": 360,
        
        # Extra kwargs to be passed to your model's c'tor.
        "custom_model_config": {
            "num_hidden_layers": 4,
            "hidden_size": 128,
            "num_layers": 1,
            "read_heads": 4,
            "nr_cells": 128,
            "cell_size": 128,
            "preprocessor_input_size": 128,
            "preprocessor_output_size": 128,
            "preprocessor": torch.nn.Sequential(torch.nn.Linear(128, 128), torch.nn.Tanh()),

        }     
    }

models_zoo = {
    "rec_2_custom_default": rec_2_custom_default,
    "rec_2_custom_default_short_mem": rec_2_custom_default_short_mem,
    "rec_2_custom_trainable_initial_state": rec_2_custom_trainable_initial_state,
    "rec_2_default": rec_2_default,
    "rec_2_default_short_mem": rec_2_default_short_mem,
    "rec_2_transformer_default": rec_2_transformer_default,
    "rec_2_relu": rec_2_relu,
    "rec_2_relu_separated_vf": rec_2_relu_separated_vf,
    "rec_2_short_mem_relu": rec_2_short_mem_relu,
    "rec_2_short_mem_relu_separated_vf": rec_2_short_mem_relu_separated_vf,
    "rec_2_elu": rec_2_elu,
    "rec_2_short_mem_elu": rec_2_short_mem_elu,
    "rec_2_elu_separated_vf": rec_2_elu_separated_vf,
    "rec_2_short_mem_elu_separated_vf": rec_2_short_mem_elu_separated_vf,
    "rec_2_swish_separated_vf": rec_2_swish_separated_vf,
    "rec_2_short_mem_swish_separated_vf": rec_2_short_mem_swish_separated_vf,
    "rec_2_relu_separated_vf_and_free_log": rec_2_relu_separated_vf_and_free_log,
    "rec_2_default_separated_vf": rec_2_default_separated_vf,
    "rec_2_default_short_mem_separated_vf": rec_2_default_short_mem_separated_vf,
    "rec_2_default_separated_vf_and_free_log": rec_2_default_separated_vf_and_free_log,
    "rec_2_relu_tanh": rec_2_relu_tanh,
    "rec_2_relu_tanh_separated_vf": rec_2_relu_tanh_separated_vf,
    "rec_2_relu_free_log_std": rec_2_relu_free_log_std,
    "rec_2_default_free_log_std": rec_2_default_free_log_std,
    "rec_2_custom": rec_2_custom,
    "rec_2_custom_like_original": rec_2_custom_like_original,
    "rec_28_default": rec_28_default,
    "rec_28_short_mem_default": rec_28_short_mem_default,
    "rec_28_swish_separated_vf": rec_28_swish_separated_vf,
    "rec_28_short_mem_swish_separated_vf": rec_28_short_mem_swish_separated_vf,
    "rec_28_swish": rec_28_swish,
    "rec_28_short_mem_swish": rec_28_short_mem_swish,
    "rec_28_default_separated_vf": rec_28_default_separated_vf,
    "rec_28_short_mem_default_separated_vf": rec_28_short_mem_default_separated_vf,
    "short_horizon_rec_7_from_rec_28_summer_end_default": short_horizon_rec_7_from_rec_28_summer_end_default,
    "short_horizon_rec_7_from_rec_28_summer_end_default_separated_vf": short_horizon_rec_7_from_rec_28_summer_end_default_separated_vf,
    "short_horizon_rec_7_from_rec_28_summer_end_default_bigger": short_horizon_rec_7_from_rec_28_summer_end_default_bigger,
    "short_horizon_rec_7_from_rec_28_summer_end_default_separated_vf_bigger": short_horizon_rec_7_from_rec_28_summer_end_default_separated_vf_bigger,
    "short_horizon_rec_7_from_rec_28_summer_end_default_separated_vf_prev_action": short_horizon_rec_7_from_rec_28_summer_end_default_separated_vf_prev_action,
    "short_horizon_rec_7_from_rec_28_summer_end_transformer_default": short_horizon_rec_7_from_rec_28_summer_end_transformer_default,
    "short_horizon_rec_7_from_rec_28_summer_end_default_short_mem": short_horizon_rec_7_from_rec_28_summer_end_default_short_mem,
    "short_horizon_rec_2_from_rec_28_summer_end_default": short_horizon_rec_2_from_rec_28_summer_end_default,
    "short_horizon_rec_2_from_rec_28_summer_end_default_separated_vf": short_horizon_rec_2_from_rec_28_summer_end_default_separated_vf,
    "short_horizon_rec_2_from_rec_28_summer_end_custom": short_horizon_rec_2_from_rec_28_summer_end_custom,
    "short_horizon_rec_2_from_rec_28_summer_end_custom_brc": short_horizon_rec_2_from_rec_28_summer_end_custom_brc,
    "short_horizon_rec_2_from_rec_28_summer_end_custom_nbrc": short_horizon_rec_2_from_rec_28_summer_end_custom_nbrc,
    "very_short_horizon_rec_2_from_rec_28_summer_end_default": very_short_horizon_rec_2_from_rec_28_summer_end_default,
    "short_horizon_rec_7_from_rec_28_summer_end_custom": short_horizon_rec_7_from_rec_28_summer_end_custom,
    "short_horizon_rec_7_from_rec_28_summer_end_custom_gru": short_horizon_rec_7_from_rec_28_summer_end_custom_gru,
    "short_horizon_rec_7_from_rec_28_summer_end_custom_brc": short_horizon_rec_7_from_rec_28_summer_end_custom_brc,
    "short_horizon_rec_7_from_rec_28_summer_end_custom_nbrc": short_horizon_rec_7_from_rec_28_summer_end_custom_nbrc,
    "short_horizon_rec_7_from_rec_28_summer_end_custom_dnc": short_horizon_rec_7_from_rec_28_summer_end_custom_dnc,
    "short_horizon_rec_7_from_rec_28_summer_end_custom_large_gru": short_horizon_rec_7_from_rec_28_summer_end_custom_large_gru,
    "short_horizon_rec_7_from_rec_28_summer_end_custom_large_nbrc": short_horizon_rec_7_from_rec_28_summer_end_custom_large_nbrc,
    "short_horizon_rec_7_from_rec_28_summer_end_custom_large_brc": short_horizon_rec_7_from_rec_28_summer_end_custom_large_brc,
    "short_horizon_rec_7_from_rec_28_summer_end_custom_gru_nbrc": short_horizon_rec_7_from_rec_28_summer_end_custom_gru_nbrc,
    "very_short_horizon_rec_7_from_rec_28_summer_end_default": very_short_horizon_rec_7_from_rec_28_summer_end_default,
    "very_short_horizon_rec_7_from_rec_28_summer_end_custom_gru_nbrc": very_short_horizon_rec_7_from_rec_28_summer_end_custom_gru_nbrc,
    "very_short_horizon_rec_7_from_rec_28_summer_end_custom_gru": very_short_horizon_rec_7_from_rec_28_summer_end_custom_gru,
    "short_horizon_rec_7_from_rec_28_summer_end_default_gaussian_init_trainable_state": short_horizon_rec_7_from_rec_28_summer_end_default_gaussian_init_trainable_state

}


def short_horizon_rec_7_from_rec_28_summer_end_dreamer_default():
    return {
            "custom_model": DreamerModel,
            # RSSM/PlaNET parameters
            "deter_size": 200,
            "stoch_size": 30,
            # CNN Decoder Encoder
            "depth_size": 32,
            # General Network Parameters
            "hidden_size": 400,
            # Action STD
            "action_init_std": 5.0,
        }
dreamer_models_zoo = {
    "short_horizon_rec_7_from_rec_28_summer_end_dreamer_default": short_horizon_rec_7_from_rec_28_summer_end_dreamer_default
}

def very_short_horizon_rec_7_from_rec_28_summer_end_skrl_ppo_crelu():
    return {
        "policy":{ 
            "clip_actions":True,
            "clip_log_std":True,
            "min_log_std":-20,
            "max_log_std":2,
            "reduction":"none", 
            "rnn_num_layers":1,
            "net_hidden_size":256,
            "rnn_hidden_size": 128,
            "sequence_length":180,
            "rnn_layer": "gru",
            "net_activation": "crelu",
            "net_num_layers": 5
        },
        "value_function":{ 
            "clip_actions":True,
            "rnn_num_layers":1,
            "net_hidden_size":256,
            "rnn_hidden_size": 128,
            "sequence_length":180,
            "rnn_layer": "gru",
            "net_activation": "crelu",
            "net_num_layers": 5
        }
    }

def very_short_horizon_rec_7_from_rec_28_summer_end_skrl_ppo_hybrid():
    return {
        "policy":{ 
            "clip_actions":True,
            "clip_log_std":True,
            "min_log_std":-20,
            "max_log_std":2,
            "reduction":"none", 
            "rnn_num_layers":1,
            "net_hidden_size":64,
            "rnn_hidden_size": 256,
            "sequence_length":180,
            "rnn_layer": "gru",
            "net_activation": "tanh",
            "net_num_layers": 2
        },
        "value_function":{ 
            "clip_actions":True,
            "rnn_num_layers":1,
            "net_hidden_size":128,
            "rnn_hidden_size": 128,
            "sequence_length":180,
            "rnn_layer": "gru",
            "net_activation": "crelu",
            "net_num_layers": 5
        }
    }

def short_horizon_rec_7_from_rec_28_summer_end_skrl_ppo_default():
    return {
        "policy":{ 
            "clip_actions":True,
            "clip_log_std":True,
            "clip_distribution": True,
            "min_log_std":-20,
            "max_log_std":0.01,
            "reduction":"mean", 
            "rnn_num_layers":1,
            "before_net_hidden_size": 64,
            "before_net_num_layers": 2,
            "before_net_activation": "tanh",
            "net_hidden_size":512,
            "rnn_hidden_size": 512,
            "sequence_length":180,
            "provide_hidden_state_at_last_layer": True,
            "rnn_layer": "gru",
            "net_activation": "tanh",
            "net_num_layers": 0
        },
        "value_function":{ 
            "clip_actions":True,
            "rnn_num_layers":1,
            "before_net_hidden_size": 256,
            "before_net_num_layers": 2,
            "before_net_activation": "tanh",
            "net_hidden_size":512,
            "rnn_hidden_size": 512,
            "sequence_length":180,
            "provide_hidden_state_at_last_layer": True,
            "rnn_layer": "gru",
            "net_activation": "tanh",
            "net_num_layers": 0
        }
    }

def short_horizon_rec_7_from_rec_28_summer_end_skrl_ppo_shared():
    return {
        "policy_value":{ 
            "clip_actions":True,
            "clip_log_std":True,
            "clip_distribution": True,
            "min_log_std":-20,
            "max_log_std":1,
            "reduction":"mean", 
            "rnn_num_layers":2,
            "before_net_hidden_size": 512,
            "before_net_num_layers": 2,
            "before_net_activation": "tanh",
            "net_hidden_size":512,
            "rnn_hidden_size": 512,
            "sequence_length":360,
            "provide_hidden_state_at_last_layer": True,
            "rnn_layer": "gru",
            "net_activation": "tanh",
            "net_num_layers": 0
        }
    }

def very_short_horizon_rec_7_from_rec_28_summer_end_skrl_ppo_default():
    return {
        "policy":{ 
            "clip_actions":True,
            "clip_log_std":True,
            "clip_distribution": True,
            "min_log_std":-20,
            "max_log_std":2,
            "reduction":"mean", 
            "rnn_num_layers":1,
            "before_net_hidden_size": 256,
            "before_net_num_layers": 2,
            "before_net_activation": "tanh",
            "net_hidden_size":64,
            "rnn_hidden_size": 512,
            "sequence_length":360,
            "rnn_layer": "gru",
            "net_activation": "tanh",
            "net_num_layers": 0
        },
        "value_function":{ 
            "clip_actions":True,
            "rnn_num_layers":1,
            "before_net_hidden_size": 128,
            "before_net_num_layers": 2,
            "before_net_activation": "tanh",
            "net_hidden_size":64,
            "rnn_hidden_size": 256,
            "sequence_length":90,
            "rnn_layer": "lstm",
            "net_activation": "tanh",
            "net_num_layers": 0
        }
    }

def rec_2_skrl_ppo_default():
    return {
        "policy":{ 
            "clip_actions":True,
            "clip_log_std":True,
            "min_log_std":-20,
            "max_log_std":2,
            "reduction":"mean", 
            "rnn_num_layers":1,
            "before_net_hidden_size": 64,
            "before_net_num_layers": 2,
            "before_net_activation": "tanh",
            "net_hidden_size":64,
            "rnn_hidden_size": 512,
            "sequence_length":180,
            "rnn_layer": "lstm",
            "net_activation": "tanh",
            "net_num_layers": 0
        },
        "value_function":{ 
            "clip_actions":True,
            "rnn_num_layers":1,
            "net_hidden_size":256,
            "rnn_hidden_size": 128,
            "sequence_length":180,
            "rnn_layer": "gru",
            "net_activation": "tanh",
            "net_num_layers": 2
        }
    }

skrl_ppo_models_zoo = {
    "very_short_horizon_rec_7_from_rec_28_summer_end_skrl_ppo_default": very_short_horizon_rec_7_from_rec_28_summer_end_skrl_ppo_default,
    "very_short_horizon_rec_7_from_rec_28_summer_end_skrl_ppo_crelu": very_short_horizon_rec_7_from_rec_28_summer_end_skrl_ppo_crelu,
    "very_short_horizon_rec_7_from_rec_28_summer_end_skrl_ppo_hybrid": very_short_horizon_rec_7_from_rec_28_summer_end_skrl_ppo_hybrid,
    "short_horizon_rec_7_from_rec_28_summer_end_skrl_ppo_default": short_horizon_rec_7_from_rec_28_summer_end_skrl_ppo_default,
    "rec_2_skrl_ppo_default": rec_2_skrl_ppo_default
} 