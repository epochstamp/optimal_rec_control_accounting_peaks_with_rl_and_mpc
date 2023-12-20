from typing import Any, Dict, Union, List, Tuple, Callable
from gym.spaces import Dict as DictSpace, Box, Space, Discrete, Tuple as TupleSpace
from base.space_converter import SpaceConverter
import numpy as np
from env.rec_env import RecEnv
from experiment_scripts.rl.space_converters.observation_space_converters.resize_and_pad_sequences_observations import ResizeAndPadSequencesObservations

from utils.utils import normalize_bounds

class ResizeAndPadExogenousObservations(ResizeAndPadSequencesObservations):

    def __init__(self,
                 current_action_space: Space,
                 current_observation_space: Space,
                 current_reward_space: Space,
                 original_action_space: Space,
                 original_observation_space: Space,
                 original_reward_space: Space,
                 members: List[str] = [],
                 number_of_past_sequence_data=10000000,
                 exogenous_space: Space = None,
                 Delta_M=1,
                 T=1
        ):
        self._exogenous_space = exogenous_space
        self._number_of_past_exogenous_prices = (min(number_of_past_sequence_data, T)//Delta_M + 1)*(number_of_past_sequence_data > 0)
        self._number_of_past_exogenous_variables_members = min(number_of_past_sequence_data, T)
        super().__init__(current_action_space,
                         current_observation_space,
                         current_reward_space,
                         original_action_space,
                         original_observation_space,
                         original_reward_space,
                         members=members,
                         number_of_past_sequence_data=number_of_past_sequence_data)
        
    def _get_observation_keys(self):
        return list(self._exogenous_space.keys())

    def _compute_number_of_past_sequence_data(self, observation_key=None):

        return self._number_of_past_exogenous_prices if observation_key[1] in ("buying_price", "selling_price") else self._number_of_past_exogenous_variables_members
    

    @staticmethod
    def get_kwargs_from_env_and_previous_converters(
        rec_env: RecEnv,
        previous_converters: List
    ):
        return {
            "members": rec_env.members,
            "Delta_M": (
                rec_env.Delta_M
            ),
            "T": rec_env.T,
            "exogenous_space": rec_env.exogenous_space
        }