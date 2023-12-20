from typing import Any, Dict, Union, List, Tuple, Callable
from gym.spaces import Dict as DictSpace, Box, Space, Discrete, Tuple as TupleSpace
from base.space_converter import SpaceConverter
import numpy as np
from env.rec_env import RecEnv
from experiment_scripts.rl.space_converters.observation_space_converters.resize_and_pad_sequences_observations import ResizeAndPadSequencesObservations

from utils.utils import normalize_bounds

class ResizeAndPadMetersObservations(ResizeAndPadSequencesObservations):

    def __init__(self,
                 current_action_space: Space,
                 current_observation_space: Space,
                 current_reward_space: Space,
                 original_action_space: Space,
                 original_observation_space: Space,
                 original_reward_space: Space,
                 members: List[str] = [],
                 Delta_P=None,
                 number_of_past_sequence_data=10000000
        ):
        self._number_of_metering_periods_behind = min(Delta_P if Delta_P is not None else 1, number_of_past_sequence_data)
        super().__init__(current_action_space,
                         current_observation_space,
                         current_reward_space,
                         original_action_space,
                         original_observation_space,
                         original_reward_space,
                         members=members,
                         number_of_past_sequence_data=number_of_past_sequence_data)
        
    def _get_observation_keys(self):
        meter_keys = ["consumption_meters", "production_meters"]
        return [(member, meter_key) for meter_key in meter_keys for member in self._members]

    def _compute_number_of_past_sequence_data(self, observation_key=None):
        return self._number_of_metering_periods_behind

    @staticmethod
    def get_kwargs_from_env_and_previous_converters(
        rec_env: RecEnv,
        previous_converters: List
    ):
        return {
            "members": rec_env.members,
            "Delta_P": (
                rec_env.Delta_P if rec_env.involve_peaks else None
            )
        }