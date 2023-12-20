from typing import Any, Dict, Union, List, Tuple, Callable
from gym.spaces import Dict as DictSpace, Box, Space, Discrete, Tuple as TupleSpace
from base.space_converter import SpaceConverter
import numpy as np
from env.rec_env import RecEnv
from experiment_scripts.rl.space_converters.observation_space_converters.resize_and_pad_exogenous_variables_observations import ResizeAndPadExogenousObservations

from utils.utils import normalize_bounds

class ResizeAndPadExogenousExogenousMembersVariablesHalfPeakPeriod(ResizeAndPadExogenousObservations):

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
                 Delta_P=1,
                 T=1
        ):
        super().__init__(current_action_space,
                         current_observation_space,
                         current_reward_space,
                         original_action_space,
                         original_observation_space,
                         original_reward_space,
                         members=members,
                         exogenous_space={
                             k:v for k,v in exogenous_space.items() if k[1] not in ("buying_price", "selling_price")
                         },
                         number_of_past_sequence_data=(Delta_M*Delta_P)//2,
                         Delta_M=Delta_M,
                         T=T)
    

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
            "Delta_P": (
                rec_env.Delta_P
            ),
            "T": rec_env.T,
            "exogenous_space": rec_env.exogenous_space
        }