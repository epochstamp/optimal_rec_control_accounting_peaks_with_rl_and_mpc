from typing import Any, Dict, Union, List, Tuple, Callable
from gym.spaces import Dict as DictSpace, Box, Space, Discrete, Tuple as TupleSpace
from base.space_converter import SpaceConverter
import numpy as np
from experiment_scripts.rl.space_converters.observation_space_converters.resize_and_pad_exogenous_variables_observations import ResizeAndPadExogenousObservations
from utils.utils import normalize_bounds

class ObserveLastExogenousVariablesOnly(ResizeAndPadExogenousObservations):

    def __init__(self,
                 current_action_space: Space,
                 current_observation_space: Space,
                 current_reward_space: Space,
                 original_action_space: Space,
                 original_observation_space: Space,
                 original_reward_space: Space,
                 members: List[str] = [],
                 exogenous_space: Space = None,
                 Delta_M=1,
                 T=1
        ):
        super().__init__(current_action_space,
                         current_observation_space,
                         current_reward_space,
                         original_action_space,
                         original_observation_space,
                         original_reward_space,
                         members=members,
                         exogenous_space=exogenous_space,
                         Delta_M=Delta_M,
                         T=T,
                         number_of_past_sequence_data=1)