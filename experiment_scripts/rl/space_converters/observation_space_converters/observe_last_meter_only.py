from typing import Any, Dict, Union, List, Tuple, Callable
from gym.spaces import Dict as DictSpace, Box, Space, Discrete, Tuple as TupleSpace
from base.space_converter import SpaceConverter
import numpy as np
from .resize_and_pad_meters_observations import ResizeAndPadMetersObservations

from utils.utils import normalize_bounds

class ObserveLastMeterOnly(ResizeAndPadMetersObservations):

    def __init__(self,
                 current_action_space: Space,
                 current_observation_space: Space,
                 current_reward_space: Space,
                 original_action_space: Space,
                 original_observation_space: Space,
                 original_reward_space: Space,
                 members: List[str] = [],
                 Delta_P=None):
        super().__init__(current_action_space,
                         current_observation_space,
                         current_reward_space,
                         original_action_space,
                         original_observation_space,
                         original_reward_space,
                         members=members,
                         Delta_P=Delta_P,
                         number_of_past_sequence_data=1)