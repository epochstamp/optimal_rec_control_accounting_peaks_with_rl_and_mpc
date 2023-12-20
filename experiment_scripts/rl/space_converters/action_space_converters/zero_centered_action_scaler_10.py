from typing import Any, Dict, Union, List, Tuple, Callable
from gym.spaces import Dict as DictSpace, Box, Space, Discrete, Tuple as TupleSpace
from base.space_converter import SpaceConverter
import numpy as np
from experiment_scripts.rl.space_converters.action_space_converters.action_scaler import ActionScaler

from utils.utils import normalize_bounds, normalize_1_1, to_0_1_range, unnormalize_1_1

class ZeroCenteredActionScaler10(ActionScaler):

    def __init__(self,
                 current_action_space: Space,
                 current_observation_space: Space,
                 current_reward_space: Space,
                 original_action_space: Space,
                 original_observation_space: Space,
                 original_reward_space: Space):
        super().__init__(current_action_space,
                         current_observation_space,
                         current_reward_space,
                         original_action_space,
                         original_observation_space,
                         original_reward_space,
                         zero_centering=True,
                         multiplier=10)