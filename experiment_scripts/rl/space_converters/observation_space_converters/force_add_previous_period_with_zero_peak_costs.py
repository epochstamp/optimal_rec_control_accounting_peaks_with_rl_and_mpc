from typing import Any, Dict, Union, List, Tuple, Callable
from gym.spaces import Dict as DictSpace, Box, Space, Discrete, Tuple as TupleSpace
from base.space_converter import SpaceConverter
import numpy as np
from env.rec_env import RecEnv
from experiment_scripts.rl.space_converters.observation_space_converters.force_add_previous_period_costs import ForceAddPreviousPeriodCosts

from utils.utils import normalize_bounds

class ForceAddPreviousPeriodWithZeroPeakCosts(ForceAddPreviousPeriodCosts):

    def __init__(self,
                 current_action_space: Space,
                 current_observation_space: Space,
                 current_reward_space: Space,
                 original_action_space: Space,
                 original_observation_space: Space,
                 original_reward_space: Space,
                 Delta_M=1,
                 Delta_P=1,
                 involve_peaks=False):
        super().__init__(current_action_space,
                         current_observation_space,
                         current_reward_space,
                         original_action_space,
                         original_observation_space,
                         original_reward_space,
                         Delta_M=Delta_M,
                         Delta_P=Delta_P,
                         involve_peaks=involve_peaks,
                         force_previous_peak_cost_to_zero=True)