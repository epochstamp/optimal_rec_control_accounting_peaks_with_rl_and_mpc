from typing import Any, Dict, Union, List, Tuple, Callable
from gym.spaces import Dict as DictSpace, Box, Space, Discrete
from base.space_converter import SpaceConverter
import numpy as np
from env.rec_env import RecEnv
from .action_squeeze import ActionSqueeze

def create_battery_squeezer(members: List[str]):
    def battery_squeezer(action_space:DictSpace):
        lst_squeeze = []
        for member in members:
            if (member, "charge") in list(action_space.keys()):
                lst_squeeze.append(
                    ((member, "charge"), (member, "discharge"))
                )
        return lst_squeeze
    return battery_squeezer


class BatteryActionSqueeze(ActionSqueeze):

    def __init__(self,
                 current_action_space: Space,
                 current_observation_space: Space,
                 current_reward_space: Space,
                 original_action_space: Space,
                 original_observation_space: Space,
                 original_reward_space: Space,
                 members=[]):
        super().__init__(current_action_space,
                         current_observation_space,
                         current_reward_space,
                         original_action_space,
                         original_observation_space,
                         original_reward_space,
                         action_space_squeezer=create_battery_squeezer(members))
        

    @staticmethod
    def get_kwargs_from_env_and_previous_converters(
        rec_env: RecEnv,
        previous_converters: List
    ):
        return {
            "members": rec_env.members
        }