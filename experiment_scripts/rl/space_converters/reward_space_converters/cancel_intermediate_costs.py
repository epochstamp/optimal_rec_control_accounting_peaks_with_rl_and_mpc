from typing import Any, Dict, List, Tuple
from gym.spaces import Dict as DictSpace, Space
from base.space_converter import SpaceConverter
import numpy as np
from pprint import pprint

from env.rec_env import RecEnv

class CancelIntermediateCosts(SpaceConverter):

    def __init__(self,
                 current_action_space: Space,
                 current_observation_space: Space,
                 current_reward_space: Space,
                 original_action_space: Space,
                 original_observation_space: Space,
                 original_reward_space: Space,
                 involve_peaks=False,
                 Delta_M=1,
                 Delta_P=1):
        self._involve_peaks = involve_peaks
        self._Delta_M = Delta_M
        self._Delta_P = Delta_P
        super().__init__(current_action_space,
                         current_observation_space,
                         current_reward_space,
                         original_action_space,
                         original_observation_space,
                         original_reward_space)
    
    def _convert_reward(self, reward: float | List[float] | Dict[Tuple[str, str] | str, float], observation: int | float | List[float] | List[int] | Dict[str | Tuple[str, str], int | float | List[float] | List[int]] = None, action: int | float | List[float] | List[int] | Dict[str | Tuple[str, str], int | float | List[float] | List[int]] = None, original_action: int | float | List[float] | List[int] | Dict[str | Tuple[str, str], int | float | List[float] | List[int]] = None, original_observation: int | float | List[float] | List[int] | Dict[str | Tuple[str, str], int | float | List[float] | List[int]] = None, original_reward: float | List[float] | Dict[Tuple[str, str] | str, float] = None, backward=False, **kwargs):
        metering_period_counter = kwargs["metering_period_counter"]
        peak_period_counter = kwargs.get("peak_period_counter", 0)
        if not backward:
            if (not self._involve_peaks and metering_period_counter == self._Delta_M) or (self._involve_peaks and peak_period_counter == self._Delta_P):
                costs = kwargs["infos"].get("costs", dict())
                r =  costs.get("original_metering_period_cost", 0.0) + costs.get("original_peak_period_cost", 0.0)
                return r
            else:
                return None
        else:
            return original_reward
        
    def _convert_reward_eval(self, reward: float | List[float] | Dict[Tuple[str, str] | str, float], observation: int | float | List[float] | List[int] | Dict[str | Tuple[str, str], int | float | List[float] | List[int]] = None, action: int | float | List[float] | List[int] | Dict[str | Tuple[str, str], int | float | List[float] | List[int]] = None, original_action: int | float | List[float] | List[int] | Dict[str | Tuple[str, str], int | float | List[float] | List[int]] = None, original_observation: int | float | List[float] | List[int] | Dict[str | Tuple[str, str], int | float | List[float] | List[int]] = None, original_reward: float | List[float] | Dict[Tuple[str, str] | str, float] = None, backward=False, **kwargs):
        metering_period_counter = kwargs["metering_period_counter"]
        peak_period_counter = kwargs.get("peak_period_counter", 0)
        if not backward:
            if (not self._involve_peaks and metering_period_counter == self._Delta_M) or (self._involve_peaks and peak_period_counter == self._Delta_P):
                costs = kwargs["infos"].get("costs", dict())
                r =  costs.get("original_metering_period_cost", 0.0) + costs.get("original_peak_period_cost", 0.0)
                return r
            else:
                return 0.0
        else:
            return original_reward
    
    @staticmethod
    def get_kwargs_from_env_and_previous_converters(
        rec_env: RecEnv,
        previous_converters: List
    ):
        return {
            "involve_peaks": rec_env.involve_peaks,
            "Delta_M": rec_env.Delta_M,
            "Delta_P": rec_env.Delta_P
        }