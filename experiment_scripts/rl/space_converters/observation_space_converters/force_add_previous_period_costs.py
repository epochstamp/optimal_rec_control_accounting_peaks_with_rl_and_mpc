from typing import Any, Dict, Union, List, Tuple, Callable
from gym.spaces import Dict as DictSpace, Box, Space, Discrete, Tuple as TupleSpace
from base.space_converter import SpaceConverter
import numpy as np
from env.rec_env import RecEnv

from utils.utils import normalize_bounds

class ForceAddPreviousPeriodCosts(SpaceConverter):

    def __init__(self,
                 current_action_space: Space,
                 current_observation_space: Space,
                 current_reward_space: Space,
                 original_action_space: Space,
                 original_observation_space: Space,
                 original_reward_space: Space,
                 Delta_M=2,
                 Delta_P=1,
                 involve_peaks=False,
                 force_previous_peak_cost_to_zero=False):
        self._Delta_M=Delta_M
        self._Delta_P=Delta_P
        self._involve_peaks=involve_peaks
        self._force_previous_peak_cost_to_zero = force_previous_peak_cost_to_zero
        super().__init__(current_action_space,
                         current_observation_space,
                         current_reward_space,
                         original_action_space,
                         original_observation_space,
                         original_reward_space)
    
    def _convert_observation_space(self):
        if not self._force_previous_peak_cost_to_zero and (("previous_metering_period_cost" in list(self._original_observation_space.keys())) or ("previous_peak_period_cost" in list(self._original_observation_space.keys()))):
            return self._current_observation_space
        if type(self._current_observation_space) == Box:
            lows = np.hstack(
                [self._current_observation_space.low, np.asarray([-1000000, -1000000])]
            )
            highs = np.hstack(
                [self._current_observation_space.high, np.asarray([1000000, 1000000])]
            )
            return Box(low=np.asarray(lows), high=np.asarray(highs))
        elif type(self._current_observation_space) == TupleSpace:
            return TupleSpace(tuple(self._current_observation_space) + tuple([Box(low=-1000000, high=1000000), Box(low=-1000000, high=1000000)]))
        elif type(self._current_observation_space) == DictSpace:
            d_space = dict(self._current_observation_space)
            d_space["previous_metering_period_cost"] = Box(low=-1000000, high=1000000)
            d_space["previous_peak_period_cost"] = Box(low=-1000000, high=1000000)
            return DictSpace(d_space)
        else:
            raise BaseException(f"Space type {type(self._action_space)} is not handled for ForceAddPreviousPeriodCosts")
    
    def convert_observation(self,
                       observation: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]],
                       action: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       reward: Union[float, List[float], Dict[Union[Tuple[str, str], str], float]]=None,
                       original_action: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       original_observation: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       original_reward: Union[float, List[float], Dict[Union[Tuple[str, str], str], float]]=None,
                       backward=False,
                       **kwargs):
        
        if not self._force_previous_peak_cost_to_zero and (("previous_metering_period_cost" in list(self._original_observation_space.keys())) or ("previous_peak_period_cost" in list(self._original_observation_space.keys()))):
            return observation
        if backward:
            if type(self._current_observation_space) == Box:
                return observation[:-2]
            elif type(self._current_observation_space) == TupleSpace:
                return observation[:-2]
            elif type(self._current_observation_space) == DictSpace:
                d_observation = dict(observation)
                d_observation.pop("previous_metering_period_cost", None)
                d_observation.pop("previous_peak_period_cost", None)
                return d_observation
            else:
                raise BaseException(f"Space type {type(self._action_space)} is not handled for ForceAddPreviousPeriodCosts")
        else:
            is_end_of_metering_period = kwargs["metering_period_counter"] == self._Delta_M
            is_end_of_peak_period = kwargs.get("peak_period_counter", 0) == self._Delta_P
            infos = kwargs["infos"]
            is_metering_period_triggered = infos.get("is_metering_period_cost_triggered", False)
            is_peak_period_triggered = infos.get("is_peak_period_cost_triggered", False)
            metering_period_cost = infos.get("costs", {}).get("original_metering_period_cost", 0.0)
            peak_period_cost = infos.get("costs", {}).get("original_metering_period_cost", 0.0)
            if is_metering_period_triggered:
                self._previous_metering_period_cost = float(metering_period_cost * (1-is_end_of_metering_period))
            if is_peak_period_triggered:
                self._previous_peak_period_cost = float(peak_period_cost * (1-is_end_of_peak_period)) * (not self._force_previous_peak_cost_to_zero)

            if type(self._current_observation_space) == Box:
                observation = np.hstack([observation, np.asarray([float(self._previous_metering_period_cost), float(self._previous_peak_period_cost)])])
            elif type(self._current_observation_space) == TupleSpace:
                observation = observation + [self._previous_metering_period_cost, self._previous_peak_period_cost]
            elif type(self._current_observation_space) == DictSpace:
                d_observation = dict(observation)
                d_observation["previous_metering_period_cost"] = self._previous_metering_period_cost
                d_observation["previous_peak_period_cost"] = self._previous_peak_period_cost
                observation = d_observation
            else:
                raise BaseException(f"Space type {type(self._action_space)} is not handled for ForceAddPreviousPeriodCosts")
            
            return observation
        
    def reset(self):
        self._previous_metering_period_cost = 0.0
        self._previous_peak_period_cost = 0.0
        
    @staticmethod
    def get_kwargs_from_env_and_previous_converters(
        rec_env: RecEnv,
        previous_converters: List
    ):
        return {
            "Delta_M": rec_env.Delta_M,
            "Delta_P": rec_env.Delta_P,
            "involve_peaks": rec_env.involve_peaks
        }