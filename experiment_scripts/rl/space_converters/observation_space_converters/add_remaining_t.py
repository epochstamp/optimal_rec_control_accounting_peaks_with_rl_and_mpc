from typing import Any, Dict, Union, List, Tuple, Callable
from gym.spaces import Dict as DictSpace, Box, Space, Discrete, Tuple as TupleSpace
from base.space_converter import SpaceConverter
import numpy as np
from env.rec_env import RecEnv

from utils.utils import normalize_bounds

class AddRemainingT(SpaceConverter):

    def __init__(self,
                 current_action_space: Space,
                 current_observation_space: Space,
                 current_reward_space: Space,
                 original_action_space: Space,
                 original_observation_space: Space,
                 original_reward_space: Space,
                 members: List[str] = [],
                 ratio=False,
                 convert_to_peak_periods=False,
                 time_horizon=1000,
                 Delta_M=1,
                 Delta_P=1):
        self._members = members
        self._ratio=ratio
        self._time_horizon = time_horizon
        self._convert_to_peak_periods = convert_to_peak_periods
        self._Delta_M = Delta_M
        self._Delta_P = Delta_P
        self._nb_time_steps_in_peak_period = Delta_M * Delta_P
        self._nb_peak_periods = (time_horizon-1)//self._nb_time_steps_in_peak_period
        super().__init__(current_action_space,
                         current_observation_space,
                         current_reward_space,
                         original_action_space,
                         original_observation_space,
                         original_reward_space)
        
        
    
    def _convert_observation_space(self):
        if type(self._original_observation_space) != DictSpace:
            raise BaseException(f"Only DictSpace type {type(self._observation_space)} is handled for RemovePeakObservations")

        return DictSpace(
                {
                    **self._current_observation_space,
                    **{
                        "remaining_t": Box(low=0.0, high=(100000.0 if not self._ratio else 1.0))
                    }
                }
            )
    
    def convert_observation(self,
                       observation: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]],
                       action: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       reward: Union[float, List[float], Dict[Union[Tuple[str, str], str], float]]=None,
                       original_action: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       original_observation: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       original_reward: Union[float, List[float], Dict[Union[Tuple[str, str], str], float]]=None,
                       backward=False,
                       **kwargs):
        if not backward:
            if type(self._current_observation_space) != DictSpace:
                raise BaseException(f"Only DictSpace type {type(self._observation_space)} is handled for RemovePeakObservations")
            else:
                infos = kwargs.get("infos", dict())
                d_observation = dict(observation)
                if self._convert_to_peak_periods:
                    current_t = self._nb_peak_periods - int(infos.get("current_t", 0) // self._nb_time_steps_in_peak_period)
                    time_horizon = self._nb_peak_periods
                else:
                    current_t = infos.get("current_t", 0.0)
                    time_horizon = self._time_horizon
                if self._ratio:
                    d_observation["remaining_t"] = 1 - (current_t / time_horizon)
                else:
                    d_observation["remaining_t"] = time_horizon - current_t
                from pprint import pprint
                return d_observation
        else:
            if type(self._current_observation_space) != DictSpace:
                raise BaseException(f"Only DictSpace type {type(self._observation_space)} is handled for RemovePeakObservations")
            else:
                d_observation = dict(observation)
                d_observation.pop("remaining_t")
                return d_observation
            
    @staticmethod
    def get_kwargs_from_env_and_previous_converters(
        rec_env: RecEnv,
        previous_converters: List
    ):
        return {
            "members": rec_env.members,
            "time_horizon": rec_env.T,
            "Delta_M": rec_env.Delta_M,
            "Delta_P": rec_env.Delta_P
        }