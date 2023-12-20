from typing import Any, Dict, Union, List, Tuple, Callable
from gym.spaces import Dict as DictSpace, Box, Space, Discrete, Tuple as TupleSpace
from base.space_converter import SpaceConverter
import numpy as np
from env.rec_env import RecEnv

from utils.utils import normalize_bounds

class ForceAddPeakCounter(SpaceConverter):

    def __init__(self,
                 current_action_space: Space,
                 current_observation_space: Space,
                 current_reward_space: Space,
                 original_action_space: Space,
                 original_observation_space: Space,
                 original_reward_space: Space,
                 Delta_M=1,
                 Delta_P=1,
                 force_to_zero=False
    ):
        self._Delta_P=Delta_P
        self._Delta_M=Delta_M
        self._force_to_zero=force_to_zero
        super().__init__(current_action_space,
                         current_observation_space,
                         current_reward_space,
                         original_action_space,
                         original_observation_space,
                         original_reward_space)
    
    def _convert_observation_space(self):
        if ("peak_period_counter" in list(self._original_observation_space.keys())):
            return self._current_observation_space
        if type(self._current_observation_space) == Box:
            
            return TupleSpace(self._current_observation_space, Discrete(self._Delta_P+1))
        elif type(self._current_observation_space) == TupleSpace:
            return TupleSpace(tuple(self._current_observation_space) + tuple([Discrete(self._Delta_P+1)]))
        elif type(self._current_observation_space) == DictSpace:
            d_space = dict(self._current_observation_space)
            d_space["peak_period_counter"] = Discrete(self._Delta_P+1)
            return DictSpace(d_space)
        else:
            raise BaseException(f"Space type {type(self._action_space)} is not handled for ForceAddPeakCounter")
    
    def convert_observation(self,
                       observation: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]],
                       action: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       reward: Union[float, List[float], Dict[Union[Tuple[str, str], str], float]]=None,
                       original_action: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       original_observation: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       original_reward: Union[float, List[float], Dict[Union[Tuple[str, str], str], float]]=None,
                       backward=False,
                       **kwargs):
        
        if (not self._force_to_zero and "peak_period_counter" in list(self._original_observation_space.keys())):
            return observation
        if backward:
            if type(self._current_observation_space) == Box:
                new_observation = observation[0]
            elif type(self._current_observation_space) == TupleSpace:
                new_observation = observation[:-1]
            elif type(self._current_observation_space) == DictSpace:
                d_observation = dict(observation)
                d_observation.pop("peak_period_counter", None)
                new_observation = d_observation
            else:
                raise BaseException(f"Space type {type(self._action_space)} is not handled for ForceAddPeakCounter")
        else:
            peak_period_counter = self._peak_period_counter if not self._force_to_zero else 0
            if type(self._observation_space) == Box:
                observation = (observation, peak_period_counter)
            elif type(self._observation_space) == TupleSpace:
                observation = observation + [peak_period_counter]
            elif type(self._observation_space) == DictSpace:
                d_observation = dict(observation)
                d_observation["peak_period_counter"] = peak_period_counter
                observation = d_observation
            else:
                raise BaseException(f"Space type {type(self._action_space)} is not handled for ForceAddPeakCounter")
            new_observation =  observation
        if not self._force_to_zero:
            if self._peak_period_counter == self._Delta_P:
                self._peak_period_counter = 0
            elif d_observation["metering_period_counter"] == self._Delta_M:
                self._peak_period_counter += 1
        return new_observation
        
    def reset(self):
        self._peak_period_counter = 0

    @staticmethod
    def get_kwargs_from_env_and_previous_converters(
        rec_env: RecEnv,
        previous_converters: List
    ):
        return {
            "Delta_P": rec_env.Delta_P,
            "Delta_M": rec_env.Delta_M
        }