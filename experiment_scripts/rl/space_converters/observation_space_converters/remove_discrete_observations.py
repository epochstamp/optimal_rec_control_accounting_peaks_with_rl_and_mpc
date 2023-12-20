from typing import Any, Dict, Union, List, Tuple, Callable
from gym.spaces import Dict as DictSpace, Box, Space, Discrete, Tuple as TupleSpace, MultiDiscrete, MultiBinary
from base.space_converter import SpaceConverter
import numpy as np

from utils.utils import normalize_bounds

class RemoveDiscreteObservations(SpaceConverter):

    def __init__(self,
                 current_action_space: Space,
                 current_observation_space: Space,
                 current_reward_space: Space,
                 original_action_space: Space,
                 original_observation_space: Space,
                 original_reward_space: Space):
        if type(current_observation_space) == DictSpace:
            self._keys_discretes = set([
                key for key,v in original_observation_space.items()
                if type(v) == Discrete or type(v) == MultiDiscrete or type(v) == MultiBinary
            ])
        elif type(current_observation_space) == TupleSpace:
            self._keys_discretes = set([
                v for v in original_observation_space.items()
                if type(v) == Discrete or type(v) == MultiDiscrete or type(v) == MultiBinary
            ])
        else:
            self._keys_discretes = set()
        self._keys_obs_to_keep = set(current_observation_space.keys()) - self._keys_discretes
        super().__init__(current_action_space,
                         current_observation_space,
                         current_reward_space,
                         original_action_space,
                         original_observation_space,
                         original_reward_space)
        
    
    def _convert_observation_space(self):
        if type(self._original_observation_space) != DictSpace:
            raise BaseException(f"Only DictSpace type {type(self._observation_space)} is handled for RemovePeakObservations")
        
        if len(self._keys_discretes) == 0:
            return self._current_observation_space
        else:
            if type(self._current_observation_space) == DictSpace:
                return DictSpace({
                    k:self._current_observation_space[k]
                    for k in self._keys_obs_to_keep
                })
            elif type(self._current_observation_space) == TupleSpace:
                TupleSpace([
                    self._current_observation_space[k]
                    for k in self._keys_obs_to_keep
                ])
            else:
                return self._current_observation_space
    
    def convert_observation(self,
                       observation: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]],
                       action: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       reward: Union[float, List[float], Dict[Union[Tuple[str, str], str], float]]=None,
                       original_action: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       original_observation: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       original_reward: Union[float, List[float], Dict[Union[Tuple[str, str], str], float]]=None,
                       backward=False,
                       **kwargs):
        if len(self._keys_discretes) == 0:
            return observation
        else:
            if backward:
                if type(self._current_observation_space) == DictSpace:
                    return {
                        **observation,
                        **{
                            k:original_observation[k]
                            for k in self._keys_discretes
                        }
                    }
                elif type(self._current_observation_space) == TupleSpace:
                    return [
                        original_observation[k]
                        for k in self._keys_discretes.union(self._keys_obs_to_keep)
                    ]
                else:
                    return observation
            else:
                if type(self._current_observation_space) == DictSpace:
                    return {
                        k:observation[k]
                        for k in self._keys_obs_to_keep
                    }
                elif type(self._current_observation_space) == TupleSpace:
                    [
                        observation[k]
                        for k in self._keys_obs_to_keep
                    ]
                else:
                    return observation