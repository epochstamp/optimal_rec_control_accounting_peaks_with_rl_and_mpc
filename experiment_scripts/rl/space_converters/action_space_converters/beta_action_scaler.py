from typing import Any, Dict, Union, List, Tuple, Callable
from gym.spaces import Dict as DictSpace, Box, Space, Discrete, Tuple as TupleSpace
from base.space_converter import SpaceConverter
import numpy as np

from utils.utils import normalize_bounds, to_0_1_range, from_0_1_range

class BetaActionScaler(SpaceConverter):

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
                         original_reward_space)
        
    
    def _convert_action_space(self):
        if type(self._current_action_space) == Box:
            return Box(low=0, high=1, shape=(len(self._current_action_space.shape[0]),))
        elif type(self._current_action_space) == TupleSpace:
            return TupleSpace([
                Box(low=0, high=1)
            ] * self._current_action_space.shape[0])
        elif type(self._current_action_space) == DictSpace:
            return DictSpace({
                key:Box(low=0, high=1) for key in self._current_action_space.keys()
            })
        else:
            raise BaseException(f"Space type {type(self._current_action_space)} is not handled for ActionStandardScaler")
    
    def convert_action(self,
                       action: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]],
                       observation: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       reward: Union[float, List[float], Dict[Union[Tuple[str, str], str], float]]=None,
                       original_action: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       original_observation: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       original_reward: Union[float, List[float], Dict[Union[Tuple[str, str], str], float]]=None,
                       backward=False,
                       **kwargs):
        if backward:
            if type(self._current_action_space) == Box:
                return from_0_1_range(action, self._current_action_space.low, self._current_action_space.high)
            elif type(self._action_space) == TupleSpace:
                return [
                    (
                        np.clip(from_0_1_range(action_value, self._current_action_space[i].low, self._current_action_space[i].high), self._current_action_space[i].low, self._current_action_space[i].high)
                    ) if type(self._current_action_space[i]) == Box else action_value
                    for i, action_value in enumerate(action)
                ]
            elif type(self._current_action_space) == DictSpace:
                action_space_keys = list(self._action_space.keys())
                normalized_action = {
                    key:
                    (   
                        np.clip(from_0_1_range(action[key], self._current_action_space[key].low, self._current_action_space[key].high), self._current_action_space[key].low, self._current_action_space[key].high)
                    ) if type(self._current_action_space[key]) == Box else action[key]
                    for key in action_space_keys
                }
                return normalized_action
            else:
                raise BaseException(f"Space type {type(self._action_space)} is not handled for ActionStandardScaler")
        else:
            if type(self._current_action_space) == Box:
                return np.clip(to_0_1_range(action, self._current_action_space.low, self._current_action_space.high), self._action_space.low, self._action_space.high)
            elif type(self._action_space) == TupleSpace:
                return [
                    (
                        np.clip(to_0_1_range(action_value, self._current_action_space[i].low, self._current_action_space[i].high), self._action_space[i].low, self._action_space[i].high)
                        
                    ) if type(self._current_action_space[i]) == Box else action_value
                    for i, action_value in enumerate(action)
                ]
            elif type(self._current_action_space) == DictSpace:
                return {
                    key:(
                        to_0_1_range(action[key], self._current_action_space[key].low, self._current_action_space[key].high)
                    ) if type(self._current_action_space[key]) == Box else action[key]
                    for key in self._action_space.keys()
                }
            else:
                raise BaseException(f"Space type {type(self._action_space)} is not handled for ActionStandardScaler")
