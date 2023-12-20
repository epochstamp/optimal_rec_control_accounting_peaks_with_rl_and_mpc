from typing import Any, Dict, Union, List, Tuple, Callable
from gym.spaces import Dict as DictSpace, Box, Space, Discrete, Tuple as TupleSpace
from base.space_converter import SpaceConverter
import numpy as np

from utils.utils import normalize_bounds, normalize_1_1, to_0_1_range, from_0_1_range, unnormalize_1_1

def to_1_1_range_zero_centered(x, min_x, max_x):
    if type(x) in (float, int):
        if x > 0:
            return x/max_x
        elif x < 0:
            return -(x/min_x)
        else:
            return 0.0
    else:
        new_x_vector = np.asarray(x)
        if len(new_x_vector[new_x_vector>0]) > 0:
            new_x_vector[new_x_vector>0] /= max_x
        if len(new_x_vector[new_x_vector<0]) > 0:
            new_x_vector[new_x_vector<0] /= -min_x
        if len(new_x_vector[new_x_vector==0]) > 0:
            new_x_vector[new_x_vector==0] = 0.0
        return new_x_vector

def from_1_1_range_zero_centered(x, min_x, max_x):
    if type(x) in (float, int):
        if x > 0:
            return x*max_x
        elif x < 0:
            return -(x*min_x)
        else:
            return 0.0
    else:
        new_x_vector = np.asarray(x)
        if len(new_x_vector[new_x_vector>0]) > 0:
            new_x_vector[new_x_vector>0] *= max_x
        if len(new_x_vector[new_x_vector<0]) > 0:
            new_x_vector[new_x_vector<0] *= -min_x
        if len(new_x_vector[new_x_vector==0]) > 0:
            new_x_vector[new_x_vector==0] = 0.0
        return new_x_vector

class ActionScaler(SpaceConverter):

    def __init__(self,
                 current_action_space: Space,
                 current_observation_space: Space,
                 current_reward_space: Space,
                 original_action_space: Space,
                 original_observation_space: Space,
                 original_reward_space: Space,
                 zero_centering: bool=False,
                 multiplier=1):
        self._zero_centering = zero_centering
        self._multiplier=multiplier
        super().__init__(current_action_space,
                         current_observation_space,
                         current_reward_space,
                         original_action_space,
                         original_observation_space,
                         original_reward_space)
        
    
    def _convert_action_space(self):
        if type(self._current_action_space) == Box:
            return Box(low=-1.0*self._multiplier, high=1.0*self._multiplier, shape=(len(self._current_action_space.shape[0]),))
        elif type(self._current_action_space) == TupleSpace:
            return TupleSpace([
                Box(low=-1.0*self._multiplier, high=1.0*self._multiplier)
            ] * self._current_action_space.shape[0])
        elif type(self._current_action_space) == DictSpace:
            return DictSpace({
                key:Box(low=-1.0*self._multiplier, high=1.0*self._multiplier) for key in self._current_action_space.keys()
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
                
                if self._zero_centering:
                    if np.any(self._current_action_space.low >= 0) or np.any(self._current_action_space.low <= 0):
                        raise BaseException("Min value must be strictly negative and max value must be strictly positive for each action")
                    return from_1_1_range_zero_centered(action/self._multiplier, self._current_action_space.low, self._current_action_space.high)
                else:
                    return unnormalize_1_1(action/self._multiplier, self._current_action_space.low, self._current_action_space.high)
            elif type(self._action_space) == TupleSpace:
                if self._zero_centering:
                    return [
                        (
                            np.clip(from_1_1_range_zero_centered(action_value/self._multiplier, self._current_action_space[i].low/self._multiplier, self._current_action_space[i].high/self._multiplier), self._current_action_space[i].low/self._multiplier, self._current_action_space[i].high/self._multiplier)
                        ) if type(self._current_action_space[i]) == Box else action_value
                        for i, action_value in enumerate(action)
                    ]
                else:
                    return [
                        (
                            np.clip(unnormalize_1_1(action_value/self._multiplier, self._current_action_space[i].low/self._multiplier, self._current_action_space[i].high/self._multiplier), self._current_action_space[i].low/self._multiplier, self._current_action_space[i].high/self._multiplier)
                        ) if type(self._current_action_space[i]) == Box else action_value
                        for i, action_value in enumerate(action)
                    ]
            elif type(self._current_action_space) == DictSpace:
                if self._zero_centering:
                    action_space_keys = list(self._action_space.keys())
                    normalized_action = {
                        key:
                        (   
                            np.clip(from_1_1_range_zero_centered(action[key]/self._multiplier, self._current_action_space[key].low/self._multiplier, self._current_action_space[key].high/self._multiplier), self._current_action_space[key].low/self._multiplier, self._current_action_space[key].high/self._multiplier)
                        ) if type(self._current_action_space[key]) == Box else action[key]
                        for key in action_space_keys
                    }
                else:
                    action_space_keys = list(self._action_space.keys())
                    normalized_action = {
                        key:
                        (   
                            np.clip(unnormalize_1_1(action[key]/self._multiplier, self._current_action_space[key].low/self._multiplier, self._current_action_space[key].high/self._multiplier), self._current_action_space[key].low/self._multiplier, self._current_action_space[key].high/self._multiplier)
                        ) if type(self._current_action_space[key]) == Box else action[key]
                        for key in action_space_keys
                    }
                return normalized_action
            else:
                raise BaseException(f"Space type {type(self._action_space)} is not handled for ActionStandardScaler")
        else:
            if type(self._current_action_space) == Box:
                if self._zero_centering:
                    new_action = np.clip(to_1_1_range_zero_centered(action, self._current_action_space.low, self._current_action_space.high), self._action_space.low/self._multiplier, self._action_space.high/self._multiplier)*self._multiplier
                else:
                    new_action = np.clip(normalize_1_1(action, self._current_action_space.low/self._multiplier, self._current_action_space.high/self._multiplier), self._action_space.low/self._multiplier, self._action_space.high/self._multiplier)*self._multiplier
                return new_action
            elif type(self._current_action_space) == TupleSpace:
                if self._zero_centering:
                    return [
                        (
                            np.clip(to_1_1_range_zero_centered(action_value, self._current_action_space[i].low, self._current_action_space[i].high), self._action_space[i].low/self._multiplier, self._action_space[i].high/self._multiplier)*self._multiplier
                            
                        ) if type(self._current_action_space[i]) == Box else action_value
                        for i, action_value in enumerate(action)
                    ]
                else:
                    return [
                        (
                            np.clip(normalize_1_1(action_value, self._current_action_space[i].low, self._current_action_space[i].high), self._action_space[i].low/self._multiplier, self._action_space[i].high/self._multiplier)*self._multiplier
                            
                        ) if type(self._current_action_space[i]) == Box else action_value
                        for i, action_value in enumerate(action)
                    ]
            elif type(self._current_action_space) == DictSpace:
                if self._zero_centering:
                    new_action = {
                        key:(
                            np.clip(to_1_1_range_zero_centered(action[key], self._current_action_space[key].low, self._current_action_space[key].high), self._action_space[key].low/self._multiplier, self._action_space[key].high/self._multiplier)*self._multiplier
                        ) if type(self._current_action_space[key]) == Box else action[key]
                        for key in self._action_space.keys()
                    }
                else:
                    new_action = {
                        key:(
                            np.clip(normalize_1_1(action[key], self._current_action_space[key].low/self._multiplier, self._current_action_space[key].high/self._multiplier), self._action_space[key].low/self._multiplier, self._action_space[key].high/self._multiplier)*self._multiplier
                        ) if type(self._current_action_space[key]) == Box else action[key]
                        for key in self._action_space.keys()
                    }
                return new_action
            else:
                raise BaseException(f"Space type {type(self._action_space)} is not handled for ActionStandardScaler")
