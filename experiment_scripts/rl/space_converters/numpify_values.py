from typing import Any, Dict, Union, List, Tuple, Callable
from gym.spaces import Dict as DictSpace, Box, Space, Discrete, Tuple as TupleSpace, MultiDiscrete
from base.space_converter import SpaceConverter
import numpy as np

from utils.utils import normalize_bounds

def numpify(v):
    if type(v) in (int, np.int64, np.int32):
        return np.asarray(v, dtype=np.int32)
    elif type(v) in (float, np.float64, np.float32):
        return np.asarray([v], dtype=np.float32)
    return v

def unnumpify(v):
    if type(v) in (np.int32, np.int64):
        return int(v)
    elif type(v) in (np.float32, np.float64):
        return float(v)
    return v

class NumpifyValues(SpaceConverter):

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
    
    def convert_observation(self,
                       observation: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]],
                       action: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       reward: Union[float, List[float], Dict[Union[Tuple[str, str], str], float]]=None,
                       original_action: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       original_observation: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       original_reward: Union[float, List[float], Dict[Union[Tuple[str, str], str], float]]=None,
                       backward=False,
                       **kwargs):
        #TODO : numpify/unnumpify
        if type(self._current_observation_space) not in [DictSpace, TupleSpace]:
            return unnumpify(observation) if backward else numpify(observation)
        else:
            if backward:
                if type(self._current_observation_space) == DictSpace:
                    new_observation = {
                        k:unnumpify(v) for k,v in observation.items()
                    }
                elif type(self._current_observation_space) == TupleSpace:
                    new_observation = [
                        unnumpify(v) for v in observation
                    ]
            else:
                if type(self._current_observation_space) == DictSpace:
                    new_observation = {
                        k:numpify(v) for k,v in observation.items()
                    }
                elif type(self._current_observation_space) == TupleSpace:
                    new_observation = [
                        numpify(v) for v in observation
                    ]
            return new_observation
        
    def convert_action(self,
                       action: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]],
                       observation: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       reward: Union[float, List[float], Dict[Union[Tuple[str, str], str], float]]=None,
                       original_action: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       original_observation: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       original_reward: Union[float, List[float], Dict[Union[Tuple[str, str], str], float]]=None,
                       backward=False,
                       **kwargs):
        #TODO : numpify/unnumpify
        if type(self._current_action_space) not in [DictSpace, TupleSpace]:
            return unnumpify(action) if backward else numpify(action)
        else:
            if backward:
                if type(self._current_action_space) == DictSpace:
                    new_action = {
                        k:unnumpify(v) for k,v in action.items()
                    }
                elif type(self._current_action_space) == TupleSpace:
                    new_action = [
                        unnumpify(v) for v in action
                    ]
            else:
                if type(self._current_action_space) == DictSpace:
                    new_action = {
                        k:numpify(v) for k,v in action.items()
                    }
                elif type(self._current_action_space) == TupleSpace:
                    new_action = [
                        numpify(v) for v in action
                    ]
            return new_action