from typing import Any, Dict, Union, List, Tuple, Callable
from gym.spaces import Dict as DictSpace, Box, Space, Discrete, Tuple as TupleSpace
from base.space_converter import SpaceConverter
import numpy as np
from env.rec_env import RecEnv

from utils.utils import normalize_bounds

class ResizeAndPadSequencesObservations(SpaceConverter):

    def __init__(self,
                 current_action_space: Space,
                 current_observation_space: Space,
                 current_reward_space: Space,
                 original_action_space: Space,
                 original_observation_space: Space,
                 original_reward_space: Space,
                 members: List[str] = [],
                 number_of_past_sequence_data=10000000):
        self._members = members
        self._number_of_past_sequence_data = number_of_past_sequence_data
        observation_keys = self._get_observation_keys()
        self._observation_keys = set([
            key for key in original_observation_space.keys()
            if key in observation_keys
        ])
        
        super().__init__(current_action_space,
                         current_observation_space,
                         current_reward_space,
                         original_action_space,
                         original_observation_space,
                         original_reward_space)

    def _get_observation_keys(self):
        raise NotImplementedError()

    def _compute_number_of_past_sequence_data(self, observation_key=None):
        return self._number_of_past_sequence_data      
    
    def _convert_observation_space(self):
        if type(self._original_observation_space) != DictSpace:
            raise BaseException(f"Only DictSpace type {type(self._observation_space)} is handled for ResizeAndPadSequencesObservations")
        
        if self._observation_keys == set():
            return self._current_observation_space
        new_observation_space = {
            k:(s if k not in self._observation_keys else 
               Box(0, 10000, shape=[self._compute_number_of_past_sequence_data(observation_key=k)])
               if self._compute_number_of_past_sequence_data(observation_key=k) > 0
               else None) for k,s in self._current_observation_space.items()
        }
        new_observation_space = DictSpace({
            k:s for k,s in new_observation_space.items() if s is not None
        })
        return new_observation_space
    
    def convert_observation(self,
                       observation: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]],
                       action: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       reward: Union[float, List[float], Dict[Union[Tuple[str, str], str], float]]=None,
                       original_action: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       original_observation: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       original_reward: Union[float, List[float], Dict[Union[Tuple[str, str], str], float]]=None,
                       backward=False,
                       **kwargs):
        if self._observation_keys == set():
            return observation
        if backward:
            if type(self._current_observation_space) != DictSpace:
                raise BaseException(f"Only DictSpace type {type(self._observation_space)} is handled for ResizeAndPadSequencesObservations")
            else:
                d_observation = dict(observation)
                for key_observation in self._observation_keys:
                    d_observation[key_observation] = original_observation[key_observation]
                return d_observation
        else:
            if type(self._current_observation_space) != DictSpace:
                raise BaseException(f"Only DictSpace type {type(self._observation_space)} is handled for ResizeAndPadSequencesObservations")
            d_observation = dict(observation)
            for key_observation in self._observation_keys:
                number_of_past_sequence_data = self._compute_number_of_past_sequence_data(observation_key=key_observation)
                if number_of_past_sequence_data > 0:
                    d_observation[key_observation] = d_observation[key_observation][-number_of_past_sequence_data:]
                    if len(d_observation[key_observation]) < number_of_past_sequence_data:
                        d_observation[key_observation] = np.hstack([[0.0]*(number_of_past_sequence_data - len(d_observation[key_observation])), d_observation[key_observation]])
                else:
                    d_observation.pop(key_observation, None)
            return d_observation

    @staticmethod
    def get_kwargs_from_env_and_previous_converters(
        rec_env: RecEnv,
        previous_converters: List
    ):
        return {
            "members": rec_env.members
        }