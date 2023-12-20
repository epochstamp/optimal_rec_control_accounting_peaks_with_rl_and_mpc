from typing import Any, Dict, Union, List, Tuple, Callable
from gym.spaces import Dict as DictSpace, Box, Space, Discrete, Tuple as TupleSpace
from base.space_converter import SpaceConverter
import numpy as np
from env.rec_env import RecEnv

from utils.utils import normalize_bounds

class RemoveControllableAssetsObservations(SpaceConverter):

    def __init__(self,
                 current_action_space: Space,
                 current_observation_space: Space,
                 current_reward_space: Space,
                 original_action_space: Space,
                 original_observation_space: Space,
                 original_reward_space: Space,
                 controllable_assets_state_keys: List[str] = []):
        self._controllable_assets_state_keys = {
            ("#".join(k) if type(k) == tuple and k not in current_observation_space else k) for k in controllable_assets_state_keys
        }
        super().__init__(current_action_space,
                         current_observation_space,
                         current_reward_space,
                         original_action_space,
                         original_observation_space,
                         original_reward_space)
        
    
    def _convert_observation_space(self):
        if type(self._current_observation_space) != DictSpace:
            raise BaseException(f"Only DictSpace type {type(self._observation_space)} is handled for RemoveControllableAssetsObservations")
        
        if self._controllable_assets_state_keys == []:
            return self._current_observation_space
        else:
            return DictSpace(
                {
                    k:s for k,s in self._current_observation_space.items() if k not in self._controllable_assets_state_keys
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
        if self._controllable_assets_state_keys == []:
            return observation
        if backward:
            if type(self._current_observation_space) != DictSpace:
                raise BaseException(f"Only DictSpace type {type(self._observation_space)} is handled for RemoveControllableAssetsObservations")
            else:
                d_observation = dict(observation)
                for key_controllable_assets_state in self._controllable_assets_state_keys:
                    d_observation[key_controllable_assets_state] = original_observation[key_controllable_assets_state]
                return d_observation
        else:
            if type(self._current_observation_space) != DictSpace:
                raise BaseException(f"Only DictSpace type {type(self._observation_space)} is handled for RemoveControllableAssetsObservations")
            else:
                d_observation = dict(observation)
                for key_controllable_assets_state in self._controllable_assets_state_keys:
                    d_observation.pop(key_controllable_assets_state, None)
                return d_observation
            
    @staticmethod
    def get_kwargs_from_env_and_previous_converters(
        rec_env: RecEnv,
        previous_converters: List
    ):
        return {
            "controllable_assets_state_keys": list(rec_env.controllable_assets_state_space.keys())
        }