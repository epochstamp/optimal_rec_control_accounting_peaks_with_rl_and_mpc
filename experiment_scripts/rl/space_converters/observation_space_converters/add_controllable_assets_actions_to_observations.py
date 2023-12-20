from typing import Any, Dict, Union, List, Tuple, Callable
from gym.spaces import Dict as DictSpace, Box, Space, Discrete, Tuple as TupleSpace
from base.space_converter import SpaceConverter
import numpy as np
from env.rec_env import RecEnv

from utils.utils import normalize_bounds

class AddControllableAssetsActionsToObservations(SpaceConverter):

    def __init__(self,
                 current_action_space: Space,
                 current_observation_space: Space,
                 current_reward_space: Space,
                 original_action_space: Space,
                 original_observation_space: Space,
                 original_reward_space: Space):
        controllable_assets_action_keys = list(current_action_space.keys())
        self._controllable_assets_action_keys = {
            k for k in controllable_assets_action_keys if k not in ("metering_period_trigger", "peak_period_trigger")
        }
        super().__init__(current_action_space,
                         current_observation_space,
                         current_reward_space,
                         original_action_space,
                         original_observation_space,
                         original_reward_space)
        
    
    def _convert_observation_space(self):
        if type(self._current_action_space) != DictSpace and type(self._current_observation_space) != DictSpace:
            raise BaseException(f"Only DictSpace type {type(self._action_space)} is handled for AddControllableAssetsActionsToObservations")
        
        if self._controllable_assets_action_keys == []:
            return self._current_observation_space
        else:
            return DictSpace({
                **self._current_observation_space,
                **{
                    key:self._current_action_space[key]
                    for key in self._controllable_assets_action_keys
                }
            })
    
    def convert_observation(self,
                       observation: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]],
                       action: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       reward: Union[float, List[float], Dict[Union[Tuple[str, str], str], float]]=None,
                       original_action: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       original_observation: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       original_reward: Union[float, List[float], Dict[Union[Tuple[str, str], str], float]]=None,
                       backward=False,
                       **kwargs):
        if self._controllable_assets_action_keys== []:
            return observation
        if not backward:
            if type(self._current_observation_space) != DictSpace:
                raise BaseException(f"Only DictSpace type {type(self._observation_space)} is handled for AddControllableAssetsActionsToObservations")
            else:
                d_observation = {
                    **observation,
                    **{
                        key_controllable_assets_action: action[key_controllable_assets_action]
                        for key_controllable_assets_action in self._controllable_assets_action_keys
                    }
                }
                return d_observation
        else:
            if type(self._current_observation_space) != DictSpace:
                raise BaseException(f"Only DictSpace type {type(self._observation_space)} is handled for RemoveControllableAssetsObservations")
            else:
                d_observation = {
                    k:obs for k, obs in observation.items()
                    if k not in self._controllable_assets_action_keys
                }
                return d_observation