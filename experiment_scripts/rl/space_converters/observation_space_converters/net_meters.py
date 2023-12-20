from typing import Any, Dict, Union, List, Tuple, Callable
from gym.spaces import Dict as DictSpace, Box, Space, Discrete, Tuple as TupleSpace
from base.space_converter import SpaceConverter
import numpy as np
from env.rec_env import RecEnv

from utils.utils import normalize_bounds

class NetMeters(SpaceConverter):

    def __init__(self,
                 current_action_space: Space,
                 current_observation_space: Space,
                 current_reward_space: Space,
                 original_action_space: Space,
                 original_observation_space: Space,
                 original_reward_space: Space,
                 members: List[str] = []):
        self._meter_keys = (
            "consumption_meters", "production_meters"
        )
        self._keys_meters_per_member = set([
            key for key in original_observation_space.keys()
            if len(key) > 1 and key[1] in self._meter_keys
        ])
        self._members = members
        super().__init__(current_action_space,
                         current_observation_space,
                         current_reward_space,
                         original_action_space,
                         original_observation_space,
                         original_reward_space)
        
    
    def _convert_observation_space(self):
        if type(self._original_observation_space) != DictSpace:
            raise BaseException(f"Only DictSpace type {type(self._observation_space)} is handled for NetMeters")
        
        if self._keys_meters_per_member == set():
            return self._current_observation_space
        else:
            return DictSpace({
                **{
                    k:s for k,s in self._current_observation_space.items() if k not in self._keys_meters_per_member
                },
                **{
                    (member, "net_meters"): Box(-10000000, 10000000, shape=self._current_observation_space[(member, "consumption_meters")].shape) for member in self._members
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
        if self._keys_meters_per_member == []:
            return observation
        if type(self._current_observation_space) != DictSpace:
            raise BaseException(f"Only DictSpace type {type(self._observation_space)} is handled for NetMeters")
        if backward:
            d_observation = dict(observation)
            for key_meter in self._keys_meters_per_member:
                _, meter_key = key_meter
                d_observation[key_meter] = original_observation[key_meter][-self._current_observation_space[(self._members[0], meter_key)].shape[0]:]
            d_observation = {
                k:v for k,v in d_observation.items() if len(k) < 2 or k[1] != "net_meters"
            }
            return d_observation
        else:
            d_observation = dict(observation)
            for member in self._members:
                d_observation[(member, "net_meters")] = np.asarray(d_observation[(member, "consumption_meters")]) - np.asarray(d_observation[(member, "production_meters")])
                d_observation.pop((member, "consumption_meters"))
                d_observation.pop((member, "production_meters"))
            return d_observation
            
    @staticmethod
    def get_kwargs_from_env_and_previous_converters(
        rec_env: RecEnv,
        previous_converters: List
    ):
        return {
            "members": rec_env.members
        }