from typing import Any, Dict, Union, List, Tuple, Callable
from gym.spaces import Dict as DictSpace, Box, Space, Discrete, Tuple as TupleSpace
from base.space_converter import SpaceConverter
import numpy as np
from env.rec_env import RecEnv
from copy import deepcopy

from utils.utils import normalize_bounds

class CombineMetersAndExogenous(SpaceConverter):

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
            if key[1] in self._meter_keys
        ])
        self._keys_exogenous_per_member = set([
            key for key in original_observation_space.keys()
            if key[1] in self._meter_keys
        ])
        self._members = members
        self._exo_keys = (
            "consumption", "production"
        )
        self._exo_keys_per_member = {
            (member, exo_key) for member in self._members for exo_key in self._exo_keys
        }
        super().__init__(current_action_space,
                         current_observation_space,
                         current_reward_space,
                         original_action_space,
                         original_observation_space,
                         original_reward_space)
        
    
    def _convert_observation_space(self):
        if type(self._original_observation_space) != DictSpace:
            raise BaseException(f"Only DictSpace type {type(self._observation_space)} is handled for SumMeters")
        
        if self._keys_meters_per_member == set():
            return self._current_observation_space
        else:
            #print(dict(self._current_observation_space).keys())
            #exit()
            return DictSpace({
                **{
                    k:s for k,s in self._current_observation_space.items() if k not in self._keys_meters_per_member #and k not in self._exo_keys_per_member
                },
                **{
                    (member, meter): Box(0, 10000000, shape=self._current_observation_space[(member, meter)].shape)
                    for member in self._members for meter in self._meter_keys
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
            raise BaseException(f"Only DictSpace type {type(self._observation_space)} is handled for SumMeters")
        if backward:
            d_observation = deepcopy(observation)
            for member in self._members:
                #if (member, "consumption") in original_observation:
                #    d_observation[(member, "consumption")] = original_observation[(member, "consumption")]
                #if (member, "production") in original_observation:
                #    d_observation[(member, "production")] = original_observation[(member, "production")]
                for meter_key in self._meter_keys:
                    key_meter = (member, meter_key)
                    key_meter_forward = (member, meter_key)
                    d_observation[key_meter] = original_observation[key_meter][-self._current_observation_space[(self._members[0], meter_key)].shape[0]:]
                    d_observation.pop(key_meter_forward, None)
            return d_observation
        else:
            from time import time
            d_observation = deepcopy(observation)
            for member in self._members:
                
                for meter_key in self._meter_keys:
                    key_meter = (member, meter_key)
                    key_meter_forward = (member, meter_key)
                    meters = np.asarray(d_observation.pop(key_meter, None))
                    if "production" in meter_key:
                        net_consumption_production = np.maximum(np.asarray(d_observation.get((member, "production"), 0.0)) - np.asarray(d_observation.get((member, "consumption"), 0.0)), 0)
                    elif "consumption" in meter_key:
                        net_consumption_production = np.maximum(np.asarray(d_observation.get((member, "consumption"), 0.0)) - np.asarray(d_observation.get((member, "production"), 0.0)), 0)
                    meters += net_consumption_production
                    d_observation[key_meter_forward] = meters
                #d_observation.pop((member, "consumption"), None)
                #d_observation.pop((member, "production"), None)
            return d_observation
            
    @staticmethod
    def get_kwargs_from_env_and_previous_converters(
        rec_env: RecEnv,
        previous_converters: List
    ):
        return {
            "members": rec_env.members
        }