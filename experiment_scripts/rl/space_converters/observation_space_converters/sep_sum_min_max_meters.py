from typing import Any, Dict, Union, List, Tuple, Callable
from gym.spaces import Dict as DictSpace, Box, Space, Discrete, Tuple as TupleSpace
from base.space_converter import SpaceConverter
import numpy as np
from env.rec_env import RecEnv

from utils.utils import normalize_bounds

class SepSumMinMaxMeters(SpaceConverter):

    def __init__(self,
                 current_action_space: Space,
                 current_observation_space: Space,
                 current_reward_space: Space,
                 original_action_space: Space,
                 original_observation_space: Space,
                 original_reward_space: Space,
                 members: List[str] = []):
        self._members = members
        super().__init__(current_action_space,
                         current_observation_space,
                         current_reward_space,
                         original_action_space,
                         original_observation_space,
                         original_reward_space)
        
    
    def _convert_observation_space(self):
        if type(self._original_observation_space) != DictSpace:
            raise BaseException(f"Only DictSpace type {type(self._observation_space)} is handled for SepSumMinMaxNetMeters")
        
        return DictSpace({
            **{
                k:s for k,s in self._current_observation_space.items() if "net_meters" not in k
            },
            **{
                (member, type_meter): Box(-10000000, 10000000) for member in self._members for type_meter in ("sum_cons_meter", "sum_prod_meter", "min_meter", "max_meter")
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
        if type(self._current_observation_space) != DictSpace:
            raise BaseException(f"Only DictSpace type {type(self._observation_space)} is handled for SepSumMinMaxNetMeters")
        if backward:
            d_observation = dict(observation)
            for member in self._members:
                d_observation[(member, "net_meters")] = original_observation[(member, "consumption_meters")][-self._current_observation_space[(self._members[0], "net_meters")].shape[0]:] - original_observation[(member, "production_meters")][-self._current_observation_space[(self._members[0], "net_meters")].shape[0]:]
            d_observation = {
                k:v for k,v in d_observation.items() if len(k) < 2 or k[1] not in ("sum_cons_meter", "sum_prod_meter", "min_meter", "max_meter")
            }
            return d_observation
        else:
            d_observation = dict(observation)
            for member in self._members:
                d_observation[(member, "max_meter")] = max(d_observation[(member, "net_meters")])
                d_observation[(member, "min_meter")] = min(d_observation[(member, "net_meters")])
                d_observation[(member, "sum_cons_meter")] = sum([e for e in d_observation[(member, "net_meters")] if e >= 0])
                d_observation[(member, "sum_prod_meter")] = sum([e for e in d_observation[(member, "net_meters")] if e <= 0])
                d_observation.pop((member, "net_meters"))
            return d_observation
            
    @staticmethod
    def get_kwargs_from_env_and_previous_converters(
        rec_env: RecEnv,
        previous_converters: List
    ):
        return {
            "members": rec_env.members
        }