from typing import Any, Dict, Union, List, Tuple, Callable
from gym.spaces import Dict as DictSpace, Box, Space, Discrete, Tuple as TupleSpace
from base.space_converter import SpaceConverter
import numpy as np
from env.rec_env import RecEnv

from utils.utils import normalize_bounds

class NetProdCons(SpaceConverter):

    def __init__(self,
                 current_action_space: Space,
                 current_observation_space: Space,
                 current_reward_space: Space,
                 original_action_space: Space,
                 original_observation_space: Space,
                 original_reward_space: Space,
                 members: List[str] = []):
        self._members = members
        self._prodcons_keys = (
            "consumption", "production"
        )
        self._keys_prodcons_per_member = set([
            key for key in original_observation_space.keys()
            if len(key) > 1 and key[1] in self._prodcons_keys
        ])
        self._current_observation_space_keys = None
        super().__init__(current_action_space,
                         current_observation_space,
                         current_reward_space,
                         original_action_space,
                         original_observation_space,
                         original_reward_space)
        
        
    
    def _convert_observation_space(self):
        

        if type(self._original_observation_space) != DictSpace:
            raise BaseException(f"Only DictSpace type {type(self._observation_space)} is handled for NetProdCons")
        if self._current_observation_space_keys is None:
            self._current_observation_space_keys = list(self._current_observation_space.keys())

        if self._keys_prodcons_per_member == set():
            return self._current_observation_space
        else:
            return DictSpace({
                **{
                    k:s for k,s in self._current_observation_space.items() if k not in self._keys_prodcons_per_member
                },
                **{
                    (member, "net_prod_cons"): Box(-10000000, 10000000, shape=(
                        self._current_observation_space[(member, "production")] if (member, "production") in self._current_observation_space_keys else self._current_observation_space[(member, "consumption")]
                    ).shape) for member in self._members if (member, "consumption") in self._current_observation_space_keys or (member, "production") in self._current_observation_space_keys
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
        if self._keys_prodcons_per_member == []:
            return observation
        if type(self._current_observation_space) != DictSpace:
            raise BaseException(f"Only DictSpace type {type(self._observation_space)} is handled for NetProdCons")
        if backward:
            d_observation = dict(observation)
            for key_prodcons in self._keys_prodcons_per_member:
                _, meter_key = key_prodcons
                d_observation[key_prodcons] = original_observation[key_prodcons][-self._current_observation_space[key_prodcons].shape[0]:]
            d_observation = {
                k:v for k,v in d_observation.items() if len(k) < 2 or k[1] != "net_prod_cons"
            }
            return d_observation
        else:
            d_observation = dict(observation)
            for member in self._members:
                if (member, "consumption") in d_observation or (member, "production") in d_observation:
                    d_observation[(member, "net_prod_cons")] = np.asarray(d_observation.get((member, "consumption"), 0.0)) - np.asarray(d_observation.get((member, "production"), 0.0))
                    d_observation.pop((member, "consumption"), None)
                    d_observation.pop((member, "production"), None)
                #print(d_observation[(member, "net_meters")])
            #print(d_observation)
            return d_observation
            
    @staticmethod
    def get_kwargs_from_env_and_previous_converters(
        rec_env: RecEnv,
        previous_converters: List
    ):
        return {
            "members": rec_env.members
        }