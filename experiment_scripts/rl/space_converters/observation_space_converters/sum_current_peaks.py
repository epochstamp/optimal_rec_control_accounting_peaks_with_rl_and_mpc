from typing import Any, Dict, Union, List, Tuple, Callable
from experiment_scripts.rl.space_converters.observation_space_converters.add_current_peaks_observations import AddCurrentPeaksObservations
from gym.spaces import Dict as DictSpace, Box, Space, Discrete, Tuple as TupleSpace
from base.space_converter import SpaceConverter
import numpy as np
from env.rec_env import RecEnv

from utils.utils import normalize_bounds

class SumCurrentPeaks(SpaceConverter):

    def __init__(self,
                 current_action_space: Space,
                 current_observation_space: Space,
                 current_reward_space: Space,
                 original_action_space: Space,
                 original_observation_space: Space,
                 original_reward_space: Space,
                 members: List[str] = []):
        self._members = members
        self._lst_peaks = [
            "current_offtake_peaks",
            "current_injection_peaks"
        ]
        super().__init__(current_action_space,
                         current_observation_space,
                         current_reward_space,
                         original_action_space,
                         original_observation_space,
                         original_reward_space)
        
        
    
    def _convert_observation_space(self):
        if type(self._original_observation_space) != DictSpace:
            raise BaseException(f"Only DictSpace type {type(self._observation_space)} is handled for RemovePeakObservations")

        return DictSpace(
            {
                **{
                    k:s for k,s in self._current_observation_space.items() if type(k) != tuple or ("current_offtake_peaks" not in k and "current_injection_peaks" not in k)
                },
                **{
                    peak_type: Box(low=0.0, high=1000000.0)
                    for peak_type in self._lst_peaks
                }
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
        if self._lst_peaks == []:
            return observation
        if backward:
            if type(self._current_observation_space) != DictSpace:
                raise BaseException(f"Only DictSpace type {type(self._observation_space)} is handled for RemovePeakObservations")
            else:
                d_observation = dict(observation)
                for key_peak in self._lst_peaks:
                    for member in self._members:
                        d_observation[(member, key_peak)] = kwargs["infos"].get(key_peak, dict()).get(member, 0.0)
                    d_observation.pop(key_peak)
                return d_observation
        else:
            if type(self._current_observation_space) != DictSpace:
                raise BaseException(f"Only DictSpace type {type(self._observation_space)} is handled for RemovePeakObservations")
            else:
                d_observation = dict(observation)
                for key_peak in self._lst_peaks:
                    d_observation[key_peak] = 0.0
                    for member in self._members:
                        d_observation[key_peak] += float(d_observation[(member, key_peak)])
                        d_observation.pop((member, key_peak))
                from pprint import pprint
                return d_observation
            
    @staticmethod
    def get_kwargs_from_env_and_previous_converters(
        rec_env: RecEnv,
        previous_converters: List
    ):
        return {
            "members": rec_env.members
        }