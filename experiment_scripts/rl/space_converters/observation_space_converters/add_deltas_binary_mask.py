from typing import Any, Dict, Union, List, Tuple, Callable
from gym.spaces import Dict as DictSpace, Box, Space, Discrete, Tuple as TupleSpace
from base.space_converter import SpaceConverter
import numpy as np
from env.rec_env import RecEnv

from utils.utils import normalize_bounds

class AddDeltasBinaryMask(SpaceConverter):

    def __init__(self,
                 current_action_space: Space,
                 current_observation_space: Space,
                 current_reward_space: Space,
                 original_action_space: Space,
                 original_observation_space: Space,
                 original_reward_space: Space,
                 Delta_M=2,
                 Delta_P=1):
        super().__init__(current_action_space,
                         current_observation_space,
                         current_reward_space,
                         original_action_space,
                         original_observation_space,
                         original_reward_space)
        self._Delta_M=Delta_M
        self._Delta_P=Delta_P
    
    def _convert_observation_space(self):
        if type(self._current_observation_space) == Box:
            lows = np.hstack(
                [self._current_observation_space.low, np.asarray([0.0, 0.0])]
            )
            highs = np.hstack(
                [self._current_observation_space.high, np.asarray([1.0, 1.0])]
            )
            return Box(low=np.asarray(lows), high=np.asarray(highs))
        elif type(self._current_observation_space) == TupleSpace:
            return TupleSpace(tuple(self._current_observation_space) + tuple([Discrete(2), Discrete(2)]))
        elif type(self._current_observation_space) == DictSpace:
            d_space = dict(self._current_observation_space)
            d_space["binary_mask_delta_m"] = Discrete(2)
            d_space["binary_mask_delta_p"] = Discrete(2)
            return DictSpace(d_space)
        else:
            raise BaseException(f"Space type {type(self._action_space)} is not handled for AddDeltaBinaryMask")
    
    def convert_observation(self,
                       observation: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]],
                       action: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       reward: Union[float, List[float], Dict[Union[Tuple[str, str], str], float]]=None,
                       original_action: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       original_observation: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       original_reward: Union[float, List[float], Dict[Union[Tuple[str, str], str], float]]=None,
                       backward=False,
                       **kwargs):
        if backward:
            if type(self._current_observation_space) == Box:
                return observation[:-2]
            elif type(self._current_observation_space) == TupleSpace:
                return observation[:-2]
            elif type(self._current_observation_space) == DictSpace:
                d_observation = dict(observation)
                d_observation.pop("binary_mask_delta_m")
                d_observation.pop("binary_mask_delta_p")
                return d_observation
            else:
                raise BaseException(f"Space type {type(self._action_space)} is not handled for AddDeltaBinaryMask")
        else:
            binary_mark_metering_period = (
                1 if original_observation["metering_period_counter"] == self._Delta_M else 0
            )
            binary_mark_peak_period = (
                1 if original_observation["peak_period_counter"] == self._Delta_P else 0
            )
            if type(self._current_observation_space) == Box:
                return np.hstack([observation, np.asarray([float(binary_mark_metering_period), float(binary_mark_peak_period)])])
            elif type(self._current_observation_space) == TupleSpace:
                return action + [binary_mark_metering_period, binary_mark_peak_period]
            elif type(self._current_observation_space) == DictSpace:
                d_observation = dict(observation)
                d_observation["binary_mask_delta_m"] = binary_mark_metering_period
                d_observation["binary_mask_delta_p"] = binary_mark_peak_period
                return d_observation
            else:
                raise BaseException(f"Space type {type(self._action_space)} is not handled for AddDeltaBinaryMask")

    @staticmethod
    def get_kwargs_from_env_and_previous_converters(
        rec_env: RecEnv,
        previous_converters: List
    ):
        return {
            "Delta_M": rec_env.Delta_M,
            "Delta_P": rec_env.Delta_P
        }