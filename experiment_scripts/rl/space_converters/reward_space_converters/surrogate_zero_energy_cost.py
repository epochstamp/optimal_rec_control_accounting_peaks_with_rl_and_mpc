from typing import Any, Dict, List, Tuple, Union
from gym.spaces import Dict as DictSpace, Space, Box
from base.space_converter import SpaceConverter
import numpy as np
from pprint import pprint

from env.rec_env import RecEnv

class SurrogateZeroEnergyCost(SpaceConverter):

    def __init__(self,
                 current_action_space: Space,
                 current_observation_space: Space,
                 current_reward_space: Space,
                 original_action_space: Space,
                 original_observation_space: Space,
                 original_reward_space: Space,
                 members=[],
                 involve_peaks=False,
                 Delta_M=1,
                 Delta_P=1,
                 convert_obs=True,
                 convert_rew=True,
                 with_peak_costs=False):
        self._involve_peaks = involve_peaks
        self._Delta_M = Delta_M
        self._Delta_P = Delta_P
        self._members = members
        self._convert_obs = convert_obs
        self._convert_rew = convert_rew
        self._with_peak_costs = with_peak_costs
        super().__init__(current_action_space,
                         current_observation_space,
                         current_reward_space,
                         original_action_space,
                         original_observation_space,
                         original_reward_space)
        
    def _convert_observation_space(self):
        if not self._convert_obs:
            return self._current_observation_space
        if type(self._original_observation_space) != DictSpace:
            raise BaseException(f"Only DictSpace type {type(self._observation_space)} is handled for RemovePeakObservations")

        if self._with_peak_costs:
            return DictSpace(
                {
                    **self._current_observation_space,
                    **{
                        "previous_net_balance_cost": Box(low=0.0, high=1000000.0)
                    },
                    **{
                        "previous_peak_slope_cost": Box(low=0.0, high=1000000.0)
                    }
                }
            )
        else:
            return DictSpace(
                {
                    **self._current_observation_space,
                    **{
                        "previous_net_balance_cost": Box(low=0.0, high=1000000.0)
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
        if not self._convert_obs:
            return observation
        if not backward:
            if type(self._current_observation_space) != DictSpace:
                raise BaseException(f"Only DictSpace type {type(self._observation_space)} is handled for RemovePeakObservations")
            else:
                d_observation = dict(observation)
                metering_period_counter = kwargs["metering_period_counter"]
                peak_period_counter = kwargs.get("peak_period_counter", 0)
                previous_observation = kwargs.get("current_observation", None)
                previous_net_balance_cost, previous_peak_slope_cost = (
                    self._compute_surrogate_costs(observation, previous_observation=previous_observation, metering_period_counter=metering_period_counter, peak_period_counter=peak_period_counter, compute_delta=False)
                )
                d_observation["previous_net_balance_cost"] = (previous_net_balance_cost if metering_period_counter < self._Delta_M else 0)
                if self._with_peak_costs:
                    d_observation["previous_peak_slope_cost"] = previous_peak_slope_cost if peak_period_counter < self._Delta_P else 0
                return d_observation
        else:
            if type(self._current_observation_space) != DictSpace:
                raise BaseException(f"Only DictSpace type {type(self._observation_space)} is handled for RemovePeakObservations")
            else:
                d_observation = dict(observation)
                d_observation.pop("previous_net_balance_cost", None)
                d_observation.pop("previous_peak_slope_cost", None)
                return d_observation
            
    def _compute_surrogate_costs(self, observation, previous_observation=None, metering_period_counter=0, peak_period_counter=0, compute_delta=True):
        current_balance = float(sum([observation[(member, "net_meters")] for member in self._members])**2)
        current_balance_cost = current_balance
        if compute_delta and metering_period_counter < self._Delta_M:
            current_balance_cost = current_balance_cost - observation["previous_net_balance_cost"]
        current_peak_cost = 0.0
        if self._with_peak_costs:
            offtake_peak_slope = sum([(observation[(member, "current_offtake_peaks")]) for member in self._members])
            injection_peak_slope = sum([(observation[(member, "current_injection_peaks")]) for member in self._members])
            peak_slope_cost = offtake_peak_slope + injection_peak_slope
            current_peak_cost = peak_slope_cost
            if compute_delta and peak_period_counter < self._Delta_P:
                current_peak_cost -= observation["previous_peak_slope_cost"]
        return current_balance_cost*0.2, current_peak_cost*0.5
    
    
    def _convert_reward(self, reward: float | List[float] | Dict[Tuple[str, str] | str, float], observation: int | float | List[float] | List[int] | Dict[str | Tuple[str, str], int | float | List[float] | List[int]] = None, action: int | float | List[float] | List[int] | Dict[str | Tuple[str, str], int | float | List[float] | List[int]] = None, original_action: int | float | List[float] | List[int] | Dict[str | Tuple[str, str], int | float | List[float] | List[int]] = None, original_observation: int | float | List[float] | List[int] | Dict[str | Tuple[str, str], int | float | List[float] | List[int]] = None, original_reward: float | List[float] | Dict[Tuple[str, str] | str, float] = None, backward=False, **kwargs):
        if not self._convert_rew:
            return original_reward
        if not backward:
            metering_period_counter = kwargs["metering_period_counter"]
            peak_period_counter = kwargs.get("peak_period_counter", 0)
            previous_observation = kwargs.get("current_observation", None)
            new_reward = sum(self._compute_surrogate_costs(observation, previous_observation=previous_observation, metering_period_counter=metering_period_counter, peak_period_counter=peak_period_counter, compute_delta=True))
            return new_reward
        else:
            return original_reward
    
    @staticmethod
    def get_kwargs_from_env_and_previous_converters(
        rec_env: RecEnv,
        previous_converters: List
    ):
        return {
            "members": rec_env.members,
            "involve_peaks": rec_env.involve_peaks,
            "Delta_M": rec_env.Delta_M,
            "Delta_P": rec_env.Delta_P
        }
    

class SurrogateZeroEnergyCostRew(SurrogateZeroEnergyCost):
    def __init__(self,
                 current_action_space: Space,
                 current_observation_space: Space,
                 current_reward_space: Space,
                 original_action_space: Space,
                 original_observation_space: Space,
                 original_reward_space: Space,
                 members=[],
                 involve_peaks=False,
                 Delta_M=1,
                 Delta_P=1,
                 with_peak_costs=False):
        super().__init__(
            current_action_space,
            current_observation_space,
            current_reward_space,
            original_action_space,
            original_observation_space,
            original_reward_space,
            members=members,
            involve_peaks=involve_peaks,
            Delta_M=Delta_M,
            Delta_P=Delta_P,
            convert_obs=False,
            convert_rew=True,
            with_peak_costs=with_peak_costs
        )


class SurrogateZeroEnergyCostObs(SurrogateZeroEnergyCost):
    def __init__(self,
                 current_action_space: Space,
                 current_observation_space: Space,
                 current_reward_space: Space,
                 original_action_space: Space,
                 original_observation_space: Space,
                 original_reward_space: Space,
                 members=[],
                 involve_peaks=False,
                 Delta_M=1,
                 Delta_P=1,
                 with_peak_costs=False):
        super().__init__(
            current_action_space,
            current_observation_space,
            current_reward_space,
            original_action_space,
            original_observation_space,
            original_reward_space,
            members=members,
            involve_peaks=involve_peaks,
            Delta_M=Delta_M,
            Delta_P=Delta_P,
            convert_obs=True,
            convert_rew=False,
            with_peak_costs=with_peak_costs
        )

class SurrogateZeroEnergyCostRewWithPeaks(SurrogateZeroEnergyCostRew):
    def __init__(self,
                 current_action_space: Space,
                 current_observation_space: Space,
                 current_reward_space: Space,
                 original_action_space: Space,
                 original_observation_space: Space,
                 original_reward_space: Space,
                 members=[],
                 involve_peaks=False,
                 Delta_M=1,
                 Delta_P=1):
        super().__init__(
            current_action_space,
            current_observation_space,
            current_reward_space,
            original_action_space,
            original_observation_space,
            original_reward_space,
            members=members,
            involve_peaks=involve_peaks,
            Delta_M=Delta_M,
            Delta_P=Delta_P,
            with_peak_costs=True,
        )


class SurrogateZeroEnergyCostObsWithPeaks(SurrogateZeroEnergyCostObs):
    def __init__(self,
                 current_action_space: Space,
                 current_observation_space: Space,
                 current_reward_space: Space,
                 original_action_space: Space,
                 original_observation_space: Space,
                 original_reward_space: Space,
                 members=[],
                 involve_peaks=False,
                 Delta_M=1,
                 Delta_P=1):
        super().__init__(
            current_action_space,
            current_observation_space,
            current_reward_space,
            original_action_space,
            original_observation_space,
            original_reward_space,
            members=members,
            involve_peaks=involve_peaks,
            Delta_M=Delta_M,
            Delta_P=Delta_P,
            with_peak_costs=True
        )