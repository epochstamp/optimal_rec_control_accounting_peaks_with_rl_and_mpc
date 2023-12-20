from typing import Any, Dict, List, Tuple
from gym.spaces import Dict as DictSpace, Space
from base.space_converter import SpaceConverter
import numpy as np

class RemoveCost(SpaceConverter):

    def __init__(self,
                 current_action_space: Space,
                 current_observation_space: Space,
                 current_reward_space: Space,
                 original_action_space: Space,
                 original_observation_space: Space,
                 original_reward_space: Space,
                 key_cost_to_remove: str,
                 when_to_keep_costs = None,
                 apply_for_train=True,
                 apply_for_eval=True):
        self._key_cost_to_remove = key_cost_to_remove
        self._when_to_keep_costs = when_to_keep_costs
        self._apply_for_train=apply_for_train
        self._apply_for_eval=apply_for_eval
        super().__init__(current_action_space,
                         current_observation_space,
                         current_reward_space,
                         original_action_space,
                         original_observation_space,
                         original_reward_space)
        
    def _inner_convert_reward_space(self, backward=False):
        if type(self._current_reward_space) != DictSpace:
            raise BaseException(f"Only DictSpace type {type(self._observation_space)} is handled for RemoveControllableAssetsCosts")
        if not backward:
            return {
                k:s for k,s in self._current_reward_space.items() if k != self._key_cost_to_remove or self._when_to_keep_costs is None
            }
        else:
            if self._when_to_keep_costs is None:
                return {
                    **self._current_reward_space,
                    **{
                        self._key_cost_to_remove: self._original_reward_space[self._key_cost_to_remove]
                    }
                }
            else:
                return self._current_reward_space
            
    def _convert_reward_space(self, backward=False):
        return self._inner_convert_reward_space(backward=backward) if self._apply_for_train else self._current_reward_space
    
    def _convert_reward_space_eval(self, backward=False):
        return self._inner_convert_reward_space(backward=backward) if self._apply_for_eval else self._current_reward_space
    
    def _inner_convert_reward(self, reward: float | List[float] | Dict[Tuple[str, str] | str, float], observation: int | float | List[float] | List[int] | Dict[str | Tuple[str, str], int | float | List[float] | List[int]] = None, action: int | float | List[float] | List[int] | Dict[str | Tuple[str, str], int | float | List[float] | List[int]] = None, original_action: int | float | List[float] | List[int] | Dict[str | Tuple[str, str], int | float | List[float] | List[int]] = None, original_observation: int | float | List[float] | List[int] | Dict[str | Tuple[str, str], int | float | List[float] | List[int]] = None, original_reward: float | List[float] | Dict[Tuple[str, str] | str, float] = None, backward=False, **kwargs):
        if type(self._current_reward_space) != DictSpace:
            raise BaseException(f"Only DictSpace type {type(self._observation_space)} is handled for RemoveControllableAssetsCosts")
        tau_m = original_observation["metering_period_counter"]
        tau_p = original_observation.get("peak_period_counter", None)
        if not backward:
            return {
                k:v for k,v in reward.items() if k != self._key_cost_to_remove and (self._when_to_keep_costs is None or self._when_to_keep_costs(tau_m, tau_p))
            }
        else:
            return {
                **self._current_reward_space,
                **{
                    self._key_cost_to_remove: original_reward[self._key_cost_to_remove]
                }
            }
        
    def _convert_reward(self, reward: float | List[float] | Dict[Tuple[str, str] | str, float], observation: int | float | List[float] | List[int] | Dict[str | Tuple[str, str], int | float | List[float] | List[int]] = None, action: int | float | List[float] | List[int] | Dict[str | Tuple[str, str], int | float | List[float] | List[int]] = None, original_action: int | float | List[float] | List[int] | Dict[str | Tuple[str, str], int | float | List[float] | List[int]] = None, original_observation: int | float | List[float] | List[int] | Dict[str | Tuple[str, str], int | float | List[float] | List[int]] = None, original_reward: float | List[float] | Dict[Tuple[str, str] | str, float] = None, backward=False, **kwargs):
        return self._inner_convert_reward() if self._apply_for_train else reward
    
    def _convert_reward_eval(self, reward: float | List[float] | Dict[Tuple[str, str] | str, float], observation: int | float | List[float] | List[int] | Dict[str | Tuple[str, str], int | float | List[float] | List[int]] = None, action: int | float | List[float] | List[int] | Dict[str | Tuple[str, str], int | float | List[float] | List[int]] = None, original_action: int | float | List[float] | List[int] | Dict[str | Tuple[str, str], int | float | List[float] | List[int]] = None, original_observation: int | float | List[float] | List[int] | Dict[str | Tuple[str, str], int | float | List[float] | List[int]] = None, original_reward: float | List[float] | Dict[Tuple[str, str] | str, float] = None, backward=False, **kwargs):
        return self._inner_convert_reward() if self._apply_for_eval else reward