from typing import Any, Dict, Union, List, Tuple, Callable
from gym.spaces import Dict as DictSpace, Box, Space, Discrete
from base.space_converter import SpaceConverter
import numpy as np
from env.rec_env import RecEnv
from utils.utils import epsilonify

class Discretize_Action(SpaceConverter):

    def __init__(self,
                 current_action_space: Space,
                 current_observation_space: Space,
                 current_reward_space: Space,
                 original_action_space: Space,
                 original_observation_space: Space,
                 original_reward_space: Space
    ):
        super().__init__(current_action_space,
                         current_observation_space,
                         current_reward_space,
                         original_action_space,
                         original_observation_space,
                         original_reward_space)
        
    
    def _convert_action_space(self):
        if type(self._current_action_space) != DictSpace or len(list(self._current_action_space.keys())) > 1:
            raise BaseException("Non-dict actions or more than 1 action is not handled atm")
        self._action_key = list(self._current_action_space.keys())[0]
        return DictSpace(
            {
                self._action_key: Discrete(3)
            }
        )
    
    def convert_action(self,
                       action: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]],
                       observation: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       reward: Union[float, List[float], Dict[Union[Tuple[str, str], str], float]]=None,
                       original_action: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       original_observation: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       original_reward: Union[float, List[float], Dict[Union[Tuple[str, str], str], float]]=None,
                       backward=False,
                       **kwargs):
        if type(self._current_action_space) != DictSpace or len(list(self._current_action_space.keys())) > 1:
            raise BaseException("Non-dict actions or more than 1 action is not handled atm", self._current_action_space)
        if not backward:
            
            if action[self._action_key] < 0:
                return {
                    self._action_key: 0
                }
            elif epsilonify(action[self._action_key]) == 0:
                return {
                    self._action_key: 1
                }
            else:
                return {
                    self._action_key: 2
                }

        else:
            if action[self._action_key] == 0:
                new_action = {
                    self._action_key: float(self._current_action_space[self._action_key].low)
                }
            elif action[self._action_key] == 1:
                new_action = {
                    self._action_key: 0.0
                }
            elif action[self._action_key] == 2:
                new_action = {
                    self._action_key: float(self._current_action_space[self._action_key].high)
                }
            else:
                raise BaseException("Unknown action", action[self._action_key])
            #print(action, new_action)
            return new_action