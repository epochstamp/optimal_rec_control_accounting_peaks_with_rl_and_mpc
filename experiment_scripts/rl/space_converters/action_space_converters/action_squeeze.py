from typing import Any, Dict, Union, List, Tuple, Callable
from gym.spaces import Dict as DictSpace, Box, Space, Discrete
from base.space_converter import SpaceConverter
import numpy as np
from env.rec_env import RecEnv

class ActionSqueeze(SpaceConverter):

    def __init__(self,
                 current_action_space: Space,
                 current_observation_space: Space,
                 current_reward_space: Space,
                 original_action_space: Space,
                 original_observation_space: Space,
                 original_reward_space: Space,
                 action_space_squeezer: Callable[[Union[Box, Discrete, List[Space], DictSpace]], Union[List[Tuple[int, int]], List[Tuple[str, str]], List[Tuple[Tuple[str, str], Tuple[str, str]]], List[int], List[str], List[Tuple[str, str]]]]= lambda action_spaces: []):
        self._action_space_squeezer = action_space_squeezer
        self._action_spaces_to_squeeze = self._action_space_squeezer(current_action_space)
        super().__init__(current_action_space,
                         current_observation_space,
                         current_reward_space,
                         original_action_space,
                         original_observation_space,
                         original_reward_space)
        
    
    def _convert_action_space(self):
        if self._action_spaces_to_squeeze == []:
            return self._current_action_space
        if type(self._action_spaces_to_squeeze[0][0]) == int:
            lb_list = [-self._current_action_space[action_space_to_squeeze[0]].high for action_space_to_squeeze in self._action_spaces_to_squeeze]
            ub_list = [self._current_action_space[action_space_to_squeeze[1]].high for action_space_to_squeeze in self._action_spaces_to_squeeze]
            return Box(low=np.asarray(lb_list), high=np.asarray(ub_list))
        else:
            new_action_space = {
                (key_1, key_2): Box(low=-self._current_action_space[key_2].high, high=self._current_action_space[key_1].high)
            for (key_1, key_2) in self._action_spaces_to_squeeze}
            return DictSpace(new_action_space)
    
    def convert_action(self,
                       action: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]],
                       observation: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       reward: Union[float, List[float], Dict[Union[Tuple[str, str], str], float]]=None,
                       original_action: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       original_observation: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       original_reward: Union[float, List[float], Dict[Union[Tuple[str, str], str], float]]=None,
                       backward=False,
                       **kwargs):
        if self._action_spaces_to_squeeze == []:
            return action
        if not backward:
            
            if type(self._action_spaces_to_squeeze[0][0]) == int:
                return [action[action_space_to_squeeze[0]] - action[action_space_to_squeeze[1]] for action_space_to_squeeze in self._action_spaces_to_squeeze]
            else:
                new_action = {
                    (key_1, key_2): action[key_1] - action[key_2]
                for key_1, key_2 in self._action_spaces_to_squeeze}
                return new_action
        else:
            if type(self._action_spaces_to_squeeze[0][0]) == int:
                
                list_actions = [0]*self._current_action_space.shape[0]
                for i, (i1, i2) in enumerate(self._action_spaces_to_squeeze):
                    list_actions[i1] = max(action[i], 0)
                    list_actions[i2] = -min(action[i], 0)
                return list_actions
            else:
                d_actions = dict()
                for k1, k2 in self._action_spaces_to_squeeze:
                    k = (k1, k2)
                    d_actions[k1] = action[k] if action[k] > 0 else 0
                    d_actions[k2] = -action[k] if action[k] < 0 else 0
                return d_actions

    @staticmethod
    def get_kwargs_from_env_and_previous_converters(
        rec_env: RecEnv,
        previous_converters: List
    ):
        return {
            "action_space_squeezer": None
        }