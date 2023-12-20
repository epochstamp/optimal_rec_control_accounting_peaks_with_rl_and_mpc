from typing import Any, Dict, Union, List, Tuple, Callable
from gym.spaces import Dict as DictSpace, Box, Space, Discrete
from base.space_converter import SpaceConverter

class SequentialSpaceConverter(SpaceConverter):

    def __init__(self,
                 current_action_space: Space,
                 current_observation_space: Space,
                 current_reward_space: Space,
                 original_action_space: Space,
                 original_observation_space: Space,
                 original_reward_space: Space,
                 space_converters: List[SpaceConverter]):
        self._space_converters_list = list(space_converters)
        self._space_converters_reversed_list = list(reversed(self._space_converters_list))
        super().__init__(current_action_space,
                         current_observation_space,
                         current_reward_space,
                         original_action_space,
                         original_observation_space,
                         original_reward_space)
        

    def _space_converters(self, backward=False):
        if backward:
            return self._space_converters_reversed_list
        else:
            return self._space_converters_list
        
    def _convert_space(self, current_space, space_converter_method_getter, backward=False):
        new_space = current_space
        for space_converter in self._space_converters(backward=backward):
            new_space = space_converter_method_getter(space_converter)(backward=backward)
        return new_space
    
    def _convert_space_value(self, current_value, current_values, space_value_converter_method_getter, backward=False, eval_env=False, **kwargs):
        new_value = current_value
        for space_converter in self._space_converters(backward=backward):
            new_value = space_value_converter_method_getter(space_converter)(new_value, backward=backward, **{**kwargs, **current_values})
        return new_value
    
    def _build_values(
        self,
        keys_to_skip=set(),
        **kwargs
    ):
        return {
            k:v for k,v in kwargs.items() if k not in keys_to_skip
        }

    def convert_action_space(self, backward=False):
        return self._convert_space(
            self._current_action_space,
            lambda space_converter: space_converter.convert_action_space,
            backward=backward
        )
    
        
    def convert_observation_space(self, backward=False):
        return self._convert_space(
            self._current_observation_space,
            lambda space_converter: space_converter.convert_observation_space,
            backward=backward
        )
    
    def _convert_reward_space(self, backward=False, **kwargs):
        return self._convert_space(
            self._current_reward_space,
            lambda space_converter: space_converter._convert_reward_space,
            backward=backward
        )
    
    def _convert_reward_space_eval(self, backward=False, **kwargs):
        return self._convert_space(
            self._current_reward_space,
            lambda space_converter: space_converter._convert_reward_space_eval,
            backward=backward
        )
        
    def convert(self,
                action: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                observation: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                reward: Union[float, List[float], Dict[Union[Tuple[str, str], str], float]]=None,
                original_action: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                original_observation: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                original_reward: Union[float, List[float], Dict[Union[Tuple[str, str], str], float]]=None,
                backward=False,
                eval_env=False,
                **kwargs):
        current_action = action
        current_observation = observation
        current_reward = reward
        for space_converter in self._space_converters(backward=backward):
            current_action_temp = None
            current_observation_temp = None
            current_reward_temp = None
            if current_action is not None:
                current_action_temp = space_converter.convert_action(
                    current_action,
                    observation=current_observation,
                    reward=current_reward,
                    original_action=original_action,
                    original_observation=original_observation,
                    original_reward=original_reward,
                    backward=backward,
                    **kwargs)
            if current_observation is not None:
                current_observation_temp = space_converter.convert_observation(
                    current_observation,
                    action=current_action,
                    reward=current_reward,
                    original_action=original_action,
                    original_observation=original_observation,
                    original_reward=original_reward,
                    backward=backward,
                    **kwargs)
            if current_reward is not None:
                current_reward_temp = space_converter.convert_reward(
                    current_reward,
                    observation=current_observation,
                    action=current_action,
                    original_action=original_action,
                    original_observation=original_observation,
                    original_reward=original_reward,
                    backward=backward,
                    eval=eval_env,
                    **kwargs)
            current_action = current_action_temp
            current_observation = current_observation_temp
            current_reward = current_reward_temp
        return current_observation, current_action, current_reward
    
    def convert_action(self,
                       action: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]],
                       observation: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       reward: Union[float, List[float], Dict[Union[Tuple[str, str], str], float]]=None,
                       original_action: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       original_observation: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       original_reward: Union[float, List[float], Dict[Union[Tuple[str, str], str], float]]=None,
                       backward=False,
                       **kwargs):
        values = self._build_values(
            key_to_skip={"action", "backward", "eval"},
            action=action,
            observation=observation,
            reward=reward,
            original_action=original_action,
            original_observation=original_observation,
            original_reward=original_reward,
            **kwargs
        )
        return self._convert_space_value(
            action,
            values,
            lambda space_value_converter: space_value_converter.convert_action,
            backward=backward,
            **kwargs
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
        values = self._build_values(
            key_to_skip= {"observation", "backward", "eval"},
            action=action,
            observation=observation,
            reward=reward,
            original_action=original_action,
            original_observation=original_observation,
            original_reward=original_reward,
            **kwargs
        )
        return self._convert_space_value(
            observation,
            values,
            lambda space_value_converter: space_value_converter.convert_observation,
            backward=backward,
            **kwargs
        )
    
    
    def _convert_reward(self,
                       reward: Union[float, List[float], Dict[Union[Tuple[str, str], str], float]],
                       observation: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       action: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       original_action: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       original_observation: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       original_reward: Union[float, List[float], Dict[Union[Tuple[str, str], str], float]]=None,
                       backward=False,
                       **kwargs):
        values = self._build_values(
            key_to_skip={"reward", "backward"},
            action=action,
            observation=observation,
            reward=reward,
            original_action=original_action,
            original_observation=original_observation,
            original_reward=original_reward
        )
        return self._convert_space_value(
            reward,
            values,
            lambda space_value_converter: space_value_converter._convert_reward,
            backward=backward,
            **kwargs
        )
        
    def _convert_reward_eval(self,
                       reward: Union[float, List[float], Dict[Union[Tuple[str, str], str], float]],
                       observation: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       action: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       original_action: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       original_observation: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       original_reward: Union[float, List[float], Dict[Union[Tuple[str, str], str], float]]=None,
                       backward=False,
                       **kwargs):
        values = self._build_values(
            key_to_skip={"reward", "backward"},
            action=action,
            observation=observation,
            reward=reward,
            original_action=original_action,
            original_observation=original_observation,
            original_reward=original_reward
        )
        return self._convert_space_value(
            reward,
            values,
            lambda space_value_converter: space_value_converter._convert_reward_eval,
            backward=backward,
            **kwargs
        )
        
        
    def reset(self):
        for space_converter in self._space_converters():
            space_converter.reset()