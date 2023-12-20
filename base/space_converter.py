from typing import Dict, Any, Union, List, Tuple
from gym.spaces import Space
from env.rec_env import RecEnv


class SpaceConverter(object):

    def __init__(self,
                 current_action_space: Space,
                 current_observation_space: Space,
                 current_reward_space: Space,
                 original_action_space: Space,
                 original_observation_space: Space,
                 original_reward_space: Space):
        self._current_action_space = current_action_space
        self._current_observation_space = current_observation_space
        self._current_reward_space = current_reward_space
        self._original_action_space = original_action_space
        self._original_observation_space = original_observation_space
        self._original_reward_space = original_reward_space

        self._action_space = self.convert_action_space(
        )
        self._observation_space = self.convert_observation_space(
        )
        self._reward_space = self.convert_reward_space(
        )
        self.reset()


    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self) :
        return self._observation_space
    
    @property
    def reward_space(self):
        return self._reward_space

    def convert_action_space(self,
                              backward=False):
        if backward:
            return self._current_action_space
        else:
            return self._convert_action_space()
        
    def _convert_action_space(self):
        return self._current_action_space
    
    def convert_observation_space(self,
                                  backward=False):
        if backward:
            return self._current_observation_space
        else:
            return self._convert_observation_space()
        
    def _convert_observation_space(self):
        return self._current_observation_space
    
    def convert_reward_space(self,
                              backward=False,
                              eval=False):
        if eval:
            return self._convert_reward_space_eval(
                backward=backward
            )
        else:
            if backward:
                return self._current_reward_space
            else:
                return self._convert_reward_space(
                    backward=backward
                )
        
    def _convert_reward_space(self, backward=False, **kwargs):
        return self._current_reward_space
    
    def _convert_reward_space_eval(self, backward=False, **kwargs):
        return self._original_reward_space
    
    def convert_action(self,
                       action: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]],
                       observation: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       reward: Union[float, List[float], Dict[Union[Tuple[str, str], str], float]]=None,
                       original_action: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       original_observation: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       original_reward: Union[float, List[float], Dict[Union[Tuple[str, str], str], float]]=None,
                       backward=False,
                       **kwargs):
        return action
    
    
    
    def convert_observation(self,
                       observation: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]],
                       action: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       reward: Union[float, List[float], Dict[Union[Tuple[str, str], str], float]]=None,
                       original_action: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       original_observation: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       original_reward: Union[float, List[float], Dict[Union[Tuple[str, str], str], float]]=None,
                       backward=False,
                       **kwargs):
        return observation
    
    
    def convert_reward(self,
                       reward: Union[float, List[float], Dict[Union[Tuple[str, str], str], float]],
                       observation: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       action: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       original_action: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       original_observation: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       original_reward: Union[float, List[float], Dict[Union[Tuple[str, str], str], float]]=None,
                       backward=False,
                       eval=False,
                       **kwargs):
        if eval:
            return self._convert_reward_eval(
                reward,
                observation=observation,
                action=action,
                original_action=original_action,
                original_observation=original_observation,
                original_reward=original_reward,
                backward=backward,
                **kwargs
            )
        else:
            return self._convert_reward(
                reward,
                observation=observation,
                action=action,
                original_action=original_action,
                original_observation=original_observation,
                original_reward=original_reward,
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
        return reward
        
    def _convert_reward_eval(self,
                             reward: Union[float, List[float], Dict[Union[Tuple[str, str], str], float]],
                             observation: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                             action: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                             original_action: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                             original_observation: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                             original_reward: Union[float, List[float], Dict[Union[Tuple[str, str], str], float]]=None,
                             backward=False,
                             **kwargs):
        return reward
    
    def reset(self):
        pass

    @staticmethod
    def get_kwargs_from_env_and_previous_converters(
        rec_env: RecEnv,
        previous_converters: List
    ):
        return dict()
    
    