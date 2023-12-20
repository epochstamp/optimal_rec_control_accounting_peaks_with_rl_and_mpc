from typing import Any, Dict, Union, List, Tuple, Callable
from gym.spaces import Dict as DictSpace, Box, Space, Discrete, Tuple as TupleSpace, MultiDiscrete
from base.space_converter import SpaceConverter
from env.rec_env import RecEnv
import numpy as np

from utils.utils import flatten, normalize_bounds

class FlattenAll(SpaceConverter):

    def __init__(self,
                 current_action_space: Space,
                 current_observation_space: Space,
                 current_reward_space: Space,
                 original_action_space: Space,
                 original_observation_space: Space,
                 original_reward_space: Space,
                 discrete_to_box=False,
                 separate_exogenous_from_state=False,
                 exogenous_space=None,
                 transform_to_dict_when_separate=False,
                 repeat_exogenous_value_in_state=False,
                 members=[]):
        self._transform_to_dict_when_separate = transform_to_dict_when_separate
        self._repeat_exogenous_value_in_state = repeat_exogenous_value_in_state
        self._exogenous_space = {
            **exogenous_space,
            **{
                (member, "net_prod_cons"): Box(-10000000, 10000000, shape=(
                        exogenous_space[(member, "production")] if (member, "production") in exogenous_space else exogenous_space
                    ).shape) for member in members
            }
        }
        self._exogenous_keys = list(self._exogenous_space.keys())
        if type(current_observation_space) == DictSpace:
            if separate_exogenous_from_state:
                self._state_keys_boxes = [k for k in current_observation_space.keys() if type(current_observation_space[k]) == Box and (repeat_exogenous_value_in_state or k not in self._exogenous_keys)]
                self._exogenous_keys_boxes = [k for k in current_observation_space.keys() if k in self._exogenous_keys]
                self._obs_keys_discretes = [k for k in current_observation_space.keys() if type(current_observation_space[k]) == Discrete]
            else:
                self._obs_keys_boxes = [k for k in current_observation_space.keys() if type(current_observation_space[k]) == Box]
                self._obs_keys_discretes = [k for k in current_observation_space.keys() if type(current_observation_space[k]) == Discrete]
        elif type(current_observation_space) == TupleSpace:
            self._obs_keys_boxes = [k for k in range(len(current_observation_space)) if type(current_observation_space[k]) == Box]
            self._obs_keys_discretes = [k for k in range(len(current_observation_space)) if type(current_observation_space[k]) == Discrete]

        if type(current_action_space) == DictSpace:
            self._act_keys_boxes = [k for k in current_action_space.keys() if type(current_action_space[k]) == Box]
            self._act_keys_discretes = [k for k in current_action_space.keys() if type(current_action_space[k]) == Discrete]
        elif type(current_action_space) == TupleSpace:
            self._act_keys_boxes = [k for k in range(len(current_action_space)) if type(current_action_space[k]) == Box]
            self._act_keys_discretes = [k for k in range(len(current_action_space)) if type(current_action_space[k]) == Discrete]
        self._discrete_to_box = discrete_to_box
        self._separate_exogenous_from_state = separate_exogenous_from_state
        self._exogenous_space = exogenous_space
        
        super().__init__(current_action_space,
                         current_observation_space,
                         current_reward_space,
                         original_action_space,
                         original_observation_space,
                         original_reward_space)
        

    
    def _convert_observation_space(self):
        if type(self._current_observation_space) not in [DictSpace, TupleSpace]:
            return self._current_observation_space
        else:
            if self._separate_exogenous_from_state and len(self._exogenous_keys_boxes) > 0:
                boxes_state = [self._current_observation_space[k] for k in self._state_keys_boxes]
                boxes_exogenous = [self._current_observation_space[k] for k in self._exogenous_keys_boxes]
                low_state_boxes = np.hstack(flatten([[-1000000]*max(sum(box.shape), 1) for box in boxes_state]))
                high_state_boxes = np.hstack(flatten([[1000000]*max(sum(box.shape), 1) for box in boxes_state]))
                low_exogenous_boxes = np.hstack(flatten([[-1000000]*max(sum(box.shape), 1) for box in boxes_exogenous]))
                high_exogenous_boxes = np.hstack(flatten([[1000000]*max(sum(box.shape), 1) for box in boxes_exogenous]))
                discretes = [self._current_observation_space[k] for k in self._obs_keys_discretes]
                multidiscrete = None
                box_state = Box(low=low_state_boxes, high=high_state_boxes, dtype=np.float32)
                box_exogenous = Box(low=low_exogenous_boxes, high=high_exogenous_boxes, dtype=np.float32)
                if discretes != []:
                    n_discrete = np.hstack([d.n for d in discretes])
                    if not self._discrete_to_box:
                        multidiscrete = MultiDiscrete(n_discrete, dtype=np.int32)
                        whole_space=TupleSpace((box_state, multidiscrete, box_exogenous))
                    else:
                        new_state_lows = np.asarray([-10000] * len(discretes), dtype=np.float32) 
                        new_state_highs = np.asarray([10000] * len(discretes), dtype=np.float32) 
                        box_state = Box(low=np.hstack([low_state_boxes, new_state_lows]), high=np.hstack([new_state_highs, high_state_boxes]))
                        whole_space=TupleSpace((box_state, box_exogenous))
                else:
                    whole_space=TupleSpace((box_state, box_exogenous))
                if self._transform_to_dict_when_separate:
                    #print(box_state.shape)
                    #print(whole_space)
                    whole_space = DictSpace({
                        "states": box_state,
                        "exogenous": box_exogenous
                    })
                    #print(sum([(sum(v.shape)) for v in list(self._current_observation_space.values())]))
                    #print(sum([(sum(v.shape)) for v in list(whole_space.values())]))
                    #print(self._state_keys_boxes)
                    if multidiscrete is not None:
                        whole_space["multidiscrete"] = multidiscrete
                return whole_space
                    
            else:
                
                boxes = [self._current_observation_space[k] for k in self._obs_keys_boxes]
                discretes = [self._current_observation_space[k] for k in self._obs_keys_discretes]
                low_boxes = np.hstack([box.low for box in boxes])
                high_boxes = np.hstack([box.high for box in boxes])
                box = Box(low=low_boxes, high=high_boxes, dtype=np.float32)
                multidiscrete=None
                if discretes != []:
                    n_discrete = np.hstack([d.n for d in discretes])
                    if not self._discrete_to_box:
                        
                        multidiscrete = MultiDiscrete(n_discrete, dtype=np.int32)
                    else:
                        new_lows = np.asarray([0.0] * len(discretes), dtype=np.int32) 
                        new_highs = n_discrete.astype(np.float32)
                        box = Box(low=np.hstack([low_boxes, new_lows]), high=np.hstack([high_boxes, new_highs]))
                return TupleSpace((box, multidiscrete)) if multidiscrete is not None else box
        
        

    def _convert_action_space(self):
        if type(self._current_action_space) not in [DictSpace, TupleSpace]:
            return self._current_action_space
        else:
            boxes = [self._current_action_space[k] for k in self._act_keys_boxes]
            discretes = [self._current_action_space[k] for k in self._act_keys_discretes]
            box = None
            if len(boxes) > 0:
                low_boxes = np.hstack([box.low for box in boxes])
                high_boxes = np.hstack([box.high for box in boxes])
            
                box = Box(low=low_boxes, high=high_boxes, dtype=np.float32)
            multidiscrete=None
            if discretes != []:
                n_discrete = np.hstack([d.n for d in discretes], dtype=np.int32)
                if len(discretes) > 1:
                    multidiscrete = MultiDiscrete(n_discrete, dtype=np.int32)
                else:
                    multidiscrete = discretes[0]
                """
                if not self._discrete_to_box:
                    if len(discretes) > 1:
                        multidiscrete = MultiDiscrete(n_discrete, dtype=np.int32)
                    else:
                        multidiscrete = discretes[0]
                else:
                    new_lows = np.asarray([0.0] * len(discretes), dtype=np.int32) 
                    new_highs = n_discrete.astype(np.float32)
                    box = Box(low=np.hstack([low_boxes, new_lows]), high=np.hstack([high_boxes, new_highs]))
                """
            if box is None and multidiscrete is not None:
                space = multidiscrete
            elif box is not None and multidiscrete is None:
                space = box
            else:
                space=TupleSpace((box, multidiscrete)) 
            return space
        
    def _convert_reward_space(self, backward=False):
        
        if not backward and type(self._current_reward_space) in (DictSpace, TupleSpace):
            return Box(low=-10000, high=10000)
        else:
            return self._current_reward_space
    
    def convert_observation(self,
                       observation: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]],
                       action: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       reward: Union[float, List[float], Dict[Union[Tuple[str, str], str], float]]=None,
                       original_action: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       original_observation: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       original_reward: Union[float, List[float], Dict[Union[Tuple[str, str], str], float]]=None,
                       backward=False,
                       **kwargs):
        if type(self._current_observation_space) not in [DictSpace, TupleSpace]:
            return observation
        else:
            if backward:
                if self._separate_exogenous_from_state and len(self._exogenous_keys_boxes) > 0:
                    if self._discrete_to_box:
                        obs_idxs = list(self._state_keys_boxes) + list(self._exogenous_keys_boxes)
                    else:
                        obs_idxs = list(self._state_keys_boxes) + list(self._obs_keys_discretes) + list(self._exogenous_keys_boxes)
                    obs_idxs = [k for k in obs_idxs if k in list(self._current_observation_space.keys())]
                else:
                    obs_idxs = list(self._obs_keys_boxes) + list(self._obs_keys_discretes)
                if type(self._current_observation_space) == DictSpace:
                    new_observation = dict()
                else:
                    new_observation = [None]*len(obs_idxs)
                if type(observation) in (tuple, list):
                    if self._separate_exogenous_from_state and len(self._exogenous_keys_boxes) > 0 and not self._discrete_to_box:
                        current_observation = list(observation[0]) + list(observation[1]) + list(observation[2])
                    else:
                        current_observation = list(observation[0]) + list(observation[1])
                elif type(observation) == dict:
                    if self._separate_exogenous_from_state and len(self._exogenous_keys_boxes) > 0 and not self._discrete_to_box:
                        current_observation = list(observation["states"]) + list(observation["multidiscrete"]) + list(observation["exogenous"])
                    else:
                        current_observation = list(observation["states"]) + list(observation["exogenous"])
                else:
                    current_observation = observation
                for i, obs_idx in enumerate(obs_idxs):
                    new_observation[obs_idx] = current_observation[i]
            else:
                if self._separate_exogenous_from_state and len(self._exogenous_keys_boxes) > 0:
                    continuous_state_values = np.hstack([observation[k] for k in self._state_keys_boxes]).astype(np.float32)
                    continuous_exogenous_values = np.hstack([observation[k] for k in self._exogenous_keys_boxes]).astype(np.float32)
                    discrete_values = None
                    if len(self._obs_keys_discretes) > 0:
                        discrete_values = np.hstack([observation[k] for k in self._obs_keys_discretes]).astype(np.int32)
                        if not self._discrete_to_box:
                            new_observation = (continuous_state_values, discrete_values, continuous_exogenous_values)
                        else:
                            discrete_values = discrete_values.astype(np.float32)
                            continuous_state_values = np.hstack([continuous_state_values, discrete_values])
                            new_observation = (continuous_state_values, continuous_exogenous_values)
                    else:
                        new_observation = (continuous_state_values, continuous_exogenous_values)
                    if self._transform_to_dict_when_separate:
                        new_observation = {
                            "states": continuous_state_values,
                            "exogenous": continuous_exogenous_values
                        }
                        if discrete_values is not None:
                            new_observation["multidiscrete"] = discrete_values

                else:
                    continuous_values = np.hstack([observation[k] for k in self._obs_keys_boxes]).astype(np.float32)
                    new_observation = continuous_values
                    if len(self._obs_keys_discretes) > 0:
                        discrete_values = np.hstack([observation[k] for k in self._obs_keys_discretes]).astype(np.int32)
                        if not self._discrete_to_box:
                            new_observation = (continuous_values, discrete_values)
                        else:
                            discrete_values = discrete_values.astype(np.float32)
                            new_observation = np.hstack([continuous_values, discrete_values])
            #if not backward:
                #from pprint import pprint
                #pprint(observation)
                #exit()
            return new_observation
        
    def convert_action(self,
                       action: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]],
                       observation: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       reward: Union[float, List[float], Dict[Union[Tuple[str, str], str], float]]=None,
                       original_action: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       original_observation: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       original_reward: Union[float, List[float], Dict[Union[Tuple[str, str], str], float]]=None,
                       backward=False,
                       **kwargs):
        if type(self._current_action_space) not in [DictSpace, TupleSpace]:
            return action
        else:
            if backward:
                act_idxs = self._act_keys_boxes + self._act_keys_discretes
                if type(self._current_action_space) == DictSpace:
                    new_action = dict()
                else:
                    new_action = [None]*len(act_idxs)
                if type(action) not in (tuple, list):
                    action = [action]
                for i, act_idx in enumerate(act_idxs):
                    new_action[act_idx] = action[i]
            else:
                continuous_values=[]
                if len(self._act_keys_boxes) > 0:
                    continuous_values = np.hstack([action[k] for k in self._act_keys_boxes]).astype(dtype=np.float32)
                
                new_action = continuous_values
                if len(self._act_keys_discretes) > 0:
                    discrete_values = np.hstack([action[k] for k in self._act_keys_discretes]).astype(dtype=np.int32)
                    if not self._discrete_to_box:
                        if continuous_values != []:
                            new_action = (continuous_values, discrete_values)
                        else:
                            new_action = discrete_values
                    else:
                        if continuous_values != []:
                            discrete_values = discrete_values.astype(np.float32)
                            new_action = np.hstack([continuous_values, discrete_values]).astype(dtype=np.float32)
                        else:
                            new_action = discrete_values
            return new_action
        
    def _convert_reward(self,
                       reward: Union[float, List[float], Dict[Union[Tuple[str, str], str], float]],
                       observation: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       action: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       original_action: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       original_observation: Union[int, float, List[float], List[int], Dict[Union[str, Tuple[str, str]], Union[int, float, List[float], List[int]]]]=None,
                       original_reward: Union[float, List[float], Dict[Union[Tuple[str, str], str], float]]=None,
                       backward=False,
                       **kwargs):
        if backward:
            return original_reward
        else:
            if type(reward) == dict:
                return sum(list(reward.values()))
            elif type(reward) in (tuple, list):
                return sum(reward)
            else:
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
        if backward:
            return original_reward
        else:
            if type(reward) == dict:
                return sum(list(reward.values()))
            elif type(reward) in (tuple, list):
                return sum(reward)
            else:
                return reward
            
    @staticmethod
    def get_kwargs_from_env_and_previous_converters(
        rec_env: RecEnv,
        previous_converters: List
    ):
        return {
            "exogenous_space": rec_env.exogenous_space,
            "members": rec_env.members
        }