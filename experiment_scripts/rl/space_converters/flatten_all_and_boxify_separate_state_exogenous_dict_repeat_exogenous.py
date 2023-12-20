from gym.spaces import Space
from experiment_scripts.rl.space_converters.flatten_all import FlattenAll

class FlattenAllAndSeparateBoxifyStateExogenousDictRepeatExogenous(FlattenAll):

    def __init__(self,
                 current_action_space: Space,
                 current_observation_space: Space,
                 current_reward_space: Space,
                 original_action_space: Space,
                 original_observation_space: Space,
                 original_reward_space: Space,
                 exogenous_space=None,
                 members=[]):
        super().__init__(current_action_space,
                         current_observation_space,
                         current_reward_space,
                         original_action_space,
                         original_observation_space,
                         original_reward_space,
                         discrete_to_box=True,
                         separate_exogenous_from_state=True,
                         transform_to_dict_when_separate=True,
                         exogenous_space=exogenous_space,
                         repeat_exogenous_value_in_state=True,
                         members=members)