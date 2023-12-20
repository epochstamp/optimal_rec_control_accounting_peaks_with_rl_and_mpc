from gym.spaces import Space
from experiment_scripts.rl.space_converters.reward_space_converters.remove_cost import RemoveCost
from experiment_scripts.generic.trigger_zoo import metering_period_trigger_global_bill_functions

class RemoveMeteringPeriodCosts(RemoveCost):

    def __init__(self,
                 current_action_space: Space,
                 current_observation_space: Space,
                 current_reward_space: Space,
                 original_action_space: Space,
                 original_observation_space: Space,
                 original_reward_space: Space):
        super().__init__(
            current_action_space,
            current_observation_space,
            current_reward_space,
            original_action_space,
            original_observation_space,
            original_reward_space,
            key_cost_to_remove="metering_period_cost",
            when_to_keep_costs=metering_period_trigger_global_bill_functions["default"]
        )