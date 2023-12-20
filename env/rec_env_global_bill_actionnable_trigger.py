from typing import Optional, Tuple, Union
from gym.core import ObsType, ActType
from gym.spaces import Dict as DictSpace, Box, Discrete
import numpy as np
from env.rec_env import RecEnv
from env.rec_env_global_bill_observe_current_peaks import RecEnvGlobalBillTrigger
from env.rec_env_global_bill_wrapper import RecEnvGlobalBillWrapper
from utils.utils import epsilonify, merge_dicts
from exceptions import *
from copy import deepcopy

"""
TODO : Carefully check that class to introduce virtual costs.


"""
class RecEnvGlobalActionnableBillTrigger(RecEnvGlobalBillTrigger):

    def __init__(
            self,
            rec_env: Union[RecEnv, RecEnvGlobalBillWrapper],
            metering_period_cost_trigger= lambda tau_m, tau_p: False,
            peak_period_cost_trigger= lambda tau_m, tau_p: False,
            global_bill_optimiser_greedy_init = False,
            incremental_build_flag = False,
            penalty_actionnable_triggers=0,
            **kwargs
        ):
        super().__init__(
            rec_env,
            metering_period_cost_trigger=metering_period_cost_trigger,
            peak_period_cost_trigger=peak_period_cost_trigger,
            global_bill_optimiser_greedy_init=global_bill_optimiser_greedy_init,
            incremental_build_flag=incremental_build_flag

        )
        self._penalty_actionnable_triggers = penalty_actionnable_triggers
        trigger_period_actions = {
            "metering_period_trigger": Discrete(1)
        }
        if self.involve_peaks:
            trigger_period_actions["peak_period_trigger"] = Discrete(1)
        self.action_space = DictSpace({
            **dict(self.action_space),
            **trigger_period_actions
        })
        self.global_bill_adaptative_optimiser.greedy_init = global_bill_optimiser_greedy_init
        self.global_bill_adaptative_optimiser.incremental_build_flag = incremental_build_flag
    
    def _get_additionals_kwargs(self):
        return {
            "metering_period_cost_trigger": self._metering_period_cost_trigger,
            "peak_period_cost_trigger": self._peak_period_cost_trigger,
            "global_bill_optimiser_greedy_init": self.global_bill_adaptative_optimiser.greedy_init,
            "incremental_build_flag": self.global_bill_adaptative_optimiser.incremental_build_flag,
            "penalty_actionnable_triggers": self._penalty_actionnable_triggers
        }

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        next_observation, underlying_cost, terminated,  truncated, info = super().step(
            action
        )
        if terminated:
            return next_observation, underlying_cost, terminated, truncated, info

        metering_period_trigger = bool(action["metering_period_trigger"])
        peak_period_trigger = bool(action["peak_period_trigger"])
        cost = underlying_cost
        wrapped_info = dict(info)
        if not (info["is_metering_period_cost_triggered"] or info["is_peak_period_cost_triggered"]):
            if metering_period_trigger or peak_period_trigger:
                current_state = self.compute_current_state()
                current_exogenous_sequences = self.compute_current_exogenous_sequences()
                metering_period_cost, peak_period_cost, _, _ = self.global_bill_adaptative_optimiser.optimise_global_bill(
                    current_state,
                    current_exogenous_sequences
                )
                
                wrapped_info["is_metering_period_cost_triggered"] = metering_period_trigger
                wrapped_info["is_peak_period_cost_triggered"] = peak_period_trigger
                wrapped_info["costs"]["original_metering_period_cost"] = metering_period_cost if metering_period_cost is not None else 0.0
                wrapped_info["costs"]["original_peak_period_cost"] = peak_period_cost if peak_period_cost is not None else 0.0
                wrapped_info["costs"]["metering_period_cost"] = metering_period_cost if metering_period_cost is not None else 0.0
                wrapped_info["costs"]["peak_period_cost"] = peak_period_cost if peak_period_cost is not None else 0.0
                cost = (
                    underlying_cost
                    + self._penalty_actionnable_triggers * (int(metering_period_trigger) + int(peak_period_trigger))
                )
        
        self._current_observation = next_observation
        self._current_observation_dict = self._compute_current_observation_dict()
        
        return next_observation, cost, terminated, truncated, wrapped_info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[ObsType, dict]:
        super().reset(seed=seed, options=options)
        self._virtual_previous_costs = self._reset_virtual_previous_costs()
        self._current_observation = self._compute_current_observation()
        self._current_observation_dict = self._compute_current_observation_dict()
        return self._current_observation

    def _compute_current_observation(self):
        return {
            **self._compute_current_observation(),
            **self._virtual_previous_costs
        }

    def compute_current_state(self):
        return {
            **self.compute_current_state(),
            **self._virtual_previous_costs
        }

    def _compute_current_observation_dict(self):
        return {
            **self._compute_current_observation_dict(),
            **{
                "virtual_previous_costs": deepcopy(self._virtual_previous_costs),
            }
        }

    def _reset_virtual_previous_costs(self):
        virtual_previous_costs = {
            "virtual_previous_metering_period_cost": 0
        }
        return virtual_previous_costs if "virtual_previous_peak_period_cost" not in list(self._observation_space.keys()) else {
            **virtual_previous_costs,
            **{
                "virtual_previous_peak_period_cost": 0.0
            }
        }

    def _update_virtual_previous_costs(self, counter_states, metering_period_cost, peak_period_cost=None):
        if ("peak_period_counter" not in counter_states and self.is_end_of_metering_period(counter_states)) or self.is_end_of_peak_period(counter_states):
            self._virtual_previous_costs["virtual_previous_metering_period_cost"] = 0.0
        else:
            self._virtual_previous_costs["virtual_previous_metering_period_cost"] = epsilonify(metering_period_cost)

        if peak_period_cost is not None:
            if self.is_end_of_peak_period(counter_states):
                self._virtual_previous_costs["virtual_previous_peak_period_cost"] = 0.0
            else:
                self._virtual_previous_costs["virtual_previous_peak_period_cost"] = epsilonify(peak_period_cost)

