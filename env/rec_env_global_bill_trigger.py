from typing import Optional, Tuple, Union
from gym.core import ObsType, ActType
from gym.spaces import Dict as DictSpace, Box, Discrete
import numpy as np
from env.rec_env import RecEnv
from env.rec_env_global_bill_wrapper import RecEnvGlobalBillWrapper
from utils.utils import epsilonify, merge_dicts
from exceptions import *
from copy import deepcopy
class RecEnvGlobalBillTrigger(RecEnvGlobalBillWrapper):

    def __init__(self,
                 rec_env: Union[RecEnv, RecEnvGlobalBillWrapper],
                 metering_period_cost_trigger= lambda tau_m, tau_p: False,
                 peak_period_cost_trigger= lambda tau_m, tau_p: False,
                 global_bill_optimiser_greedy_init = False,
                 incremental_build_flag = False,
                 **kwargs
        ):
        super().__init__(rec_env, **kwargs)
        self._wrapped_rec_env = rec_env
        self.action_space = rec_env.action_space
        self._metering_period_cost_trigger = metering_period_cost_trigger
        self._peak_period_cost_trigger = peak_period_cost_trigger
        
        previous_metering_period_cost_state = dict()
        if "previous_metering_period_cost" not in self._wrapped_observation_space_keys:
            # Counter for number of metering periods in the current peak period
            previous_metering_period_cost_state = {
                "previous_metering_period_cost": Box(-10000000, 10000000, dtype=np.float32, shape=())
            }

        previous_peak_period_cost_state = dict()
        if self._wrapped_rec_env.involve_peaks and "previous_peak_period_cost" not in self._wrapped_observation_space_keys:
            # Counter for number of metering periods in the current peak period
            previous_peak_period_cost_state = {
                "previous_peak_period_cost": Box(0, 10000000, dtype=np.float32, shape=())
            }
        self.observation_space = DictSpace(
            {
                **dict(self._wrapped_rec_env.observation_space),
                **previous_metering_period_cost_state,
                **previous_peak_period_cost_state
            }
        )
        self.global_bill_adaptative_optimiser.greedy_init = global_bill_optimiser_greedy_init
        self.global_bill_adaptative_optimiser.incremental_build_flag = incremental_build_flag

    def _get_additionals_kwargs(self):
        return {
            "metering_period_cost_trigger": self._metering_period_cost_trigger,
            "peak_period_cost_trigger": self._peak_period_cost_trigger,
            "global_bill_optimiser_greedy_init": self.global_bill_adaptative_optimiser.greedy_init,
            "incremental_build_flag": self.global_bill_adaptative_optimiser.incremental_build_flag
        }

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        next_observation, underlying_cost, terminated,  truncated, info = super().step(
            action
        )
        if terminated:
            return self._current_observation, underlying_cost, True, True, info
        observation_for_global_bill = (
            self._compute_current_observation() if self.compute_global_bill_on_next_observ else self._current_observation
        )
        current_state_for_global_bill = (
            self._compute_current_state() if self.compute_global_bill_on_next_observ else self._current_state
        )
        current_exogenous_sequences_for_global_bill = (
            self._compute_current_exogenous_sequences() if self.compute_global_bill_on_next_observ else self._current_exogenous_sequences
        )
        observation_dict_for_global_bill = (
            self._compute_current_observation_dict() if self.compute_global_bill_on_next_observ else self._current_observation_dict
        )
        metering_period_counter = observation_for_global_bill["metering_period_counter"]
        peak_period_counter = None
        if self.involve_peaks:
            peak_period_counter = observation_for_global_bill["peak_period_counter"]
        metering_period_trigger = self._metering_period_cost_trigger(
            metering_period_counter,
            peak_period_counter
        )
        peak_period_trigger = self._peak_period_cost_trigger(
            metering_period_counter,
            peak_period_counter
        )
        metering_period_cost = None
        wrapped_info = dict(info)
        if info["is_metering_period_cost_triggered"] or info["is_peak_period_cost_triggered"]:
            metering_period_cost = info["is_metering_period_cost_triggered"] * wrapped_info["costs"]["metering_period_cost"]
            peak_period_cost = info["is_peak_period_cost_triggered"] * wrapped_info["costs"]["peak_period_cost"]
        elif not self._wrapped_rec_env.disable_global_bill_trigger and (metering_period_trigger or peak_period_trigger):
            
            current_state = current_state_for_global_bill
            current_exogenous_sequences = current_exogenous_sequences_for_global_bill
            metering_period_cost, peak_period_cost, offtake_peaks, injection_peaks = self.global_bill_adaptative_optimiser.optimise_global_bill(
                current_state,
                current_exogenous_sequences
            )
            metering_period_cost *= metering_period_trigger
            if peak_period_cost is None:
                peak_period_cost = 0
            peak_period_cost *= peak_period_trigger
            
            wrapped_info["is_metering_period_cost_triggered"] = metering_period_trigger
            wrapped_info["is_peak_period_cost_triggered"] = peak_period_trigger
            wrapped_info["current_offtake_peaks"] = offtake_peaks
            wrapped_info["current_injection_peaks"] = injection_peaks
        else:
            metering_period_cost = 0.0
            peak_period_cost = 0.0
        wrapped_info["costs"]["original_metering_period_cost"] = metering_period_cost
        wrapped_info["costs"]["original_peak_period_cost"] = peak_period_cost
        next_metering_period_trigger = self._metering_period_cost_trigger(
            self.counters_states["metering_period_counter"],
            self.counters_states["peak_period_counter"]
        )
        next_peak_period_trigger = self._peak_period_cost_trigger(
            self.counters_states["metering_period_counter"],
            self.counters_states["peak_period_counter"]
        )
        wrapped_info["next_step_cost_triggered"] = wrapped_info["next_step_cost_triggered"] or next_metering_period_trigger or next_peak_period_trigger
        
        peak_period_cost = 0.0 if not self.involve_peaks else peak_period_cost
        previous_metering_period_cost = self._current_observation["previous_metering_period_cost"]
        previous_peak_period_cost = self._current_observation["previous_peak_period_cost"]
        if previous_peak_period_cost is None:
            previous_peak_period_cost = 0
        wrapped_info["costs"]["metering_period_cost"] = ((metering_period_cost - previous_metering_period_cost) if wrapped_info["is_metering_period_cost_triggered"] else 0.0)
        wrapped_info["costs"]["peak_period_cost"] = ((peak_period_cost - previous_peak_period_cost) if wrapped_info["is_peak_period_cost_triggered"] else 0.0)
        cost = (
            0*wrapped_info["costs"]["controllable_assets_cost"]
            + wrapped_info["costs"]["metering_period_cost"]
            + wrapped_info["costs"]["peak_period_cost"]
        )
        
        if wrapped_info["is_metering_period_cost_triggered"] or wrapped_info["is_peak_period_cost_triggered"]:
            self._update_previous_costs(
                observation_dict_for_global_bill["counters_states"],
                metering_period_cost,
                peak_period_cost
            )
        
        
        
        

        self._current_state = self.compute_current_state()
        self._current_exogenous_sequences = self.compute_current_exogenous_sequences()        
        self._current_observation = self._compute_current_observation()
        self._current_observation_dict = self._compute_current_observation_dict()
        return self._current_observation, cost, terminated, truncated, wrapped_info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[ObsType, dict]:
        self._current_observation = super().reset(seed=seed, options=options)
        
        if "previous_metering_period_cost" not in self._wrapped_observation_space_keys:
            self._previous_costs = self._reset_previous_costs()
            self._current_observation = {
                **self._current_observation,
                **self._previous_costs
            }
            self._current_observation_dict = {
                **self._current_observation_dict,
                **{
                    "previous_costs": dict(self._previous_costs)
                }
            }
        self._current_state = self.compute_current_state()
        self._current_exogenous_sequences = self.compute_current_exogenous_sequences()     
        return self._current_observation

    def _compute_current_observation(self):
        return {
            **self._wrapped_rec_env._compute_current_observation(),
            **self._previous_costs
        }

    def compute_current_state(self):
        return {
            **self._wrapped_rec_env.compute_current_state(),
            **self._previous_costs
        }

    def _compute_current_observation_dict(self):
        return {
            **self._wrapped_rec_env._compute_current_observation_dict(),
            **{
                "previous_costs": dict(self._previous_costs)
            }
        }

    def _reset_previous_costs(self):
        previous_costs = {
            "previous_metering_period_cost": 0
        }
        return previous_costs if "previous_peak_period_cost" in self._wrapped_observation_space_keys else {
            **previous_costs,
            **{
                "previous_peak_period_cost": 0.0
            }
        }

    def _update_previous_costs(self, counter_states, metering_period_cost, peak_period_cost=None):
        if ("peak_period_counter" not in counter_states and self.is_end_of_metering_period(counter_states)) or self.is_end_of_peak_period(counter_states):
            self._previous_costs["previous_metering_period_cost"] = 0.0
        else:
            self._previous_costs["previous_metering_period_cost"] = epsilonify(metering_period_cost)

        if peak_period_cost is not None:
            if self.is_end_of_peak_period(counter_states):
                self._previous_costs["previous_peak_period_cost"] = 0.0
            else:
                self._previous_costs["previous_peak_period_cost"] = epsilonify(peak_period_cost)

