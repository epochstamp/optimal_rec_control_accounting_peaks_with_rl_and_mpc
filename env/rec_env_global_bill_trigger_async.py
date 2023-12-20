from typing import Optional, Tuple, Union
from gym.core import ObsType, ActType
from gym.spaces import Dict as DictSpace, Box, Discrete
import numpy as np
from env.rec_env import RecEnv
from env.rec_env_global_bill_trigger import RecEnvGlobalBillTrigger
from env.rec_env_global_bill_wrapper import RecEnvGlobalBillWrapper
from utils.utils import epsilonify, merge_dicts
from exceptions import *
from copy import deepcopy
import threading

class RecEnvGlobalBillTriggerAsync(RecEnvGlobalBillTrigger):

    def __init__(self,
                 rec_env: Union[RecEnv, RecEnvGlobalBillWrapper],
                 metering_period_cost_trigger= lambda tau_m, tau_p: False,
                 peak_period_cost_trigger= lambda tau_m, tau_p: False,
                 global_bill_optimiser_greedy_init = False,
                 incremental_build_flag = False,
                 **kwargs
        ):
        super().__init__(rec_env, metering_period_cost_trigger=metering_period_cost_trigger, peak_period_cost_trigger=peak_period_cost_trigger, global_bill_optimiser_greedy_init=global_bill_optimiser_greedy_init, incremental_build_flag=incremental_build_flag, **kwargs)
        self._wrapped_rec_env._disable_global_bill_trigger = True
        self._task = None
        self._data = None

    def _optimise_global_bill_for_next_step(self, next_state, next_current_exogenous_sequences):
        self._data = self.global_bill_adaptative_optimiser.optimise_global_bill(
            next_state,
            next_current_exogenous_sequences
        )

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]: 
        return self._step(action) 

    def _create_optimise_global_bill_for_next_step_task_if_needed(self):
        metering_period_counter = self._current_observation["metering_period_counter"]
        peak_period_counter = None
        if self.involve_peaks:
            peak_period_counter = self._current_observation["peak_period_counter"]
        metering_period_trigger = self._metering_period_cost_trigger(
            metering_period_counter,
            peak_period_counter
        )
        peak_period_trigger = self._peak_period_cost_trigger(
            metering_period_counter,
            peak_period_counter
        )
        metering_period_trigger = self._metering_period_cost_trigger(
            metering_period_counter,
            peak_period_counter
        )
        peak_period_trigger = self._peak_period_cost_trigger(
            metering_period_counter,
            peak_period_counter
        )
        is_metering_period_cost_triggered = metering_period_trigger or (not self._wrapped_rec_env._involve_peaks and self.is_end_of_metering_period(self._current_observation_dict["counters_states"])) or self.is_end_of_peak_period(self._current_observation_dict["counters_states"])
        is_peak_period_cost_triggered = peak_period_trigger or self.is_end_of_peak_period(self._current_observation_dict["counters_states"])
        if is_metering_period_cost_triggered or is_peak_period_cost_triggered:
            t = time.time()
            self._task = threading.Thread(target=self._optimise_global_bill_for_next_step(self._current_state, self._current_exogenous_sequences))
            print("thread start time taken")
            self._task.start()
        else:
            self._task = None
            self._data = None

    def _step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        next_observation, underlying_cost, terminated,  truncated, info = super().step(
            action
        )
        if terminated:
            return self._current_observation, underlying_cost, True, True, info
        metering_period_counter = self._current_observation["metering_period_counter"]
        peak_period_counter = None
        if self.involve_peaks:
            peak_period_counter = self._current_observation["peak_period_counter"]
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
        if not self._wrapped_rec_env.disable_global_bill_trigger and (info["is_metering_period_cost_triggered"] or info["is_peak_period_cost_triggered"]):
            metering_period_cost = info["is_metering_period_cost_triggered"] * wrapped_info["costs"]["metering_period_cost"]
            peak_period_cost = info["is_peak_period_cost_triggered"] * wrapped_info["costs"]["peak_period_cost"]

        elif metering_period_trigger or peak_period_trigger or (self._wrapped_rec_env.disable_global_bill_trigger and (info["is_metering_period_cost_triggered"] or info["is_peak_period_cost_triggered"])):
            
            current_state = self._current_state
            current_exogenous_sequences = self._current_exogenous_sequences
            if self._task is None:
                metering_period_cost, peak_period_cost, offtake_peaks, injection_peaks = self.global_bill_adaptative_optimiser.optimise_global_bill(
                    current_state,
                    current_exogenous_sequences
                )
            else:
                self._task.join()
                metering_period_cost, peak_period_cost, offtake_peaks, injection_peaks = self._data
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
        
        peak_period_cost = 0.0 if not self.involve_peaks else peak_period_cost
        previous_metering_period_cost = self._current_observation["previous_metering_period_cost"]
        previous_peak_period_cost = self._current_observation["previous_peak_period_cost"]
        if previous_peak_period_cost is None:
            previous_peak_period_cost = 0
        wrapped_info["costs"]["metering_period_cost"] = ((metering_period_cost - previous_metering_period_cost) if wrapped_info["is_metering_period_cost_triggered"] else 0.0)
        wrapped_info["costs"]["peak_period_cost"] = ((peak_period_cost - previous_peak_period_cost) if wrapped_info["is_peak_period_cost_triggered"] else 0.0)
        cost = (
            wrapped_info["costs"]["controllable_assets_cost"]
            + wrapped_info["costs"]["metering_period_cost"]
            + wrapped_info["costs"]["peak_period_cost"]
        )
        
        if wrapped_info["is_metering_period_cost_triggered"] or wrapped_info["is_peak_period_cost_triggered"]:
            self._update_previous_costs(
                self._current_observation_dict["counters_states"],
                metering_period_cost,
                peak_period_cost
            )
        
        
        
        

        self._current_state = self.compute_current_state()
        self._current_exogenous_sequences = self.compute_current_exogenous_sequences()        
        self._current_observation = self._compute_current_observation()
        self._current_observation_dict = self._compute_current_observation_dict()
        self._create_optimise_global_bill_for_next_step_task_if_needed()
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

