from typing import Optional, Tuple
from gym.core import ObsType, ActType
from gym.spaces import Dict as DictSpace, Box, Discrete
import numpy as np
from env.rec_env import RecEnv
from env.rec_env_global_bill_wrapper import RecEnvGlobalBillWrapper
from utils.utils import epsilonify, merge_dicts
from exceptions import *
from copy import deepcopy
class RecEnvGlobalBillObserveCurrentPeaks(RecEnvGlobalBillWrapper):

    def __init__(self,
                 rec_env: RecEnv,
                 metering_period_cost_trigger= lambda tau_m, tau_p: False,
                 peak_period_cost_trigger= lambda tau_m, tau_p: False,
                 global_bill_optimiser_greedy_init = False,
                 incremental_build_flag = False
        ):
        super().__init__(rec_env)
        self._wrapped_rec_env = rec_env
        self.action_space = rec_env.action_space
        self._metering_period_cost_trigger = metering_period_cost_trigger
        self._peak_period_cost_trigger = peak_period_cost_trigger
        
        lst_peak_states = []
        members = rec_env.members
        if self._wrapped_rec_env.involve_peaks:
            # Counter for number of metering periods in the current peak period
            lst_peak_states += [
                {
                    "previous_peak_period_cost": Box(0, 10000, dtype=np.float32, shape=())
                }
            ]
            if self._wrapped_rec_env.involve_current_peaks:
                if rec_env._current_offtake_peak_cost > 0:
                    # Current offtake-peak state for each member
                    lst_peak_states += [
                        {
                            (member, "current_offtake_peak"): Box(0, 10000, dtype=np.float32, shape=())
                            for member in members
                        }
                    ]
                if rec_env._current_injection_peak_cost > 0:
                    # Current injection-peak state for each member
                    lst_peak_states += [
                        {
                            (member, "current_injection_peak"): Box(0, 10000, dtype=np.float32, shape=())
                            for member in members
                        }
                    ]
        peak_states = merge_dicts(lst_peak_states) if lst_peak_states != [] else dict()
        self.observation_space = DictSpace(
            {
                **dict(self._wrapped_rec_env.observation_space),
                **{
                    "previous_metering_cost": Box(low=-10000, high=10000)
                },
                **peak_states
            }
        )
        self.global_bill_adaptative_optimiser.greedy_init = global_bill_optimiser_greedy_init
        self.global_bill_adaptative_optimiser.incremental_build_flag = incremental_build_flag

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        next_observation, underlying_cost, terminated,  truncated, info = self._wrapped_rec_env.step(
            action
        )
        if terminated:
            return next_observation, underlying_cost, terminated, truncated, info

        


    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[ObsType, dict]:
        super().reset(seed=seed, options=options)
        self._peaks_states = self._reset_peaks()
        self._previous_costs = self._reset_previous_costs()
        return self._compute_current_observation()

    def _compute_current_observation(self):
        return {
            **self._compute_current_observation(),
            **self._previous_costs,
            **self._peaks_states
        }

    def compute_current_state(self):
        return {
            **self.compute_current_state(),
            **self._previous_costs,
            **self._peaks_states
        }

    def _compute_current_observation_dict(self):
        return {
            **self._compute_current_observation_dict(),
            **{
                "peaks_states": deepcopy(self._peaks_states),
                "previous_costs": deepcopy(self._previous_costs),
            }
        }

    def _reset_previous_costs(self):
        previous_costs = {
            "previous_metering_period_cost": 0
        }
        return previous_costs if "previous_peak_period_cost" not in list(self._observation_space.keys()) else {
            **previous_costs,
            **{
                "previous_peak_period_cost": 0.0
            }
        }

            

    def _reset_peaks(self):
        return merge_dicts([
            {
                (member, "historical_offtake_peaks"): []
                for member in self._members
            },
            {
                (member, "historical_injection_peaks"): []
                for member in self._members
            },
            {
                (member, "current_offtake_peak"): 0.0
                for member in self._members
            },
            {
                (member, "current_injection_peak"): 0.0
                for member in self._members
            }
        ])

    def _update_previous_costs(self, counter_states, metering_period_cost, peak_period_cost):
        if metering_period_cost is not None:
            if ("peak_period_counter" not in counter_states and self._is_end_of_metering_period(counter_states)) or self._is_end_of_peak_period(counter_states):
                self._previous_costs["previous_metering_period_cost"] = 0.0
            else:
                self._previous_costs["previous_metering_period_cost"] = epsilonify(metering_period_cost)

        if peak_period_cost is not None:
            if self._is_end_of_peak_period(counter_states):
                self._previous_costs["previous_peak_period_cost"] = 0.0
            else:
                self._previous_costs["previous_peak_period_cost"] = epsilonify(peak_period_cost)

    def _update_peaks(self, counter_states, peaks_states, new_peaks=None):
        for member in self._members:
            current_offtake_peak = peaks_states[(member, "current_offtake_peaks")] if new_peaks is None or (member, "offtake_peaks") not in list(new_peaks.keys()) else new_peaks
            current_injection_peak = peaks_states[(member, "current_injection_peaks")] if new_peaks is None or (member, "injection_peaks") not in list(new_peaks.keys()) else new_peaks
            # History offtake/injection peak state update
            if self._Delta_P_prime > 0:
                if self._is_end_of_peak_period(counter_states):
                    if len(peaks_states[(member, "historical_offtake_peaks")]) < self._Delta_P_prime:
                        if self._historical_offtake_peak_cost > 0:
                            self._peaks_states[(member, "historical_offtake_peaks")].append(current_offtake_peak)
                        if self._historical_injection_peak_cost > 0:
                            self._peaks_states[(member, "historical_injection_peaks")].append(current_injection_peak)
                    else:
                        if self._historical_offtake_peak_cost > 0:
                            self._peaks_states[(member, "historical_offtake_peaks")] = (peaks_states[(member, "historical_offtake_peaks")] + [current_offtake_peak])[1:]
                        if self._historical_injection_peak_cost > 0:
                            self._peaks_states[(member, "historical_injection_peaks")] = (peaks_states[(member, "historical_injection_peaks")] + [current_injection_peak])[1:]
            
            # Current offtake/injection peak state update
            if self._is_end_of_peak_period(counter_states):
                self._peaks_states[(member, "current_offtake_peak")] = 0.0
                self._peaks_states[(member, "current_injection_peak")] = 0.0  
            elif new_peaks is not None and self._is_end_of_metering_period(counter_states):
                self._peaks_states[(member, "current_offtake_peak")] = float(current_offtake_peak)
                self._peaks_states[(member, "current_injection_peak")] = float(current_injection_peak)
            if member == "C":
                pass#print(peaks_states[(member, "current_offtake_peak")], self._peaks_states[(member, "historical_offtake_peaks")] , self._is_end_of_peak_period(counter_states))

       
