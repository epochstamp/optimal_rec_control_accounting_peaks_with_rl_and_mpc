from __future__ import annotations
from copy import deepcopy
from typing import List, Optional, Tuple, Union
from gym import Env
from gym.core import ObsType, ActType, RenderFrame
from distributions.time_step_sampler import TimeStepSampler
from env.counter_utils import future_counters
from env.rec_env import RecEnv
from env.rec_env_global_bill_wrapper import RecEnvGlobalBillWrapper
import numpy as np

from utils.utils import epsilonify

class RecEnvGlobalBillMembersAgg(RecEnvGlobalBillWrapper):

    def __init__(self,
                 rec_env: Union[RecEnvGlobalBillWrapper, RecEnv],
                 return_true_global_bill=False,
                 **kwargs
        ):
        super().__init__(
            rec_env, **kwargs
        )
        self._return_true_global_bill = return_true_global_bill

        
    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        next_observation, underlying_cost, terminated,  truncated, info = super().step(
            action
        )
        state = self._current_observation
        exogenous_prices = self._current_observation
        cost = underlying_cost
        new_info = info

        #TODO : merge all consumption and production meters and use maximum buying price / minimum selling price along with peak costs
        if info["is_metering_period_cost_triggered"] or info["is_peak_period_cost_triggered"]:
            consumption_meter_states = {
                member: state[(member, "consumption_meters")] for member in self.members
            }
            production_meter_states = {
                member: state[(member, "production_meters")] for member in self.members
            }
            buying_prices = {
                member: exogenous_prices[(member, "buying_price")] for member in self.members
            }
            selling_prices = {
                member: exogenous_prices[(member, "selling_price")] for member in self.members
            }
            consumption_meter_states = np.asarray([
                consumption_meter_states[member] for member in self.members
            ])
            production_meter_states = np.asarray([
                production_meter_states[member] for member in self.members
            ])
            buying_prices = np.asarray([
                buying_prices[member] for member in self.members
            ])
            selling_prices = np.asarray([
                selling_prices[member] for member in self.members
            ])
            net_consumption_meter_states = np.maximum(consumption_meter_states - production_meter_states, 0.0)
            net_production_meter_states = np.maximum(production_meter_states - consumption_meter_states, 0.0)
            agg_net_consumption_meter_state = np.sum(net_consumption_meter_states, axis=0)
            agg_net_production_meter_state = np.sum(net_production_meter_states, axis=0)
            rec_consumption_meter_state = np.maximum(agg_net_consumption_meter_state - agg_net_production_meter_state, 0)
            rec_production_meter_state = np.maximum(agg_net_production_meter_state - agg_net_consumption_meter_state, 0)
            max_buying_price = np.max(buying_prices, axis=0)
            min_selling_price = np.min(selling_prices, axis=0)
            metering_period_cost = np.dot(max_buying_price, rec_consumption_meter_state) - np.dot(min_selling_price, rec_production_meter_state)
            peak_period_cost = (
                np.max(rec_consumption_meter_state) * epsilonify(self.current_offtake_peak_cost, epsilon=1e-6)
                + np.max(rec_production_meter_state) * epsilonify(self.current_injection_peak_cost, epsilon=1e-6)
            )
            new_cost = metering_period_cost + peak_period_cost

            #TODO : Carefully overwrite peak period cost and metering period cost by taking into account number of time steps in current peak period

            if net_consumption_meter_states.shape[1] > net_consumption_meter_states.shape[0]:
                
                print(agg_net_consumption_meter_state.shape)
                exit()
            self._current_observation = self._compute_current_observation()
            print(net_consumption_meter_states.shape)

        return self._current_observation, cost, terminated, truncated, new_info