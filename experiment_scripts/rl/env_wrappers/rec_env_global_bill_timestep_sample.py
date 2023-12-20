from __future__ import annotations
from copy import deepcopy
from typing import List, Optional, Tuple, Union
from gym import Env
from gym.core import ObsType, ActType, RenderFrame
from distributions.time_step_sampler import TimeStepSampler
from env.counter_utils import future_counters
from env.rec_env import RecEnv
from env.rec_env_global_bill_wrapper import RecEnvGlobalBillWrapper
class RecEnvGlobalBillTimestepSample(RecEnvGlobalBillWrapper):

    def __init__(self,
                 rec_env: Union[RecEnvGlobalBillWrapper, RecEnv],
                 time_step_sampler: TimeStepSampler,
                 max_T: float = None,
                 **kwargs
        ):
        super().__init__(
            rec_env, **kwargs
        )
        self._time_step_sampler = time_step_sampler
        self._max_T = max_T
        self._full_wrapped_rec_env = self._wrapped_rec_env.clone()
        self._full_wrapped_rec_env.reset()
        self._metering_period_counter, _ = future_counters(0, 0, self._full_wrapped_rec_env.T, Delta_M=self._wrapped_rec_env.Delta_M, Delta_P=self._wrapped_rec_env.Delta_P)
        self._nb_metering_periods = sum([1 for tau_m in self._metering_period_counter if tau_m == self._wrapped_rec_env.Delta_M])

        
    @property
    def T(self):
        return self._max_T
    
    def _get_additionals_kwargs(self):
        return {
            "max_T": self._max_T,
            "time_step_sampler": deepcopy(self._time_step_sampler)
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[ObsType, dict]:
        obs = super().reset(
            seed=seed, options=options
        )
        self._wrapped_rec_env.T = self._max_T
        t0 = self._time_step_sampler.sample()
        elapsed_metering_periods = sum([1 for tau_m in self._metering_period_counter[:t0] if tau_m == self._wrapped_rec_env.Delta_M])
        remaining_metering_periods = sum([1 for tau_m in self._metering_period_counter[:self._max_T] if tau_m == self._wrapped_rec_env.Delta_M])
        exogenous_variables_members_items = list(self._full_wrapped_rec_env.observe_all_members_exogenous_variables().items())
        self._wrapped_rec_env.set_exogenous_variables_members(
            {
                k:v[t0:t0+self._max_T] for k,v in exogenous_variables_members_items
            }
        )
        obs_prices = self._full_wrapped_rec_env.observe_all_raw_prices_exogenous_variables()
        self._wrapped_rec_env.set_buying_prices({
            member: obs_prices[(member, "buying_price")][elapsed_metering_periods: elapsed_metering_periods + remaining_metering_periods] for member in self._full_wrapped_rec_env.members
        })
        self._wrapped_rec_env.set_selling_prices({
            member: obs_prices[(member, "selling_price")][elapsed_metering_periods: elapsed_metering_periods + remaining_metering_periods] for member in self._full_wrapped_rec_env.members
        })
        return self._compute_current_observation()
