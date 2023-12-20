from __future__ import annotations
from copy import deepcopy
from typing import List, Optional, Tuple, Union
from gymnasium import Env
from gymnasium.core import ObsType, ActType, RenderFrame
from distributions.time_step_sampler import TimeStepSampler
from env.counter_utils import future_counters
from env.rec_env import RecEnv
from env.rec_env_global_bill_wrapper import RecEnvGlobalBillWrapper
import numpy as np

from utils.utils import epsilonify

class RecEnvGlobalBillDiscountedCost(Env):

    def __init__(self,
                 rec_env:Union[RecEnvGlobalBillWrapper, RecEnv],
                 gamma:float = 0.99,
                 **kwargs
        ):
        self._wrapped_rec_env = rec_env
        rec_env_source = rec_env
        while type(rec_env_source) != RecEnv:
            rec_env_source = rec_env_source.wrapped_rec_env
        T = rec_env_source.T
        self.observation_space = self._wrapped_rec_env.observation_space
        self.action_space = self._wrapped_rec_env.action_space
        nb_time_steps_in_peak_period = rec_env_source.Delta_M * rec_env_source.Delta_P
        nb_peak_periods = (T-1)//nb_time_steps_in_peak_period
        self._gammas = [(gamma**nb_time_steps_in_peak_period)] * (nb_time_steps_in_peak_period+1)
        if nb_peak_periods > 1:
            for _ in range(nb_peak_periods-1):
                self._gammas.extend([self._gammas[-1]*(gamma**nb_time_steps_in_peak_period)]*(nb_time_steps_in_peak_period))
        self._gammas = np.asarray(self._gammas, dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[ObsType, dict]:
        self._t = 0
        return self._wrapped_rec_env.reset(seed=seed, options=options)


        
    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        next_observation, underlying_cost, terminated,  truncated, info = self._wrapped_rec_env.step(
            action
        )
        discounted_cost = underlying_cost*self._gammas[self._t]
        self._t += 1

        return next_observation, discounted_cost, terminated, truncated, info
    
class RecEnvGlobalBillNegateReward(Env):

    def __init__(self,
                 rec_env,
                 **kwargs
        ):
        self.wrapped_rec_env = rec_env
        self.observation_space = self.wrapped_rec_env.observation_space
        self.action_space = self.wrapped_rec_env.action_space
        
    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        next_observation, underlying_cost, terminated,  truncated, info = self.wrapped_rec_env.step(
            action
        )

        return next_observation, -underlying_cost, terminated, truncated, info
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[ObsType, dict]:
        return self.wrapped_rec_env.reset(seed=seed, options=options)