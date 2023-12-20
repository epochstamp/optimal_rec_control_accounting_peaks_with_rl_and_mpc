from __future__ import annotations
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union
from gym import Env
from gym.core import ObsType, ActType, RenderFrame
from env.rec_env import RecEnv
class RecEnvGlobalBillWrapper(Env):

    def __init__(self,
                 rec_env: Union[RecEnvGlobalBillWrapper, RecEnv],
                 **kwargs
        ):
        self._wrapped_rec_env = deepcopy(rec_env)
        
        self.action_space = self._wrapped_rec_env.action_space
        self.observation_space = self._wrapped_rec_env.observation_space
        self.reward_range = self._wrapped_rec_env.reward_range
        self._wrapped_observation_space_keys = set(self.observation_space.keys())
        self._wrapped_action_space_keys = set(self.action_space.keys())

    def _get_additionals_kwargs(self):
        return {}

    def clone(self, **env_kwargs):
        cloned_rec_env = self._wrapped_rec_env.clone(**env_kwargs)
        return self.__class__(
            cloned_rec_env,
            **self._get_additionals_kwargs()
        )
    
    @property
    def type_solver(self):
        return self._wrapped_rec_env._type_solver

    @property
    def compute_global_bill_on_next_observ(self):
        return self._wrapped_rec_env._compute_global_bill_on_next_observ

    @property
    def global_bill_adaptative_optimiser(self):
        return self._wrapped_rec_env.global_bill_adaptative_optimiser
    
    @property
    def involve_peaks(self):
        return self._wrapped_rec_env.involve_peaks
    
    @property
    def involve_current_peaks(self):
        return self._wrapped_rec_env.involve_current_peaks
    
    @property
    def involve_historical_peaks(self):
        return self._wrapped_rec_env.involve_historical_peaks

    @property
    def Delta_C(self):
        return self._wrapped_rec_env.Delta_C

    @property
    def Delta_M(self):
        return self._wrapped_rec_env.Delta_M
    
    @property
    def Delta_P(self):
        return self._wrapped_rec_env.Delta_P
    
    @property
    def wrapped_rec_env(self):
        return self._wrapped_rec_env

    @property
    def Delta_P_prime(self):
        return self._wrapped_rec_env.Delta_P_prime
    
    @property
    def members(self):
        return self._wrapped_rec_env.members
    
    @property
    def T(self):
        return self._wrapped_rec_env.T
    
    @property
    def feasible_actions_controllable_assets(self):
        return self._wrapped_rec_env.feasible_actions_controllable_assets
    
    @property
    def consumption_function(self):
        return self._wrapped_rec_env.consumption_function
    
    @property
    def production_function(self):
        return self._wrapped_rec_env.production_function
    
    @property
    def controllable_assets_state_space(self):
        return self._wrapped_rec_env.controllable_assets_state_space
    
    @property
    def controllable_assets_action_space(self):
        return self._wrapped_rec_env.controllable_assets_action_space
    
    @property
    def controllable_assets_dynamics(self):
        return self._wrapped_rec_env.controllable_assets_dynamics
    
    @property
    def cost_function_controllable_assets(self):
        return self._wrapped_rec_env.cost_function_controllable_assets
    
    @property
    def exogenous_space(self):
        return self._wrapped_rec_env.exogenous_space
    
    @property
    def counters_states(self):
        return self._wrapped_rec_env._counters_states
    
    @property
    def current_offtake_peak_cost(self):
        return self._wrapped_rec_env.current_offtake_peak_cost
    
    @property
    def current_injection_peak_cost(self):
        return self._wrapped_rec_env.current_injection_peak_cost
    
    @property
    def historical_offtake_peak_cost(self):
        return self._wrapped_rec_env.historical_offtake_peak_cost
    
    @property
    def historical_injection_peak_cost(self):
        return self._wrapped_rec_env.historical_injection_peak_cost
    
    @property
    def env_name(self):
        return self._wrapped_rec_env.env_name
    
    @property
    def projector(self):
        return self._wrapped_rec_env.projector
    
    @property
    def rec_import_fees(self):
        return self._wrapped_rec_env.rec_import_fees
    
    @property
    def rec_export_fees(self):
        return self._wrapped_rec_env.rec_export_fees
    
    @projector.setter
    def projector(self, projector):
        self._wrapped_rec_env.projector = projector

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        return self._wrapped_rec_env.step(action)
        

        


    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[ObsType, dict]:
        self._current_observation = self._wrapped_rec_env.reset(seed=seed, options=options)
        self._current_observation_dict = self._wrapped_rec_env._compute_current_observation_dict()
        return self._current_observation

    def _compute_current_observation(self):
        return self._wrapped_rec_env._compute_current_observation()

    def compute_current_state(self):
        return self._wrapped_rec_env.compute_current_state()

    def compute_current_exogenous_sequences(self):
        return self._wrapped_rec_env.compute_current_exogenous_sequences()

    def _compute_current_observation_dict(self):
        return self._wrapped_rec_env._compute_current_observation_dict()
    
    def _observe_members_exogenous_variables(self):
        return self._wrapped_rec_env._observe_members_exogenous_variables()
    
    def observe_all_members_exogenous_variables(self):
        return self._wrapped_rec_env.observe_all_members_exogenous_variables()
    
    def observe_all_exogenous_variables(self):
        return self._wrapped_rec_env.observe_all_exogenous_variables()


    def _observe_prices_exogenous_variables(self):
        return self._wrapped_rec_env._observe_prices_exogenous_variables()
    
    def observe_all_raw_prices_exogenous_variables(self):
        return self._wrapped_rec_env.observe_all_raw_prices_exogenous_variables()

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass 

    def is_end_of_metering_period(self, counter_states):
        return self._wrapped_rec_env.is_end_of_metering_period(counter_states=counter_states)

    def is_end_of_peak_period(self, counter_states):
        return self._wrapped_rec_env.is_end_of_peak_period(counter_states=counter_states)
    
    def set_buying_prices(self, buying_prices_members: Dict[str, List[float]]):
        self._wrapped_rec_env.set_buying_prices(buying_prices_members)

    def set_selling_prices(self, selling_prices_members: Dict[str, List[float]]):
        self._wrapped_rec_env.set_selling_prices(selling_prices_members)
