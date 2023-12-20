from base.exogenous_provider import ExogenousProvider
from typing import Dict, Any, List
import numpy as np

from env.counter_utils import future_counters
from env.rec_env import RecEnv
from copy import deepcopy, copy

class PseudoForecastExogenousProvider(ExogenousProvider):

    def __init__(self, exogenous_variables_members: Dict[str, List[float]], exogenous_prices: Dict[str, List[float]], stochastic_env: RecEnv, Delta_M=1, alpha=0.99, alpha_fading=1.0, fixed_sequence="input"):
        super().__init__(
            exogenous_variables_members,
            exogenous_prices,
            Delta_M=Delta_M
        )
        self._stochastic_env = stochastic_env
        self._stochastic_env_2 = None
        self._fixed_sequence = fixed_sequence
        if self._fixed_sequence in ("sample", "extreme_sample"):
            self._stochastic_env_2 = self._stochastic_env
        self._alpha = alpha
        
        if self._fixed_sequence == "input":
            self._exogenous_variables_members_fixed_sequence = exogenous_variables_members

        self._weight_realisation = np.asarray([alpha**(t) for t in range(10000)])
        self._weight_fixed = 1 - self._weight_realisation
        
    def reset(self):
        if self._fixed_sequence in ("sample", "extreme_sample"):
            self._exogenous_variables_members_fixed_sequence = self._stochastic_env_2.sample_members_exogenous_variables()

    def automate_multi_sequence_sampling(self) -> bool:
        return True
    
    def _forecast(self, fixed_sequence, realisation):
        len_realisation = len(realisation)
        weight_realisation = self._weight_realisation[:len_realisation]
        weight_fixed = self._weight_fixed[:len_realisation]
        forecasted_sequence = weight_realisation*realisation + weight_fixed * fixed_sequence
        return forecasted_sequence


    def sample_future_sequence(self, exogenous_variables_members: Dict[Any, List[float]], exogenous_prices: Dict[str, List[float]], length: int = 1) -> Dict[Any, List[float]]:
        if self._fixed_sequence == "extreme_sample":
            self._exogenous_variables_members_fixed_sequence = self._stochastic_env_2.sample_members_exogenous_variables()
        exogenous_variables_members_realisation = self._stochastic_env.observe_all_members_exogenous_variables()
        current_timestamp = len(list(exogenous_variables_members.values())[0])
        
        current_tau_m = len(list(exogenous_prices.values())[0])
        length_tau_m = int(np.ceil(length//self._Delta_M))
        return ({
                    k: self._forecast(self._exogenous_variables_members_fixed_sequence[k][current_timestamp: current_timestamp+length+1], exogenous_variables_members_realisation[k][current_timestamp: current_timestamp+length+1]) for k in self._exogenous_variables_members.keys()
                  },
                {
                    k: list(v)[current_tau_m: current_tau_m+length_tau_m+1] for k, v in self._exogenous_prices.items()
                  }
                )