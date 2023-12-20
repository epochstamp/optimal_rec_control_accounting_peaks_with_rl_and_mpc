from base.exogenous_provider import ExogenousProvider
from typing import Dict, Any, List
import numpy as np

from env.counter_utils import future_counters

class PerfectForesightExogenousProvider(ExogenousProvider):

    def __init__(self, exogenous_variables_members: Dict[str, List[float]], exogenous_prices: Dict[str, List[float]], Delta_M=1):
        super().__init__(
            exogenous_variables_members,
            exogenous_prices,
            Delta_M=Delta_M
        )
        

    def automate_multi_sequence_sampling(self) -> bool:
        return True

    def sample_future_sequence(self, exogenous_variables_members: Dict[Any, List[float]], exogenous_prices: Dict[str, List[float]], length: int = 1) -> Dict[Any, List[float]]:
        
        current_timestamp = len(list(exogenous_variables_members.values())[0])
        current_tau_m = len(list(exogenous_prices.values())[0])
        length_tau_m = int(np.ceil(length//self._Delta_M))
        return ({
                    k: v[current_timestamp: current_timestamp+length+1] for k, v in self._exogenous_variables_members.items()
                  },
                {
                    k: list(v)[current_tau_m: current_tau_m+length_tau_m+1] for k, v in self._exogenous_prices.items()
                  }
                )