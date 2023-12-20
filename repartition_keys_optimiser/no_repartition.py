from base.repartition_keys_optimiser import RepartitionKeysOptimiser
from typing import Any, List, Tuple, Dict

class NoRepartition(RepartitionKeysOptimiser):

    def __init__(self, **kwargs):
        self._empty_rep_keys = None

    def optimise_repartition_keys(self, members: List[str], counter_states: Dict[str, int], meters_states: Dict[Tuple[str, str], int], exogenous_variables_prices: Dict[Tuple[str, str], float], surrogate:bool=False, Delta_C:float=1.0, Delta_M:int=1, Delta_P: int = 1, offtake_peak_cost:float=0, injection_peak_cost:float=0, peak_states:Dict[Tuple[str, str], float]=None):
        if self._empty_rep_keys is None:
            self._empty_rep_keys = {**{
                (member, "rec_import"): 0.0 for member in members
            }, **{
                (member, "rec_export"): 0.0 for member in members
            }}
        return self._empty_rep_keys