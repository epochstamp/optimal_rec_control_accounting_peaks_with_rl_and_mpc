from abc import ABCMeta, abstractmethod
from typing import Any, List, Tuple, Dict

class RepartitionKeysOptimiser(object, metaclass=ABCMeta):

    def __init__(self, **kwargs):
        pass

    def optimise_repartition_keys(self, members: List[str], metering_period_counter_state: int, peak_period_counter_state: int, consumption_meters_states: Dict[str, float], production_meters_states: Dict[str, float], buying_prices: Dict[str, float], selling_prices: Dict[str, float], Delta_C:float=1.0, Delta_M:int=1, Delta_P:int=1, Delta_P_prime:int=0, current_offtake_peak_cost:float=0, current_injection_peak_cost:float=0, historical_offtake_peak_cost:float=0, historical_injection_peak_cost:float=0, peak_states:Dict[Tuple[str, str], float]=None):
        raise NotImplementedError()