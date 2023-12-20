from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Any, List, Union, Tuple, Dict
import numpy as np

class ExogenousProvider(object, metaclass=ABCMeta):

    def __init__(self, exogenous_variables_members: Dict[str, List[float]], exogenous_prices: Dict[str, List[float]], Delta_M=1):
        self._exogenous_variables_members = exogenous_variables_members
        self._exogenous_prices = exogenous_prices
        self._Delta_M = Delta_M

    @abstractmethod
    def automate_multi_sequence_sampling(self) -> bool:
        raise NotImplementedError()

    def sample_future_sequences(self, exogenous_variables_members: Dict[Any, List[float]], exogenous_prices: Dict[str, List[float]], length: int = 1, n_samples=1) -> List[Dict[Any, List[float]]]:
        """
            Returns action given a state
        """
        if self.automate_multi_sequence_sampling():
            future_sequences = [
                self.sample_future_sequence(exogenous_variables_members, exogenous_prices, length=length) for _ in range(n_samples)
            ]
            return tuple(zip(*future_sequences))
        else:
            return self._sample_future_sequences(exogenous_variables_members, exogenous_prices, length=length, n_samples=n_samples)
        
    def _sample_future_sequences(self, exogenous_variables_members: Dict[Any, List[float]], exogenous_prices: Dict[str, List[float]], length: int = 1, n_samples=1) -> List[Dict[Any, List[float]]]:
        raise NotImplementedError()

    def sample_future_sequence(self, exogenous_variables_members: Dict[Any, List[float]], exogenous_prices: Dict[str, List[float]], length: int = 1) -> Dict[Any, List[float]]:
        raise NotImplementedError()
    
    def reset(self):
        pass
