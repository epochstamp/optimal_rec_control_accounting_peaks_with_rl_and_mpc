from base.conditional_probability_distribution import ConditionalProbabilityDistribution
from typing import Any, List, Union, Dict, Tuple
import numpy as np
from abc import ABCMeta, abstractmethod

class TimeSerieDiscreteSampling(ConditionalProbabilityDistribution):

    def __init__(self, initial_time_serie: List[float], max_length: int = None):
        self._initial_time_serie = list(initial_time_serie)
        self._len_initial_time_serie = len(self._initial_time_serie)
        if max_length is None:
            self._max_length = len(initial_time_serie)-1
        else:
            self._max_length=max_length
        self._most_probable = None

    def support(self) -> Tuple[float, float]:
        """
            Support of the distribution

            Returns:
        """
        return (min(self._initial_time_serie), max(self._initial_time_serie))
    
    @abstractmethod
    def _get_t_prob_discrete_distribution(self, t):
        raise NotImplementedError()
    
    @abstractmethod
    def _sample_t_from_discrete_distribution(self):
        raise NotImplementedError()

    def sample(self, c: List = [], length=1) -> Union[List, Any]:
        t = self._sample_t_from_discrete_distribution()
        return self._initial_time_serie[t: t+min(length, self._max_length)]

    def prob(self, x: List, c: List = []) -> float:
        """Compute p(x).

        """
        if len(x) == self._len_initial_time_serie:
            return 1.0
        elif len(x) > self._len_initial_time_serie:
            return 0.0
        #Look for t
        for t in range(self._len_initial_time_serie - len(x)):
            if x == self._initial_time_serie[t:t+self._len_initial_time_serie]:
                return self._get_t_prob_discrete_distribution(t)
        return 0.0

    def log_prob(self, x, c: List = []) -> float:
        return np.log(self.prob(x))

    def most_probable(self, c: List = []):
        """Most probable data points.
        """
        if self._most_probable is None:
            max_prob_t = -np.inf
            for t in range(self._len_initial_time_serie - self._max_length):
                prob_t = self._prob(self._len_initial_time_serie[t:t+self._max_length])
                if prob_t > max_prob_t:
                    max_prob_t = prob_t
                    self._most_probable = self._len_initial_time_serie[t:t+self._max_length]
        return self._most_probable
