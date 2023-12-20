from base.probability_distribution import ProbabilityDistribution
from typing import Any, List, Union, Dict, Tuple
import numpy as np
from abc import ABCMeta, abstractmethod
from random import choices

class TimeStepSampler(ProbabilityDistribution):

    def __init__(self, t_range: List[int], t_weights: List[float]):
        self._t_range = t_range
        self._len_t_range = len(t_range)
        self._most_probable = None
        if sum(t_weights) == 0:
            t_weights = [1.0/self._len_t_range] * self._len_t_range
        self._t_weights = list(np.asarray(t_weights) / np.sum(t_weights))

    def support(self) -> Tuple[float, float]:
        """
            Support of the distribution

            Returns:
        """
        return (min(self._initial_time_serie), max(self._initial_time_serie))

    def sample(self, length=1) -> int:
        return int(choices(self._t_range, weights=self._t_weights, k=1)[0])

    def prob(self, x: int) -> float:
        return self._t_weights[x]

    def log_prob(self, x) -> float:
        return float(np.log(self.prob(x)))

    def most_probable(self, c: List = []):
        """Most probable data points.
        """
        t_argmax = max(zip(self._t_range, range(self._len_t_range)))[1]
        return self._t_range[t_argmax]
