from base.conditional_probability_distribution import ConditionalProbabilityDistribution
from typing import Any, List, Union, Dict, Tuple
import numpy as np
from abc import ABCMeta, abstractmethod
from base.probability_distribution import ProbabilityDistribution
from scipy.stats._distn_infrastructure import rv_continuous
from abc import ABCMeta, abstractmethod, abstractproperty

from utils.utils import normalize_bounds

class TimeSerieScalingNoiser(ProbabilityDistribution):

    def __init__(self, initial_time_serie: List[float], scipy_based_distribution: rv_continuous, args_scipy_based_distribution: list, kwargs_scipy_based_distribution: dict, max_error_scale=1.0):
        self._initial_time_serie = np.asarray(initial_time_serie)
        self._len_initial_time_serie = len(self._initial_time_serie)
        self._most_probable = None
        self._distribution = scipy_based_distribution
        self._args_scipy_based_distribution = args_scipy_based_distribution
        self._kwargs_scipy_based_distribution = kwargs_scipy_based_distribution
        self._min_noise, self._max_noise = self._support_noise()
        self._max_error_scale = max_error_scale

    def support(self) -> Tuple[float, float]:
        """
            Support of the distribution

            Returns:
        """
        return (min(self._initial_time_serie)*(1-self._max_error_scale), max(self._initial_time_serie)*(1+self._max_error_scale))
    
    @abstractmethod
    def _support_noise(self) -> Tuple[float, float]:
        raise NotImplementedError()
    
    def _post_process_noise(self, noise):
        return noise

    def sample(self, length=1) -> Union[List, Any]:
        noise = self._distribution.rvs(*self._args_scipy_based_distribution, size=length, **self._kwargs_scipy_based_distribution)
        noise = normalize_bounds(noise, -self._max_error_scale, self._max_error_scale, self._min_noise, self._max_noise, convert_to_float=False)
        noise = self._post_process_noise(noise)
        return self._initial_time_serie*(1+noise)

    def prob(self, x: List) -> float:
        return 1e-6

    def log_prob(self, x) -> float:
        return np.log(self.prob(x))

    def most_probable(self):
        """Most probable data points.
        """
        return None
