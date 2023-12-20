from base.conditional_probability_distribution import ConditionalProbabilityDistribution
from typing import Any, List, Union, Dict, Tuple
import numpy as np
from abc import ABCMeta, abstractmethod
from base.probability_distribution import ProbabilityDistribution
from scipy.stats._distn_infrastructure import rv_continuous
from abc import ABCMeta, abstractmethod, abstractproperty

from utils.utils import normalize_bounds

class TimeSerieScalingAndAdditiveNoiser(ProbabilityDistribution):

    def __init__(self, initial_time_serie: List[float], scipy_based_distribution: rv_continuous, args_scipy_based_distribution: list, kwargs_scipy_based_distribution: dict, max_error_scale=1.0, max_error_scale_support=1.0, max_error_additive=0.1):
        self._initial_time_serie = np.asarray(initial_time_serie)
        self._len_initial_time_serie = len(self._initial_time_serie)
        self._most_probable = None
        self._distribution = scipy_based_distribution
        self._args_scipy_based_distribution = args_scipy_based_distribution
        self._kwargs_scipy_based_distribution = kwargs_scipy_based_distribution
        self._max_error_scale = max_error_scale
        self._max_error_additive = max_error_additive
        self._max_error_scale_support = max_error_scale_support
        self._min_noise_scaling, self._max_noise_scaling = self._support_noise_scaling()
        self._min_noise_additive, self._max_noise_additive = self._support_noise_additive()
        if "random_state" in self._kwargs_scipy_based_distribution:
            self._random_state_copy_scale = None
            self._random_state_copy_additive = None
            self._random_state:np.random.RandomState = self._kwargs_scipy_based_distribution["random_state"]
            self._kwargs_scipy_based_distribution.pop("random_state")
        self._id = id(self)
    def support(self) -> Tuple[float, float]:
        """
            Support of the distribution

            Returns:
        """
        return (0.0, max(self._initial_time_serie)*(1+self._max_error_scale_support))
    
    @abstractmethod
    def _support_noise_additive(self) -> Tuple[float, float]:
        raise NotImplementedError()
    
    @abstractmethod
    def _support_noise_scaling(self) -> Tuple[float, float]:
        raise NotImplementedError()
    
    def _post_process_noises(self, noise_scaling, noise_additive):
        return noise_scaling, noise_additive

    def sample(self, length=1) -> Union[List, Any]:
        if self._random_state_copy_scale is None or self._random_state_copy_additive is None or self._id != id(self):
            self._id = id(self)
            random_state_copy_scale = np.random.randint(1, 100000)
            random_state_copy_additive = np.random.randint(1, 100000)
            self._random_state_copy_scale = np.random.RandomState(random_state_copy_scale)
            self._random_state_copy_additive = np.random.RandomState(random_state_copy_additive)
        #print(id(self._random_state_copy_scale))
        noise_scaling = self._distribution.rvs(*self._args_scipy_based_distribution, size=length, **{**self._kwargs_scipy_based_distribution, **{"random_state": self._random_state_copy_scale }})
        noise_additive = self._distribution.rvs(*self._args_scipy_based_distribution, size=length, **{**self._kwargs_scipy_based_distribution, **{"random_state": self._random_state_copy_additive}})
        
        noise_scaling = np.clip(noise_scaling, *self._support_noise_scaling())
        noise_additive = np.clip(noise_additive, *self._support_noise_additive())
        noise_scaling = normalize_bounds(noise_scaling, -self._max_error_scale, self._max_error_scale, self._min_noise_scaling, self._max_noise_scaling, convert_to_float=False)
        noise_additive = normalize_bounds(noise_additive, -self._max_error_additive, self._max_error_additive, self._min_noise_additive, self._max_noise_additive, convert_to_float=False)
        noise_scaling, noise_additive = self._post_process_noises(noise_scaling, noise_additive)
        min_value, max_value = self.support()
        new_time_serie = np.clip(self._initial_time_serie*(1+noise_scaling) + (noise_additive)*max(self._initial_time_serie), min_value, max_value)
        return new_time_serie

    def prob(self, x: List) -> float:
        return 1e-6

    def log_prob(self, x) -> float:
        return np.log(self.prob(x))

    def most_probable(self):
        """Most probable data points.
        """
        return None
