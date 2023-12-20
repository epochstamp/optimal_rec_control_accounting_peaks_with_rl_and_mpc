from base.conditional_probability_distribution import ConditionalProbabilityDistribution
from typing import Any, List, Union, Dict, Tuple
import numpy as np
from abc import ABCMeta, abstractmethod
from scipy.stats._distn_infrastructure import rv_continuous
from abc import ABCMeta, abstractmethod, abstractproperty
from distributions.time_serie_scaling_and_additive.time_serie_scaling_and_additive_noiser import TimeSerieScalingAndAdditiveNoiser

from utils.utils import normalize_bounds

class TimeSerieRedScalingAndAdditiveNoiser(TimeSerieScalingAndAdditiveNoiser):

    def __init__(self, initial_time_serie: List[float], scipy_based_distribution: rv_continuous, args_scipy_based_distribution: list, kwargs_scipy_based_distribution: dict, max_error_scale=1.0, max_error_additive=0.1, r=0.5, max_error_scale_support=1.0):
        super().__init__(
            initial_time_serie,
            scipy_based_distribution,
            args_scipy_based_distribution,
            kwargs_scipy_based_distribution,
            max_error_scale=max_error_scale,
            max_error_additive=max_error_additive,
            max_error_scale_support=max_error_scale_support
        )
        self._r = r
        self._sqrt_r = np.sqrt(1-(self._r*self._r))

    def _post_process_noises(self, noise_scaling, noise_additive):
        red_noise_scaling = [noise_scaling[0]]
        red_noise_additive = [noise_additive[0]]
        if self._min_noise_scaling == 0 and self._max_noise_scaling == 0:
            red_noise_scaling = [0]*len(noise_scaling)
        else:
            red_noise_scaling = [noise_scaling[0]]
            for w in noise_scaling[1:]:
                red_noise_scaling += [float(red_noise_scaling[-1] * self._r + self._sqrt_r*w)]
        if self._min_noise_additive == 0 and self._max_noise_additive == 0:
            red_noise_additive = [0]*len(noise_additive)
        else:
            red_noise_additive = [noise_additive[0]]
            for w in noise_additive[1:]:
                red_noise_additive += [float(red_noise_additive[-1] * self._r + self._sqrt_r*w)]
        return np.asarray(red_noise_scaling), np.asarray(red_noise_additive) 
