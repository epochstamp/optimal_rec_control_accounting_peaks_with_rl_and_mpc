from base.conditional_probability_distribution import ConditionalProbabilityDistribution
from typing import Any, List, Union, Dict, Tuple
import numpy as np
from abc import ABCMeta, abstractmethod
from scipy.stats import norm
from scipy.stats._continuous_distns import norm_gen

from distributions.time_serie_scaling_and_additive.time_serie_red_scaling_and_additive_noiser import TimeSerieRedScalingAndAdditiveNoiser

class TimeSerieRedGaussianScalingAndAdditiveNoiser(TimeSerieRedScalingAndAdditiveNoiser):

    def __init__(self, initial_time_serie: List[float], max_error_scale=1.0, max_error_additive=0.1, r=0.5, scale=1.0, max_error_scale_support=1.0, np_random_state=None):
        if np_random_state is None:
            seed = 1000000
            randomly_seed = np.random.randint(1, 1000000)
            np_random_state=np.random.RandomState(randomly_seed)
        super().__init__(
            initial_time_serie,
            norm,
            [],
            {"loc": 0, "scale": scale, "random_state": np_random_state},
            max_error_scale=max_error_scale,
            max_error_additive=max_error_additive,
            r=r,
            max_error_scale_support=max_error_scale_support
        )

    def _support_noise_additive(self) -> Tuple[float, float]:
        return (-self._max_error_additive, self._max_error_additive)
    
    def _support_noise_scaling(self) -> Tuple[float, float]:
        return (-self._max_error_scale, self._max_error_scale)
