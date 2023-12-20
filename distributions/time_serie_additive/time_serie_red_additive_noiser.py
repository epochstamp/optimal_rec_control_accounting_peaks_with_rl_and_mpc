from base.conditional_probability_distribution import ConditionalProbabilityDistribution
from typing import Any, List, Union, Dict, Tuple
import numpy as np
from abc import ABCMeta, abstractmethod
from distributions.time_serie_additive.time_serie_additive_noiser import TimeSerieAdditiveNoiser
from scipy.stats._distn_infrastructure import rv_continuous
from abc import ABCMeta, abstractmethod, abstractproperty

from utils.utils import normalize_bounds

class TimeSerieRedAdditiveNoiser(TimeSerieAdditiveNoiser):

    def __init__(self, initial_time_serie: List[float], scipy_based_distribution: rv_continuous, args_scipy_based_distribution: list, kwargs_scipy_based_distribution: dict, max_error_scale=1.0, r=0):
        super().__init__(
            initial_time_serie,
            scipy_based_distribution,
            args_scipy_based_distribution,
            kwargs_scipy_based_distribution,
            max_error_scale
        )
        self._r = r

    def _post_process_noise(self, noise):
        red_noise = [noise[0]]
        for w in noise[1:]:
            red_noise += [float(red_noise[-1] * self._r + np.sqrt(1-(self._r*self._r)))*w]
        return np.asarray(red_noise)
