from base.conditional_probability_distribution import ConditionalProbabilityDistribution
from typing import Any, List, Union, Dict, Tuple
import numpy as np
from abc import ABCMeta, abstractmethod
from distributions.time_serie_additive.time_serie_red_additive_noiser import TimeSerieRedAdditiveNoiser
from scipy.stats import norm

class TimeSerieRedGaussianAdditiveNoiser(TimeSerieRedAdditiveNoiser):

    def __init__(self, initial_time_serie: List[float], max_error_scale=1.0, r=0.5):
        super().__init__(
            initial_time_serie,
            norm,
            [],
            {"loc": 0, "scale": 1},
            max_error_scale=max_error_scale,
            r=r
        )

    def _support_noise(self) -> Tuple[float, float]:
        return (-1, 1)
