from base.conditional_probability_distribution import ConditionalProbabilityDistribution
from typing import Any, List, Union, Dict, Tuple
import numpy as np
from abc import ABCMeta, abstractmethod
from .time_serie_scaling_noiser import TimeSerieScalingNoiser
from scipy.stats import beta

class TimeSerieBetaScaling(TimeSerieScalingNoiser):

    def __init__(self, initial_time_serie: List[float], max_error_scale=1.0):
        super().__init__(
            initial_time_serie,
            beta,
            [2, 2],
            {},
            max_error_scale=max_error_scale
        )

    def _support_noise(self) -> Tuple[float, float]:
        return (0, 1)
