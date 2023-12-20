from base.conditional_probability_distribution import ConditionalProbabilityDistribution
from typing import Any, List, Union, Dict, Tuple
import numpy as np
from abc import ABCMeta, abstractmethod
from distributions.time_serie_scaling.time_serie_beta_scaling import TimeSerieBetaScaling
from distributions.time_serie_scaling.time_serie_red_scaling_noiser import TimeSerieRedScalingNoiser
from .time_serie_scaling_noiser import TimeSerieScalingNoiser
from scipy.stats import beta

class TimeSerieRedBetaScaling(TimeSerieRedScalingNoiser):

    def __init__(self, initial_time_serie: List[float], max_error_scale=1.0, r=0.5):
        super().__init__(
            initial_time_serie,
            beta,
            [2, 2],
            {},
            max_error_scale=max_error_scale,
            r=r
        )

    def _support_noise(self) -> Tuple[float, float]:
        return (0, 1)
