from typing import Any, List, Union, Dict, Tuple
import numpy as np

from distributions.time_serie_discrete_sampling import TimeSerieDiscreteSampling

class TimeSerieUniformDiscreteSampling(TimeSerieDiscreteSampling):

    
    def _get_t_prob_discrete_distribution(self, t):
        return 1.0 / (self._len_initial_time_serie - self._max_length)
    
    def _sample_t_from_discrete_distribution(self):
        t = np.random.randint(0, self._len_initial_time_serie - self._max_length)
        return t
