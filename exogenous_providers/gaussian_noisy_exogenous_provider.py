from base.exogenous_provider import ExogenousProvider
from typing import Dict, Any, List
from scipy.stats import truncnorm
import numpy as np
import random
from utils.utils import chunks, flatten

class GaussianNoisyExogenousProvider(ExogenousProvider):

    def __init__(self, exogenous_sequences: Dict[Any, List[float]], error: float=0, Delta_M=1):
        super().__init__(
            exogenous_sequences,
            Delta_M=Delta_M
        )
        self._error = error
        

    def automate_multi_sequence_sampling(self) -> bool:
        return True

    def _compute_noisy_timeserie(self, timeserie, length, repeat_noise = False, current_timestamp=1):
        if self._error == 0 or length == 0:
            return timeserie
        Fs = 8
        f = 1 
        if repeat_noise:
            timeserie_chunks = list(chunks(timeserie[::-1], self._Delta_M))[::-1]
            sample = len(timeserie_chunks)
        else:
            sample = length 
        y = np.random.normal(0, np.std(timeserie), len(timeserie)) * self._error
        #print(y)
        timeserie_noised = None
        if repeat_noise:
            timeserie_noised = list(flatten([list(np.sqrt(np.square(y[i]+np.array(timeserie_chunks[i])))) for i in np.arange(sample)]))
        else:
            timeserie_noised = list(np.sqrt((y + timeserie)**2))
        return timeserie_noised

    def sample_future_sequence(self, exogenous_sequences: Dict[Any, List[float]], length: int = 1) -> Dict[Any, List[float]]:
        current_timestamp = len(list(exogenous_sequences.values())[0])
        future_sequences = dict()
        
        
        for k,v in self._exogenous_sequences:
            future_sequences[k] = self._compute_noisy_timeserie(v[current_timestamp: current_timestamp+length], length, repeat_noise="buying_price" in k or "selling_price" in k, current_timestamp=current_timestamp)
        return future_sequences