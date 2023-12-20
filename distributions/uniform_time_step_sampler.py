from typing import Any, List, Union, Dict, Tuple
from distributions.time_step_sampler import TimeStepSampler

class UniformTimeStepSampler(TimeStepSampler):

    def __init__(self, t_range: List[int]):
        super().__init__(t_range, [1]*len(t_range))
