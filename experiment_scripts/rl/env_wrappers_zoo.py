from env.rec_env_global_bill_wrapper import RecEnvGlobalBillWrapper
from env.rec_env import RecEnv
from utils.utils import flatten
from .env_wrappers.rec_env_global_bill_timestep_sample import RecEnvGlobalBillTimestepSample
from distributions.uniform_time_step_sampler import UniformTimeStepSampler
from typing import Dict, List, Union

def create_uniform_time_sampler(new_T: int):
    def wrap(rec_env: Union[RecEnv, RecEnvGlobalBillWrapper]):
        if new_T >= rec_env.T:
            raise BaseException("Time sampling is not supported when new_T >= rec_env.T")
        return RecEnvGlobalBillTimestepSample(
            rec_env,
            UniformTimeStepSampler(range(rec_env.T - new_T)),
            max_T=new_T
        )
    return wrap

env_wrapper_sequences: Dict[str, List[RecEnvGlobalBillWrapper]] = {
    "none": [],
    "rec_2_uniform_exogenous_sampler_from_timestep": [create_uniform_time_sampler(101)]
}

def wrap(rec_env, wrappers_ids:List[str]):
    wrappers = list(flatten([
        env_wrapper_sequences[wrapper_id] for wrapper_id in wrappers_ids
    ]))
    current_env = rec_env
    for wrapper in wrappers:
        current_env = wrapper(current_env)
    return current_env