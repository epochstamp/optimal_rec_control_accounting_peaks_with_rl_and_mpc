from copy import deepcopy
import os

from ray.rllib.core.models.base import Model
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec

from experiment_scripts.rl.compute_optimal_z_score import compute_optimal_z_score
from utils.utils import rec_gamma_sequence
os.environ["OPENBLAS_NUM_THREADS"] = str(1) # export OPENBLAS_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = str(1) # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = str(1) # export NUMEXPR_NUM_THREADS=1
os.environ["OMP_NUM_THREADS"] = str(1)
os.environ["MKL_NUM_THREADS"] = str(1)
from hashlib import sha256
import warnings
import zipfile
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.utils.filter import MeanStdFilter
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.evaluation.episode import Episode
from ray.rllib.env.vector_env import VectorEnvWrapper, _VectorizedGymEnv
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.utils.typing import AgentID, EnvType, ModelGradients, MultiAgentPolicyConfigDict, PartialAlgorithmConfigDict, PolicyID, PolicyState, SampleBatchType
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from experiment_scripts.rl.action_distributions import action_distribution_zoo
import ray
from ray.tune.registry import ENV_CREATOR, _global_registry
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode, RolloutWorker
from gymnasium.spaces import Box, Space, Tuple as TupleSpace
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from typing import Any, Callable, Container, Dict, List, Tuple, Type
from .rec_env_rl_libs_gymnasium_gym_wrapper import RecEnvRlLibsGymnasiumGymWrapper
from .rec_env_rl_libs_wrapper import RecEnvRlLibsWrapper
from typing import Union, Optional
from env.rec_env import RecEnv
from .env_wrappers_zoo import env_wrapper_sequences, wrap
import glob
from .rl_envs_zoo import create_rl_env_creators, rl_envs
from ray.rllib.algorithms.callbacks import make_multi_callbacks
from .models_zoo import models_zoo
from .space_converters_zoo import space_converter_sequences
from envs import create_env
from env.counter_utils import future_counters
import os
import array
import json
import struct
import click
import wandb
import time
import torch as tc
import numpy as np
import random
from pprint import pprint
from envs import create_env_fcts
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.utils.annotations import override
import threading
from itertools import islice
from .parallel_worker_set_wrapper import ParallelWorkerSetWrapper
from multiprocess import Queue, Manager, shared_memory, Semaphore
  
  
def chunk(arr_range, arr_size): 
    arr_range = iter(arr_range) 
    return iter(lambda: tuple(islice(arr_range, arr_size)), ()) 

os.environ["WANDB_SILENT"] = "true"

value_target_values = []
gamma_global = None
obs_z_score_global = None
rew_z_score_global = None
NUM_OPTIM_ROLLOUTS = None
DEBUG_MODE = None
GLOBAL_MEMORY=dict()
LOCAL_MEMORY=None
GAMMA_SEQUENCE=None
OUTPUT_PATH_MODEL=None


#algo.gamma=lst_gammas[i]


def get_random_states():
    return random.getstate(), np.random.get_state(), tc.random.get_rng_state()

def restore_random_states(random_states):
    random_state, np_random_state, tc_random_state = random_states
    random.setstate(random_state)
    np.random.set_state(np_random_state)
    tc.random.set_rng_state(tc_random_state)


def init_worker(memory_dict):
    global LOCAL_MEMORY
    LOCAL_MEMORY = memory_dict

def reset_env(ids):
    global GLOBAL_MEMORY
    env_id, id = ids
    env = GLOBAL_MEMORY[env_id]["envs"][id]
    obs_and_infos = env.reset()
    return env, obs_and_infos

def step_env(ids):
    global GLOBAL_MEMORY
    env_id, id = ids
    env = GLOBAL_MEMORY[env_id]["envs"][id] 
    action = GLOBAL_MEMORY[env_id]["actions"][id] 
    trans_tuple = env.step(action)
    return env, trans_tuple

def reset_env_mono(env):
    obs_and_infos = env.reset()
    return obs_and_infos

def step_env_mono(env_action):
    env, action = env_action
    trans_tuple = env.step(action)
    return trans_tuple

class TransitionWorker(threading.Thread):
    def __init__(self, q, envs, *args, **kwargs):
        self.q = q
        self.envs = envs
        super().__init__(*args, **kwargs)
        self.data = []
        self.data_append = self.data.append
        self._run=True

    def stop_run(self):
        self._run = False

    def start_run(self) -> None:
        self._run = True

    def reset(self):
        self.data = []
    def run(self):
        while self._run:
            work=None
            try:
                work = self.q.get(block=True, timeout=0.00001)  # 3s timeout
            except:
                pass
            if work is not None:
                if work == "reset_data":
                    self.data = []
                else:
                    if work[0] == "reset":
                        data = self.envs[work[1]].reset()
                    elif work[0] == "step":
                        data = self.envs[work[1]].step(work[2])
                    else:
                        print("I don't recognize this task:", "[" + str(work) + "]")
                    self.data.append((work[1],) +  data)
                self.q.task_done()

class _ParallelVectorizedGymEnv(_VectorizedGymEnv):
    def __init__(self,
                 env_train_id,
                 env_eval_id,
                 num_cpus,
                 existing_envs = None,
                 num_envs: int = 1):
        global GLOBAL_MEMORY
        self._env_train_id = env_train_id
        self._env_eval_id = env_eval_id
        self._existing_env = existing_envs[0]
        super().__init__(self._make_env,
                            existing_envs,
                            num_envs)
        self._final_env_id = self._env_train_id if not self._existing_env.eval_env else self._env_eval_id
        self._lst_num_envs = list(range(len(self.envs)))
        self._combined_lst_num_envs = [(self._final_env_id, idx) for idx in self._lst_num_envs]
        self._num_envs = num_envs
        #self._pool = Pool(min(num_cpus, num_envs))
        
        
        #q.join()  # blocks until the queue is empty.

    def _make_env(self, index=0):
        return _global_registry.get(ENV_CREATOR, self._env_train_id if not self._existing_env.eval_env else self._env_eval_id)()

    @override(_VectorizedGymEnv)
    def vector_reset(
            self, *, seeds = None, options = None
        ):
        # Use reset_at(index) to restart and retry until
        # we successfully create a new env.
        #with Pool(self._num_procs) as pool:
        lst_obs_infos = map(reset_env_mono, self.envs)
        obs_and_infos = zip(*lst_obs_infos)
        obs_and_infos = [list(l) for l in obs_and_infos]
        """
        for q in self._qs:
            q.put_nowait("reset")
        for i, q in enumerate(self._qs):
            t = time.time()
            q.join()
            print(time.time() - t)
            obs_and_infos_threads.extend(self._workers[i].data)
        """
            
        return obs_and_infos

    
    @override(_VectorizedGymEnv)
    def vector_step(
        self, actions
    ):
        lst_steps_info = map(step_env_mono, [(self.envs[i], actions[i]) for i in self._lst_num_envs])
        trans_tuple = zip(*lst_steps_info)
        trans_tuples = tuple([list(l) for l in trans_tuple])
        if (trans_tuples[-2][0] or trans_tuples[-3][0]):
            current_t = trans_tuples[-1][0]['current_t']
            setting = "training" if not self._existing_env.eval_env else "evaluation"
            #print(f"Just finished an episode of {current_t} steps in a {setting} setting")
        return trans_tuples
    """
    @override(_VectorizedGymEnv)
    def vector_step(
        self, actions
    ):
        t = time.time()
        obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = (
            [],
            [],
            [],
            [],
            [],
        )
        for i in range(self.num_envs):
            try:
                results = self.envs[i].step(actions[i])
            except Exception as e:
                if self.restart_failed_sub_environments:
                    self.restart_at(i)
                    results = e, 0.0, True, True, {}
                else:
                    raise e

            obs, reward, terminated, truncated, info = results

            if not isinstance(info, dict):
                raise ValueError(
                    "Info should be a dict, got {} ({})".format(info, type(info))
                )
            obs_batch.append(obs)
            reward_batch.append(reward)
            terminated_batch.append(terminated)
            truncated_batch.append(truncated)
            info_batch.append(info)
        print(time.time() - t)
        return obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch
    """

def create_shared_memories(n, m):
    return ({i:shared_memory.SharedMemory(create=True, size=m).name for i in range(n)}, m)
manager = Manager()
global_shared_memories = manager.dict()


class MetricsCollector(DefaultCallbacks):

    def __init__(self, *args, legacy_callbacks_dict: Dict[str, callable] = None, shared_memory=True, **kwargs):
        super().__init__(legacy_callbacks_dict=legacy_callbacks_dict)
        global global_shared_memories
        self._shared_memory = shared_memory
        if shared_memory:
            """
            self._discounted_sums_shared_memory_names, self._discounted_sums_shared_memory_size = global_shared_memories["discounted_sum"]
            self._undiscounted_sums_shared_memory_names, self._undiscounted_sums_shared_memory_size = global_shared_memories["undiscounted_sum"]
            self._switch_eval_shared_memory_name= global_shared_memories["switch_eval"][0][0]
            self._value_target_shared_memory_names, self._value_target_shared_memory_size = global_shared_memories["value_targets"]
            self._vec_reset_sizes = dict()
            for size in (self._discounted_sums_shared_memory_size, self._undiscounted_sums_shared_memory_size, self._value_target_shared_memory_size):
                if size not in self._vec_reset_sizes.keys():
                    self._vec_reset_sizes[size] = bytearray([0]*size)
            """
            self._discounted_sums_shared_memory = global_shared_memories["discounted_sums"]
            self._undiscounted_sums_shared_memory = global_shared_memories["undiscounted_sums"]
            self._value_target_shared_memory = global_shared_memories["value_targets"]
        else:
            self._discounted_sums = []
            self._undiscounted_sums = []
            self._value_targets = []
            self._switch_eval = False
        self._reset()
    def _reset(self):
        self._reset_metrics()


    def _reset_metrics(self):
        if self._shared_memory:
            self._discounted_sums_shared_memory[:] = []
            self._undiscounted_sums_shared_memory[:] = []
            self._value_target_shared_memory[:] = []
        else:
            self._discounted_sums = []
            self._undiscounted_sums = []
            self._value_targets = []
        
    def on_evaluate_start(self, *, algorithm: Algorithm, **kwargs) -> None:
        self._reset()

    def on_evaluate_end(self, *, algorithm: Algorithm, evaluation_metrics: dict, **kwargs) -> None:
        if self._shared_memory:
            discounted_expected_returns = self._get_discounted_sums()
            undiscounted_expected_returns = self._get_undiscounted_sums()
            value_targets = self._get_value_targets()
            evaluation_metrics["evaluation"]["custom_metrics"]["discounted_rewards_mean"] = (
                np.mean(discounted_expected_returns)
            )
            evaluation_metrics["evaluation"]["custom_metrics"]["discounted_rewards_var"] = (
                np.std(discounted_expected_returns, ddof=1) if len(discounted_expected_returns) > 1
                else 0.0
            )
            evaluation_metrics["evaluation"]["custom_metrics"]["undiscounted_rewards_mean"] = (
                np.mean(undiscounted_expected_returns)
            )
            evaluation_metrics["evaluation"]["custom_metrics"]["undiscounted_rewards_var"] = (
                np.std(undiscounted_expected_returns, ddof=1) if len(undiscounted_expected_returns) > 1
                else 0.0
            )

            evaluation_metrics["evaluation"]["custom_metrics"]["value_targets_mean"] = (
                np.mean(value_targets)
            )
            evaluation_metrics["evaluation"]["custom_metrics"]["value_targets_var"] = (
                np.std(value_targets, ddof=1) if len(value_targets) > 1
                else 0.0
            )
            evaluation_metrics["evaluation"]["custom_metrics"]["value_targets_min"] = (
                np.min(value_targets)
            )
            evaluation_metrics["evaluation"]["custom_metrics"]["value_targets_max"] = (
                np.max(value_targets)
            )
        else:
            pass
            #value_targets_lst = evaluation_metrics["evaluation"]["hist_stats"]["value_targets_lst"]
            #undiscounted_rewards_lst = evaluation_metrics["evaluation"]["hist_stats"]["undiscounted_rewards_lst"]
            #discounted_rewards_lst = evaluation_metrics["evaluation"]["hist_stats"]["discounted_rewards_lst"]
            #evaluation_metrics["evaluation"]["custom_metrics"]["value_targets_var"] = (
            #    np.std(value_targets_lst, ddof=1)
            #    if len(value_targets_lst) > 1
            #    else 0.0
            #)
            #evaluation_metrics["evaluation"]["custom_metrics"]["undiscounted_rewards_var"] = (
            #    np.std(undiscounted_rewards_lst, ddof=1)
            #    if len(undiscounted_rewards_lst) > 1
            #    else 0.0
            #)
            #evaluation_metrics["evaluation"]["custom_metrics"]["discounted_rewards_var"] = (
            #    np.std(discounted_rewards_lst, ddof=1)
            #    if len(discounted_rewards_lst) > 1
            #    else 0.0
            #)
        self._reset()

    def _collect_undiscounted_sum(self, postprocessed_batch):
        return self._collect_discounted_sum(postprocessed_batch, discounted=False)
    
    def _collect_value_targets(self, postprocessed_batch):
        return postprocessed_batch["vf_preds"]

    def _collect_discounted_sum(self, postprocessed_batch, discounted=True):
        global GAMMA_SEQUENCE
        #episode.custom_metrics["discounted_rewards"] = np.dot(postprocessed_batch["rewards"], self._gammas[:len_rewards])
        if discounted:
            len_rewards = postprocessed_batch["rewards"].shape[0]
            discounted_sum = np.dot(postprocessed_batch["rewards"], GAMMA_SEQUENCE[:len_rewards])
        else:
            discounted_sum = np.sum(postprocessed_batch["rewards"])
        return discounted_sum
        #global_discounted_expected_returns_queue.put_nowait(episode.custom_metrics["discounted_rewards"])

    def _update_value_targets(self, postprocessed_batch, collect=False):
        value_targets = self._collect_value_targets(postprocessed_batch)
        if self._shared_memory:
            self._value_target_shared_memory.extend(value_targets)
        else:
            self._value_targets.extend(value_targets)
        if collect:
            return value_targets

    def _get_value_targets(self):
        lst_value_targets = []
        if self._shared_memory:
            lst_value_targets = self._value_target_shared_memory
        else:
            lst_value_targets = self._value_targets
        return lst_value_targets

    def _update_discounted_sum(self, postprocessed_batch, collect=False):
        #print(self._get_unroll_id(postprocessed_batch))
        discounted_sum = self._collect_discounted_sum(postprocessed_batch)
        if self._shared_memory:
            self._discounted_sums_shared_memory.append(discounted_sum)
        else:
            self._discounted_sums.append(discounted_sum)
        if collect:
            return discounted_sum

    def _get_discounted_sums(self):
        lst_discounted_sums = []
        if self._shared_memory:
            lst_discounted_sums = self._discounted_sums_shared_memory
        else:
            lst_discounted_sums = self._discounted_sums
        return lst_discounted_sums
    
    def _get_unroll_id(self, postprocessed_batch):
        return postprocessed_batch["unroll_id"][0]-1
    
    def _update_undiscounted_sum(self, postprocessed_batch, collect=False):
        undiscounted_sum = self._collect_undiscounted_sum(postprocessed_batch)
        if self._shared_memory:
            self._undiscounted_sums_shared_memory.append(undiscounted_sum)
        else:
            self._undiscounted_sums.append(undiscounted_sum)
        if collect:
            return undiscounted_sum

    def _get_undiscounted_sums(self):
        lst_undiscounted_sums = []
        if self._shared_memory:
            lst_undiscounted_sums = self._undiscounted_sums_shared_memory
        else:
            lst_undiscounted_sums = self._undiscounted_sums
        return lst_undiscounted_sums
    
    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[PolicyID, Policy], episode: Episode | EpisodeV2 | Exception, env_index: int | None = None, **kwargs) -> None:
        if not self._shared_memory:
            episode.is_eval = base_env.vector_env.envs[0].eval_env

    def on_postprocess_trajectory(
        self,
        *,
        worker: RolloutWorker,
        episode: Episode,
        agent_id: str,
        policy_id: str,
        policies: Dict[str, Policy],
        postprocessed_batch: SampleBatch,
        original_batches: Dict[str, Tuple[Policy, SampleBatch]],
        **kwargs
    ):
        #postprocessed_batch["infos"])
        #if postprocessed_batch["infos"][0]["eval_env"]:
        if episode.is_eval:
            discounted_sum = self._update_discounted_sum(postprocessed_batch, collect=not self._shared_memory)
            undiscounted_sum = self._update_undiscounted_sum(postprocessed_batch, collect=not self._shared_memory)
            value_targets = self._update_value_targets(postprocessed_batch, collect=not self._shared_memory)
            if not self._shared_memory:
                episode.custom_metrics["value_targets"] = value_targets
                #episode.hist_data["value_targets_lst"] = value_targets
                episode.custom_metrics["undiscounted_rewards"] = undiscounted_sum
                #episode.hist_data["undiscounted_rewards_lst"] = [undiscounted_sum]
                episode.custom_metrics["discounted_rewards"] = discounted_sum
                #episode.hist_data["discounted_rewards_lst"] = [discounted_sum]
                
        
        #if worker.env.eval_env:
            #print(self._get_unroll_id(postprocessed_batch))
            #self._incr_counter()
        
        #if postprocessed_batch["infos"][0]["eval_env"]:
            #print("here")
            #print(len(postprocessed_batch["vf_preds"].tolist()))
            #for v in postprocessed_batch["vf_preds"].tolist():
                #global_value_stats_queue.put(-v)

class SequentialMetricsCollector(MetricsCollector):

    def __init__(self, *args, legacy_callbacks_dict: Dict[str, Any] = None, shared_memory=None, **kwargs):
        super().__init__(*args, legacy_callbacks_dict=legacy_callbacks_dict, shared_memory=False, **kwargs)

def apply_func_in_nested_dict(f, dic):
    return {
        k:(f(d) if isinstance(d, tc.Tensor) else apply_func_in_nested_dict(f, d)) for k, d in dic.items()
    }

class JustOutputModel(DefaultCallbacks):

    def on_algorithm_init(self, *, algorithm: Algorithm, **kwargs) -> None:
        global OUTPUT_PATH_MODEL
        from dill import dump, load
        env = algorithm.env_creator()
        obs, info = env.reset()
        state_in =  algorithm.get_policy().model.get_initial_state()
        model:PPOTorchRLModule= algorithm.get_policy().model
        obs = tc.from_numpy(obs)[None, None, :]
        state_in = apply_func_in_nested_dict(lambda x: x[None, :], state_in)
        policy = algorithm.get_policy()
        model.config = None
        os.makedirs(OUTPUT_PATH_MODEL, exist_ok=True)
        tc.save(model, OUTPUT_PATH_MODEL+"/model.m")
        
        with open(OUTPUT_PATH_MODEL+"/inputs.dat", "wb") as finputs:
            dump(obs, finputs)
        with open(OUTPUT_PATH_MODEL+"/state.dat", "wb") as fstate:
            dump(state_in, fstate)
        exit()


def validate_space_converter(ctx, param, value):
    space_converters = value.split("#")
    invalid_space_converters = [
        v for v in space_converters if v not in space_converter_sequences.keys()
    ]
    if len(invalid_space_converters) > 0:
        raise click.BadParameter(f"These space converter ids are not registered : {invalid_space_converters}")
    return value

def abrefy(s, separator="_"):
    return "".join([e[0] for e in s.split(separator)])

@click.command()
@click.option('--env', "env", type=click.Choice(list(create_env_fcts.keys())), help='Environment to launch for training', required=True)
@click.option('--env-wrappers', "env_wrappers", type=click.Choice(list(env_wrapper_sequences.keys())), help='Reduce training environment time horizon (useful for sampling different exogenous variables starts). Multiple wrappers possible, sep by # character', default=None)
@click.option('--env-valid', "env_valid", type=click.Choice(list(create_env_fcts.keys())), help='Environment to launch for validation (default : same as training)', default=None, callback=lambda c, p, v: v if v is not None else c.params['env'])
@click.option('--env-eval', "env_eval", type=click.Choice(list(create_env_fcts.keys())), help='Environment to launch for testing (default : same as training)', default=None, callback=lambda c, p, v: v if v is not None else c.params['env'])
@click.option('--rl-env', "rl_env", type=click.Choice(list(rl_envs.keys())), help='RL env configuration for training', default="rl")
@click.option('--rl-env-eval', "rl_env_eval", type=click.Choice(list(rl_envs.keys())), help='RL env configuration for eval (default : same as training)', default="rl")
@click.option('--T', "T", type=int, default=None, help='Time horizon T (default : auto env). Cannot be greater than auto env')
@click.option('--Delta-M', "Delta_M", type=int, default=2, help='Delta_M.')
@click.option('--Delta-P', "Delta_P", type=int, default=1, help='Delta_P.')
@click.option('--Delta-P-prime', "Delta_P_prime", type=int, default=1, help='Delta_P_prime.')
@click.option('--random-seed', "random_seed", type=int, default=1, help='Random seed.')
@click.option('--remove-current-peak-costs/--no-remove-current-peak-costs', "remove_current_peak_costs", is_flag=True, help='Whether current peak costs are removed.')
@click.option('--remove-historical-peak-costs/--no-remove-historical-peak-costs', "remove_historical_peak_costs", is_flag=True, help='Whether historical peak costs are removed.')
@click.option('--erase-file/--no-erase-file', "erase_file", is_flag=True, help='Whether result file is erased.')
@click.option('--stdout/--no-stdout', "stdout", is_flag=True, help='Whether the result is print instead of being saved.')
@click.option('--multiprice/--no-multiprice', "multiprice", is_flag=True, help='Whether (buying) are changing per metering period.')
@click.option('--space-converter', "space_converter", type=str, help='Space converter (can use several with # separator)', default="no_converter", callback=validate_space_converter)
@click.option("--mean-std-filter-mode", "mean_std_filter_mode", type=click.Choice(["no_filter", "only_obs", "obs_and_rew", "obs_optim", "rew_optim", "obs_and_rew_optim", "obs_multi_optim", "rew_multi_optim", "obs_and_rew_multi_optim"]), help="Choose whether observation and/or is zscored by running mean/std")
@click.option('--model-config', "model_config", type=click.Choice(models_zoo.keys()), default=None, help="Model config available from models zoo (see experiment_scripts/rl/models_zoo.py)", callback = lambda c, p, v: v if v is not None else c.params['env'] + "_default")
@click.option('--gamma', "gamma", type=float, help='Discount factor gamma', default=0.99)
@click.option('--gamma-policy', "gamma_policy", type=str, help='Discount factor gamma for RL (either single value of 3 values separated by # for gamma scheduling)')
@click.option('--lambda-gae', "lambda_gae", type=float, help='GAE Lambda value', default=1.0)
@click.option('--clip-param', "clip_param", type=str, help='PPO clip param (either single value or three values separated by # for schedule)', default="0.3")
@click.option('--vf-clip-param', "vf_clip_param", type=float, help='PPO clip param for VF', default=10.0)
@click.option('--vf-coeff', "vf_coeff", type=float, help='Value function coeff', default=1.0)
@click.option('--entropy-coeff', "entropy_coeff", type=float, help='Entropy coeff', default=0.0)
@click.option('--kl-coeff', "kl_coeff", type=float, help='KL coeff', default=0.2)
@click.option('--kl-target', "kl_target", type=float, help='KL target', default=0.01)
@click.option('--lr', "learning_rate", type=str, help='Learning rate (either one or three values for schedule)', default="5e-05")
@click.option('--bs', "batch_size", type=int, help='Batch size', default=64)
@click.option("--gc", "gc", type=float, help="Gradient clipping value (0 for default clipping per algo)", default=0)
@click.option('--ne', "number_of_episodes", type=int, help='Number of episodes per training iter', default=1)
@click.option('--n-sgds', "n_sgds", type=int, help='Number of SGD passes', default=10)
@click.option('--action-weights-divider', "action_weights_divider", type=float, help='Divider of the weights of the output action layer', default=1.0)
@click.option("--action-dist", "action_dist", type=click.Choice(list(action_distribution_zoo.keys())), default="default", help="Choice of action distribution for policy")
@click.option('--n-gpus', "n_gpus", type=int, help='Number of gpus', default=0)
@click.option('--n-cpus', "n_cpus", type=int, help='Number of cpus', default=1)
@click.option('--use-wandb/--no-use-wandb', "use_wandb", is_flag=True, help='Whether to use Weight and Biases')
@click.option("--wandb-project", "wandb_project", default="rlstudy", help="Wandb project name")
@click.option("--wandb-offline", "wandb_offline", is_flag=True, help="Whether to turn wandb offline")
@click.option("--gymnasium-wrap", "gymnasium_wrap", is_flag=True, help="Whether to wrap envs with Gymnasium wrapper (useful for latest versions of Ray)")
@click.option("--root-dir", "root_dir", default=os.path.expanduser('~'), help="Root directory")
@click.option("--tmp-dir", "tmp_dir", default=os.path.expanduser('~'), help="Temp directory")
@click.option('--time-iter', "time_iter", is_flag=True, help='Whether to display iteration/evaluaton time.')
@click.option('--sha-folders', "sha_folders", is_flag=True, help='Whether to rename results folders on sha256 (parameters are registered in a separate json).')
@click.option('--tar-gz-results', "tar_gz_results", is_flag=True, help='Whether to compress results files on a single archive ((except parameters files)).')
@click.option('--n-iters', "n_iters", type=int, help='Number of iterations.', default=201)
@click.option('--ne-eval', "number_of_episodes_eval", type=int, help='Number of episodes per evaluation iter', default=10)
@click.option('--framework', "framework", type=click.Choice(["tf2", "torch"]), help="Framework to use for deep nn", default="torch")
@click.option('--compute-global-bill-on-next-obs', "compute_global_bill_on_next_obs", is_flag=True, help='Whether global rec bill is computed on next observation.')
@click.option('--debug', "debug", is_flag=True, help='Debug mode.')
@click.option('--just-output-model', "just_output_model", is_flag=True, help='Just output neural network models.')
@click.option('--just-output-model-path', "just_output_model_path", type=str, default="$HOME/OneDrive/rec_plots/models/", help='Output folder path for neural network models.')
@click.option('--evaluation-interval', "evaluation_interval", type=int, help='Evaluation interval', default=1)
def run_experiment(env, env_wrappers, env_valid, env_eval, rl_env, rl_env_eval, T, Delta_M, Delta_P, Delta_P_prime, random_seed, remove_current_peak_costs, remove_historical_peak_costs, erase_file, stdout, multiprice, space_converter, mean_std_filter_mode, model_config, gamma, gamma_policy, lambda_gae, clip_param, vf_clip_param, vf_coeff, entropy_coeff, kl_coeff, kl_target, learning_rate, batch_size, gc, number_of_episodes, n_sgds, action_weights_divider, action_dist, n_gpus, n_cpus, use_wandb, wandb_project, wandb_offline, gymnasium_wrap, root_dir, tmp_dir, time_iter, sha_folders, tar_gz_results, n_iters, number_of_episodes_eval, framework, compute_global_bill_on_next_obs, debug, just_output_model, just_output_model_path, evaluation_interval):
    global gamma_global, NUM_OPTIM_ROLLOUTS, DEBUG_MODE, global_shared_memories, GAMMA_SEQUENCE, manager, OUTPUT_PATH_MODEL
    OUTPUT_PATH_MODEL = just_output_model_path+"/"+model_config
    NUM_OPTIM_ROLLOUTS=number_of_episodes_eval
    DEBUG_MODE=debug
    
    random.seed(random_seed)
    np.random.seed(random_seed)
    tc.manual_seed(random_seed)
    tc.set_num_interop_threads(1)
    tc.set_num_threads(1)
    value_target_values = []
    folder = root_dir + "/rec_experiments/RL/"
    n_samples = 1
    gamma_policy_steps = None
    gamma_policy_start = None
    gamma_policy_end = None
    gamma_policy_values = gamma_policy.split("#")
     
    if len(gamma_policy_values) == 1:
        gamma_policy = gamma_policy if float(gamma_policy_values[0]) != 0 else gamma

    elif len(gamma_policy_values) == 3:
        gamma_policy_start, gamma_policy_end, gamma_policy_steps = tuple([float(gp) for gp in gamma_policy_values])
        gamma_policy_steps = int(gamma_policy_steps)
    else:
        raise BaseException("Not a valid value for gamma_policy:", gamma_policy)
    
    clip_param_steps = None
    clip_param_start = None
    clip_param_end = None
    clip_param_values = clip_param.split("#")
    if len(clip_param_values) == 1:
        clip_param_start = float(clip_param_values[0])
    else:
        clip_param_start, clip_param_end, clip_param_steps = tuple(float(cp) for cp in clip_param_values)
        clip_param_steps = int(clip_param_steps)
        

    current_offtake_peak_cost = None if not remove_current_peak_costs else 0
    current_injection_peak_cost = None if not remove_current_peak_costs else 0
    historical_offtake_peak_cost = None if not remove_historical_peak_costs else 0
    historical_injection_peak_cost = None if not remove_historical_peak_costs else 0
    remove_peaks_costs = remove_current_peak_costs and remove_historical_peak_costs
    rec_env_train, infos_rec_env_train = create_env(
        id_env=env,
        env_name=f"{env_eval}_train",
        Delta_M=Delta_M,
        Delta_P=Delta_P,
        Delta_P_prime=Delta_P_prime,
        current_offtake_peak_cost=current_offtake_peak_cost,
        current_injection_peak_cost=current_injection_peak_cost,
        historical_offtake_peak_cost=historical_offtake_peak_cost,
        historical_injection_peak_cost=historical_injection_peak_cost,
        multiprice=multiprice,
        disable_warnings=False,
        T=T,
        n_cpus_global_bill_optimiser=1,
        seed=random_seed,
        compute_global_bill_on_next_obs=compute_global_bill_on_next_obs,
        type_solver=("mosek" if "dense" in rl_env else "cvxpy")
    )
    #t=time.time()
    #rec_env_train.reset()
    ##print(time.time() - t)
    #t=time.time()
    #rec_env_train.reset()
    #print(time.time() - t)
    #exit()
    if env_wrappers is not None:
        rec_env_train = wrap(rec_env_train, env_wrappers.split("#"))
    rec_env_train._n_cpus_global_bill_optimiser=1
    rec_env_eval, infos_rec_env_eval = create_env(
        id_env=env_eval,
        env_name=f"{env_eval}_eval",
        Delta_M=Delta_M,
        Delta_P=Delta_P,
        Delta_P_prime=Delta_P_prime,
        current_offtake_peak_cost=current_offtake_peak_cost,
        current_injection_peak_cost=current_injection_peak_cost,
        historical_offtake_peak_cost=historical_offtake_peak_cost,
        historical_injection_peak_cost=historical_injection_peak_cost,
        multiprice=multiprice,
        disable_warnings=False,
        T=T,
        n_cpus_global_bill_optimiser=1,
        time_optim=time_iter,
        seed=random_seed,
        type_solver=("mosek" if "dense" in rl_env else "cvxpy")
    )
    rec_env_eval._n_cpus_global_bill_optimiser = 1
    rec_env_valid, infos_rec_env_valid = None, None
    if env_eval != env_valid:
        rec_env_valid, infos_rec_env_valid = create_env(
            id_env=env_valid,
            env_name=f"{env_valid}_valid",
            Delta_M=Delta_M,
            Delta_P=Delta_P,
            Delta_P_prime=Delta_P_prime,
            current_offtake_peak_cost=current_offtake_peak_cost,
            current_injection_peak_cost=current_injection_peak_cost,
            historical_offtake_peak_cost=historical_offtake_peak_cost,
            historical_injection_peak_cost=historical_injection_peak_cost,
            multiprice=multiprice,
            disable_warnings=False,
            T=T,
            n_cpus_global_bill_optimiser=1,
            seed=random_seed
        )
        rec_env_valid._n_cpus_global_bill_optimiser=1
    assert(rec_env_train.T == rec_env_eval.T and (rec_env_valid is None or rec_env_eval.T == rec_env_valid.T))
    T = rec_env_train.T
    future_counter_tau_dm, future_counter_tau_dp = future_counters(
        0, 0, duration=T-1, Delta_M=Delta_M, Delta_P=Delta_P
    )
    model_config_dict = models_zoo[model_config]()
    is_custom_model = model_config_dict.pop("is_custom", False)
    trainable_initial_state_type = model_config_dict.pop("initial_state_type", "zeros")
    if not ((future_counter_tau_dp[-1] == Delta_P or (remove_peaks_costs and future_counter_tau_dm[-1] == Delta_M)) and not ("optim" in mean_std_filter_mode and "flatten_and_boxify" not in space_converter and False) and not (batch_size < model_config_dict.get("max_seq_len", batch_size) and False)):
        print("Conflicting case detected, exit")
    else:
        if gamma_policy_steps is not None:
            if gamma_policy_steps < 0:
                lst_gammas = [gamma_policy_start]*n_iters
            elif gamma_policy_steps == 0:
                lst_gammas = [gamma_policy_end]*n_iters
            elif gamma_policy_steps > 0:
                lst_gammas = list(np.linspace(gamma_policy_start, gamma_policy_end, gamma_policy_steps))
                lst_gammas += [gamma] * max(n_iters - gamma_policy_steps + 1, 0)
                lst_gammas = lst_gammas[1:]
        else:
            lst_gammas = [float(gamma_policy)]*n_iters
        
        if clip_param_steps is not None:
            if clip_param_steps <= 0:
                lst_clip_param = [clip_param_start]*n_iters
            else:
                lst_clip_param = list(np.linspace(clip_param_start, clip_param_end, clip_param_steps))
                lst_clip_param += [clip_param_end] * max(n_iters - clip_param_steps + 1, 0)
                lst_clip_param = lst_clip_param[1:]
        else:
            lst_clip_param = [clip_param_start]*n_iters
        """
        if mean_std_filter_mode == "obs_and_rew":
                lst_callbacks += [NormRewardsCallback]
            elif mean_std_filter_mode == "obs_and_rew_optim":
                lst_callbacks += [OptiNormObsRewardsCallback]
            elif mean_std_filter_mode == "obs_optim":
                lst_callbacks += [OptiNormObsCallback]
            elif mean_std_filter_mode == "rew_optim":
                lst_callbacks += [OptiNormRewCallback]
            elif mean_std_filter_mode == "obs_and_rew_multi_optim":
                lst_callbacks += [MultiOptiNormObsRewardsCallback]
            elif mean_std_filter_mode == "obs_multi_optim":
                lst_callbacks += [MultiOptiNormObsCallback]
            elif mean_std_filter_mode == "rew_multi_optim":
                lst_callbacks += [MultiOptiNormRewCallback]
        """
        obs_optim = mean_std_filter_mode in (
            "obs_and_rew_optim",
            "obs_optim",
            "obs_multi_optim",
            "obs_and_rew_multi_optim"
        )
        rew_optim = mean_std_filter_mode in (
            "obs_and_rew_optim",
            "rew_optim",
            "rew_multi_optim",
            "obs_and_rew_multi_optim"
        )
        num_optim_rollouts = (
            1 if "multi" not in mean_std_filter_mode
            else number_of_episodes_eval
        )
        env_train_id, env_eval_id, env_eval_creator, env_valid_creator = create_rl_env_creators(
            rl_env, rl_env_eval, rec_env_train, rec_env_eval, space_converter, gymnasium_wrap=gymnasium_wrap, infos_rec_env_train=infos_rec_env_train, infos_rec_env_eval=infos_rec_env_eval, rec_env_valid=rec_env_valid, infos_rec_env_valid=infos_rec_env_valid, members_with_controllable_assets=infos_rec_env_train["members_with_controllable_assets"],
            gamma=gamma, obs_optim=obs_optim, rew_optim=rew_optim, num_optim_rollouts=num_optim_rollouts, verbose=debug
        )
        
        multiprice_str = "multi" if multiprice else "mono"
        
        hyper_parameters = {
            "model_config": model_config,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "number_of_episodes": number_of_episodes,
            "gradient_clipping_norm": gc,
            "number_of_sgds": n_sgds,
            "gamma": gamma,
            "gamma_policy": gamma_policy,
            "mean_std_filter_mode": mean_std_filter_mode,
            "lambda_gae": lambda_gae,
            "entropy_coeff": entropy_coeff,
            "kl_coeff": kl_coeff,
            "kl_target": kl_target,
            "clip_param": clip_param,
            "vf_clip_param": vf_clip_param,
            "vf_coeff": vf_coeff,
            "action_weights_divider": action_weights_divider,
            "action_dist": action_dist,
            
        }
        space_converter_str_seq = space_converter.split("#")
        #space_converter_str = "_".join(["".join([r2[0] for r2 in r.split("_")]) for r in space_converter_str_seq])
        config_wandb = {**hyper_parameters, **{
            "env_name": env,
            "env_eval_name": env_eval,
            "env_valid_name": env_valid,
            "rl_env": rl_env,
            "rl_env_eval": rl_env_eval,
            "multiprice": multiprice,
            "Delta_M": Delta_M,
            "space_converter": space_converter,
            "random_seed": random_seed,
            "env_wrappers": env_wrappers,
            "number_of_episodes_eval": number_of_episodes_eval
            
        }}
        hyper_parameters_slashes_str = "/".join([
            f"{k}={v}" for k,v in hyper_parameters.items()
        ])
        #hyper_parameters_underscore_str = "_".join([
        #    f"{abrefy(str(k))}={abrefy(str(v))}" for k,v in hyper_parameters.items()
        #]).replace(".", "_").replace("True", "T").replace("False", "F")
        prefix = f"multiprice={multiprice_str}/evaluation_interval={evaluation_interval}/env={env}/number_of_episodes_eval={number_of_episodes_eval}/env_wrappers={env_wrappers}/env_eval={env_eval}/env_valid={env_valid}/rl_env={rl_env}/rl_env_eval={rl_env_eval}/space_converter={space_converter}/{hyper_parameters_slashes_str}/random_seed={random_seed}/Delta_M={Delta_M}"
        #group_wandb=f"{env_str}_{env_eval_str}_{rl_env_str}_{rl_env_eval_str}_{multiprice_str}_{space_converter_str}_{hyper_parameters_underscore_str}_{Delta_M}"
        if remove_peaks_costs:
            suffix = ""
        else:
            suffix = prefix + f"/Delta_P={Delta_P}/Delta_P_prime={Delta_P_prime}/"
            #group_wandb += f"_{Delta_P}_{Delta_P_prime}"
            config_wandb = {
                **config_wandb,
                **{"Delta_P":Delta_P, "Delta_P_prime":Delta_P_prime}
            }
        if sha_folders:
            suffix = sha256(suffix.encode('utf-8')).hexdigest()+"/"
        path = folder + suffix
        pathfile_random_state = path+'random_states.rs'
        pathfile = path+'result$i$.json'
        pathlastfile = path+f'result{n_iters-1}.json'
        pathlock = path+'result.lock'
        pathdone = path+"done.lock"
        path_best_policy = path+"best_policy_checkpoint/"
        full_path_checkpoint = None
        
        if not (stdout or erase_file or (not os.path.isfile(pathdone) and not os.path.isfile(pathlock))):
            print("Locked or already computed, exit")
        else:
            if n_cpus > 0:
                num_cpus_parallel = n_cpus
            else:
                num_cpus_parallel = os.cpu_count()
            num_cpus = 0
            #ray.init(ignore_reinit_error=True, num_cpus=0, num_gpus=0, _temp_dir=f"{tmp_dir}/tmp", include_dashboard=False)
            #ray.init(include_dashboard=False)
            if not stdout:
                os.makedirs(path, exist_ok=True)
                with open(pathlock, 'w') as _: 
                    pass
            gamma_global = gamma
            GAMMA_SEQUENCE = rec_gamma_sequence(gamma, Delta_M=rec_env_eval.Delta_M, Delta_P=rec_env_eval.Delta_P, T=(T))
            if gc == 0:
                gc = None
            #global_value_stats_queue = global_value_stats_queue(maxsize=number_of_episodes_eval*T)
            #global_discounted_expected_returns_queue = global_discounted_expected_returns_queue(maxsize=number_of_episodes_eval*T)
            
            num_rollout_workers = (0 if num_cpus_parallel<=1 else num_cpus_parallel)
            num_rollout_workers_eval = (0 if num_cpus_parallel<=1 else num_cpus_parallel)
            num_rollout_workers_eval = min(num_rollout_workers_eval, number_of_episodes_eval)
            num_rollout_workers = min(num_rollout_workers, number_of_episodes)
            num_envs_per_worker = number_of_episodes//(max(num_rollout_workers, 1))
            num_envs_per_worker_eval = number_of_episodes_eval//(max(num_rollout_workers_eval, 1))
            
            explore_eval = number_of_episodes_eval>1
            evaluation_config = {"explore": explore_eval, "env": env_eval_id, "env_config": config_wandb, "num_envs_per_worker": num_envs_per_worker_eval, "create_env_on_local_worker": True}
            if action_dist != "default":
                model_config_dict["custom_action_dist"] = action_dist
            framework_kwargs = {"framework": "torch"} if framework == "torch" else {"framework": "tf2", "eager_tracing": True}
            if framework == "torch":
                pass
                framework_kwargs["torch_compile_learner"] = True
                framework_kwargs["torch_compile_learner_dynamo_backend"] = "ipex"
                framework_kwargs["torch_compile_learner_dynamo_mode"] = "max-autotune"
                #framework_kwargs["torch_compile_worker"] = True
                #framework_kwargs["torch_compile_worker_dynamo_backend"] = "onnxrt"
                #framework_kwargs["torch_compile_worker_dynamo_mode"] = "default"
            lr_args = learning_rate.split("#")
            lr_schedule = None
            if len(lr_args) == 3:
                lr_start = float(lr_args[0])
                lr_end = float(lr_args[1])
                lr_steps = int(lr_args[2]) * T * number_of_episodes
                #lr_schedule = list(np.linspace(lr_start, lr_end, lr_steps))
                #lr_iters = n_iters * T * number_of_episodes
                #lr_schedule += [lr_end] * max(lr_iters - lr_steps + 1, 0)
                #lr_schedule = list(enumerate(lr_schedule))
                #print(lr_steps)
                #exit()
                lr_schedule = [
                    [0, lr_start],
                    [lr_steps, lr_end]
                ]
            kwargs_ppo = (({
                "lr": float(learning_rate)
                }) if lr_schedule is None else {"lr": lr_schedule})
            #print(kwargs_ppo)
            #exit()
            from ray.tune.logger import NoopLogger
            def create_noop_logger(_):
                return NoopLogger(None, root_dir)
            if num_cpus > 1:
                from ray.runtime_env import RuntimeEnv
                runtime_env={
                    "OPENBLAS_NUM_THREADS":"1",
                    "VECLIB_MAXIMUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1"
                }
                runtime_env = RuntimeEnv(
                    env_vars=runtime_env
                )
                ray.init()
                #ray.init(local_mode=True)
                #ray.init(num_cpus=num_cpus)
                #ray.init(ignore_reinit_error=True, num_gpus=0, num_cpus=(num_cpus if num_cpus > 1 else 0), log_to_driver=debug, configure_logging=debug, include_dashboard=debug, _temp_dir=f"{tmp_dir}/tmp", runtime_env=runtime_env)
                #ray.init(ignore_reinit_error=True, num_gpus=0, num_cpus=num_cpus, log_to_driver=debug, configure_logging=debug, include_dashboard=debug, _temp_dir=f"{tmp_dir}/tmp", runtime_env=runtime_env)
                #ray.init(ignore_reinit_error=True, num_cpus=num_cpus, num_gpus=0, _temp_dir=f"{tmp_dir}/tmp", include_dashboard=False, configure_logging=False, log_to_driver=False)
            
            #num_rollout_workers = (0 if num_cpus_parallel<=1 else num_cpus_parallel)
            evaluation_duration_unit = "episodes" if number_of_episodes_eval <= 1 else "timesteps"
            evaluation_duration = (
                ((T*number_of_episodes_eval)) // (num_envs_per_worker_eval)
                if number_of_episodes_eval > 1 else number_of_episodes_eval
            )
            config = (
                PPOConfig()
                .environment(env_train_id, env_config={**config_wandb, **{"seed": random_seed}}, disable_env_checking=True)
                .debugging(
                    logger_creator = create_noop_logger
                )
                .framework(**framework_kwargs)
                .rollouts(create_env_on_local_worker=True, num_rollout_workers=0, num_envs_per_worker=num_envs_per_worker, observation_filter=("MeanStdFilter" if mean_std_filter_mode in ("only_obs", "obs_and_rew") else "NoFilter"))
                .evaluation(evaluation_sample_timeout_s=1800, evaluation_num_workers=0, evaluation_duration_unit=evaluation_duration_unit, evaluation_duration=evaluation_duration, evaluation_config=evaluation_config, evaluation_interval=evaluation_interval)
                .training(
                    train_batch_size=(T)*number_of_episodes,
                    sgd_minibatch_size=batch_size,
                    num_sgd_iter=n_sgds,
                    grad_clip=gc,
                    model=model_config_dict,
                    gamma=float(gamma_policy_start if gamma_policy_start is not None else gamma_policy),
                    lambda_=lambda_gae,
                    vf_loss_coeff=vf_coeff,
                    kl_coeff=kl_coeff,
                    kl_target=kl_target,
                    entropy_coeff=entropy_coeff,
                    clip_param=clip_param_start,
                    vf_clip_param=vf_clip_param,
                    **kwargs_ppo
                )
                
                
            )
                
            if False and (num_rollout_workers > 1 or num_rollout_workers_eval>1):
                def custom_parallel_vector_env(env):
                    
                    

                    # Convert gym.Env to VectorEnv ...
                    num_envs = num_envs_per_worker if not env.eval_env else num_envs_per_worker_eval
                    env = _ParallelVectorizedGymEnv(
                        env_train_id,
                        env_eval_id,
                        num_cpus_parallel,
                        existing_envs=[env],
                        num_envs=num_envs
                    )
                    # ... then the resulting VectorEnv to a BaseEnv.
                    env = VectorEnvWrapper(env)
                    return env
                config["custom_vector_env"] = custom_parallel_vector_env
            
            rl_module_spec = config.get_default_rl_module_spec()
            from ray.rllib.utils.nested_dict import NestedDict
            from typing import Mapping
            if not just_output_model and not explore_eval:
                class FixedPPOTorchRLModule(PPOTorchRLModule):
                    def _forward_inference(self, batch: NestedDict) -> Mapping[str, Any]:
                        return self._forward_exploration(batch)
                rl_module_spec.module_class = FixedPPOTorchRLModule
            from ray.rllib.core.models.configs import (
                ActorCriticEncoderConfig,
                MLPHeadConfig,
                FreeLogStdMLPHeadConfig,
            )
            from ray.rllib.models import MODEL_DEFAULTS
            from ray.rllib.algorithms.ppo.ppo_catalog import _check_if_diag_gaussian
            from ray.rllib.models.utils import get_filter_config
            from ray.rllib.core.models.configs import (
                CNNEncoderConfig,
                MLPEncoderConfig,
                RecurrentEncoderConfig,
            )
            from ray.rllib.utils import override
            from ray.rllib.core.models.catalog import Catalog
            
            class CustomPPOCatalog(PPOCatalog):
                def get_action_dist_cls(self, framework: str) -> Model:
                    if action_dist != "default":
                        return action_distribution_zoo[action_dist]
                    else:
                        return super().get_action_dist_cls(framework)
                
                def __init__(
                    self,
                    observation_space,
                    action_space,
                    model_config_dict,
                ):
                    """Initializes the PPOCatalog.

                    Args:
                        observation_space: The observation space of the Encoder.
                        action_space: The action space for the Pi Head.
                        model_config_dict: The model config to use.
                    """
                    super().__init__(
                        observation_space=observation_space,
                        action_space=action_space,
                        model_config_dict=model_config_dict,
                    )
                    if is_custom_model:
                        # Replace EncoderConfig by ActorCriticEncoderConfig
                        self.actor_critic_encoder_config = ActorCriticEncoderConfig(
                            base_encoder_config=self._encoder_config,
                            shared=self._model_config_dict["vf_share_layers"],
                        )
                        post_fcnet_hiddens = (
                            self._model_config_dict["post_fcnet_hiddens"]["pi"]
                            if type(self._model_config_dict["post_fcnet_hiddens"]) == dict
                            else self._model_config_dict["post_fcnet_hiddens"]
                        )
                        vf_post_fcnet_hiddens = (
                            self._model_config_dict["post_fcnet_hiddens"].get("vf", post_fcnet_hiddens)
                            if type(self._model_config_dict["post_fcnet_hiddens"]) == dict
                            else self._model_config_dict["post_fcnet_hiddens"]
                        )
                        post_fcnet_activation = (
                            self._model_config_dict["post_fcnet_activation"]["pi"]
                            if type(self._model_config_dict["post_fcnet_activation"]) == dict
                            else self._model_config_dict["post_fcnet_activation"]
                        )
                        vf_post_fcnet_activation = (
                            self._model_config_dict["post_fcnet_activation"].get("vf", post_fcnet_activation)
                            if type(self._model_config_dict["post_fcnet_activation"]) == dict
                            else self._model_config_dict["post_fcnet_activation"]
                        )
                        self.pi_and_vf_head_hiddens = post_fcnet_hiddens
                        self.pi_and_vf_head_activation = post_fcnet_activation
                        self.vf_head_hiddens = vf_post_fcnet_hiddens
                        self.vf_head_activation = vf_post_fcnet_activation
                        # We don't have the exact (framework specific) action dist class yet and thus
                        # cannot determine the exact number of output nodes (action space) required.
                        # -> Build pi config only in the `self.build_pi_head` method.
                        self.pi_head_config = None
                        self.pi_layer_norm = (
                            self._model_config_dict["post_fcnet_hiddens"].get("pi_layer_norm", False)
                            if type(self._model_config_dict["post_fcnet_hiddens"]) == dict
                            else False
                        )
                        self.vf_layer_norm = (
                            self._model_config_dict["post_fcnet_hiddens"].get("vf_layer_norm", self.pi_layer_norm)
                            if type(self._model_config_dict["post_fcnet_hiddens"]) == dict
                            else self.pi_layer_norm
                        )
                        self.vf_head_config = MLPHeadConfig(
                            input_dims=self.latent_dims,
                            hidden_layer_dims=self.vf_head_hiddens,
                            hidden_layer_activation=self.vf_head_activation,
                            output_layer_activation="linear",
                            output_layer_dim=1,
                            hidden_layer_use_layernorm=self.vf_layer_norm
                        )
                
                def build_pi_head(self, framework: str) -> Model:
                    """Builds the policy head.

                    The default behavior is to build the head from the pi_head_config.
                    This can be overridden to build a custom policy head as a means of configuring
                    the behavior of a PPORLModule implementation.

                    Args:
                        framework: The framework to use. Either "torch" or "tf2".

                    Returns:
                        The policy head.
                    """
                    # Get action_distribution_cls to find out about the output dimension for pi_head
                    if is_custom_model:
                        action_distribution_cls = self.get_action_dist_cls(framework=framework)
                        if self._model_config_dict["free_log_std"]:
                            _check_if_diag_gaussian(
                                action_distribution_cls=action_distribution_cls, framework=framework
                            )
                        required_output_dim = action_distribution_cls.required_input_dim(
                            space=self.action_space, model_config=self._model_config_dict
                        )
                        # Now that we have the action dist class and number of outputs, we can define
                        # our pi-config and build the pi head.
                        pi_head_config_class = (
                            FreeLogStdMLPHeadConfig
                            if self._model_config_dict["free_log_std"]
                            else MLPHeadConfig
                        )
                        self.pi_head_config = pi_head_config_class(
                            input_dims=self.latent_dims,
                            hidden_layer_dims=self.pi_and_vf_head_hiddens,
                            hidden_layer_activation=self.pi_and_vf_head_activation,
                            output_layer_dim=required_output_dim,
                            output_layer_activation="linear",
                            hidden_layer_use_layernorm=self.pi_layer_norm
                        )

                        pi_head = self.pi_head_config.build(framework=framework)
                        return pi_head
                    else:
                        return super().build_pi_head(framework)

                @classmethod
                def get_tokenizer_config(
                    cls,
                    observation_space,
                    model_config_dict,
                    view_requirements = None,
                ):
                    """Returns a tokenizer config for the given space.

                    This is useful for recurrent / tranformer models that need to tokenize their
                    inputs. By default, RLlib uses the models supported by Catalog out of the box to
                    tokenize.

                    You should override this method if you want to change the custom tokenizer
                    inside current encoders that Catalog returns without providing the recurrent
                    network as a whole. For example, if you want to define some custom CNN layers
                    as a tokenizer for a recurrent encoder that already includes the recurrent
                    layers and handles the state.

                    Args:
                        observation_space: The observation space to use.
                        model_config_dict: The model config to use.
                        view_requirements: The view requirements to use if anything else than
                            observation_space is to be encoded. This signifies an advanced use case.
                    """
                    if is_custom_model:
                        return cls._get_encoder_config(
                            observation_space=observation_space,
                            # Use model_config_dict without flags that would end up in complex models
                            model_config_dict={
                                **model_config_dict,
                                **{"use_lstm": False, "use_attention": False},
                            },
                            view_requirements=view_requirements
                        )
                    else:
                        return PPOCatalog.get_tokenizer_config(cls, observation_space, model_config_dict, view_requirements=view_requirements)
                    
                def build_actor_critic_encoder(self, framework: str):
                    """Builds the ActorCriticEncoder.

                    The default behavior is to build the encoder from the encoder_config.
                    This can be overridden to build a custom ActorCriticEncoder as a means of
                    configuring the behavior of a PPORLModule implementation.

                    Args:
                        framework: The framework to use. Either "torch" or "tf2".

                    Returns:
                        The ActorCriticEncoder.
                    """
                    actor_critic_encoder = self.actor_critic_encoder_config.build(framework=framework)
                    if is_custom_model and trainable_initial_state_type != "zeros":
                        from .rnn.trainable_initial_state_wrapper import TrainableInitialStateWrapper
                        actor_critic_encoder = TrainableInitialStateWrapper(actor_critic_encoder, type_initial_state=trainable_initial_state_type)

                    return actor_critic_encoder

                @classmethod
                def _get_encoder_config(
                    cls,
                    observation_space,
                    model_config_dict,
                    action_space=None,
                    view_requirements=None,
                ):
                    """Returns an EncoderConfig for the given input_space and model_config_dict.

                    Encoders are usually used in RLModules to transform the input space into a
                    latent space that is then fed to the heads. The returned EncoderConfig
                    objects correspond to the built-in Encoder classes in RLlib.
                    For example, for a simple 1D-Box input_space, RLlib offers an
                    MLPEncoder, hence this method returns the MLPEncoderConfig. You can overwrite
                    this method to produce specific EncoderConfigs for your custom Models.

                    The following input spaces lead to the following configs:
                    - 1D-Box: MLPEncoderConfig
                    - 3D-Box: CNNEncoderConfig
                    # TODO (Artur): Support more spaces here
                    # ...

                    Args:
                        observation_space: The observation space to use.
                        model_config_dict: The model config to use.
                        action_space: The action space to use if actions are to be encoded. This
                            is commonly the case for LSTM models.
                        view_requirements: The view requirements to use if anything else than
                            observation_space or action_space is to be encoded. This signifies an
                            advanced use case.

                    Returns:
                        The encoder config.
                    """
                    if is_custom_model:
                        # TODO (Artur): Make it so that we don't work with complete MODEL_DEFAULTS
                        model_config_dict = {**MODEL_DEFAULTS, **model_config_dict}

                        activation = model_config_dict["fcnet_activation"]
                        output_activation = model_config_dict["fcnet_activation"]
                        fcnet_hiddens = model_config_dict["fcnet_hiddens"]
                        # TODO (sven): Move to a new ModelConfig object (dataclass) asap, instead of
                        #  "linking" into the old ModelConfig (dict)! This just causes confusion as to
                        #  which old keys now mean what for the new RLModules-based default models.
                        encoder_latent_dim = (
                            model_config_dict["encoder_latent_dim"] or (fcnet_hiddens[-1] if len(fcnet_hiddens) > 0 else None)
                        )
                        use_lstm = model_config_dict["use_lstm"]
                        use_attention = model_config_dict["use_attention"]

                        if use_lstm or type(use_lstm) == dict:
                            if type(use_lstm) == dict:
                                recurrent_layer_type = use_lstm.get("rnn_type", "lstm")
                                num_layers = use_lstm.get("num_layers", 1)
                            else:
                                num_layers=1
                                recurrent_layer_type="lstm"
                            encoder_config = RecurrentEncoderConfig(
                                input_dims=observation_space.shape,
                                recurrent_layer_type=recurrent_layer_type,
                                hidden_dim=model_config_dict["lstm_cell_size"],
                                batch_major=not model_config_dict["_time_major"],
                                num_layers=num_layers,
                                tokenizer_config=(cls.get_tokenizer_config(
                                    observation_space,
                                    model_config_dict,
                                    view_requirements,
                                ) if encoder_latent_dim is not None else None),
                            )
                        elif use_attention:
                            raise NotImplementedError
                        else:
                            # TODO (Artur): Maybe check for original spaces here
                            # input_space is a 1D Box
                            if isinstance(observation_space, Box) and len(observation_space.shape) == 1:
                                # In order to guarantee backward compatability with old configs,
                                # we need to check if no latent dim was set and simply reuse the last
                                # fcnet hidden dim for that purpose.
                                if model_config_dict["encoder_latent_dim"]:
                                    hidden_layer_dims = model_config_dict["fcnet_hiddens"]
                                else:
                                    hidden_layer_dims = model_config_dict["fcnet_hiddens"][:-1]
                                encoder_config = MLPEncoderConfig(
                                    input_dims=observation_space.shape,
                                    hidden_layer_dims=hidden_layer_dims,
                                    hidden_layer_activation=activation,
                                    output_layer_dim=encoder_latent_dim,
                                    output_layer_activation=output_activation,
                                )

                            # input_space is a 3D Box
                            elif (
                                isinstance(observation_space, Box) and len(observation_space.shape) == 3
                            ):
                                if not model_config_dict.get("conv_filters"):
                                    model_config_dict["conv_filters"] = get_filter_config(
                                        observation_space.shape
                                    )

                                encoder_config = CNNEncoderConfig(
                                    input_dims=observation_space.shape,
                                    cnn_filter_specifiers=model_config_dict["conv_filters"],
                                    cnn_activation=model_config_dict["conv_activation"],
                                    cnn_use_layernorm=model_config_dict.get(
                                        "conv_use_layernorm", False
                                    ),
                                )
                            # input_space is a 2D Box
                            elif (
                                isinstance(observation_space, Box) and len(observation_space.shape) == 2
                            ):
                                # RLlib used to support 2D Box spaces by silently flattening them
                                raise ValueError(
                                    f"No default encoder config for obs space={observation_space},"
                                    f" lstm={use_lstm} and attention={use_attention} found. 2D Box "
                                    f"spaces are not supported. They should be either flattened to a "
                                    f"1D Box space or enhanced to be a 3D box space."
                                )
                            # input_space is a possibly nested structure of spaces.
                            else:
                                # NestedModelConfig
                                raise ValueError(
                                    f"No default encoder config for obs space={observation_space},"
                                    f" lstm={use_lstm} and attention={use_attention} found."
                                )

                        return encoder_config
                    else:
                        return PPOCatalog._get_encoder_config(observation_space, model_config_dict, action_space=action_space, view_requirements=view_requirements)
            rl_module_spec.catalog_class = CustomPPOCatalog
            config = config.rl_module(_enable_rl_module_api=True, rl_module_spec=rl_module_spec)
            config["simple_optimizer"] = True
            metrics_collector_class = (
                SequentialMetricsCollector if num_rollout_workers_eval <= 1 else MetricsCollector
            )
            lst_callbacks = [metrics_collector_class]
            if num_rollout_workers_eval > 1:
                global_shared_memories["discounted_sums"] = manager.list()#create_shared_memories(number_of_episodes_eval, 20)
                global_shared_memories["undiscounted_sums"] = manager.list()#create_shared_memories(number_of_episodes_eval, 20)
                global_shared_memories["value_targets"] = manager.list()#create_shared_memories(number_of_episodes_eval*(T), 20)
            #lst_callbacks += [ClipAdvantageByPercentile]
            #global_shared_memories["discounted_returns"] = create_shared_memories(num_rollout_workers_eval, number_of_episodes_eval)
            #global_shared_memories["value_targets"] = dict()
            #create_shared_memories(num_rollout_workers_eval, number_of_episodes_eval)
            if num_rollout_workers > 1 or num_rollout_workers_eval>1:
                    
                class CustomParallelWorking(DefaultCallbacks):

                    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[PolicyID, Policy], episode: Episode | EpisodeV2, env_index: int | None = None, **kwargs) -> None:
                        episode.worker_idx = worker.worker_index
                        episode.is_eval = worker.env.eval_env

                    def on_postprocess_trajectory(self, *, worker: RolloutWorker, episode: Episode, agent_id: AgentID, policy_id: PolicyID, policies: Dict[PolicyID, Policy], postprocessed_batch: SampleBatch, original_batches: Dict[AgentID, Tuple[Policy, SampleBatch]], **kwargs) -> None:
                        postprocessed_batch["unroll_id"] += episode.worker_idx
    
                    def on_algorithm_init(
                        self,
                        *,
                        algorithm: "Algorithm",
                        **kwargs,
                    ) -> None:
                        if num_rollout_workers > 1:
                            algorithm.workers = ParallelWorkerSetWrapper(
                                algorithm.workers,
                                number_of_processes=num_rollout_workers,
                                num_env_per_worker=num_envs_per_worker,
                                id_worker_set="train"
                            )
                        if num_rollout_workers_eval > 1:
                            algorithm.evaluation_workers = ParallelWorkerSetWrapper(
                                algorithm.evaluation_workers,
                                number_of_processes=num_rollout_workers_eval,
                                num_env_per_worker=num_envs_per_worker_eval,
                                id_worker_set="eval"
                            )
                lst_callbacks = [CustomParallelWorking] + lst_callbacks
            if just_output_model:
                lst_callbacks = [JustOutputModel] + lst_callbacks
            if lst_callbacks != []:
                multi_parametrized_callbacks = make_multi_callbacks(lst_callbacks)
                config = config.callbacks(multi_parametrized_callbacks)
            algo = config.build()  # 2. build the algorithm,
            policy = algo.get_policy()
            offset_iter = 0
            if use_wandb and not wandb_offline:
                max_retries = 7
                seconds_to_wait = 2
                wandb_connected=False
                exception=None
                run = None
                while not wandb_connected and max_retries > 0: 
                    try:
                        run = wandb.init(config=config_wandb, project=wandb_project, entity="samait", mode=("offline" if wandb_offline else "online"))
                        wandb_connected=True
                    except BaseException as _:
                        time.sleep(seconds_to_wait)
                        seconds_to_wait*=2
                        max_retries -= 1
                if not wandb_connected:
                    warnings.warn("Wandb cannot be online, Trying to switch to offline mode")
                    try:
                        run = wandb.init(config=config_wandb, project=wandb_project, entity="samait", mode="offline")
                        warnings.warn("Successfully switching in offline mode.")
                    except BaseException as e:
                        exception = e
                if exception is not None:
                    raise BaseException("Something went wrong with wandb, even in offline mode. Details: " + str(exception))
                if run is not None:
                    with open(path+"wandb_data.json", "w+") as wandb_data_file:
                        json.dump(
                            {
                                "run_id": run.id,
                                "run_name": run.name,
                                "config_wandb": config_wandb,
                                "project": wandb_project,
                                "entity": "samait"
                            },
                            wandb_data_file
                        )
                
                
            stats = []
            if time_iter:
                print("Start training...", flush=True)
            best_validate_expected_effective_bill = np.inf
            best_validate_undiscounted_expected_effective_bill = np.inf
            validate_expected_effective_bill = 0
            validate_undiscounted_expected_effective_bill = 0
            
            
            
            if env_eval != env_valid:
                evaluation_workers = WorkerSet(
                    env_creator=env_eval_creator,
                    validate_env=None,
                    default_policy_class=algo.get_default_policy_class(algo.config),
                    config=algo.evaluation_config,
                    num_workers=0,
                    logdir=algo.logdir,
                )
                validation_workers = WorkerSet(
                    env_creator=env_valid_creator,
                    validate_env=None,
                    default_policy_class=algo.get_default_policy_class(algo.config),
                    config=algo.evaluation_config,
                    num_workers=0,
                    logdir=algo.logdir,
                )
                algo.evaluation_workers = evaluation_workers
            if action_weights_divider > 1.0:
                policy = algo.get_policy()
                init_weights = deepcopy(policy.get_weights())
                pi_shape = algo.get_policy().model.pi.get_output_specs().shape[1]
                for k, w in policy.get_weights().items():
                    if "pi" in k and w.shape[0] == pi_shape:
                        init_weights[k][:w.shape[0]//2] /= action_weights_divider
                policy.set_weights(init_weights)

            if sha_folders:
                with open(path+"parameters.json", 'w') as fp:
                    json.dump(config_wandb, fp)
            #print(algo.get_policy().get_weights().keys())
            for i in range(n_iters):
                #print(algo.get_policy().get_weights().keys())
                #print()
                #from torchviz import make_dot
                if time_iter:
                    print(f"Training at iteration {i}...", flush=True)
                    t = time.time()
                    global_t = t
                
                train_stats = algo.train()  # 3. train it,
                #algo.get_policy().config["model"]["max_seq_len"] = min(algo.get_policy().config["model"]["max_seq_len"]+1, batch_size)
                #from pprint import pprint
                #print(len(train_stats["sampler_results"]["hist_stats"]["episode_reward"]))
                #print(len(set(train_stats["sampler_results"]["hist_stats"]["episode_reward"])))
                #pprint(train_stats)
                #exit()
                policy_loss = train_stats["info"]["learner"]["default_policy"]["policy_loss"]
                critic_loss = train_stats["info"]["learner"]["default_policy"]["vf_loss"]
                grad_gnorm = train_stats["info"]["learner"]["default_policy"]["gradients_default_optimizer_global_norm"]
                cur_lr = train_stats["info"]["learner"]["default_policy"]["curr_lr"]
                cur_kl_coeff = train_stats["info"]["learner"]["default_policy"]["curr_kl_coeff"]
                kl_loss = train_stats["info"]["learner"]["default_policy"]["mean_kl_loss"]
                entropy = train_stats["info"]["learner"]["default_policy"]["entropy"]
                entropy_coeff = train_stats["info"]["learner"]["default_policy"]["curr_entropy_coeff"]
                explained_critic_variance = train_stats["info"]["learner"]["default_policy"]["vf_explained_var"]
                if time_iter:
                    print("Training time (in seconds):", time.time() - t, flush=True)
                    t = time.time()
                    print("Evaluate...", flush=True)
                

                """
                random_states = get_random_states()
                if env_eval != env_valid:
                    algo.evaluation_workers = evaluation_workers
                algo:PPO
                evaluation_complete = algo.evaluate()
                if env_eval != env_valid:
                    restore_random_states(random_states)
                """
                if "evaluation" in train_stats:
                    evaluation = train_stats["evaluation"]
                    #pprint(evaluation["custom_metrics"])
                    std_episode = evaluation["custom_metrics"].get(
                        "discounted_rewards_var",
                        evaluation["custom_metrics"].get("discounted_rewards_var_mean", 0.0)
                    )
                    
                    #print(len(evaluation["sampler_results"]["hist_stats"]["episode_reward"]))
                    #print(len(set(evaluation["sampler_results"]["hist_stats"]["episode_reward"])))
                    expected_return = -evaluation["custom_metrics"]["discounted_rewards_mean"]
                    

                    undiscounted_expected_return = -evaluation["episode_reward_mean"]
                    if np.isnan(undiscounted_expected_return):
                        undiscounted_expected_return = -evaluation["custom_metrics"]["undiscounted_rewards_mean"]
                    if time_iter:
                        print("Evaluate time (in seconds)", time.time() - t, flush=True)

                    if time_iter:
                        t = time.time()
                        print("Validate...", flush=True)
                    
                    
                    validation = evaluation
                    
                    
                    validate_expected_effective_bill = -validation["custom_metrics"]["discounted_rewards_mean"]
                    validate_undiscounted_expected_effective_bill = -validation["episode_reward_mean"]
                    if np.isnan(validate_undiscounted_expected_effective_bill):
                        validate_undiscounted_expected_effective_bill = -validation["custom_metrics"]["undiscounted_rewards_mean"]
                    
                    if time_iter:
                        print("Validate time (in seconds)", time.time() - t, flush=True)
                    
                    is_best_validate_policy = best_validate_expected_effective_bill < validate_expected_effective_bill
                    best_validate_expected_effective_bill = min(best_validate_expected_effective_bill, validate_expected_effective_bill)
                    best_validate_undiscounted_expected_effective_bill = min(best_validate_undiscounted_expected_effective_bill, validate_undiscounted_expected_effective_bill)
                    critic_average = evaluation["custom_metrics"].get("value_targets_mean", 0.0)
                    critic_var = evaluation["custom_metrics"].get("value_targets_var", 0.0)
                    critic_max = evaluation["custom_metrics"].get("value_targets_max", 0.0)
                    critic_min = evaluation["custom_metrics"].get("value_targets_min", 0.0)
                    cur_lr = train_stats["info"]["learner"]["default_policy"]["curr_lr"]
                    stat = {
                        "Best Expected Effective Bill": float(best_validate_expected_effective_bill),
                        "Expected Effective Bill": float(expected_return),
                        "Expected Effective Bill Std": float(std_episode),
                        "Policy Loss": float(policy_loss),
                        "Critic Loss": float(critic_loss),
                        "Clipped Gradient": float(grad_gnorm),
                        "Explained Critic Variance": float(explained_critic_variance),
                        "Critic Average": float(critic_average),
                        "Critic Variance": float(critic_var),
                        "Critic Max": float(critic_max),
                        "Critic Min": float(critic_min),
                        "KL coeff": float(cur_kl_coeff),
                        "KL loss": float(kl_loss),
                        "Entropy": float(entropy),
                        "Current learning rate": float(cur_lr)
                        
                    }
                    if gamma_policy_steps is not None and gamma_policy_steps > 0:
                        def change_pol_gamma(pol):
                            pol.config["gamma"] = lst_gammas[i]
                        algo.workers.foreach_policy(
                            lambda pol, pol_id: change_pol_gamma(pol)
                        )
                    if clip_param_steps is not None and clip_param_steps > 0:
                        def change_pol_clip_param(pol):
                            pol.config["clip_param"] = lst_clip_param[i]
                        algo.workers.foreach_policy(
                            lambda pol, pol_id: change_pol_clip_param(pol)
                        )
                    if not stdout:
                        if not wandb_offline:
                            f_result_pathname = pathfile.replace("$i$", f"_{i}")
                            with open(f_result_pathname, 'w') as fp:
                                json.dump(stat, fp)
                            if tar_gz_results:
                                with zipfile.ZipFile(path+"results.zip","a") as tar_results:
                                    tar_results.write(f_result_pathname, os.path.basename(f_result_pathname))
                                    os.remove(f_result_pathname)
                        """
                        if full_path_checkpoint is not None:
                            shutil.rmtree(full_path_checkpoint, ignore_errors=True)
                        full_path_checkpoint = algo.save(checkpoint_dir=path, prevent_upload=True)
                        with open(pathfile_random_state, "wb") as random_state_file:
                            pickle.dump(random_states, random_state_file)
                        if is_best_validate_policy:
                            shutil.copytree(
                                full_path_checkpoint+"/policies",
                                path_best_policy,
                                dirs_exist_ok=True
                            )
                        """
                        pass
                    if use_wandb:
                        """
                        while True:
                            runs = wandb.Api().runs(path='samait/mpcstudy')
                            runs = [run for run in runs if run.id == id_wandb]
                            if len(runs) == 0:
                                break
                            for run in runs:
                                print(dir(run))
                                run.wait_until_finished()
                        """
                        
                        #wandb.mark_preempting()
                        #run.wait_until_finished()
                        
                        
                        #wandb.log({f"Expected Effective Bill": expected_return})
                        #wandb.log({f"Policy loss": policy_loss})
                        #wandb.log({f"Critic loss": critic_loss})
                        #wandb.log({f"Clipped gradient": grad_gnorm})
                        if not wandb_offline:
                            try:
                                if time_iter:
                                    print("Log to wandb...", flush=True)
                                    t = time.time()
                                wandb.log(stat)
                                if time_iter:
                                    print("Log to wandb done in", time.time() - t, "seconds", flush=True)
                            except BaseException as e:
                                warnings.warn(f"Log went wrong with commit, trying without commit... (Details: {e})")
                                wandb.log(stat, commit=False)
                        else:
                            pathwandbfile = path+f'post_to_wandb_{i}.json'
                            with open(pathwandbfile, 'w') as fp:
                                json.dump({
                                    "config":config_wandb,
                                    "wandb_project": wandb_project,
                                    "entity": "samait",
                                    "results": stat

                                }, fp)
                            if tar_gz_results:
                                with zipfile.ZipFile(path+"post_to_wandb.zip","a") as tar_results:
                                    tar_results.write(pathwandbfile, os.path.basename(pathwandbfile))
                                os.remove(pathwandbfile)

                    if stdout:
                        print("Stats at iteration", i, flush=True)
                        pprint(stat)
                if time_iter:
                    print("Global iteration time:", time.time() - global_t, "seconds")
            if use_wandb:
                wandb.finish()
            if not stdout:
                with open(pathdone, 'w') as _: 
                    pass
                os.remove(pathlock)
                

if __name__ == '__main__':
    run_experiment()