import os
os.environ["OPENBLAS_NUM_THREADS"] = str(1) # export OPENBLAS_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = str(1) # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = str(1) # export NUMEXPR_NUM_THREADS=1
os.environ["OMP_NUM_THREADS"] = str(1)
os.environ["MKL_NUM_THREADS"] = str(1)
from hashlib import sha256

import warnings
import zipfile
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dreamer import DreamerConfig
from ray.rllib.utils.filter import MeanStdFilter
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.utils.typing import AgentID, EnvType, PolicyID
from experiment_scripts.rl.action_distributions import action_distribution_zoo
import ray
import pickle
import shutil
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode, RolloutWorker
from gymnasium.spaces import Box
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from typing import Any, Dict, Tuple
from .rec_env_rl_libs_gymnasium_gym_wrapper import RecEnvRlLibsGymnasiumGymWrapper
from .rec_env_rl_libs_wrapper import RecEnvRlLibsWrapper
from typing import Union, Optional
from env.rec_env import RecEnv
from env.rec_env_global_bill_wrapper import RecEnvGlobalBillWrapper
from .env_wrappers_zoo import env_wrapper_sequences, wrap
import glob
from exogenous_providers.perfect_foresight_exogenous_provider import PerfectForesightExogenousProvider
from .rl_envs_zoo import create_rl_env_creators, rl_envs
from ray.rllib.algorithms.callbacks import make_multi_callbacks
from .models_zoo import models_zoo, dreamer_models_zoo
from .space_converters_zoo import space_converter_sequences
from envs import create_env
from env.counter_utils import future_counters
import os
import json
import click
import wandb
import time
import torch as tc
import numpy as np
import random
from pprint import pprint
from envs import create_env_fcts
from policies.model_predictive_control_policy_cplex import ModelPredictiveControlPolicyCplex
from click_option_group import MutuallyExclusiveOptionGroup, optgroup, RequiredMutuallyExclusiveOptionGroup
from ray.rllib.algorithms.dreamerv3.dreamerv3 import DreamerV3Config
from .compute_optimal_z_score import compute_optimal_z_score

os.environ["WANDB_SILENT"] = "true"

value_target_values = []
gamma_global = None
obs_z_score_global = None
rew_z_score_global = None
NUM_OPTIM_ROLLOUTS = None

#algo.gamma=lst_gammas[i]


def get_random_states():
    return random.getstate(), np.random.get_state(), tc.random.get_rng_state()

def restore_random_states(random_states):
    random_state, np_random_state, tc_random_state = random_states
    random.setstate(random_state)
    np.random.set_state(np_random_state)
    tc.random.set_rng_state(tc_random_state)

class OptiNormObsRewardsAbstractCallback(DefaultCallbacks):
    def __init__(self, *args, legacy_callbacks_dict: Dict[str, callable] = None, do_multi_reset=False, **kwargs):
        super().__init__(legacy_callbacks_dict=legacy_callbacks_dict)
        self._do_multi_reset = do_multi_reset
        if do_multi_reset:
            self._number_of_resets = 1
        else:
            self._number_of_resets = NUM_OPTIM_ROLLOUTS
        self._init=False

    def _norm_rew(self):
        raise NotImplementedError()
    
    def _norm_obs(self):
        raise NotImplementedError()
    
    def on_episode_created(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        env_index: int,
        episode: Union[Episode, EpisodeV2],
        **kwargs,
    ) -> None:
        global obs_z_score_global, rew_z_score_global
        sub_envs = None
        if self._norm_obs():
            if obs_z_score_global is None and not worker.env.eval_env:
                self._set_global_z_score(worker.env)
            if obs_z_score_global is not None:
                #print(id(worker.env))
                z_score_mean_obs, z_score_std_obs = obs_z_score_global
                #print(worker.env.eval_env)
                #worker.foreach_env(
                #    lambda env: set_obs_z_score_mean(env)
                #)
                #worker.foreach_env(
                #    lambda env: set_obs_z_score_std(env)
                #)
                sub_envs = base_env.get_sub_environments()
                worker.env.obs_z_score_mean = z_score_mean_obs
                worker.env.obs_z_score_std = z_score_std_obs
                worker.env.observation_space = Box(low=-np.inf, high=np.inf, shape=worker.env.observation_space.shape)
                if sub_envs[-1].obs_z_score_mean is None:
                    for env in sub_envs:
                        env.obs_z_score_mean = z_score_mean_obs
                        env.obs_z_score_std = z_score_std_obs
                        env.observation_space = Box(low=-np.inf, high=np.inf, shape=env.observation_space.shape)
        if self._norm_rew():
            if rew_z_score_global is not None:
                z_score_mean_rew, z_score_std_rew = rew_z_score_global
                if not sub_envs[-1].eval_env and sub_envs[-1].rew_z_score_mean is None:
                    if sub_envs is None:
                        sub_envs = base_env.get_sub_environments()
                    worker.env.rew_z_score_mean = z_score_mean_rew
                    worker.env.rew_z_score_std = z_score_std_rew 
                    for env in sub_envs:
                        env.rew_z_score_mean = z_score_mean_rew
                        env.rew_z_score_std = z_score_std_rew 
    
    def _set_global_z_score(self, env_source: Union[RecEnvRlLibsGymnasiumGymWrapper, RecEnvRlLibsWrapper] ):
        global obs_z_score_global, rew_z_score_global
        if obs_z_score_global is None and rew_z_score_global is None:
            if env_source is not None and not env_source.eval_env:
                if self._number_of_resets is None:
                    self._number_of_resets = NUM_OPTIM_ROLLOUTS if self._do_multi_reset else 1
                rec_env: Union[RecEnv, RecEnvGlobalBillWrapper, RecEnvRlLibsWrapper] = env_source.wrapped_rec_env
                observs = []
                rewards = []
                members_with_controllable_assets = rec_env.members_with_controllable_assets
                while not isinstance(rec_env, RecEnv) and not isinstance(rec_env, RecEnvGlobalBillWrapper):
                    rec_env = rec_env.wrapped_rec_env
                print("Starting env optimal rollup")
                t = time.time()
                for _ in range(self._number_of_resets):
                    initial_obs = env_source.reset()
                    
                    if type(env_source) == RecEnvRlLibsGymnasiumGymWrapper:
                        initial_obs, _ = initial_obs
                    
                    
                    
                    
                    exogenous_provider = PerfectForesightExogenousProvider(
                        rec_env.observe_all_members_exogenous_variables(),
                        rec_env.observe_all_raw_prices_exogenous_variables(),
                        Delta_M=rec_env.Delta_M
                    )
                    optimal_policy = ModelPredictiveControlPolicyCplex(
                        rec_env.members,
                        rec_env.controllable_assets_state_space,
                        rec_env.controllable_assets_action_space,
                        rec_env.feasible_actions_controllable_assets,
                        rec_env.consumption_function,
                        rec_env.production_function,
                        rec_env.controllable_assets_dynamics,
                        exogenous_provider,
                        rec_env.cost_function_controllable_assets,
                        T=rec_env.T,
                        max_length_samples=rec_env.T,
                        Delta_C=rec_env.Delta_C,
                        Delta_M=rec_env.Delta_M,
                        Delta_P_prime=rec_env.Delta_P_prime,
                        Delta_P=rec_env.Delta_P,
                        n_threads=1,
                        small_penalty_control_actions=0,
                        net_consumption_production_mutex_before=1000000,
                        gamma=gamma_global,
                        rescaled_gamma_mode="rescale_terminal",
                        solver_config="none",
                        verbose=False,
                        solution_chained_optimisation=False,
                        disable_env_ctrl_assets_constraints=True,
                        rec_import_fees=rec_env.rec_import_fees,
                        rec_export_fees=rec_env.rec_export_fees,
                        current_offtake_peak_cost = rec_env.current_offtake_peak_cost,
                        current_injection_peak_cost = rec_env.current_injection_peak_cost,
                        historical_offtake_peak_cost = rec_env.historical_offtake_peak_cost,
                        historical_injection_peak_cost = rec_env.historical_injection_peak_cost,
                        members_with_controllable_assets=members_with_controllable_assets
                    )
                    actions = optimal_policy.sequence_of_actions(
                        rec_env.compute_current_state(),
                        rec_env._observe_members_exogenous_variables(),
                        rec_env._observe_prices_exogenous_variables()
                    )
                    previous_observ = rec_env._compute_current_observation()
                    observs += [initial_obs]
                    
                    if type(initial_obs) in (tuple, list):
                        observs[-1] = observs[-1][0]
                    rewards = []
                    previous_reward = {
                        "metering_period_cost": 0.0,
                        "peak_period_cost": 0.0,
                        "controllable_assets_cost": 0.0
                    }
                    
                    last_action = {
                        k:0 for k in actions[-1].keys()
                    }
                    for a in actions + [last_action]:
                        a_converted = env_source.space_convert_act(a, obs=previous_observ, reward=previous_reward)
                        next_observ, reward, _, _, infos = env_source.step(a_converted)
                        previous_observ = rec_env._compute_current_observation()
                        previous_reward = {
                            "metering_period_cost": infos["costs"]["metering_period_cost"],
                            "peak_period_cost": infos["costs"]["peak_period_cost"],
                            "controllable_assets_cost": infos["costs"]["controllable_assets_cost"]
                        }
                        if type(next_observ) in (list, tuple):
                            next_observ = next_observ[0]
                        if infos["is_peak_period_cost_triggered"] or infos["is_metering_period_cost_triggered"]:
                            rewards += [reward]
                        observs += [next_observ]
                    #print("Optimal env rollup took", time.time() - t, "seconds")
                print("Optimal env rollup done", time.time() - t, "seconds")
                if self._norm_obs():
                    z_score_mean_obs = np.mean(np.vstack(observs), axis=0)
                    z_score_std_obs = np.std(np.vstack(observs), axis=0) + 1e-6
                    obs_z_score_global = (z_score_mean_obs, z_score_std_obs)
                if self._norm_rew():
                    z_score_mean_rew = np.mean(np.vstack(rewards))
                    z_score_std_rew = np.std(np.vstack(rewards)) + 1e-6
                    rew_z_score_global = (z_score_mean_rew, z_score_std_rew)
    
    def on_algorithm_init(
        self,
        *,
        algorithm: "Algorithm",
        **kwargs,
    ) -> None:
        self._init = True

    def on_sub_environment_created(self, *, worker: RolloutWorker, sub_environment: EnvType, env_context: EnvContext, env_index: int | None = None, **kwargs) -> None:
        global obs_z_score_global, rew_z_score_global
        if self._norm_obs() or self._norm_rew():
            if (obs_z_score_global is None or rew_z_score_global is None) and not worker.env.eval_env:
                self._set_global_z_score(worker.env)
        if self._norm_obs():
            worker.env.observation_space = Box(low=-np.inf, high=np.inf, shape=worker.env.observation_space.shape)
            if obs_z_score_global is not None:
                #print(id(worker.env))
                z_score_mean_obs, z_score_std_obs = obs_z_score_global
                worker.env.obs_z_score_mean = z_score_mean_obs
                worker.env.obs_z_score_std = z_score_std_obs
                
        if not worker.env.eval_env and self._norm_rew():
            if rew_z_score_global is not None:
                z_score_mean_rew, z_score_std_rew = rew_z_score_global
                
                worker.env.rew_z_score_mean = z_score_mean_rew
                worker.env.rew_z_score_std = z_score_std_rew 

        
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
        pass
        #postprocessed_batch["infos"])
        #print(worker.env.eval_env)
        #global obs_z_score_global
        #if obs_z_score_global is not None:
            #if self._norm_obs():
                #z_score_mean_obs, z_score_std_obs = obs_z_score_global
                #postprocessed_batch["obs"] = (postprocessed_batch["obs"] - z_score_mean_obs) / (z_score_std_obs)
    
    def on_sample_end(self, *, worker: RolloutWorker, samples: SampleBatch, **kwargs) -> None:
        #global obs_z_score_global, rew_z_score_global
        pass
        #if self._norm_rew():
            #samples = samples["default_policy"]
            #z_score_mean_rew, z_score_std_rew = rew_z_score_global
            #is_metering_or_peak_trigger = np.asarray([
            ##    info["is_peak_period_cost_triggered"] or info["is_metering_period_cost_triggered"] for info in samples["infos"]
            #])
            #samples["rewards"][is_metering_or_peak_trigger] = (samples["rewards"][is_metering_or_peak_trigger] - float(z_score_mean_rew)) / (float(z_score_std_rew))
            
class MultiOptiNormObsRewardsAbstractCallback(OptiNormObsRewardsAbstractCallback):
    def __init__(self, *args, legacy_callbacks_dict: Dict[str, Any] = None, **kwargs):
        super().__init__(*args, legacy_callbacks_dict=legacy_callbacks_dict, do_multi_reset=True, **kwargs)

class OptiNormObsRewardsCallback(OptiNormObsRewardsAbstractCallback):

    def _norm_rew(self):
        return True
    
    def _norm_obs(self):
        return True
    
class OptiNormObsCallback(OptiNormObsRewardsAbstractCallback):
    
    def _norm_rew(self):
        return False
    
    def _norm_obs(self):
        return True
    
class OptiNormRewCallback(OptiNormObsRewardsAbstractCallback):
    
    def _norm_rew(self):
        return True
    
    def _norm_obs(self):
        return False
    
class MultiOptiNormObsRewardsCallback(MultiOptiNormObsRewardsAbstractCallback):

    def _norm_rew(self):
        return True
    
    def _norm_obs(self):
        return True
    
class MultiOptiNormObsCallback(MultiOptiNormObsRewardsAbstractCallback):
    
    def _norm_rew(self):
        return False
    
    def _norm_obs(self):
        return True
    
class MultiOptiNormRewCallback(MultiOptiNormObsRewardsAbstractCallback):
    
    def _norm_rew(self):
        return True
    
    def _norm_obs(self):
        return False

        

class NormRewardsCallback(DefaultCallbacks):
    def __init__(self, *args, legacy_callbacks_dict: Dict[str, callable] = None, **kwargs):
        super().__init__(legacy_callbacks_dict=legacy_callbacks_dict)
        self._running_filter_mean_std_reward = None
        

    def on_learn_on_batch(
        self, *, policy: Policy, train_batch: SampleBatch, result: dict, **kwargs
    ):
        if self._running_filter_mean_std_reward is None:
            self._running_filter_mean_std_reward = MeanStdFilter((), clip=np.inf)
        mean_advantages = tc.mean(train_batch["advantages"])
        std_advantages = tc.std(train_batch["advantages"])
        #print("advantages after norm", postprocessed_batch["advantages"])
        train_batch["advantages"] = (train_batch["advantages"] - mean_advantages) / (std_advantages + 1e-6)
        #print("advantages after norm", postprocessed_batch["advantages"])
class ValueStatsTracker(DefaultCallbacks):

    def __init__(self, *args, legacy_callbacks_dict: Dict[str, callable] = None, **kwargs):
        super().__init__(legacy_callbacks_dict=legacy_callbacks_dict)

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
        if (worker.env is not None and worker.env.eval_env) or postprocessed_batch["infos"][-1].get("eval_env", False) or postprocessed_batch["infos"][-1].get(0, dict()).get("agent0", dict()).get("eval_env", False):
            episode.custom_metrics["value_targets"] = -postprocessed_batch["vf_preds"]
            episode.custom_metrics["value_targets_var"] = np.var(-postprocessed_batch["vf_preds"], ddof=1)

class DiscountedEffectiveBill(DefaultCallbacks):
    
    def __init__(self, *args, legacy_callbacks_dict: Dict[str, callable] = None, **kwargs):
        super().__init__(legacy_callbacks_dict=legacy_callbacks_dict)
        self._gammas = None

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
        global gamma_global
        if (worker.env is not None and worker.env.eval_env) or postprocessed_batch["infos"][-1].get("eval_env", False) or postprocessed_batch["infos"][-1].get(0, dict()).get("agent0", dict()).get("eval_env", False):
            if self._gammas is None:
                gamma = gamma_global#policies["default_policy"].config["gamma"]
                
                T = worker.env.spec.max_episode_steps
                wrapped_env = worker.env._wrapped_rec_env
                while type(wrapped_env) != RecEnv:
                    wrapped_env = wrapped_env._wrapped_rec_env
                nb_time_steps_in_peak_period = wrapped_env.Delta_M * wrapped_env.Delta_P
                nb_peak_periods = (T-1)//nb_time_steps_in_peak_period
                self._gammas = [(gamma**nb_time_steps_in_peak_period)] * (nb_time_steps_in_peak_period+1)
                if nb_peak_periods > 1:
                    for _ in range(nb_peak_periods-1):
                        self._gammas.extend([self._gammas[-1]*(gamma**nb_time_steps_in_peak_period)]*(nb_time_steps_in_peak_period))
                self._gammas = np.asarray(self._gammas, dtype=np.float32)
            len_rewards = postprocessed_batch["rewards"].shape[0]
            episode.custom_metrics["discounted_rewards"] = np.dot(postprocessed_batch["rewards"], self._gammas[:len_rewards])
    
                
"""
class ActNormCallBack(DefaultCallbacks):
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
        from pprint import pprint
        infos = postprocessed_batch["infos"][0]
        action_moments = infos["actions_moments"]
        actions_means = list(OrderedDict(sorted(action_moments["mean"].items())).values())
        actions_std = list(OrderedDict(sorted(action_moments["std"].items())).values())
        if "actions" in postprocessed_batch:
            postprocessed_batch["actions"] = np.asarray([np.asarray([(action[i] - actions_means[i]) / (actions_std[i]) for i in range(len(action))]) for action in postprocessed_batch["actions"]])

"""
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
def run_experiment(env, env_wrappers, env_valid, env_eval, rl_env, rl_env_eval, T, Delta_M, Delta_P, Delta_P_prime, random_seed, remove_current_peak_costs, remove_historical_peak_costs, erase_file, stdout, multiprice, space_converter, mean_std_filter_mode, model_config, gamma, gamma_policy, lambda_gae, clip_param, vf_clip_param, vf_coeff, entropy_coeff, kl_coeff, kl_target, learning_rate, batch_size, gc, number_of_episodes, n_sgds, action_weights_divider, action_dist, n_gpus, n_cpus, use_wandb, wandb_project, wandb_offline, gymnasium_wrap, root_dir, tmp_dir, time_iter, sha_folders, tar_gz_results, n_iters, number_of_episodes_eval, framework, compute_global_bill_on_next_obs):
    global gamma_global, NUM_OPTIM_ROLLOUTS
    NUM_OPTIM_ROLLOUTS=number_of_episodes_eval
    
    random.seed(random_seed)
    np.random.seed(random_seed)
    tc.manual_seed(random_seed)
    tc.set_num_threads(n_cpus)
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
        n_cpus_global_bill_optimiser=n_cpus,
        seed=random_seed,
        compute_global_bill_on_next_obs=compute_global_bill_on_next_obs
    )
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
        n_cpus_global_bill_optimiser=n_cpus,
        time_optim=time_iter,
        seed=random_seed
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
            n_cpus_global_bill_optimiser=n_cpus,
            seed=random_seed
        )
        rec_env_valid._n_cpus_global_bill_optimiser=1
    assert(rec_env_train.T == rec_env_eval.T and (rec_env_valid is None or rec_env_eval.T == rec_env_valid.T))
    T = rec_env_train.T
    future_counter_tau_dm, future_counter_tau_dp = future_counters(
        0, 0, duration=T-1, Delta_M=Delta_M, Delta_P=Delta_P
    )
    model_config_dict = dict()#dreamer_models_zoo[model_config]()
    if not ((future_counter_tau_dp[-1] == Delta_P or (remove_peaks_costs and future_counter_tau_dm[-1] == Delta_M)) and not ("optim" in mean_std_filter_mode and "flatten_and_boxify" not in space_converter)):
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
        
        env_train_id, env_eval_id, env_eval_creator, env_valid_creator, env_train_creator = create_rl_env_creators(
            rl_env, rl_env_eval, rec_env_train, rec_env_eval, space_converter, gymnasium_wrap=False, infos_rec_env_train=infos_rec_env_train, infos_rec_env_eval=infos_rec_env_eval, rec_env_valid=rec_env_valid, infos_rec_env_valid=infos_rec_env_valid, members_with_controllable_assets=infos_rec_env_train["members_with_controllable_assets"], gym_register=True, return_rec_env_train_creator=True
        )
        multiprice_str = "multi" if multiprice else "mono"
        
        hyper_parameters = {
            "model_config": model_config,
            "learning_rate": learning_rate,
            "number_of_episodes": number_of_episodes,
            "gradient_clipping_norm": gc,
            "gamma": gamma,
            "gamma_policy": gamma_policy,
            "mean_std_filter_mode": mean_std_filter_mode,
            "lambda_gae": lambda_gae,
            "kl_coeff": kl_coeff,
            "action_weights_divider": action_weights_divider,
            "action_dist": action_dist
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
        prefix = f"multiprice={multiprice_str}//env={env}/number_of_episodes_eval={number_of_episodes_eval}/env_wrappers={env_wrappers}/env_eval={env_eval}/env_valid={env_valid}/rl_env={rl_env}/rl_env_eval={rl_env_eval}/space_converter={space_converter}/{hyper_parameters_slashes_str}/random_seed={random_seed}/Delta_M={Delta_M}"
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
                num_cpus = n_cpus
            else:
                num_cpus = os.cpu_count()
            
            #ray.init(ignore_reinit_error=True, num_cpus=0, num_gpus=0, _temp_dir=f"{tmp_dir}/tmp", include_dashboard=False)
            #ray.init(include_dashboard=False)
            if not stdout:
                os.makedirs(path, exist_ok=True)
                with open(pathlock, 'w') as _: 
                    pass
            gamma_global = gamma
            if gc == 0:
                gc = None
            evaluation_config = {"explore":False, "env": env_eval_id, "env_config": config_wandb, "num_envs_per_worker": number_of_episodes_eval}
            
            if action_dist != "default":
                model_config_dict["custom_action_dist"] = action_dist
            framework_kwargs = {"framework": "torch"} if framework == "torch" else {"framework": "tf2", "eager_tracing": True}

            lr_args = learning_rate.split("#")
            lr_schedule = None
            if len(lr_args) == 3:
                lr_start = float(lr_args[0])
                lr_end = float(lr_args[1])
                lr_steps = int(lr_args[2]) * T * number_of_episodes
                #lr_schedule = list(np.linspace(lr_start, lr_end, lr_steps))
                lr_iters = n_iters * T * number_of_episodes
                #lr_schedule += [lr_end] * max(lr_iters - lr_steps + 1, 0)
                #lr_schedule = list(enumerate(lr_schedule))
                lr_schedule = [
                    [0, lr_start],
                    [lr_steps, lr_end]
                ]
            kwargs_dreamer = (({
                "lr": float(learning_rate)
                }) if lr_schedule is None else {"lr_schedule": lr_schedule})
            import dreamerv3
            from dreamerv3 import embodied
            from embodied.envs import from_gym
            from dreamerv3.train import make_replay
            config = embodied.Config(dreamerv3.configs['defaults'])
            config = config.update(dreamerv3.configs['large'])
            env_train = env_train_creator()
            env_eval = env_eval_creator()
            #config = config.update(dreamerv3.configs['multicpu'])
            config = config.update({
                'envs.amount': 1,
                'run.train_ratio': 64,
                'run.eval_every': 1,
                'run.log_every': 10,
                'batch_size': 1,
                "batch_length": 181,
                'jax.prealloc': False,
                'encoder.cnn_keys': '$^',
                'decoder.cnn_keys': '$^',
                'encoder.mlp_keys': 'vector',
                'decoder.mlp_keys': 'vector',
                'jax.platform': 'cpu',
                "imag_horizon": env_train.T,
                #'rssm.units': 256,
                "jax.precision": "float32",
                "horizon": env_train.T,
                "run.eval_initial": False,
                "run.eval_eps":10,
                "rssm.unroll": False,
                "run.train_fill": 10*env_train.T,
                "run.expl_until": 0,
                "retnorm.decay": 0.9999
            })

            logdir = None
            step = embodied.Counter()
            logger = embodied.Logger(step, [
                embodied.logger.TerminalOutput(),
                # embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
                # embodied.logger.TensorBoardOutput(logdir),
                # embodied.logger.WandBOutput(logdir.name, config),
                # embodied.logger.MLFlowOutput(logdir.name),
            ])

            
            kwargs_optim = None
            if mean_std_filter_mode == "obs_and_rew_optim":
                kwargs_optim = {
                    "include_obs": True,
                    "include_rew": True,
                    "num_rollouts": 1
                }
            elif mean_std_filter_mode == "obs_optim":
                kwargs_optim = {
                    "include_obs": True,
                    "include_rew": False,
                    "num_rollouts": 1
                }
            elif mean_std_filter_mode == "rew_optim":
                kwargs_optim = {
                    "include_obs": False,
                    "include_rew": True,
                    "num_rollouts": 1
                }
            elif mean_std_filter_mode == "obs_and_rew_multi_optim":
                kwargs_optim = {
                    "include_obs": True,
                    "include_rew": True,
                    "num_rollouts": number_of_episodes
                }
            elif mean_std_filter_mode == "obs_multi_optim":
                kwargs_optim = {
                    "include_obs": True,
                    "include_rew": False,
                    "num_rollouts": number_of_episodes
                }
            elif mean_std_filter_mode == "rew_multi_optim":
                kwargs_optim = {
                    "include_obs": False,
                    "include_rew": True,
                    "num_rollouts": number_of_episodes
                }
            obs_z_score_mean = None
            obs_z_score_std = None
            rew_z_score_mean = None
            rew_z_score_std = None
            if kwargs_optim is not None:
                from copy import deepcopy
                obs_z_score, rew_z_score = compute_optimal_z_score(deepcopy(env_train), gamma=gamma, **kwargs_optim)
                #print(obs_z_score)
                #print(rew_z_score)
                #print(type(train_env))
                if obs_z_score is not None:
                    obs_z_score_mean, obs_z_score_std = obs_z_score 
                    env_train.obs_z_score_mean = obs_z_score_mean
                    env_train.obs_z_score_std = obs_z_score_std
                    env_eval.obs_z_score_mean = obs_z_score_mean
                    env_eval.obs_z_score_std = obs_z_score_std
                if rew_z_score is not None:
                    rew_z_score_mean, rew_z_score_std = rew_z_score 
                    env_train.rew_z_score_mean = rew_z_score_mean
                    env_train.rew_z_score_std = rew_z_score_std
            

            env = from_gym.FromGym(env_train, obs_key='vector')  # Or obs_key='vector'.
            env = dreamerv3.wrap_env(env, config)
            env = embodied.BatchEnv([env], parallel=False)

            eval_env = from_gym.FromGym(env_eval, obs_key='vector')  # Or obs_key='vector'.
            eval_env = dreamerv3.wrap_env(eval_env, config)
            eval_env = embodied.BatchEnv([eval_env], parallel=False)
            """
            env = make("Pendulum-v1")#crafter.Env()  # Replace this with your Gym env.
            env = from_gym.FromGym(env, obs_key='vector')  # Or obs_key='vector'.
            env = dreamerv3.wrap_env(env, config)
            env = embodied.BatchEnv([env], parallel=False)
            eval_env=env
            """
            agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
            replay = make_replay(config, logdir)
            eval_replay = make_replay(config, logdir, is_eval=True)

            args = embodied.Config(
                **config.run, logdir=logdir,
                batch_steps=config.batch_size * config.batch_length)
            embodied.run.train_eval(agent, env, eval_env, replay, eval_replay, logger, args)
                

if __name__ == '__main__':
    run_experiment()