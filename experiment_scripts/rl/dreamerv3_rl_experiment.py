import os
os.environ["OPENBLAS_NUM_THREADS"] = str(1) # export OPENBLAS_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = str(1) # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = str(1) # export NUMEXPR_NUM_THREADS=1
os.environ["OMP_NUM_THREADS"] = str(1)
os.environ["MKL_NUM_THREADS"] = str(1)
from hashlib import sha256
import warnings
import zipfile
from experiment_scripts.rl.action_distributions import action_distribution_zoo
import pickle
import shutil
from gymnasium.spaces import Box
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
from gym import make
from .compute_optimal_z_score import compute_optimal_z_score


os.environ["WANDB_SILENT"] = "true"

value_target_values = []
gamma_global = None
obs_z_score_global = None
rew_z_score_global = None
NUM_OPTIM_ROLLOUTS = None


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
@click.option('--model-config', "model_config", type=click.Choice(dreamer_models_zoo.keys()), default=None, help="Model config available from models zoo (see experiment_scripts/rl/models_zoo.py)", callback = lambda c, p, v: v if v is not None else c.params['env'] + "_default")
@click.option('--gamma', "gamma", type=float, help='Discount factor gamma', default=0.99)
@click.option('--gamma-policy', "gamma_policy", type=str, help='Discount factor gamma for RL (either single value of 3 values separated by # for gamma scheduling)')
@click.option('--lambda-gae', "lambda_gae", type=float, help='GAE Lambda value', default=0.95)
@click.option('--kl-coeff', "kl_coeff", type=float, help='KL coeff', default=1.0)
@click.option('--lr', "learning_rate", type=float, help="Learning rate", default=8e-5)
@click.option('--critic-lr', "critic_learning_rate", type=float, help='Learning rate for critic', default=8e-5)
@click.option('--td-lr', "td_learning_rate", type=float, help='Learning rate for model dynamics', default=6e-04)    
@click.option("--gc", "gc", type=float, help="Gradient clipping value (0 for default clipping per algo)", default=100)
@click.option('--ne', "number_of_episodes", type=int, help='Number of episodes per training iter', default=1)
@click.option('--dreamer-train-iters', "dreamer_train_iters", type=int, help='Training iterations per data collection from real env', default=1)
@click.option('--free-nats', "free_nats", type=int, help='Free nats', default=3)
@click.option("--gaussian-noise", "gaussian_noise", type=float, help="Dreamer gaussian noise", default=0.3)
@click.option('--num-steps-sampled-before-learning-starts', "num_steps_sampled_before_learning_starts", type=int, help='Number of timesteps before Dreamer learning', default=0)
@click.option('--prefill-timesteps', "prefill_timesteps", type=int, help='Prefill timesteps', default=None)
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
def run_experiment(env, env_wrappers, env_valid, env_eval, rl_env, rl_env_eval, T, Delta_M, Delta_P, Delta_P_prime, random_seed, remove_current_peak_costs, remove_historical_peak_costs, erase_file, stdout, multiprice, space_converter, mean_std_filter_mode, model_config, gamma, gamma_policy, lambda_gae, kl_coeff, learning_rate, critic_learning_rate, td_learning_rate, gc, number_of_episodes, dreamer_train_iters, free_nats, gaussian_noise, num_steps_sampled_before_learning_starts, prefill_timesteps, action_weights_divider, action_dist, n_gpus, n_cpus, use_wandb, wandb_project, wandb_offline, gymnasium_wrap, root_dir, tmp_dir, time_iter, sha_folders, tar_gz_results, n_iters, number_of_episodes_eval):
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
        seed=random_seed
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
    model_config_dict = dreamer_models_zoo[model_config]()
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
        
        env_train_id, env_eval_id, env_eval_creator, env_valid_creator = create_rl_env_creators(
            rl_env, rl_env_eval, rec_env_train, rec_env_eval, space_converter, gymnasium_wrap=gymnasium_wrap, infos_rec_env_train=infos_rec_env_train, infos_rec_env_eval=infos_rec_env_eval, rec_env_valid=rec_env_valid, infos_rec_env_valid=infos_rec_env_valid, members_with_controllable_assets=infos_rec_env_train["members_with_controllable_assets"], gym_register=True
        )
        if prefill_timesteps is None:
            prefill_timesteps = T*number_of_episodes
        multiprice_str = "multi" if multiprice else "mono"
        
        hyper_parameters = {
            "model_config": model_config,
            "learning_rate": learning_rate,
            "critic_learning_rate": critic_learning_rate,
            "td_learning_rate": td_learning_rate,
            "number_of_episodes": number_of_episodes,
            "gradient_clipping_norm": gc,
            "gamma": gamma,
            "gamma_policy": gamma_policy,
            "mean_std_filter_mode": mean_std_filter_mode,
            "lambda_gae": lambda_gae,
            "kl_coeff": kl_coeff,
            "action_weights_divider": action_weights_divider,
            "action_dist": action_dist,
            "dreamer_train_iters": dreamer_train_iters,
            "free_nats": free_nats,
            "gaussian_noise": gaussian_noise,
            "num_steps_sampled_before_learning_starts": num_steps_sampled_before_learning_starts,
            "prefill_timesteps": prefill_timesteps
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

            

            
            train_env = make(env_train_id).env.env
            eval_env = make(env_eval_id).env.env
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
                obs_z_score, rew_z_score = compute_optimal_z_score(deepcopy(train_env), gamma=gamma, **kwargs_optim)
                #print(obs_z_score)
                #print(rew_z_score)
                #print(type(train_env))
                if obs_z_score is not None:
                    obs_z_score_mean, obs_z_score_std = obs_z_score 
                    train_env.obs_z_score_mean = obs_z_score_mean
                    train_env.obs_z_score_std = obs_z_score_std
                    eval_env.obs_z_score_mean = obs_z_score_mean
                    eval_env.obs_z_score_std = obs_z_score_std
                if rew_z_score is not None:
                    rew_z_score_mean, rew_z_score_std = rew_z_score 
                    train_env.rew_z_score_mean = rew_z_score_mean
                    train_env.rew_z_score_std = rew_z_score_std

            import dill
            with open('rec_7_envs.env', 'wb') as env_train_file:
                dill.dump({
                    
                    "env_train": (env_train_id, train_env),
                    "env_eval": (env_eval_id, eval_env),
                    "obs_z_score_mean": obs_z_score_mean,
                    "obs_z_score_std": obs_z_score_std,
                    "rew_z_score_mean": rew_z_score_mean,
                    "rew_z_score_std": rew_z_score_std
                    
                
                }, env_train_file)
            
                

if __name__ == '__main__':
    run_experiment()