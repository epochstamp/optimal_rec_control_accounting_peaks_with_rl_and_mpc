import os


os.environ["OPENBLAS_NUM_THREADS"] = str(1) # export OPENBLAS_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = str(1) # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = str(1) # export NUMEXPR_NUM_THREADS=1
os.environ["OMP_NUM_THREADS"] = str(1)
os.environ["MKL_NUM_THREADS"] = str(1)
from hashlib import sha256
from experiment_scripts.rl.action_distributions import action_distribution_zoo
import tqdm
from .skrl_components.skrl_ppo_rnn import Policy, Value
from skrl.agents.torch.sac import SAC_DEFAULT_CONFIG
from skrl.agents.torch.sac import SAC_RNN as SAC
import gymnasium
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer, ParallelTrainer, ManualTrainer
from skrl.utils import set_seed
from .env_wrappers_zoo import env_wrapper_sequences, wrap
from .rl_envs_zoo import create_rl_env_creators, rl_envs
from .models_zoo import skrl_ppo_models_zoo
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
from envs import create_env_fcts
from .compute_optimal_z_score import compute_optimal_z_score
from experiment_scripts.rl.env_wrappers.rec_env_global_bill_discounted_cost import RecEnvGlobalBillDiscountedCost, RecEnvGlobalBillNegateReward
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
@click.option('--model-config', "model_config", type=click.Choice(skrl_ppo_models_zoo.keys()), default=None, help="Model config available from models zoo (see experiment_scripts/rl/models_zoo.py)", callback = lambda c, p, v: v if v is not None else c.params['env'] + "_default")
@click.option('--gamma', "gamma", type=float, help='Discount factor gamma', default=0.99)
@click.option('--gamma-policy', "gamma_policy", type=str, help='Discount factor gamma for RL (either single value of 3 values separated by # for gamma scheduling)')
@click.option('--entropy-init-value', "entropy_init_value", type=float, help='Entropy initial value', default=0.2)
@click.option('--entropy-target', "entropy_target", type=float, help='Entropy target', default=0.002)
@click.option('--lr-actor', "learning_rate_actor", type=str, help='Actor learning rate (either one or three values for schedule)', default="5e-06")
@click.option('--lr-critic', "learning_rate_critic", type=str, help='Critic learning rate (either one or three values for schedule)', default="5e-06")
@click.option('--lr-entropy', "learning_rate_entropy", type=str, help='Entropy learning rate (either one or three values for schedule)', default="5e-06")
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
@click.option('--compute-global-bill-on-next-obs', "compute_global_bill_on_next_obs", is_flag=True, help='Whether global rec bill is computed on next observation.')
@click.option('--memory-size', "memory_size", type=int, default=1000000, help='Memory size')
def run_experiment(env, env_wrappers, env_valid, env_eval, rl_env, rl_env_eval, T, Delta_M, Delta_P, Delta_P_prime, random_seed, remove_current_peak_costs, remove_historical_peak_costs, erase_file, stdout, multiprice, space_converter, mean_std_filter_mode, model_config, gamma, gamma_policy, entropy_init_value, entropy_target, learning_rate_actor, learning_rate_critic, learning_rate_entropy, batch_size, gc, number_of_episodes, n_sgds, action_weights_divider, action_dist, n_gpus, n_cpus, use_wandb, wandb_project, wandb_offline, gymnasium_wrap, root_dir, tmp_dir, time_iter, sha_folders, tar_gz_results, n_iters, number_of_episodes_eval, compute_global_bill_on_next_obs, memory_size):
    global gamma_global, NUM_OPTIM_ROLLOUTS
    NUM_OPTIM_ROLLOUTS=number_of_episodes_eval
    
    random.seed(random_seed)
    np.random.seed(random_seed)
    tc.manual_seed(random_seed)
    set_seed(random_seed)
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
    model_config_dict = skrl_ppo_models_zoo[model_config]()
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
            rl_env, rl_env_eval, rec_env_train, rec_env_eval, space_converter, gymnasium_wrap=True, infos_rec_env_train=infos_rec_env_train, infos_rec_env_eval=infos_rec_env_eval, rec_env_valid=rec_env_valid, infos_rec_env_valid=infos_rec_env_valid, members_with_controllable_assets=infos_rec_env_train["members_with_controllable_assets"], gym_register=True, return_rec_env_train_creator=True
        )
        multiprice_str = "multi" if multiprice else "mono"
        
        hyper_parameters = {
            "model_config": model_config,
            "learning_rate_actor": learning_rate_actor,
            "learning_rate_critic": learning_rate_critic,
            "learning_rate_entropy": learning_rate_entropy,
            "number_of_episodes": number_of_episodes,
            "gradient_clipping_norm": gc,
            "gamma": gamma,
            "gamma_policy": gamma_policy,
            "mean_std_filter_mode": mean_std_filter_mode,
            "entropy_target": entropy_target,
            "action_weights_divider": action_weights_divider,
            "action_dist": action_dist,
            "number_of_sgds": n_sgds,
            "entropy_initial_value": entropy_init_value
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

            env_train = env_train_creator()
            env_eval = env_eval_creator()

            
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
            gymnasium.envs.registration.register(id="env_train", entry_point=lambda: env_train, max_episode_steps=T)
            gymnasium.envs.registration.register(id="env_eval", entry_point=lambda: RecEnvGlobalBillDiscountedCost(RecEnvGlobalBillNegateReward(env_eval), gamma=gamma), max_episode_steps=T)
            env_train = gymnasium.vector.make("env_train", num_envs=min(number_of_episodes, num_cpus), asynchronous=num_cpus>1, disable_env_checker=True)
            env_train = wrap_env(env_train)
            env_eval = gymnasium.vector.make("env_eval", num_envs=min(number_of_episodes, num_cpus), asynchronous=num_cpus>1, disable_env_checker=True)
            env_eval = wrap_env(env_eval)
             
            memory = RandomMemory(memory_size=memory_size, num_envs=env_train.num_envs, device="cpu", replacement=False)
            memory_length = memory_size
            policy_kwargs = model_config_dict["policy"]
            value_function_kwargs = model_config_dict["value_function"]
            policy = Policy(
                env_train.observation_space, env_train.action_space, device="cpu", num_envs=env_train.num_envs, divide_action_layer_by=action_weights_divider, **policy_kwargs
            )
            value = lambda: Value(
                env_train.observation_space, env_train.action_space, device="cpu", num_envs=env_train.num_envs, include_actions_in_input=True, **value_function_kwargs
            )
            models = {}
            models["policy"] = policy
            models["critic_1"] = value()
            models["critic_2"] = value()
            models["target_critic_1"] = value()
            models["target_critic_2"] = value()
            cfg = SAC_DEFAULT_CONFIG.copy()
            cfg["rollouts"] = memory_length  # memory_size
            cfg["gradient_steps"] = n_sgds
            cfg["batch_size"] = batch_size
            cfg["discount_factor"] = float(gamma_policy)
            cfg["learning_rate_actor"] = float(learning_rate_actor)
            cfg["learning_rate_critic"] = float(learning_rate_critic)
            cfg["learning_rate_entropy"] = float(learning_rate_entropy)
            cfg["learn_entropy"] = float(learning_rate_entropy) > 0.0
            cfg["initial_entropy_value"] = entropy_init_value
            cfg["grad_norm_clip"] = gc
            cfg["kl_threshold"] = 0
            cfg["learning_starts"] = batch_size
            #cfg["state_preprocessor"] = RunningStandardScaler
            #cfg["state_preprocessor_kwargs"] = {"size": env_train.observation_space, "device": "cpu"}
            #cfg["value_preprocessor"] = RunningStandardScaler
            #cfg["value_preprocessor_kwargs"] = {"size": 1, "device": "cpu"}

            cfg["experiment"] = dict()
            
            #cfg["experiment"]["wandb"] = use_wandb
            #cfg["experiment"]["directory"] = root_dir
            #cfg["experiment"]["experiment_name"] = sha256(suffix.encode('utf-8')).hexdigest()
            #cfg["experiment"]["write_interval"] = memory_length
            #cfg["experiment"]["wandb_kwargs"] = {
            #    "config": config_wandb,
            #    "project": wandb_project,
            #    "entity":"samait"
            #}
            
            # logging to TensorBoard and write checkpoints (in timesteps)
            #cfg["experiment"]["write_interval"] = 500
            #cfg["experiment"]["checkpoint_interval"] = 5000
            #cfg["experiment"]["directory"] = "runs/torch/PendulumNoVel"
            agent = SAC(models=models,
                        memory=memory,
                        cfg=cfg,
                        observation_space=env_train.observation_space,
                        action_space=env_train.action_space,
                        device="cpu"
                    )
            """
            cfg_eval = dict(cfg)
            cfg_eval["rollouts"] = T*number_of_episodes_eval
            cfg_eval["experiment"] = dict()
            cfg_eval["experiment"]["wandb"] = use_wandb
            cfg_eval["experiment"]["directory"] = root_dir
            cfg_eval["experiment"]["experiment_name"] = sha256(suffix.encode('utf-8')).hexdigest()
            cfg_eval["experiment"]["write_interval"] = 10
            cfg_eval["experiment"]["wandb_kwargs"] = {
                "config": config_wandb,
                "project": wandb_project,
                "entity":"samait"
            }
            
            agent_eval = PPO(models=models,
                        memory=memory_eval,
                        cfg=cfg_eval,
                        observation_space=env_eval.observation_space,
                        action_space=env_eval.action_space,
                        device="cpu"
                    )
            """
            # configure and instantiate the RL trainer
            cfg_trainer = {"timesteps": n_iters*memory_length, "headless": True, "disable_progressbar": True}
            trainer = ManualTrainer(cfg=cfg_trainer, env=env_train, agents=[agent])
            train_mode = True
            if use_wandb:
                pass
                #run = wandb.init(config=config_wandb, project=wandb_project, entity="samait", mode=("offline" if wandb_offline else "online"))
            min_mean_expected_return=float("+inf")
            IT = n_iters*memory_length+1 
            if stdout:
                ranger = tqdm.tqdm(list(range(1, IT)))
            else:
                ranger = range(1, IT)          
            for i in ranger:
                trainer.train(timestep = (i-1)%(cfg["learning_starts"]+1), timesteps = IT)
                if i >= cfg["learning_starts"] and i % (cfg["learning_starts"]+1) == 0:
                    print(f"EVALUATE AT ITER {i}...")
                    t=0
                    states, infos = env_eval.reset()
                    with tc.no_grad():
                        if number_of_episodes_eval == 1:
                            number_of_eval = 1
                        else:
                            number_of_eval = 1
                        trainer.agents.set_running_mode("eval")
                        mean_expected_return = 0
                        for _ in range(number_of_eval):
                            
                            rewards_sums = None
                            terminated = False
                            truncated= False
                            while not truncated and not terminated:
                                actions = trainer.agents.act(states, timestep=t, timesteps=T)[0]
                                next_states, rewards, terminated, truncated, infos = env_eval.step(actions)
                                if rewards_sums is None:
                                    rewards_sums = np.zeros_like(rewards)
                                rewards_sums += rewards.numpy()
                                states = next_states
                                t += 1
                                if type(terminated) != bool:
                                    terminated = tc.all(terminated)
                                    truncated = tc.all(truncated)
                            trainer.agents.set_running_mode("train")
                            mean_expected_return += (1.0/number_of_eval) * np.mean(rewards_sums)
                        min_mean_expected_return = min(min_mean_expected_return, mean_expected_return)
                        data = {
                            f"Expected Effective Bill": mean_expected_return,
                            f"Best Expected Effective Bill": min_mean_expected_return
                        }
                        if stdout:
                            print(data)
                        elif use_wandb:
                            pass
                            #wandb.log({f"Expected Effective Bill": mean_expected_return})

                
                
                            
                        
            #evaluator = SequentialTrainer(cfg=cfg_trainer, env=env_eval, agents=[agent_eval])
            #for _ in range(n_iters):
                # start training
                #
                #trainer.train()
                #agent_eval.policy.load_state_dict(agent.policy.state_dict(), strict=False)
                #evaluator.eval()

if __name__ == '__main__':
    run_experiment()