"""
from tests.create_test_env import create_test_env
from tests.create_long_simple_env import create_long_simple_env
from tests.create_first_env import create_first_env
"""
from hashlib import sha256
import os

from experiment_scripts.mpc.exogenous_data_provider_zoo import exogenous_data_provider_zoo, create_exogenous_data_provider
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6
import warnings
from env.rec_env_global_bill_trigger import RecEnvGlobalBillTrigger
from envs import create_env
from env.counter_utils import future_counters
import json
import click
import wandb
import time
import filelock
from time import time, sleep
import random
import numpy as np
from envs import create_env_fcts
from policies.replay_policy import ReplayPolicy
from utils.run_policy import run_policy
from experiment_scripts.mpc.mpc_policies_zoo import mpc_policies, create_mpc_policy
from experiment_scripts.generic.trigger_zoo import create_triggers_global_bill_function, metering_period_trigger_global_bill_functions, peak_period_trigger_global_bill_functions
from ..mpc import solver_params_dict, solvers

@click.command()
@click.option('--env', "env", type=click.Choice(list(create_env_fcts.keys())), help='Environment to launch', required=True)
@click.option('--exogenous-data-provider', "exogenous_data_provider", type=click.Choice(list(exogenous_data_provider_zoo.keys())), help='Exogenous provider zoo', default="none")
@click.option('--Delta-M', "Delta_M", type=int, default=2, help='Nb of timesteps in a metering period Delta_M.')
@click.option('--Delta-P', "Delta_P", type=int, default=1, help='Nb of metering period in a peak period Delta_P.')
@click.option('--Delta-P-prime', "Delta_P_prime", type=int, default=0, help='Number of peaks period for peak billing Delta_P_prime.')
@click.option('--K', "K", type=int, default=1, help='Policy horizon K.')
@click.option('--T', "T", type=int, default=None, help='Time horizon (leave default if None).')
@click.option("--gamma", "gamma", type=float, help="Discount factor (Env side)", default=1.0)
@click.option("--gamma-policy", "gamma_policy", type=float, help="Discount factor (MPC policy side)", default=0.0)
@click.option("--solver-config", "solver_config", type=click.Choice(solver_params_dict.keys()), help="Id of Solvers pre-registered params", default="none")
@click.option("--solver", "solver", type=click.Choice(solvers), help="Id of Solvers pre-registered params", default="cplex")
@click.option('--rescaled-gamma-mode', "rescaled_gamma_mode", type=click.Choice(["no_rescale", "rescale_terminal", "rescale_delayed_terminal"]), help='Gamma rescale mode', default="no_rescale")
@click.option('--remove-current-peak-costs/--no-remove-current-peak-costs', "remove_current_peak_costs", is_flag=True, help='Whether current peak costs are removed from env.')
@click.option('--remove-historical-peak-costs/--no-remove-historical-peak-costs', "remove_historical_peak_costs", is_flag=True, help='Whether historical peak costs are removed from env.')
@click.option('--erase-file/--no-erase-file', "erase_file", is_flag=True, help='Whether result file is erased.')
@click.option('--stdout/--no-stdout', "stdout", is_flag=True, help='Whether the result is print instead of being saved.')
@click.option('--n-samples-policy', "n_samples_policy", type=int, default=1, help='Number of times to repeat exogenous sequences for policy.')
@click.option('--n-samples', "n_samples", type=int, default=1, help='Number of times MPC policy is simulated.')
@click.option('--multiprice/--no-multiprice', "multiprice", is_flag=True, help='Whether (buying) are changing per metering period.')
@click.option('--random-seed', "random_seed", type=int, default=None, help='Random seed.')
@click.option('--mpc-policy', "mpc_policy", type=click.Choice(mpc_policies.keys()), help='Policy to execute', default="perfect_foresight_mpc")
@click.option('--metering-period-trigger', "metering_period_trigger", default="default", type=click.Choice(list(metering_period_trigger_global_bill_functions.keys())), help='Metering trigger function')
@click.option('--peak-period-trigger', "peak_period_trigger", default="default", type=click.Choice(list(peak_period_trigger_global_bill_functions.keys())), help='Peak trigger function')
@click.option('--net-cons-prod-mutex-before', "net_consumption_production_mutex_before", type=int, help="Indicate until which timestep mutexes are relaxed", default=1000000)
@click.option('--use-wandb/--no-use-wandb', "use_wandb", is_flag=True, help='Whether to use Weight and Biases')
@click.option("--wandb-project", "wandb_project", default="mpcstudy", help="Wandb project name")
@click.option("--wandb-offline", "wandb_offline", is_flag=True, help="Whether wandb is set offline")
@click.option('--n-cpus', "n_cpus", type=int, help='Number of cpus', default=None)
@click.option("--root-dir", "root_dir", default=os.path.expanduser('~'), help="Root directory")
@click.option("--lock-dir", "lock_dir", default=os.path.expanduser('~'), help="Lock (usually shared) directory")
@click.option('--solver-verbose/--no-solver-verbose', "solver_verbose", is_flag=True, help='Whether solver is verbose')
@click.option('--time-policy/--no-time-policy', "time_policy", is_flag=True, help='Whether policy execution is timed')
@click.option('--small-pen-ctrl-actions', "small_penalty_controllable_actions", type=float, help='Small penalty associated to control actions', default=0)
@click.option('--global-bill-greedy-init/--no-global-bill-greedy-init', "global_bill_greedy_init", is_flag=True, help='Whether global bill optim is greedy-initialized')
@click.option('--incremental-build/--no-incremental-build', "incremental_build", is_flag=True, help='Whether global bill optim problem is incrementally built')
@click.option('--solution-chained-optimisation/--fresh-optimisation', "solution_chained_optimisation", is_flag=True, help='[EXPERIMENTAL] Whether subsequent MPC solutions are quasi-restricted to the previous one.')
@click.option('--disable-env-ctrl-assets-constraints/--enable-env-ctrl-assets-constraints', "disable_env_ctrl_assets_constraints", is_flag=True, help='Whether ctrl actions constraints are disabled')
def run_experiment(env, exogenous_data_provider, Delta_M, Delta_P, Delta_P_prime, K, T, gamma, gamma_policy, solver_config, solver, rescaled_gamma_mode, remove_current_peak_costs, remove_historical_peak_costs, erase_file, stdout, n_samples, n_samples_policy, multiprice, random_seed, mpc_policy, metering_period_trigger, peak_period_trigger, net_consumption_production_mutex_before, use_wandb, wandb_project, wandb_offline, n_cpus, root_dir, lock_dir, solver_verbose, time_policy, small_penalty_controllable_actions, global_bill_greedy_init, incremental_build, solution_chained_optimisation, disable_env_ctrl_assets_constraints):
    current_offtake_peak_cost = None if not remove_current_peak_costs else 1e-12
    current_injection_peak_cost = None if not remove_current_peak_costs else 1e-12
    historical_offtake_peak_cost = None if not remove_historical_peak_costs else 0
    historical_injection_peak_cost = None if not remove_historical_peak_costs else 0
    remove_peak_costs = remove_current_peak_costs and remove_historical_peak_costs
    
    folder = root_dir + "/rec_experiments/MPC/"
    if random_seed is None:
        random_seed = np.random.randint(1,1000000)
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    rec_env, infos = create_env(
        id_env=env,
        Delta_M=Delta_M,
        Delta_P=Delta_P,
        Delta_P_prime=Delta_P_prime,
        multiprice=multiprice,
        current_offtake_peak_cost=current_offtake_peak_cost,
        current_injection_peak_cost=current_injection_peak_cost,
        historical_offtake_peak_cost=historical_offtake_peak_cost,
        historical_injection_peak_cost=historical_injection_peak_cost,
        disable_warnings=False,
        T=T,
        n_cpus_global_bill_optimiser=1,
        ignore_ctrl_assets_constraints=disable_env_ctrl_assets_constraints,
        seed=random_seed,
        type_solver="cvxpy"
    )
    metering_period_trigger_fct, peak_period_trigger_fct = create_triggers_global_bill_function(rec_env, metering_period_trigger, peak_period_trigger)
    if metering_period_trigger != "default" or peak_period_trigger != "default":
        rec_env = RecEnvGlobalBillTrigger(
            rec_env,
            metering_period_cost_trigger=metering_period_trigger_fct,
            peak_period_cost_trigger=peak_period_trigger_fct,
            incremental_build_flag=incremental_build,
            global_bill_optimiser_greedy_init=global_bill_greedy_init
        )
    rec_env.global_bill_adaptative_optimiser.incremental_build_flag = incremental_build
    rec_env.global_bill_adaptative_optimiser.greedy_init = global_bill_greedy_init
    rec_env.global_bill_adaptative_optimiser.time_optim = False
    rec_env.global_bill_adaptative_optimiser.n_cpus = 1
    T = rec_env.T
    #if remove_peak_costs and K <= ((T-1)/Delta_M)*2 or (not remove_peak_costs and K <= ((T-1)/(Delta_M/Delta_P))*2):
    future_counter_tau_dm, future_counter_tau_dp = future_counters(
        0, 0, duration=T-1, Delta_M=Delta_M, Delta_P=Delta_P
    )
    
    
    if not ((future_counter_tau_dp[-1] == Delta_P or (remove_peak_costs and future_counter_tau_dm[-1] == Delta_M)) and not (K == 0 and n_samples_policy > 1) and not (rescaled_gamma_mode != "no_rescale" and (gamma == 1.0 or gamma_policy == 1.0))):
        print("Conflicting case encountered, exit.")
    else:
        
        if gamma_policy <= 0:
            gamma_policy = gamma
        multiprice_str = "multiprice" if multiprice else "monoprice"
        multiprice_str_abridged = "multi" if multiprice else "mono"
        small_penalty_controllable_actions_str = str(small_penalty_controllable_actions).replace(".", "")
        net_cons_prod_mutex_str = str(net_consumption_production_mutex_before)
        disable_env_ctrl_assets_constraints_str = "disabled_env_ctrl_assets_constraints" if disable_env_ctrl_assets_constraints else "enable_env_ctrl_assets_constraints"
        gamma_str = str(gamma).replace(".", "_")
        gamma_policy_str = str(gamma_policy).replace(".", "_")
        mpc_policy_str_abridged = "".join([r[0] for r in mpc_policy.split("_")])
        rescaled_gamma_mode_str = rescaled_gamma_mode
        rescaled_gamma_mode_str_abridged = "".join([r[0] for r in rescaled_gamma_mode_str.split("_")])
        solution_chained_optimisation_str = "solution_chained_optimisation" if solution_chained_optimisation else "fresh_optimisation"
        solution_chained_optimisation_str_abridged = "".join([r[0] for r in solution_chained_optimisation_str.split("_")])
        disable_env_ctrl_assets_constraints_str_abridged = "".join([r[0] for r in disable_env_ctrl_assets_constraints_str.split("_")])
        prefix = f"solution_chained_optimisation={solution_chained_optimisation_str}/env={env}/exogenous_data_provider={exogenous_data_provider}/gamma={gamma_str}/mpc_policy={mpc_policy}/gamma_policy={gamma_policy_str}/solver={solver}/solver_config={solver_config}/rescaled_gamma={rescaled_gamma_mode_str}/metering_period_trigger={metering_period_trigger}/peak_period_trigger={peak_period_trigger}/{net_cons_prod_mutex_str}/K={K}/{multiprice_str}/n_samples_policy={n_samples_policy}/n_samples={n_samples}/small_pen_ctrl_acts={small_penalty_controllable_actions_str}/{disable_env_ctrl_assets_constraints_str}/random_seed/Delta_M={Delta_M}/"
        env_abridged = "".join([(r[0] if not r.isnumeric() else r) for r in env.split("_")])
        exogenous_data_provider_abridged = "".join([(r[0] if not r.isnumeric() else r) for r in exogenous_data_provider.split("_")])
        config = {
            "env": env,
            "exogenous_data_provider": exogenous_data_provider,
            "multiprice": multiprice,
            "mpc_policy": mpc_policy,
            "n_samples":n_samples,
            "n_samples_policy":n_samples_policy,
            "Delta_M": Delta_M,
            "small_pen_ctrl_acts": small_penalty_controllable_actions_str,
            "net_cons_prod_mutex_before": net_cons_prod_mutex_str,
            "metering_period_trigger": metering_period_trigger,
            "gamma": gamma_str,
            "gamma_policy": gamma_policy_str,
            "disable_env_ctrl_assets_constraints": disable_env_ctrl_assets_constraints,
            "rescaled_gamma_mode": rescaled_gamma_mode_str,
            "solver_config": solver_config,
            "solver": solver,
            "solution_chained_optimisation": solution_chained_optimisation,
            "exogenous_data_provider": exogenous_data_provider,
            "random_seed": random_seed
        }
        suffix = prefix
        if not remove_peak_costs:
            suffix = prefix + f"peak_period_trigger={peak_period_trigger}/Delta_P={Delta_P}/Delta_P_prime={Delta_P_prime}/"
            config = {
                **config,
                **{"Delta_P":Delta_P, "Delta_P_prime":Delta_P_prime, "peak_period_trigger": peak_period_trigger}
            }
        
        group_wandb = suffix.replace("/randomseed", "").replace(f"/K={K}/", "/")
        suffix = suffix.replace("/random_seed", f"/random_seed={random_seed}")
        path = folder + suffix
        pathfile = path+'result.json'
        pathlock = path+'result.lock'
        
        id_wandb = suffix.replace(f"/K={K}/", "/")
        group_wandb = sha256(group_wandb.encode('utf-8')).hexdigest()
        id_wandb = sha256(suffix.encode('utf-8')).hexdigest()
        
        

        if not (stdout or erase_file or (not os.path.isfile(pathfile) and not os.path.isfile(pathlock))):
            print("Locked or already computed, exit")
        else:
            if not stdout:
                os.makedirs(path, exist_ok=True)
                with open(pathlock, 'w') as _: 
                    pass
            exogenous_provider = create_exogenous_data_provider(exogenous_data_provider, rec_env)
            policy = create_mpc_policy(
                rec_env,
                mpc_policy,
                K=K,
                n_cpus=1,
                small_penalty_control_actions=small_penalty_controllable_actions,
                net_consumption_production_mutex_before=net_consumption_production_mutex_before,
                gamma_policy=gamma_policy,
                rescaled_gamma_mode=rescaled_gamma_mode,
                solver_verbose=solver_verbose,
                solver_config=solver_config,
                solver=solver,
                solution_chained_optimisation=solution_chained_optimisation,
                disable_env_ctrl_assets_constraints=disable_env_ctrl_assets_constraints,
                rec_import_fees=infos["rec_import_fees"],
                rec_export_fees=infos["rec_export_fees"],
                exogenous_provider=exogenous_provider,
                members_with_controllable_assets=infos["members_with_controllable_assets"]
            )
            if K > rec_env.T and n_samples_policy == 1 and n_cpus == 1:
                rec_env.reset()
                action_sequence = policy.sequence_of_actions(rec_env.compute_current_state(), rec_env._observe_members_exogenous_variables(), rec_env._observe_prices_exogenous_variables())
                
                policy = ReplayPolicy(
                    rec_env.members,
                    rec_env.controllable_assets_state_space,
                    rec_env.controllable_assets_action_space,
                    rec_env.feasible_actions_controllable_assets,
                    rec_env.consumption_function,
                    rec_env.production_function,
                    rec_env.exogenous_space,
                    action_sequence
                )
                policy.reset()
                no_reset_first_time=True
            else:
                no_reset_first_time=False
            t = time()
            if time_policy:
                print(f"Computing MPC policy with K={K} and T={T}...")
            cost_total, undiscounted_cost_total = run_policy(
                rec_env,
                policy,
                n_samples=n_samples,
                T=rec_env.T,
                gamma=gamma,
                time_it=time_policy,
                num_cpus=n_cpus,
                no_reset_first_time=no_reset_first_time
            )
            elapsed_time = time() - t
            if time_policy:
                print(f"MPC policy computed in {elapsed_time} seconds")
            results = {
                "expected_effective_bill": cost_total,
                "undiscounted_expected_effective_bill": undiscounted_cost_total,
                "elapsed_time": elapsed_time
            }
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
                lock_acquired=False
                wandb_connected=False
                iwandb_unique_run_id = id_wandb
                if not wandb_offline:
                    iwandb_lock = lock_dir+f"/{id_wandb}_wandb.lock"
                    lock = filelock.FileLock(iwandb_lock, timeout=60)
                    lock_acquired = False
                    max_retries = 10
                    while not lock_acquired and max_retries > 0:
                        try:
                            lock.acquire()
                            lock_acquired = True
                        except BaseException as e:
                            max_retries -= 1
                            warnings.warn(f"Failed to acquire lock. Max remaining retries : {max_retries}")
                    
                    
                    runs = []
                    if lock_acquired:
                        try:
                            runs = wandb.Api().runs(path=f'samait/{wandb_project}')
                        except BaseException as e:
                            warnings.warn("Could not fetch existing wandb runs, skip")
                            runs = []
                        runs = [run for run in runs if id_wandb in run.id]
                        if len(runs) > 0:
                            iwandb_unique_run_id = runs[0].id
                        else:
                            iwandb_unique_run_id = id_wandb + "_" + str(wandb.util.generate_id())
                        max_retries = 4
                        seconds_to_wait = 1
                        wandb_connected=False
                        while not wandb_connected and max_retries > 0: 
                            try:
                                wandb.init(group=group_wandb, id=iwandb_unique_run_id, config=config, project=wandb_project, entity="samait", resume=("must" if len(runs) > 0 else False), mode="online")
                                wandb_connected=True
                            except BaseException as _:
                                sleep(seconds_to_wait)
                                seconds_to_wait*=2
                                max_retries -= 1
                        if not wandb_connected:
                            exception = None
                            warnings.warn("Wandb cannot connect, switching to offline mode")
                            try:
                                wandb.init(group=group_wandb, id=iwandb_unique_run_id, config=config, project=wandb_project, entity="samait", resume=("must" if len(runs) > 0 else False), mode="offline")
                                warnings.warn("Successfully switching in offline mode.")
                                wandb_connected = True
                            except BaseException as e:
                                exception = e
                            if exception is not None:
                                warnings.warn("Something went wrong with wandb, even in offline mode. Details: " + str(exception))
                #wandb.mark_preempting()
                #run.wait_until_finished()
                if wandb_connected:
                    wandb.define_metric("K")
                    wandb.define_metric(f"Expected Effective Bill", step_metric="K")
                    wandb.define_metric(f"Undiscounted Expected Effective Bill", step_metric="K")
                    wandb.define_metric(f"Elapsed Time", step_metric="K")
                    wandb.log({f"Expected Effective Bill": cost_total, "Undiscounted Expected Effective Bill": undiscounted_cost_total, "Elapsed Time": elapsed_time, "K": int(K)})
                    wandb.finish()
                    if lock_acquired:
                        lock.release()
                    lock = None
                else:
                    pathwandbfile = path+'post_to_wandb.json'
                    with open(pathwandbfile, 'w') as fp:
                        json.dump({
                            "group_wandb": group_wandb,
                            "id_wandb": id_wandb,
                            "config":config,
                            "wandb_project": wandb_project,
                            "entity": "samait",
                            "step_metric": K,
                            "results": results

                        }, fp)
                
                    
                if not stdout:
                    with open(pathfile, 'w') as fp:
                        json.dump(results, fp)
                        os.remove(pathlock)
                else:
                    print(results)
            else:
                if not stdout:
                    with open(pathfile, 'w') as fp:
                        json.dump(results, fp)
                        os.remove(pathlock)
                else:
                    print(results)
                

if __name__ == '__main__':
    run_experiment()