"""
from tests.create_test_env import create_test_env
from tests.create_long_simple_env import create_long_simple_env
from tests.create_first_env import create_first_env
"""
from envs import create_env, create_simple_env
from envs import create_long_simple_env
from envs import create_first_case
from envs import create_complete_first_case
from envs.first_case.create_complete_first_case_net_cons_prod import create_complete_first_case_net_cons_prod
from envs.first_case.create_first_case_net_cons_prod import create_first_case_net_cons_prod
from exogenous_providers import PerfectForesightExogenousProvider
from repartition_keys_optimiser import create_repartition_keys_optimiser, repartition_keys_optimisers
from tests.test_mpc_policies import test_mpc_policies
from env.counter_utils import future_counters
from env.repartition_keys_utils import repartition_keys_process_str
import os
import json
import click
from copy import deepcopy
import wandb
import neptune.new as neptune
import time
from threading import Semaphore
import filelock
from time import time
import random
import numpy as np
from envs import create_env_fcts
import datetime

@click.command()
@click.option('--env', "env", type=click.Choice(list(create_env_fcts.keys())), help='Environment to launch', required=True)
@click.option('--Delta-M', "Delta_M", type=int, default=2, help='Nb of timesteps in a metering period Delta_M.')
@click.option('--Delta-P', "Delta_P", type=int, default=1, help='Nb of metering period in a peak period Delta_P.')
@click.option('--Delta-P-prime', "Delta_P_prime", type=int, default=1, help='Number of peaks period for peak billing Delta_P_prime.')
@click.option('--K', "K", type=int, default=1, help='Policy horizon K.')
@click.option('--T', "T", type=int, default=None, help='Time horizon.')
@click.option('--ts', "ts", type=int, default=1, help='Number of time steps to average.')
@click.option('--remove-peak-costs/--no-remove-peak-costs', "remove_peak_costs", is_flag=True, help='Whether peak costs are removed.')
@click.option('--remove-peak-costs-policy/--no-remove-peak-costs-policy', "remove_peak_costs_policy", is_flag=True, help='Whether peak costs are removed in the policy.')
@click.option('--n-samples-policy', "n_samples_policy", type=int, default=1, help='Number of times to repeat exogenous sequences for policy.')
@click.option('--optimal-action-population-size', "optimal_action_population_size", type=int, default=1, help='Size of the optimal solution pool (1 is deterministic).')
@click.option('--policy', "policy", type=click.Choice(["original", "surrogate", "mapped"]), help='Policy to execute', default="original")
@click.option('--repartition-keys-optimiser', "repartition_keys_optimiser", default=list(repartition_keys_optimisers.keys()), type=click.Choice(list(repartition_keys_optimisers.keys())), help='Repartition key optimisers')
@click.option('--disable-net-cons-prod-mutex/--no-disable-net-cons-prod-mutex', "disable_net_consumption_production_mutex", is_flag=True, help="Whether to disable net cons prod mutex")
@click.option("--root-dir", "root_dir", default=os.path.expanduser('~'), help="Root directory")
@click.option("--time-table-id", "time_table_id", default=os.path.expanduser('~'), help="Time table id")
@click.option('--n-cpus', "n_cpus", type=int, help='Number of cpus', default=None)
@click.option('--margin', "margin", type=float, default=1.75, help='Margin multiplier.')
def run_experiment(env, Delta_M, Delta_P, Delta_P_prime, K, T, ts, remove_peak_costs, remove_peak_costs_policy, n_samples_policy, optimal_action_population_size, policy, repartition_keys_optimiser, disable_net_consumption_production_mutex, root_dir, time_table_id, n_cpus, margin):
    offtake_peak_cost = None if not remove_peak_costs else 0
    injection_peak_cost = None if not remove_peak_costs else 0
    n_samples = 1
    repartition_keys_optimiser_obj = create_repartition_keys_optimiser(
        repartition_keys_optimiser,
        id_prob = (
            "#".join(
                [
                    "local",
                    env,
                    str(policy=="surrogate"),
                    str(T),
                    str(Delta_M),
                    str(Delta_P),
                    str(Delta_P_prime)
                ]
            )
        )
    )
    rec_env = create_env(
        id_env=env,
        Delta_M=Delta_M,
        Delta_P=Delta_P,
        Delta_P_prime=Delta_P_prime,
        offtake_peak_cost=offtake_peak_cost,
        injection_peak_cost=injection_peak_cost,
        multiprice=False,
        surrogate=policy == "surrogate",
        repartition_keys_optimiser=repartition_keys_optimiser_obj,
        disable_warnings=False,
        T=T
    )
    T = rec_env.T
    members = rec_env.members
    feasible_actions_controllable_assets = rec_env.feasible_actions_controllable_assets
    consumption_function = rec_env.consumption_function
    production_function = rec_env.production_function
    exogenous_variables = rec_env.observe_all_exogenous_variables()
    #if remove_peak_costs and K <= ((T-1)/Delta_M)*2 or (not remove_peak_costs and K <= ((T-1)/(Delta_M/Delta_P))*2):
    future_counter_tau_dm, future_counter_tau_dp = future_counters(
        0, 0, duration=T-1, Delta_M=Delta_M, Delta_P=Delta_P
    )
    

    if (future_counter_tau_dp[-1] == Delta_P or (remove_peak_costs and future_counter_tau_dm[-1] == Delta_M)) and not (K == 0 and n_samples_policy > 1) and not (remove_peak_costs and remove_peak_costs_policy):
        exogenous_provider = PerfectForesightExogenousProvider(exogenous_variables, Delta_M=Delta_M)
        
        time_table_file = f"{root_dir}/time_table/time_table_{time_table_id}.json"
        d = dict()
        if os.path.isfile(time_table_file):
            with open(time_table_file, "r") as time_table_file_stream:
                try:
                    d = json.load(time_table_file_stream)
                except BaseException as _:
                    d = dict()
        key = "#".join(str(k) for k in (
            env,
            Delta_M,
            Delta_P,
            Delta_P_prime,
            policy,
            K,
            T,
            repartition_keys_optimiser,
            remove_peak_costs_policy,
            disable_net_consumption_production_mutex,
            optimal_action_population_size,
            n_cpus,
            remove_peak_costs,
            n_samples_policy))
        if key not in d:
            t = time()
            _ = test_mpc_policies(
                rec_env,
                exogenous_provider,
                K,
                members,
                feasible_actions_controllable_assets,
                consumption_function,
                production_function,
                n_samples,
                n_samples_policy=n_samples_policy,
                optimal_action_population_size=optimal_action_population_size,
                surrogate=policy in ("surrogate", "mapped"),
                convert_repartition_keys_action_from_surrogate = policy == "mapped",
                include_repartition_keys_in_action=repartition_keys_optimiser == "none",
                verbose=False,
                n_threads=n_cpus,
                time_it=False,
                small_penalty_control_actions=0,
                replay_optimal_sequence=False,
                disable_net_production_consumption_mutex=disable_net_consumption_production_mutex,
                block_actions=False,
                ts=ts,
                offtake_peak_cost_erasing=0 if remove_peak_costs_policy else None,
                injection_peak_cost_erasing=0 if remove_peak_costs_policy else None,
            )
            t_averaged_with_margin_raw = ((time() - t) / ts)*margin*T + 20
            t_averaged_with_margin = datetime.timedelta(seconds=t_averaged_with_margin_raw)
            t_averaged_with_margin = (t_averaged_with_margin.days, t_averaged_with_margin.seconds // 3600, max((t_averaged_with_margin.seconds//60)%60, 0), max(t_averaged_with_margin.seconds, 59), t_averaged_with_margin_raw)
            os.makedirs(f"{root_dir}/time_table/locks", exist_ok=True)
            
            lock = filelock.FileLock(f"{root_dir}/time_table/locks/time_table_{time_table_id}_write.lock")
            with lock:
                d = dict()
                if os.path.isfile(time_table_file):
                    with open(time_table_file, "r") as time_table_file_stream_read:
                        try:
                            d = json.load(time_table_file_stream_read)
                        except BaseException as _:
                            d = dict()
                with open(time_table_file, "w") as time_table_file_stream:
                    
                    new_d = {
                        key:t_averaged_with_margin
                    }
                    json.dump({**d, **new_d}, time_table_file_stream)


                

if __name__ == '__main__':
    run_experiment()