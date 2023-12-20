"""
from tests.create_test_env import create_test_env
from tests.create_long_simple_env import create_long_simple_env
from tests.create_first_env import create_first_env
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = str(1) # export OPENBLAS_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = str(1) # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = str(1) # export NUMEXPR_NUM_THREADS=1
import numpy as np
from envs import create_env
from env.rec_env_global_bill_trigger import RecEnvGlobalBillTrigger
from exogenous_providers import PerfectForesightExogenousProvider
from env.counter_utils import future_counters
import click
import time
from time import time
from envs import create_env_fcts
from utils.run_policy import run_policy
from experiment_scripts.generic.trigger_zoo import create_triggers_global_bill_function, metering_period_trigger_global_bill_functions, peak_period_trigger_global_bill_functions
from policies import simple_policies
import random

@click.command()
@click.option('--env', "env", type=click.Choice(list(create_env_fcts.keys())), help='Environment to launch', required=True)
@click.option('--Delta-M', "Delta_M", type=int, default=2, help='Nb of timesteps in a metering period Delta_M.')
@click.option('--Delta-P', "Delta_P", type=int, default=1, help='Nb of metering period in a peak period Delta_P.')
@click.option('--Delta-P-prime', "Delta_P_prime", type=int, default=0, help='Number of peaks period for peak billing Delta_P_prime.')
@click.option('--T', "T", type=int, default=None, help='Time horizon (leave default if None).')
@click.option("--gamma", "gamma", type=float, help="Discount factor", default=1.0)
@click.option('--remove-current-peak-costs/--no-remove-current-peak-costs', "remove_current_peak_costs", is_flag=True, help='Whether current peak costs are removed.')
@click.option('--very-small-current-peak-costs/--no-very-small-current-peak-costs', "very_small_current_peak_costs", is_flag=True, help='Whether current peak costs are very small (=1e-8).')
@click.option('--remove-historical-peak-costs/--no-remove-historical-peak-costs', "remove_historical_peak_costs", is_flag=True, help='Whether historical peak costs are removed.')
@click.option('--multiprice/--no-multiprice', "multiprice", is_flag=True, help='Whether (buying) are changing per metering period.')
@click.option('--policy', "policy", type=click.Choice(list(simple_policies.keys())), help='Simple policy to execute', required=True)
@click.option('--metering-period-trigger', "metering_period_trigger", default="default", type=click.Choice(list(metering_period_trigger_global_bill_functions.keys())), help='Metering trigger function')
@click.option('--peak-period-trigger', "peak_period_trigger", default="default", type=click.Choice(list(peak_period_trigger_global_bill_functions.keys())), help='Peak trigger function')
@click.option('--global-bill-greedy-init/--no-global-bill-greedy-init', "global_bill_greedy_init", is_flag=True, help='Whether global bill optim is greedy-initialized')
@click.option('--incremental-build/--no-incremental-build', "incremental_build", is_flag=True, help='Whether global bill optim problem is incrementally built')
@click.option('--n-cpus', "n_cpus", type=int, help='Number of cpus', default=1)
@click.option('--n-samples', "n_samples", type=int, help='Number of simple policy sampling', default=None)
@click.option('--random-seed', "random_seed", type=int, help='Random seed', default=None)
@click.option('--time-policy/--no-time-policy', "time_policy", is_flag=True, help='Whether policy execution is timed')
@click.option('--display-std/--no-display-std', "display_std", is_flag=True, help='Whether to display std')
@click.option('--force-greedy-bill-optimisation', "force_greedy_bill_optimisation", is_flag=True, help='Whether to force greedy bill optimisation')
def run_experiment(env, Delta_M, Delta_P, Delta_P_prime, T, gamma, remove_current_peak_costs, very_small_current_peak_costs, remove_historical_peak_costs, multiprice, policy, metering_period_trigger, peak_period_trigger, global_bill_greedy_init, incremental_build, n_cpus, n_samples, random_seed, time_policy, display_std, force_greedy_bill_optimisation):
    if random_seed is None:
        random_seed = np.random.randint(1, 1000000)
    np.random.seed(random_seed)
    random.seed(random_seed)
    current_offtake_peak_cost = None if (not remove_current_peak_costs and not very_small_current_peak_costs) else (1e-12 if very_small_current_peak_costs else 0)
    current_injection_peak_cost = None if (not remove_current_peak_costs and not very_small_current_peak_costs) else (1e-12 if very_small_current_peak_costs else 0)
    historical_offtake_peak_cost = None if not remove_historical_peak_costs else 0
    historical_injection_peak_cost = None if not remove_historical_peak_costs else 0
    rec_env, infos_envs = create_env(
        id_env=env,
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
        seed=random_seed,
        force_optim_no_peak_costs=force_greedy_bill_optimisation
    )
    rec_env.global_bill_adaptative_optimiser.n_cpus = 1
    rec_env.global_bill_adaptative_optimiser.incremental_build_flag = incremental_build
    rec_env.global_bill_adaptative_optimiser.greedy_init = global_bill_greedy_init
    if metering_period_trigger != "default" or peak_period_trigger != "default":
        metering_period_cost_trigger, peak_period_cost_trigger = create_triggers_global_bill_function(rec_env, id_metering_period_trigger=metering_period_trigger, id_peak_period_trigger=peak_period_trigger)
        rec_env = RecEnvGlobalBillTrigger(
            rec_env,
            metering_period_cost_trigger=metering_period_cost_trigger,
            peak_period_cost_trigger=peak_period_cost_trigger,
            global_bill_optimiser_greedy_init=global_bill_greedy_init,
            incremental_build_flag=incremental_build
        )
    T = rec_env.T
        
    members = rec_env.members
    feasible_actions_controllable_assets = rec_env.feasible_actions_controllable_assets
    consumption_function = rec_env.consumption_function
    production_function = rec_env.production_function
    #if remove_peak_costs and K <= ((T-1)/Delta_M)*2 or (not remove_peak_costs and K <= ((T-1)/(Delta_M/Delta_P))*2):
    future_counter_tau_dm, future_counter_tau_dp = future_counters(
        0, 0, duration=T-1, Delta_M=Delta_M, Delta_P=Delta_P
    )
    

    if (future_counter_tau_dp[-1] == Delta_P or (remove_current_peak_costs and remove_historical_peak_costs and future_counter_tau_dm[-1] == Delta_M)):
        
        if time_policy:
            print(f"Computing {policy} policy with T={T}...")
            t = time()
        policy_obj = simple_policies[policy](
            members,
            rec_env.controllable_assets_state_space,
            rec_env.controllable_assets_action_space,
            feasible_actions_controllable_assets,
            consumption_function,
            production_function,
            rec_env.exogenous_space,
            members_with_controllable_assets=infos_envs["members_with_controllable_assets"],
            Delta_M=Delta_M
        )
        cost_data = run_policy(
            rec_env,
            policy_obj,
            T=rec_env.T,
            gamma=gamma,
            time_it=time_policy,
            n_samples=n_samples,
            num_cpus=n_cpus,
            return_std=display_std
        )
        if display_std:
            cost_total, undiscounted_cost_total, cost_total_std, undiscounted_cost_total_std = cost_data
        else:
            cost_total, undiscounted_cost_total = cost_data
        if time_policy:
            print(f"Policy {policy} computed in {time() - t} seconds")
        results = cost_total
        if display_std:
            results = (cost_total, cost_total_std)
        print(results)
                
                

if __name__ == '__main__':
    run_experiment()