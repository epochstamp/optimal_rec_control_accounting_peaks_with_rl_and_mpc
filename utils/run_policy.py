from typing import List
from base.exogenous_provider import ExogenousProvider
from base.policy import Policy
from env.rec_env import RecEnv
from exceptions import MissingArgument
import numpy as np
from scipy.stats import norm
from base import IneqType
from time import time
from exogenous_providers.perfect_foresight_exogenous_provider import PerfectForesightExogenousProvider
from exogenous_providers.pseudo_forecast_exogenous_provider import PseudoForecastExogenousProvider
from policies.model_predictive_control_policy import ModelPredictiveControlPolicy
from policies.model_predictive_control_policy_solver_agnostic import ModelPredictiveControlPolicySolverAgnostic
from policies.no_action_policy import NoActionPolicy
from policies.replay_policy import ReplayPolicy
from utils.utils import rec_gamma_sequence
from multiprocessing import Pool
from copy import deepcopy

global_env = None
global_policy = None

def run_policy_sample(env_policy_pair_infos):
    global global_env, global_policy
    env = global_env.clone()
    policy = deepcopy(global_policy)
    if len(env_policy_pair_infos) == 2:
        gamma_sequence, T = env_policy_pair_infos
    else:
        gamma_sequence, T, time_sample = env_policy_pair_infos
    env.reset()
    if isinstance(policy, ModelPredictiveControlPolicySolverAgnostic) and isinstance(policy._exogenous_provider, PseudoForecastExogenousProvider):
        policy._exogenous_provider._stochastic_env = env
        if policy._max_length_samples > T:
            action_sequence = policy.sequence_of_actions(env.compute_current_state(), env._observe_members_exogenous_variables(), env._observe_prices_exogenous_variables())
            
            policy = ReplayPolicy(
                env.members,
                env.controllable_assets_state_space,
                env.controllable_assets_action_space,
                env.feasible_actions_controllable_assets,
                env.consumption_function,
                env.production_function,
                env.exogenous_space,
                action_sequence
            )
    policy.reset()
    
    cost_total = 0
    undiscounted_cost_total = 0
    if time_sample:
        time_pol = time()
    for tstep in range(T):
        
        current_state = env.compute_current_state()
        current_exogenous_variables_members = env._observe_members_exogenous_variables()
        current_exogenous_prices = env._observe_prices_exogenous_variables()
        action = policy.action(
            current_state,
            current_exogenous_variables_members,
            current_exogenous_prices
        )
        #print(np.prod(list(action.values())))
        #print(current_state[("PVB", "soc")])
        #print(f"s(c) (soc) (i PVB,m 0,{T-1})=", round(env.compute_current_state()[("PVB", "soc")], 3), action)
        #print(current_state[('C', 'current_offtake_peak')], current_state[('C', 'offtake_peaks')])
        #print(current_state[('PVB', 'current_injection_peak')], current_state[('PVB', 'injection_peaks')])
        #print(action)
        #print(action[("PVB", "rec_export")])
        #print(env.compute_current_state()[('PVB', 'previous_electricity_production_metering_period_meter')], env.compute_current_state()[('PVB', 'electricity_production_metering_period_meter')])
        obs, cost, is_terminated, is_truncated, info = env.step(action)
        cost = info["costs"]["metering_period_cost"] + info["costs"].get("peak_period_cost", 0.0)
        
        #print(env.compute_current_state()[('C', 'current_offtake_peak')], env.compute_current_state()[('C', 'offtake_peaks')])
        #print(info["peak_period_costs"]["repartition_keys_sum_of_peaks_penalty_per_peak_period_cost_function"])
        cost_total += gamma_sequence[tstep]*cost
        undiscounted_cost_total += cost
        current_obs = obs
        #print("instant peak cost real", info["peak_period_costs"]["repartition_keys_sum_of_peaks_penalty_per_peak_period_cost_function"])
        #print(cost)
        #print("instant peak cost real", cost)

        if is_terminated or is_truncated:
            if is_terminated:
                raise BaseException(f"Policy broke constraints at timestep {tstep}, see warning message")
            break
    if time_sample:
        print(f"Time took for sampling this MPC policy one time : {time() - time_pol} seconds")
    return (cost_total, undiscounted_cost_total)


def run_policy(env: RecEnv, policy: Policy, n_samples=1, T=1, gamma=1.0, full_trajectory=False, time_it=False, num_cpus=1, return_std=False, no_reset_first_time=False):
    global global_env, global_policy
    #print(T)
    expected_return = []
    expected_return_after_recomputation = 0
    full_trajectories = []
    undiscounted_expected_return = []
    elapsed_time = 0
    cumul_t = 0
    current_t = 0
    if n_samples is None:
        n_samples=1
    gamma_sequence = rec_gamma_sequence(gamma, Delta_M=env.Delta_M, Delta_P=env.Delta_P, T=T)
    if time_it:
        t_global = time()
    if num_cpus == 1:
        for sample in range(n_samples):
            if sample > 0 or not no_reset_first_time:
                current_obs = env.reset()
                policy.reset()
            else:
                current_obs = env._compute_current_observation()
            cost_total = 0
            undiscounted_cost_total = 0
            if full_trajectory:
                trajectory = {
                    "observations": [],
                    "actions": [],
                    "costs": []


                }
            for tstep in range(T):
                
                t, t2 = (time(),) * 2
                #if time_it:
                    
                #print(f"At timestep {tstep}:")
                #print(f"Elapsed time so far: ", current_t, "seconds")
                #print("Computing policy action...")
                current_state = env.compute_current_state()
                current_exogenous_variables_members = env._observe_members_exogenous_variables()
                current_exogenous_prices = env._observe_prices_exogenous_variables()
                action = policy.action(
                    current_state,
                    current_exogenous_variables_members,
                    current_exogenous_prices
                )
                #print(np.prod(list(action.values())))
                
                #if time_it:
                    #print(f"Action computed in {time() - t} seconds.")
                    
                    #print("Computing env transition...")
                #t = time()
                #print(current_state[("PVB", "soc")])
                #print(f"s(c) (soc) (i PVB,m 0,{T-1})=", round(env.compute_current_state()[("PVB", "soc")], 3), action)
                #print(current_state[('C', 'current_offtake_peak')], current_state[('C', 'offtake_peaks')])
                #print(current_state[('PVB', 'current_injection_peak')], current_state[('PVB', 'injection_peaks')])
                #print(action)
                #print(action[("PVB", "rec_export")])
                #print(env.compute_current_state()[('PVB', 'previous_electricity_production_metering_period_meter')], env.compute_current_state()[('PVB', 'electricity_production_metering_period_meter')])
                obs, cost, is_terminated, is_truncated, info = env.step(action)
                cost = info["costs"]["metering_period_cost"] + info["costs"].get("peak_period_cost", 0.0)
                #if time_it:
                    #print(f"Transition computed in {time() - t} seconds.")
                if full_trajectory:
                    #current_state = env._compute_current_observation_dict()
                    action = action
                    trajectory["observations"].append(current_obs)
                    trajectory["actions"].append(action)
                    trajectory["costs"].append(cost)
                
                
                #print(env.compute_current_state()[('C', 'current_offtake_peak')], env.compute_current_state()[('C', 'offtake_peaks')])
                #print(info["peak_period_costs"]["repartition_keys_sum_of_peaks_penalty_per_peak_period_cost_function"])
                cost_total += gamma_sequence[tstep]*cost
                undiscounted_cost_total += cost
                current_obs = obs
                #print("instant peak cost real", info["peak_period_costs"]["repartition_keys_sum_of_peaks_penalty_per_peak_period_cost_function"])
                #print(cost)
                #print("instant peak cost real", cost)

                if is_terminated or is_truncated:
                    if is_terminated:
                        raise BaseException(f"Policy broke constraints at timestep {tstep}, see warning message")
                    break

                current_t += time() - t2
                
            expected_return += [cost_total]
            undiscounted_expected_return += [undiscounted_cost_total]
            if full_trajectory:
                full_trajectories += [trajectory]
    else:
        
        from multiprocessing import Pool
        global_env = env
        global_policy = policy
        with Pool(num_cpus) as p:
            lst_map = [(gamma_sequence, T, time_it) for _ in range(n_samples)]
            expected_return_pairs = p.map(run_policy_sample, lst_map)
            expected_return = [pair[0] for pair in expected_return_pairs] 
            undiscounted_expected_return = [pair[1] for pair in expected_return_pairs] 
    if time_it:
        print(f"Time took to sample this policy {n_samples} times : {time() - t_global} times")
    expected_return_mean = np.mean(expected_return)
    undiscounted_expected_return_mean = np.mean(undiscounted_expected_return)
    #print(f"s(c) (soc) (i PVB,m 0,{T-1})=", round(env.compute_current_state()[("PVB", "soc")], 3))
    #print(env.compute_current_state()[("PVB", "soc")])
    #print(env.compute_current_state()[('C', 'current_offtake_peak')], env.compute_current_state()[('C', 'offtake_peaks')])
    base_return = (expected_return_mean, undiscounted_expected_return_mean)
    if return_std:
        expected_return_std = np.std(expected_return, ddof=1) if len(set(expected_return)) > 1 else 0
        undiscounted_expected_return_std = np.std(undiscounted_expected_return, ddof=1) if len(set(undiscounted_expected_return)) > 1 else 0
        base_return += (expected_return_std, undiscounted_expected_return_std)
    if full_trajectory and num_cpus == 1:
        base_return += (full_trajectories,)
    return base_return
