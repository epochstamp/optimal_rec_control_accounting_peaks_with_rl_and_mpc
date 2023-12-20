from typing import Union
from env.rec_env import RecEnv
from env.rec_env_global_bill_wrapper import RecEnvGlobalBillWrapper
from exogenous_providers.perfect_foresight_exogenous_provider import PerfectForesightExogenousProvider
from experiment_scripts.rl.rec_env_rl_libs_gymnasium_gym_wrapper import RecEnvRlLibsGymnasiumGymWrapper
from experiment_scripts.rl.rec_env_rl_libs_wrapper import RecEnvRlLibsWrapper
from policies.model_predictive_control_policy_cplex import ModelPredictiveControlPolicyCplex
import numpy as np


def compute_optimal_z_score(env_source: Union[RecEnvRlLibsGymnasiumGymWrapper, RecEnvRlLibsWrapper], num_rollouts=1, gamma=1, include_obs=True, include_rew=True, return_actions=False, verbose=False):
    if env_source is not None and not env_source.eval_env:
        if verbose:
            print("Compute optimal value")
        number_of_resets = num_rollouts
        rec_env: Union[RecEnv, RecEnvGlobalBillWrapper, RecEnvRlLibsWrapper] = env_source.wrapped_rec_env
        observs = []
        rewards = []
        members_with_controllable_assets = env_source.members_with_controllable_assets
        while not isinstance(rec_env, RecEnv) and not isinstance(rec_env, RecEnvGlobalBillWrapper):
            rec_env = rec_env.wrapped_rec_env
        T = rec_env.T
        nb_time_steps_in_peak_period = rec_env.Delta_M * rec_env.Delta_P
        nb_peak_periods = (T-1)//nb_time_steps_in_peak_period
        if gamma < 1:
            gammas = [(gamma**nb_time_steps_in_peak_period)] * (nb_time_steps_in_peak_period+1)
            if nb_peak_periods > 1:
                for _ in range(nb_peak_periods-1):
                    gammas.extend([gammas[-1]*(gamma**nb_time_steps_in_peak_period)]*(nb_time_steps_in_peak_period))
            gammas = np.asarray(gammas, dtype=np.float32)
        else:
            gammas = [1]*rec_env.T
        for _ in range(number_of_resets):
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
                gamma=gamma,
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
            if return_actions:
                converted_actions = []
            for i, a in enumerate(actions + [last_action]):
                a_converted = env_source.space_convert_act(a, obs=previous_observ, reward=previous_reward)
                if type(env_source) == RecEnvRlLibsGymnasiumGymWrapper:
                    next_observ, reward, done_1, done_2, infos = env_source.step(a_converted)
                    done = done_1 or done_2
                else:
                    next_observ, reward, done, infos = env_source.step(a_converted)
                previous_observ = rec_env._compute_current_observation()
                previous_reward = {
                    "metering_period_cost": infos["costs"]["metering_period_cost"],
                    "peak_period_cost": infos["costs"]["peak_period_cost"],
                    "controllable_assets_cost": infos["costs"]["controllable_assets_cost"]
                }
                if type(next_observ) in (list, tuple):
                    next_observ = next_observ[0]
                if infos["is_peak_period_cost_triggered"] or infos["is_metering_period_cost_triggered"]:
                    rewards += [gammas[i]*reward]
                observs += [next_observ]
                if return_actions:
                    converted_actions += [a_converted]
                if done:
                    break     
        obs_z_score = None
        rew_z_score = None
        
        if include_obs:
            if type(observs[0]) == dict:
                z_score_mean_obs = dict()
                z_score_std_obs = dict()
                for key in observs[0].keys():
                    z_score_mean_obs[key] = np.mean(np.vstack([o[key] for o in observs]), axis=0)
                    z_score_std_obs[key] = np.std(np.vstack([o[key] for o in observs]), axis=0) + 1e-6
            else:
                z_score_mean_obs = np.mean(np.vstack(observs), axis=0)
                z_score_std_obs = np.std(np.vstack(observs), axis=0) + 1e-6
            obs_z_score = (z_score_mean_obs, z_score_std_obs)
        if include_rew:
            z_score_mean_rew = np.mean(np.vstack(rewards))
            z_score_std_rew = np.std(np.vstack(rewards)) + 1e-6
            rew_z_score = (z_score_mean_rew, z_score_std_rew)
        returning = (obs_z_score, rew_z_score)
        if return_actions:
            returning += (converted_actions, )
        if verbose:
            print("Optimal value is", sum(rewards), rew_z_score)
        return returning