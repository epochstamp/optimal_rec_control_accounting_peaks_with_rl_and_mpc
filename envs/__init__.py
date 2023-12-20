from envs.first_case.create_complete_first_case import create_complete_first_case, rec_6_from_rec_28_data, rec_6_from_rec_28_data_hourly
from envs.simple.create_simple_env import create_simple_env
from envs.simple_long.create_long_simple_env import create_rec_2, create_rec_2_noisy_provider, create_rec_2_stochastic, create_rec_2_red_stochastic, create_rec_2_red_stochastic_25, create_rec_2_red_stochastic_50, create_rec_2_red_stochastic_75
from envs.simple_long.create_long_simple_env_3 import create_long_simple_env_3
from envs.create_big_rec import create_big_rec, create_big_rec_summer_begin, create_big_rec_summer_end, rec_28_summer_end_data_biweekly_peak, create_rec_28_summer_end_red_stochastic, create_short_horizon_rec_7_from_rec_28_summer_end, create_short_horizon_rec_7_from_rec_28_summer_end_red_stochastic, create_short_horizon_rec_2_from_rec_28_summer_end, create_very_short_horizon_rec_2_from_rec_28_summer_end, create_very_short_horizon_rec_7_from_rec_28_summer_end
import numpy as np
create_env_fcts = {
    "simple": create_simple_env,
    "rec_2": create_rec_2,
    "rec_2_stochastic": create_rec_2_stochastic,
    "rec_2_red_stochastic_25_1": (create_rec_2_red_stochastic, {'r': 0.25}),
    "rec_2_red_stochastic_50_1": (create_rec_2_red_stochastic, {'r': 0.5}),
    "rec_2_red_stochastic_75_1": (create_rec_2_red_stochastic, {'r': 0.75}),
    "rec_2_red_stochastic_85_1": (create_rec_2_red_stochastic, {'r': 0.85}),
    "rec_2_red_stochastic_95_1": (create_rec_2_red_stochastic, {'r': 0.95}),
    "rec_2_red_stochastic_25_05": (create_rec_2_red_stochastic, {'r': 0.25, 'scale': 0.5}),
    "rec_2_red_stochastic_50_05": (create_rec_2_red_stochastic, {'r': 0.5, 'scale': 0.5}),
    "rec_2_red_stochastic_75_05": (create_rec_2_red_stochastic, {'r': 0.75, 'scale': 0.5}),
    "rec_2_red_stochastic_85_05": (create_rec_2_red_stochastic, {'r': 0.85, 'scale': 0.5}),
    "rec_2_red_stochastic_95_05": (create_rec_2_red_stochastic, {'r': 0.95, 'scale': 0.5}),
    "rec_2_red_stochastic_50_3": (create_rec_2_red_stochastic, {'r': 0.5, 'scale': 3}),
    "rec_3": create_long_simple_env_3,
    "rec_6": create_complete_first_case,
    "rec_6_from_rec_28_data": rec_6_from_rec_28_data,
    "rec_6_from_rec_28_data_hourly": rec_6_from_rec_28_data_hourly,
    "short_horizon_rec_7_from_rec_28_summer_end": create_short_horizon_rec_7_from_rec_28_summer_end,
    "short_horizon_rec_7_from_rec_28_summer_end_red_stochastic_75_3": (create_short_horizon_rec_7_from_rec_28_summer_end_red_stochastic, {'r': 0.75, 'scale': 3}),
    "short_horizon_rec_7_from_rec_28_summer_end_red_stochastic_25_3": (create_short_horizon_rec_7_from_rec_28_summer_end_red_stochastic, {'r': 0.25, 'scale': 3}),
    "short_horizon_rec_7_from_rec_28_summer_end_red_stochastic_50_3": (create_short_horizon_rec_7_from_rec_28_summer_end_red_stochastic, {'r': 0.5, 'scale': 3, "max_error_scale":0.0}),
    "short_horizon_rec_7_from_rec_28_summer_end_red_stochastic_50_3_maxadditivenoise": (create_short_horizon_rec_7_from_rec_28_summer_end_red_stochastic, {'r': 0.5, 'scale': 3, "max_error_scale":0.0, "max_error_additive":4.0, "max_error_scale_support":4.0}),
    "short_horizon_rec_7_from_rec_28_summer_end_red_stochastic_33_6_maxadditivenoise": (create_short_horizon_rec_7_from_rec_28_summer_end_red_stochastic, {'r': 0.33, 'scale': 6, "max_error_scale":0.0, "max_error_additive":4.0, "max_error_scale_support":4.0}),
    "short_horizon_rec_7_from_rec_28_summer_end_red_stochastic_80_6_maxadditivenoise": (create_short_horizon_rec_7_from_rec_28_summer_end_red_stochastic, {'r': 0.8, 'scale': 6, "max_error_scale":0.0, "max_error_additive":4.0, "max_error_scale_support":4.0}),
    "short_horizon_rec_7_from_rec_28_summer_end_red_stochastic_80_3_maxadditivenoise": (create_short_horizon_rec_7_from_rec_28_summer_end_red_stochastic, {'r': 0.8, 'scale': 3, "max_error_scale":0.0, "max_error_additive":3.0, "max_error_scale_support":3.0}),
    "short_horizon_rec_7_from_rec_28_summer_end_red_stochastic_33_3": (create_short_horizon_rec_7_from_rec_28_summer_end_red_stochastic, {'r': 0.33, 'scale': 3}),
    "short_horizon_rec_7_from_rec_28_summer_end_red_stochastic_50_5": (create_short_horizon_rec_7_from_rec_28_summer_end_red_stochastic, {'r': 0.5, 'scale': 5}),
    "short_horizon_rec_7_from_rec_28_summer_end_red_stochastic_33_5": (create_short_horizon_rec_7_from_rec_28_summer_end_red_stochastic, {'r': 0.33, 'scale': 5}),
    "short_horizon_rec_7_from_rec_28_summer_end_red_stochastic_50_2": (create_short_horizon_rec_7_from_rec_28_summer_end_red_stochastic, {'r': 0.5, 'scale': 2}),
    "short_horizon_rec_7_from_rec_28_summer_end_red_stochastic_50_1": (create_short_horizon_rec_7_from_rec_28_summer_end_red_stochastic, {'r': 0.5, 'scale': 1, "max_error_scale":0.0, "max_error_additive":0.75, "max_error_scale_support":0.5}),
    "rec_28": create_big_rec,
    "rec_28_summer_begin": create_big_rec_summer_begin,
    "rec_28_summer_end": create_big_rec_summer_end,
    "rec_28_summer_end_data_biweekly_peak": rec_28_summer_end_data_biweekly_peak,
    "rec_28_summer_end_red_stochastic_25_1": (create_rec_28_summer_end_red_stochastic, {"r": 0.25}),
    "rec_28_summer_end_red_stochastic_50_1": (create_rec_28_summer_end_red_stochastic, {"r": 0.5}),
    "rec_28_summer_end_red_stochastic_75_1": (create_rec_28_summer_end_red_stochastic, {"r": 0.75}),
    "rec_28_summer_end_red_stochastic_85_1": (create_rec_28_summer_end_red_stochastic, {"r": 0.85}),
    "rec_28_summer_end_red_stochastic_95_1": (create_rec_28_summer_end_red_stochastic, {"r": 0.95}),
    "rec_28_summer_end_red_stochastic_25_05": (create_rec_28_summer_end_red_stochastic, {"r": 0.25, 'scale': 0.5}),
    "rec_28_summer_end_red_stochastic_50_05": (create_rec_28_summer_end_red_stochastic, {"r": 0.5, 'scale': 0.5}),
    "rec_28_summer_end_red_stochastic_75_05": (create_rec_28_summer_end_red_stochastic, {"r": 0.75, 'scale': 0.5}),
    "rec_28_summer_end_red_stochastic_85_05": (create_rec_28_summer_end_red_stochastic, {"r": 0.85, 'scale': 0.5}),
    "rec_28_summer_end_red_stochastic_95_05": (create_rec_28_summer_end_red_stochastic, {"r": 0.95, 'scale': 0.5}),
    "short_horizon_rec_2_from_rec_28_summer_end": create_short_horizon_rec_2_from_rec_28_summer_end,
    "very_short_horizon_rec_2_from_rec_28_summer_end": create_very_short_horizon_rec_2_from_rec_28_summer_end,
    "very_short_horizon_rec_7_from_rec_28_summer_end": create_very_short_horizon_rec_7_from_rec_28_summer_end
}

def create_env(id_env: str, projector=None, current_offtake_peak_cost=None, current_injection_peak_cost=None, historical_offtake_peak_cost=None, historical_injection_peak_cost=None, multiprice=False, Delta_M=2, Delta_P=1, Delta_P_prime=1, T=None, disable_warnings=True, global_bill_optimiser_enable_greedy_init=False, n_cpus_global_bill_optimiser=None, time_optim=False, ignore_ctrl_assets_constraints = False, seed=None, type_solver="mosek", force_optim_no_peak_costs=False, **kwargs):
    from env.rec_env import RecEnv
    create_env_fct = create_env_fcts[id_env]
    if type(create_env_fct) in (tuple,list):
        create_env_fct, kwarg = create_env_fct
    else:
        kwarg = dict()
    members, Delta_C, Tenv, states_controllable_assets, exogenous_variable_members, exogenous_variable_members_buying_prices, exogenous_variable_members_selling_prices, cost_function_controllable_assets, feasible_actions_controllable_assets, consumption_function, production_function, actions_controllable_assets, current_offtake_peak_cost_env, current_injection_peak_cost_env, historical_offtake_peak_cost_env, historical_injection_peak_cost_env, precision, infos = (
        create_env_fct(Delta_M=Delta_M, T=T, multiprice=multiprice, seed=seed, **{**kwargs, **kwarg})
    )
    if ignore_ctrl_assets_constraints:
        feasible_actions_controllable_assets = {}
    
    if T is None:
        T = Tenv
    else:
        T = min(Tenv, T)
    rec_env = RecEnv(
        members,
        states_controllable_assets,
        exogenous_variable_members,
        exogenous_variable_members_buying_prices,
        exogenous_variable_members_selling_prices,
        actions_controllable_assets,
        feasible_actions_controllable_assets,
        consumption_function,
        production_function,
        Delta_C=Delta_C,
        Delta_M=Delta_M,
        Delta_P=Delta_P,
        Delta_P_prime=Delta_P_prime,
        T=T,
        current_offtake_peak_cost=current_offtake_peak_cost_env if current_offtake_peak_cost is None else current_offtake_peak_cost,
        current_injection_peak_cost=current_injection_peak_cost_env if current_injection_peak_cost is None else current_injection_peak_cost,
        historical_offtake_peak_cost=historical_offtake_peak_cost_env if historical_offtake_peak_cost is None else historical_offtake_peak_cost,
        historical_injection_peak_cost=historical_injection_peak_cost_env if historical_injection_peak_cost is None else historical_injection_peak_cost, 
        cost_function_controllable_assets=cost_function_controllable_assets,
        disable_warnings=disable_warnings,
        env_name=kwargs.get("env_name", id_env),
        global_bill_optimiser_enable_greedy_init=global_bill_optimiser_enable_greedy_init,
        n_cpus_global_bill_optimiser=n_cpus_global_bill_optimiser,
        precision=precision,
        rec_import_fees=infos.get("rec_import_fees", 0.0),
        rec_export_fees=infos.get("rec_export_fees", 0.0),
        compute_global_bill_on_next_observ=kwargs.get("compute_global_bill_on_next_obs", False),
        type_solver=type_solver,
        force_optim_no_peak_costs=force_optim_no_peak_costs
    )
    rec_env.global_bill_adaptative_optimiser.n_cpus = n_cpus_global_bill_optimiser
    rec_env.global_bill_adaptative_optimiser.time_optim = time_optim
    rec_env.projector = projector
    return rec_env, infos