from base.exogenous_provider import ExogenousProvider
from env.rec_env import RecEnv
from exogenous_providers.perfect_foresight_exogenous_provider import PerfectForesightExogenousProvider
from policies.model_predictive_control_policy_gurobi import ModelPredictiveControlPolicyGurobi
from policies.model_predictive_control_policy_cplex import ModelPredictiveControlPolicyCplex
import numpy as np



def create_perfect_foresight_exogenous_provider(rec_env:RecEnv):
    rec_env.reset()
    return PerfectForesightExogenousProvider(rec_env.observe_all_members_exogenous_variables(), rec_env.observe_all_raw_prices_exogenous_variables(), Delta_M=rec_env.Delta_M)

def create_peak_costs(rec_env:RecEnv):
    return {
        "current_offtake_peak_cost": rec_env.current_offtake_peak_cost,
        "current_injection_peak_cost": rec_env.current_injection_peak_cost,
        "historical_offtake_peak_cost": rec_env.historical_offtake_peak_cost,
        "historical_injection_peak_cost": rec_env.historical_injection_peak_cost
    }

def create_null_peak_costs(rec_env:RecEnv):
    return {
        "current_offtake_peak_cost": 0.0,
        "current_injection_peak_cost": 0.0,
        "historical_offtake_peak_cost": 0.0,
        "historical_injection_peak_cost": 0.0
    }

def create_quasi_null_peak_costs(rec_env:RecEnv):
    return {
        "current_offtake_peak_cost": 1e-12,
        "current_injection_peak_cost": 1e-12,
        "historical_offtake_peak_cost": 1e-12,
        "historical_injection_peak_cost": 1e-12
    }

mpc_policies = {
    "perfect_foresight_mpc_base": {
        "exogenous_provider": create_perfect_foresight_exogenous_provider,
        "n_samples": 1
    }
}

mpc_policies["perfect_foresight_mpc_base"]

mpc_policies["perfect_foresight_mpc_non_terminal"] = {
    **mpc_policies["perfect_foresight_mpc_base"],
    **{
        "get_peak_costs": create_peak_costs
    }
}

mpc_policies["perfect_foresight_mpc_commodity_non_terminal"] = {
    **mpc_policies["perfect_foresight_mpc_base"],
    **{
        "get_peak_costs": create_null_peak_costs
    }
}

mpc_policies["perfect_foresight_mpc_commodity_peak_force_non_terminal"] = {
    **mpc_policies["perfect_foresight_mpc_base"],
    **{
        "get_peak_costs": create_quasi_null_peak_costs
    }
}

mpc_policies["perfect_foresight_mpc_commodity"] = {
    **mpc_policies["perfect_foresight_mpc_commodity_non_terminal"],
    **{
        "force_last_time_step_to_global_bill": True
    }
}

mpc_policies["perfect_foresight_mpc_commodity_peak_force"] = {
    **mpc_policies["perfect_foresight_mpc_commodity_peak_force_non_terminal"],
    **{
        "force_last_time_step_to_global_bill": True
    }
}

mpc_policies["perfect_foresight_mpc_commodity_peak_force_optimized"] = {
    **mpc_policies["perfect_foresight_mpc_commodity_peak_force"],
    **{
        "disable_sos": True
    }
}

mpc_policies["perfect_foresight_mpc"] = {
    **mpc_policies["perfect_foresight_mpc_non_terminal"],
    **{
        "force_last_time_step_to_global_bill": True
    }
}

mpc_policies.pop("perfect_foresight_mpc_base")

def create_mpc_policy(rec_env: RecEnv, id_policy: str, K:int = 1, n_cpus:int = 1, small_penalty_control_actions:float = 0, net_consumption_production_mutex_before: int = np.inf, gamma_policy: float=1.0, rescaled_gamma_mode: str="no_rescale", solver="gurobi", solver_config="none", solver_verbose=False, solution_chained_optimisation=False, disable_env_ctrl_assets_constraints=False, rec_import_fees=0.0, rec_export_fees=0.0, exogenous_provider:ExogenousProvider = None, members_with_controllable_assets=[]):
    mpc_policies_kwargs = dict(mpc_policies[id_policy])
    if exogenous_provider is None:
        exogenous_provider = mpc_policies[id_policy]["exogenous_provider"](rec_env)
    mpc_policies_kwargs.pop("exogenous_provider")
    if "get_peak_costs" in mpc_policies_kwargs:
        mpc_policies_kwargs = {
            **mpc_policies_kwargs,
            **mpc_policies_kwargs["get_peak_costs"](rec_env)
        }
        mpc_policies_kwargs.pop("get_peak_costs")
    if solver == "gurobi":
        mpc_policy_cls = ModelPredictiveControlPolicyGurobi
    elif solver == "cplex":
        mpc_policy_cls = ModelPredictiveControlPolicyCplex
    elif solver == "mosek":
        from policies.model_predictive_control_policy_mosek import ModelPredictiveControlPolicyMosek
        mpc_policy_cls = ModelPredictiveControlPolicyMosek
    return mpc_policy_cls(
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
            max_length_samples=K,
            Delta_C=rec_env.Delta_C,
            Delta_M=rec_env.Delta_M,
            Delta_P_prime=rec_env.Delta_P_prime,
            Delta_P=rec_env.Delta_P,
            n_threads=n_cpus,
            small_penalty_control_actions=small_penalty_control_actions,
            net_consumption_production_mutex_before=net_consumption_production_mutex_before,
            gamma=gamma_policy,
            rescaled_gamma_mode=rescaled_gamma_mode,
            solver_config=solver_config,
            verbose=solver_verbose,
            solution_chained_optimisation=solution_chained_optimisation,
            disable_env_ctrl_assets_constraints=disable_env_ctrl_assets_constraints,
            rec_import_fees=rec_import_fees,
            rec_export_fees=rec_export_fees,
            members_with_controllable_assets=members_with_controllable_assets,
            **mpc_policies_kwargs
        )