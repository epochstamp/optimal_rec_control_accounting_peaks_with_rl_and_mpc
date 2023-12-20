from env.rec_env import RecEnv
from envs import create_env
from exogenous_providers.pseudo_forecast_exogenous_provider import PseudoForecastExogenousProvider

def create_none(*args, **kwargs):
    return None

def create_rec_pseudo_exogenous_data_provider(env_ref_id, env_instance, alpha=0.99, fixed_sequence="input", alpha_fading=1.0):
    from envs import create_env_fcts
    rec_env, _ = create_env(env_ref_id, Delta_P_prime=0)
    rec_env.reset()
    return PseudoForecastExogenousProvider(exogenous_variables_members=rec_env.observe_all_members_exogenous_variables(), exogenous_prices=rec_env.observe_all_raw_prices_exogenous_variables(), stochastic_env=env_instance, alpha=alpha, fixed_sequence=fixed_sequence, alpha_fading=alpha_fading)
exogenous_data_provider_zoo = {
    "none": (create_none, None, None, None),
    "pseudo_forecast_rec_2_100_input": (create_rec_pseudo_exogenous_data_provider, "rec_2", 1.0, "input"),
    "pseudo_forecast_rec_2_099_input": (create_rec_pseudo_exogenous_data_provider, "rec_2", 0.99, "input"),
    "pseudo_forecast_rec_2_085_input": (create_rec_pseudo_exogenous_data_provider, "rec_2", 0.85, "input"),
    "pseudo_forecast_rec_2_050_input": (create_rec_pseudo_exogenous_data_provider, "rec_2", 0.5, "input"),
    "pseudo_forecast_rec_2_025_input": (create_rec_pseudo_exogenous_data_provider, "rec_2", 0.25, "input"),
    "pseudo_forecast_rec_2_001_input": (create_rec_pseudo_exogenous_data_provider, "rec_2", 0.01, "input"),
    "pseudo_forecast_rec_2_100_sample": (create_rec_pseudo_exogenous_data_provider, "rec_2", 1.0, "sample"),
    "pseudo_forecast_rec_2_099_sample": (create_rec_pseudo_exogenous_data_provider, "rec_2", 0.99, "sample"),
    "pseudo_forecast_rec_2_085_sample": (create_rec_pseudo_exogenous_data_provider, "rec_2", 0.85, "sample"),
    "pseudo_forecast_rec_2_050_sample": (create_rec_pseudo_exogenous_data_provider, "rec_2", 0.5, "sample"),
    "pseudo_forecast_rec_2_025_sample": (create_rec_pseudo_exogenous_data_provider, "rec_2", 0.25, "sample"),
    "pseudo_forecast_rec_2_001_sample": (create_rec_pseudo_exogenous_data_provider, "rec_2", 0.01, "sample"),
    "pseudo_forecast_rec_2_100_extreme_sample": (create_rec_pseudo_exogenous_data_provider, "rec_2", 1.0, "extreme_sample"),
    "pseudo_forecast_rec_2_099_extreme_sample": (create_rec_pseudo_exogenous_data_provider, "rec_2", 0.99, "extreme_sample"),
    "pseudo_forecast_rec_2_085_extreme_sample": (create_rec_pseudo_exogenous_data_provider, "rec_2", 0.85, "extreme_sample"),
    "pseudo_forecast_rec_2_050_extreme_sample": (create_rec_pseudo_exogenous_data_provider, "rec_2", 0.5, "extreme_sample"),
    "pseudo_forecast_rec_2_025_extreme_sample": (create_rec_pseudo_exogenous_data_provider, "rec_2", 0.25, "extreme_sample"),
    "pseudo_forecast_rec_2_001_extreme_sample": (create_rec_pseudo_exogenous_data_provider, "rec_2", 0.01, "extreme_sample"),
    "pseudo_forecast_rec_7_100_extreme_sample": (create_rec_pseudo_exogenous_data_provider, "short_horizon_rec_7_from_rec_28_summer_end", 1.0, "extreme_sample"),
    "pseudo_forecast_rec_7_099_extreme_sample": (create_rec_pseudo_exogenous_data_provider, "short_horizon_rec_7_from_rec_28_summer_end", 0.99, "extreme_sample"),
    "pseudo_forecast_rec_7_085_extreme_sample": (create_rec_pseudo_exogenous_data_provider, "short_horizon_rec_7_from_rec_28_summer_end", 0.85, "extreme_sample"),
    "pseudo_forecast_rec_7_050_extreme_sample": (create_rec_pseudo_exogenous_data_provider, "short_horizon_rec_7_from_rec_28_summer_end", 0.5, "extreme_sample"),
    "pseudo_forecast_rec_7_025_extreme_sample": (create_rec_pseudo_exogenous_data_provider, "short_horizon_rec_7_from_rec_28_summer_end", 0.25, "extreme_sample"),
    "pseudo_forecast_rec_7_001_extreme_sample": (create_rec_pseudo_exogenous_data_provider, "short_horizon_rec_7_from_rec_28_summer_end", 0.01, "extreme_sample"),
    "pseudo_forecast_rec_28_summer_end_100": (create_rec_pseudo_exogenous_data_provider, "rec_28_summer_end", 1.0),
    "pseudo_forecast_rec_28_summer_end_099": (create_rec_pseudo_exogenous_data_provider, "rec_28_summer_end", 0.99),
    "pseudo_forecast_rec_28_summer_end_085": (create_rec_pseudo_exogenous_data_provider, "rec_28_summer_end", 0.85),
    "pseudo_forecast_rec_28_summer_end_050": (create_rec_pseudo_exogenous_data_provider, "rec_28_summer_end", 0.5),
    "pseudo_forecast_rec_28_summer_end_025": (create_rec_pseudo_exogenous_data_provider, "rec_28_summer_end", 0.25),
    "pseudo_forecast_rec_28_summer_end_001": (create_rec_pseudo_exogenous_data_provider, "rec_28_summer_end", 0.01),
    "pseudo_forecast_rec_7_100_sample": (create_rec_pseudo_exogenous_data_provider, "short_horizon_rec_7_from_rec_28_summer_end", 1.0, "sample"),
    "pseudo_forecast_rec_7_100_input": (create_rec_pseudo_exogenous_data_provider, "short_horizon_rec_7_from_rec_28_summer_end", 1.0, "input"),
    "pseudo_forecast_rec_7_099_sample": (create_rec_pseudo_exogenous_data_provider, "short_horizon_rec_7_from_rec_28_summer_end", 0.99, "sample"),
    "pseudo_forecast_rec_7_0999_sample": (create_rec_pseudo_exogenous_data_provider, "short_horizon_rec_7_from_rec_28_summer_end", 0.999, "sample"),
    "pseudo_forecast_rec_7_095_sample": (create_rec_pseudo_exogenous_data_provider, "short_horizon_rec_7_from_rec_28_summer_end", 0.95, "sample"),
    "pseudo_forecast_rec_7_085_sample": (create_rec_pseudo_exogenous_data_provider, "short_horizon_rec_7_from_rec_28_summer_end", 0.85, "sample"),
    "pseudo_forecast_rec_7_050_sample": (create_rec_pseudo_exogenous_data_provider, "short_horizon_rec_7_from_rec_28_summer_end", 0.5, "sample"),
    "pseudo_forecast_rec_7_025_sample": (create_rec_pseudo_exogenous_data_provider, "short_horizon_rec_7_from_rec_28_summer_end", 0.25, "sample"),
    "pseudo_forecast_rec_7_001_sample": (create_rec_pseudo_exogenous_data_provider, "short_horizon_rec_7_from_rec_28_summer_end", 0.01, "sample"),
    "pseudo_forecast_rec_7_085_input": (create_rec_pseudo_exogenous_data_provider, "short_horizon_rec_7_from_rec_28_summer_end", 0.85, "input"),
    "pseudo_forecast_rec_7_099_input": (create_rec_pseudo_exogenous_data_provider, "short_horizon_rec_7_from_rec_28_summer_end", 0.99, "input"),
    "pseudo_forecast_rec_7_095_input": (create_rec_pseudo_exogenous_data_provider, "short_horizon_rec_7_from_rec_28_summer_end", 0.95, "input"),
    "pseudo_forecast_rec_7_050_input": (create_rec_pseudo_exogenous_data_provider, "short_horizon_rec_7_from_rec_28_summer_end", 0.5, "input"),
    "pseudo_forecast_rec_7_025_input": (create_rec_pseudo_exogenous_data_provider, "short_horizon_rec_7_from_rec_28_summer_end", 0.25, "input"),
    "pseudo_forecast_rec_7_001_input": (create_rec_pseudo_exogenous_data_provider, "short_horizon_rec_7_from_rec_28_summer_end", 0.01, "input"),
    "pseudo_forecast_rec_7_0999_input": (create_rec_pseudo_exogenous_data_provider, "short_horizon_rec_7_from_rec_28_summer_end", 0.999, "input"),
    "pseudo_forecast_rec_7_085_input_fading_075": (create_rec_pseudo_exogenous_data_provider, "short_horizon_rec_7_from_rec_28_summer_end", 0.85, "input", 0.75),
    "pseudo_forecast_rec_7_050_input_fading_075": (create_rec_pseudo_exogenous_data_provider, "short_horizon_rec_7_from_rec_28_summer_end", 0.5, "input", 0.75),
    "pseudo_forecast_rec_7_025_input_fading_075": (create_rec_pseudo_exogenous_data_provider, "short_horizon_rec_7_from_rec_28_summer_end", 0.25, "input", 0.75),
    "pseudo_forecast_rec_7_001_input_fading_075": (create_rec_pseudo_exogenous_data_provider, "short_horizon_rec_7_from_rec_28_summer_end", 0.01, "input", 0.75),
    "pseudo_forecast_rec_7_085_input_fading_050": (create_rec_pseudo_exogenous_data_provider, "short_horizon_rec_7_from_rec_28_summer_end", 0.85, "input", 0.5),
    "pseudo_forecast_rec_7_050_input_fading_050": (create_rec_pseudo_exogenous_data_provider, "short_horizon_rec_7_from_rec_28_summer_end", 0.5, "input", 0.5),
    "pseudo_forecast_rec_7_025_input_fading_050": (create_rec_pseudo_exogenous_data_provider, "short_horizon_rec_7_from_rec_28_summer_end", 0.25, "input", 0.5),
    "pseudo_forecast_rec_7_001_input_fading_050": (create_rec_pseudo_exogenous_data_provider, "short_horizon_rec_7_from_rec_28_summer_end", 0.01, "input", 0.5)
}

def create_exogenous_data_provider(id_exogenous_data_provider: str, env_instance:RecEnv):
    infos_exogenous_data_provider = exogenous_data_provider_zoo[id_exogenous_data_provider]
    if len(infos_exogenous_data_provider) == 4:
        create_fct, env_ref_id, alpha, fixed_sequence = infos_exogenous_data_provider
        alpha_fading=1.0
    else:
        create_fct, env_ref_id, alpha, fixed_sequence, alpha_fading = infos_exogenous_data_provider
    return create_fct(env_ref_id, env_instance, alpha=alpha, fixed_sequence=fixed_sequence, alpha_fading=alpha_fading)