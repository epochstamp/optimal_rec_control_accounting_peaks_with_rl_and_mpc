from utils.utils import net_value

def number_of_time_steps_elapsed_in_peak_period(s_tau_delta_m, s_tau_delta_p, Delta_M=2, Delta_P=2):
    if s_tau_delta_p == Delta_P:
        return Delta_M*Delta_P
    elif s_tau_delta_m == Delta_M:
        return s_tau_delta_p * Delta_M
    else:
        return (s_tau_delta_p*Delta_M + s_tau_delta_m)

def no_prorata(current_timestep, max_timestep, s_tau_delta_m, s_tau_delta_p, Delta_M=2, Delta_P=2):
    return 1.0

def one_over_k(current_timestep, max_timestep, s_tau_delta_m, s_tau_delta_p, Delta_M=2, Delta_P=2):
    return 1.0/max_timestep

def elapsed_timesteps_in_peak_period(current_timestep, max_timestep, s_tau_delta_m, s_tau_delta_p, Delta_M=2, Delta_P=2):
    if s_tau_delta_p == Delta_P:
        return 1
    else:
        return number_of_time_steps_elapsed_in_peak_period(s_tau_delta_m, s_tau_delta_p, Delta_M=Delta_M, Delta_P=Delta_P)/(Delta_M*Delta_P)

def elapsed_metering_periods_in_peak_period(current_timestep, max_timestep, s_tau_delta_m, s_tau_delta_p, Delta_M=2, Delta_P=2):
    return ((s_tau_delta_p)/(Delta_P)) if s_tau_delta_p < Delta_P else 1.0

def shifted_elapsed_metering_periods_in_peak_period(current_timestep, max_timestep, s_tau_delta_m, s_tau_delta_p, Delta_M=2, Delta_P=2):
    return ((s_tau_delta_p+1)/(Delta_P)) if s_tau_delta_p < Delta_P else 1.0

def one_over_elapsed_timesteps_in_peak_period(current_timestep, max_timestep, s_tau_delta_m, s_tau_delta_p, Delta_M=2, Delta_P=2):
    if s_tau_delta_p == Delta_P:
        return 1
    else:
        return (1 / (number_of_time_steps_elapsed_in_peak_period(s_tau_delta_m, s_tau_delta_p, Delta_M=Delta_M, Delta_P=Delta_P))) if s_tau_delta_m>0 else 0

def one_over_elapsed_metering_period_in_peak_period(current_timestep, max_timestep, s_tau_delta_m, s_tau_delta_p, Delta_M=2, Delta_P=2):
    return (1.0/(s_tau_delta_p+1)) if s_tau_delta_p < Delta_P else 1.0

def one_minus_one_over_elapsed_timesteps_in_peak_period(current_timestep, max_timestep, s_tau_delta_m, s_tau_delta_p, Delta_M=2, Delta_P=2):
    return (
        (1 - (one_over_elapsed_timesteps_in_peak_period(current_timestep, max_timestep, s_tau_delta_m, s_tau_delta_p, Delta_M=Delta_M, Delta_P=Delta_P)))
        if s_tau_delta_p < Delta_P
        else 1.0
    )

def elapsed_timesteps_in_peak_period_with_K(current_timestep, max_timestep, s_tau_delta_m, s_tau_delta_p, Delta_M=2, Delta_P=2):
    nb_t = number_of_time_steps_elapsed_in_peak_period(s_tau_delta_m, s_tau_delta_p, Delta_M=Delta_M, Delta_P=Delta_P)
    return (
        (max((max_timestep - current_timestep+1)/(nb_t+1), elapsed_timesteps_in_peak_period(current_timestep, max_timestep, s_tau_delta_m, s_tau_delta_p, Delta_M=Delta_M, Delta_P=Delta_P)))
        if s_tau_delta_p < Delta_P
        else 1.0
    )

def one_over_elapsed_timesteps_in_peak_period_with_K(current_timestep, max_timestep, s_tau_delta_m, s_tau_delta_p, Delta_M=2, Delta_P=2):
    return (
        (max(1/(max_timestep-current_timestep+1), one_over_elapsed_timesteps_in_peak_period(current_timestep, max_timestep, s_tau_delta_m, s_tau_delta_p, Delta_M=Delta_M, Delta_P=Delta_P)))
        if s_tau_delta_p < Delta_P
        else 1.0
    )

def one_minus_one_over_elapsed_timesteps_in_peak_period_with_K(current_timestep, max_timestep, s_tau_delta_m, s_tau_delta_p, Delta_M=2, Delta_P=2):
    return (
        (max(1 - 1/(max_timestep-current_timestep+1), one_minus_one_over_elapsed_timesteps_in_peak_period(current_timestep, max_timestep, s_tau_delta_m, s_tau_delta_p, Delta_M=Delta_M, Delta_P=Delta_P)))
        if s_tau_delta_p < Delta_P
        else 1.0
    )

def one_minus_elapsed_timesteps_in_peak_period(current_timestep, max_timestep, s_tau_delta_m, s_tau_delta_p, Delta_M=2, Delta_P=2):
    if s_tau_delta_p == Delta_P:
        return 1.0
    else:
        return 1-elapsed_timesteps_in_peak_period(current_timestep, max_timestep, s_tau_delta_m, s_tau_delta_p, Delta_M=Delta_M, Delta_P=Delta_P)

surrogate_prorata_modes = {
    "no_prorata": no_prorata,
    "one_over_k": one_over_k,
    "elapsed_timesteps_in_peak_period": elapsed_timesteps_in_peak_period,
    "one_over_elapsed_timesteps_in_peak_period": one_over_elapsed_timesteps_in_peak_period,
    "one_over_elapsed_metering_period_in_peak_period": one_over_elapsed_metering_period_in_peak_period,
    "one_minus_one_over_elapsed_timesteps_in_peak_period": one_minus_one_over_elapsed_timesteps_in_peak_period,
    "elapsed_metering_periods_in_peak_period": elapsed_metering_periods_in_peak_period,
    "shifted_elapsed_metering_periods_in_peak_period": shifted_elapsed_metering_periods_in_peak_period,
    "elapsed_timesteps_in_peak_period_with_K": elapsed_timesteps_in_peak_period_with_K,
    "one_minus_elapsed_timesteps_in_peak_period": one_minus_elapsed_timesteps_in_peak_period,
    "one_over_elapsed_timesteps_in_peak_period_with_K": one_over_elapsed_timesteps_in_peak_period_with_K,
    "one_minus_one_over_elapsed_timesteps_in_peak_period_with_K": one_minus_one_over_elapsed_timesteps_in_peak_period_with_K
}

