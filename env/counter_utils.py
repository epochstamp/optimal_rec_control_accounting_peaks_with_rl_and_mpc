from typing import Dict, List, Tuple

def future_counters(tau_dm_state, tau_dp_state, duration: int = 1, Delta_M=1, Delta_P=1) -> Tuple[List[int], List[int]]:
    nb_timestep_elapsed_metering_periods = [tau_dm_state]
    nb_metering_periods_elapsed_peak_periods = [tau_dp_state]
    for i in range(duration):
        tau_metering_period = nb_timestep_elapsed_metering_periods[-1]
        tau_peak_period = nb_metering_periods_elapsed_peak_periods[-1]
        if tau_metering_period < Delta_M:
            nb_timestep_elapsed_metering_periods.append(tau_metering_period + 1)
        else:
            nb_timestep_elapsed_metering_periods.append(1)
        if tau_peak_period < Delta_P and tau_metering_period == Delta_M - 1:
            nb_metering_periods_elapsed_peak_periods.append(tau_peak_period + 1)
        elif tau_peak_period >= Delta_P:
            nb_metering_periods_elapsed_peak_periods.append(0)
        else:
            nb_metering_periods_elapsed_peak_periods.append(tau_peak_period)
    return (nb_timestep_elapsed_metering_periods[1:], nb_metering_periods_elapsed_peak_periods[1:])