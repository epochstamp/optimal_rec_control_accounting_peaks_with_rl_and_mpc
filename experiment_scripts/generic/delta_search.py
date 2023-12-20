#from env.counter_utils import future_counters
import click
from itertools import product
import numpy as np
import math
from time import sleep

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

@click.command()
@click.option('--T', "T", type=int, help='Time horizon', required=True)
@click.option('--deltas-start', "deltas_start", type=int, nargs=2, help='Delta_M and Delta_P search starts', default=[2, 1])
@click.option('--deltas', "deltas", type=int, nargs=2, help='Provide to avoid search Delta_M and Delta_P', default=[None, None])
def run_experiment(T, deltas, deltas_start):
    
    
    Delta_M_test, Delta_P_test = tuple(deltas)
    if Delta_M_test is not None and Delta_P_test is not None:
        future_counter_tau_dm, future_counter_tau_dp = future_counters(
            0, 0, duration=T-1, Delta_M=Delta_M_test, Delta_P=Delta_P_test
        )
        if future_counter_tau_dp[-1] == Delta_P_test:
            print(f"Delta_M={Delta_M_test}, Delta_P={Delta_P_test} ends in a peak period end")
    else:
        Delta_M_start, Delta_P_start = deltas_start
        L = math.ceil(math.sqrt(T)*2)#int(np.ceil(np.sqrt(T)*2))
        for Delta_M in range(Delta_M_start, L):
            future_counter_tau_dm, future_counter_tau_dp = future_counters(
                0, 0, duration=T-1, Delta_M=Delta_M, Delta_P=1
            )
            if future_counter_tau_dm[-1] == Delta_M:
                print(f"Delta_M={Delta_M} ends in a metering period end")
        for Delta_M, Delta_P in product(range(Delta_M_start, L), range(Delta_P_start, L*2)):
            future_counter_tau_dm, future_counter_tau_dp = future_counters(
                0, 0, duration=T-1, Delta_M=Delta_M, Delta_P=Delta_P
            )
            if future_counter_tau_dp[-1] == Delta_P:
                print(f"Delta_M={Delta_M}, Delta_P={Delta_P} ends in a peak period end")
    
    

                

if __name__ == '__main__':
    run_experiment()