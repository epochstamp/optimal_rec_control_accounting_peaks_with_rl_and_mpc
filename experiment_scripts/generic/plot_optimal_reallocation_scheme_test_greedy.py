from env.global_bill_adaptative_optimiser import GlobalBillAdaptativeOptimiser
from envs import create_env
import os
import click
import numpy as np
from envs import create_env_fcts
from utils.utils import unique_consecutives_values, flatten
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import time
import plotly.io as pio
from itertools import product
pio.kaleido.scope.mathjax = None
from time import time
from utils.utils import epsilonify
    


@click.command()
@click.option('--n-members', "n_members", type=int, help='Number of members', default=2)
@click.option('--seed', type=int, default=None, help="random seed for generating meters and prices")
@click.option('--billing-period', "billing_period", type=int, default=2, help="billing period size")
def run_examples(n_members, seed, billing_period):
    if seed is None:
        seed = np.random.randint(1, 1000000)
    random_gen = np.random.RandomState(seed=seed)
    offtake_peak_cost = 1e-12
    injection_peak_cost = 1e-12
    Delta_M = 1
    Delta_P = billing_period
    Delta_C = 1
    members = list([i for i in range(n_members)])
    buying_prices = {
        (member, "buying_price"):np.maximum(0.2,np.round(np.asarray([0.1 + (i+1)*0.1*random_gen.random()]*billing_period), 6)) for i, member in enumerate(members) 
    }
    selling_prices = {
        (member, "selling_price"): np.round(np.asarray([0.001 + (i+1)*0.001*random_gen.random()]*billing_period), 6) for i, member in enumerate(members) 
    }
    
    
    product_buying_prices = np.vstack(list(flatten(buying_prices.values())))
    product_selling_prices = np.vstack(list(flatten(selling_prices.values())))
    #rec_import_fees = np.max(product_buying_prices) - np.min(product_buying_prices)
    #rec_export_fees = np.max(product_selling_prices) - np.min(product_selling_prices)
    rec_import_fees = max(
        abs(p[0] - p[1]) for p in product(*list(buying_prices.values()))
    ) + 0.001
    rec_export_fees = max(
        abs(p[0] - p[1]) for p in product(*list(selling_prices.values()))
    ) + 0.001
    print(rec_import_fees, rec_export_fees)
    #rec_import_fees = 0.105
    #rec_export_fees = 0.0
    print(rec_import_fees, rec_export_fees)
    optimal_reallocation_schemer = GlobalBillAdaptativeOptimiser(
        members,
        current_offtake_peak_cost=offtake_peak_cost,
        current_injection_peak_cost=injection_peak_cost,
        Delta_M=Delta_M,
        Delta_C=Delta_C,
        Delta_P=Delta_P,
        rec_import_fees=rec_import_fees,
        rec_export_fees=rec_export_fees,
        dpp_compile=False,
        activate_optim_no_peak_costs=False,
        force_optim_no_peak_costs=False

    )

    optimal_reallocation_schemer_force_greedy = GlobalBillAdaptativeOptimiser(
        members,
        current_offtake_peak_cost=offtake_peak_cost,
        current_injection_peak_cost=injection_peak_cost,
        Delta_M=Delta_M,
        Delta_C=Delta_C,
        Delta_P=Delta_P,
        rec_import_fees=rec_import_fees,
        rec_export_fees=rec_export_fees,
        dpp_compile=False,
        activate_optim_no_peak_costs=True,
        force_optim_no_peak_costs=True

    )
    
    optimal_reallocation_schemer.reset()
    optimal_reallocation_schemer_force_greedy.reset()
    consumption_meters = {
        (member, "consumption_meters"):np.round(random_gen.uniform(low=0, high=10000, size=billing_period), billing_period) for member in members
    }
    production_meters = {
        (member, "production_meters"):np.round(random_gen.uniform(low=0, high=10000, size=billing_period), billing_period) for member in members
    }
    state = {
        **consumption_meters,
        **production_meters
    }
    state["metering_period_counter"] = 1
    state["peak_period_counter"] = billing_period
    
    exogenous_prices = {
        **buying_prices,
        **selling_prices
    }
    t = time()
    print("Solving...")
    metering_period_expr, peak_period_cost, offtake_peaks, injection_peaks, rec_imports, rec_exports, grid_imports, grid_exports = (
        optimal_reallocation_schemer.optimise_global_bill(state, exogenous_prices, detailed_solution=True)
    )
    metering_period_expr_force_peak, peak_period_cost_force_peak, offtake_peaks_force_peak, injection_peaks_force_peak, rec_imports_force_peak, rec_exports_force_peak, grid_imports_force_peak, grid_exports_force_peak = (
        optimal_reallocation_schemer_force_greedy.optimise_global_bill(state, exogenous_prices, detailed_solution=True)
    )
    if True or abs(metering_period_expr - metering_period_expr_force_peak) >= 1e-6:
        print("Difference of result between forcing greedy and letting LP optim, diff:", abs(metering_period_expr - metering_period_expr_force_peak))
        print(metering_period_expr)
        print(metering_period_expr_force_peak)
        print("REC imports")
        print(rec_imports)
        print(rec_imports_force_peak)
        print("REC exports")
        print(rec_exports)
        print(rec_exports_force_peak)
    else:
        print("OK!")

if __name__ == "__main__":
    run_examples()