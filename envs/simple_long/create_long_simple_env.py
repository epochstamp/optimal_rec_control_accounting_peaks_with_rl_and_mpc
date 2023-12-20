from distributions.time_serie_additive.time_serie_beta_additive_noiser import TimeSerieBetaAdditiveNoiser
from distributions.time_serie_scaling_and_additive.time_serie_red_gaussian_scaling_and_additive_noiser import TimeSerieRedGaussianScalingAndAdditiveNoiser
from env.rec_env import RecEnv, InFeasibleActionProcess
from exceptions import MissingArgument
import numpy as np
from scipy.stats import norm, truncnorm
from base import IneqType
from utils.utils import epsilonify, merge_dicts
from itertools import product
from tsaug import TimeWarp, Crop, Quantize, Drift, Reverse, Convolve, Resize
import pandas as pd
import matplotlib.pyplot as plt
import operator

def create_rec_2(Delta_C=1, Delta_M=4, offtake_peak_cost=None, injection_peak_cost=None, T = 101, multiprice=False, **kwargs):
    members = ["C", "PVB"]
    if offtake_peak_cost is None:
        offtake_peak_cost = 1
    if injection_peak_cost is None:
        injection_peak_cost = 1
    if T is None:
        T = 101
    else:
        T = min(101, T)
    def create_soc_dynamics(Delta_C = 1.0, charge_efficiency=1.0, discharge_efficiency=1.0):
        def soc_dynamics(soc, s, e, u, op_add=operator.add, op_sub=operator.sub, op_mul=operator.mul, op_div=operator.truediv):
            return op_add(soc, op_mul(Delta_C, op_sub(op_mul(charge_efficiency, u[("PVB", "charge")]), op_div(u[("PVB", "discharge")], discharge_efficiency))))
        return soc_dynamics
    states_controllable_assets = {
        "PVB": {
            "soc": (0.33, 0.0, 1.0, create_soc_dynamics(Delta_C, charge_efficiency=1.0, discharge_efficiency=1.0))
        }
    }
    actions_controllable_assets = {
        "PVB": {
            "charge": (0.0, 0.05),
            "discharge": (0.0, 0.1)
        }
    }

    x = np.arange(-50, 51, 1)

    #create range of y-values that correspond to normal pdf with mean=0 and sd=1 
    PVB_profile = list(np.round((norm.pdf(x,2,2)*2 + norm.pdf(x,-38,1.5)*1.8 + norm.pdf(x,-16,1.7)*1.5 + norm.pdf(x,24,1.7)*1.3 + norm.pdf(x,30,1.7)*0.9)*2, 2))
    C_profile = list(np.round((norm.pdf(x,5,2)*2.5 + norm.pdf(x,-7,2)*1.5 + norm.pdf(x,-36,2)*1.9 + norm.pdf(x,-21,2)*1.5 + norm.pdf(x,27,2)*0.7 + norm.pdf(x,19,2)*1.1 + norm.pdf(x,44,1.7)*1.5)*2, 2))
    nb_metering_periods = int(np.ceil(T/Delta_M))
    exogenous_variable_members = {
        "C": {
            "consumption": C_profile
        },
        "PVB": {
            "production": PVB_profile
        }
    }
    if multiprice:
        exogenous_variable_members_buying_prices = {
            "C": [(0.12 if i%2 == 0 else 0.1) for i in range(nb_metering_periods)],
            "PVB": [(0.1 if i%2 == 0 else 0.12) for i in range(nb_metering_periods)]
        }

        exogenous_variable_members_selling_prices = {
            "PVB": [0.01]*nb_metering_periods,
            "C": [0.01]*nb_metering_periods
        }

    else:
        exogenous_variable_members_buying_prices = {
            "C": [0.1]*nb_metering_periods,
            "PVB": [0.12]*nb_metering_periods
        }

        exogenous_variable_members_selling_prices = {
            "PVB": [0.01]*nb_metering_periods,
            "C": [0.01]*nb_metering_periods
        }

    def create_soc_charge_dynamics_constraint(Delta_C = 1, charge_efficiency=1, discharge_efficiency=1):
        def soc_charge_dynamics(s, _, u, op_add=operator.add, op_sub=operator.sub, op_mul=operator.mul, op_div=operator.truediv, op_idx = lambda l, i: l[i]):
            value = op_add(s[("PVB", "soc")], op_mul(Delta_C, op_mul(charge_efficiency, u[("PVB", "charge")])))
            return value, 1, IneqType.LOWER_OR_EQUALS
        return soc_charge_dynamics

    def create_soc_discharge_dynamics_constraint(Delta_C = 1, charge_efficiency=1, discharge_efficiency=1):
        def soc_discharge_dynamics(s, _, u, op_add=operator.add, op_sub=operator.sub, op_mul=operator.mul, op_div=operator.truediv, op_idx = lambda l, i: l[i]):
            value = op_sub(s[("PVB", "soc")], op_mul(Delta_C, op_div(u[("PVB", "discharge")], discharge_efficiency)))
            return value, 0, IneqType.GREATER_OR_EQUALS
        return soc_discharge_dynamics
    
    def charge_discharge_mutex_constraint(s, _, u, **kwargs):
        return u[("PVB", "charge")], u[("PVB", "discharge")], IneqType.MUTEX

    def create_discharge_fees_cost_function(fees=1e-6):
        def discharge_fees(s, _, u, sprime):
            return fees * u[("PVB", "discharge")]
        return discharge_fees

    feasible_actions_controllable_assets = {
        "soc_charge_dynamics_member_B": create_soc_charge_dynamics_constraint(charge_efficiency=1.0),
        "soc_discharge_dynamics_member_B": create_soc_discharge_dynamics_constraint(discharge_efficiency=1.0),
        "charge_discharge_mutex": charge_discharge_mutex_constraint
    }

    def null(s, e, u=None, op_add=operator.add, op_sub=operator.sub, op_mul=operator.mul, op_div=operator.truediv, op_idx = lambda l, i: l[i]):
        return 0.0

    def exogenous_only_consumption(s, e, u=None, op_add=operator.add, op_sub=operator.sub, op_mul=operator.mul, op_div=operator.truediv, op_idx = lambda l, i: l[i]):
        return op_idx(e[("C", "consumption")], -1)

    def create_battery_discharge_and_exogenous_production(Delta_C = 1):
        def battery_discharge_production(s, e, u=None, op_add=operator.add, op_sub=operator.sub, op_mul=operator.mul, op_div=operator.truediv, op_idx = lambda l, i: l[i]):
            if u is None:
                raise MissingArgument("Action needs to be provided for the battery discharge production")
            return op_add(op_mul(u[("PVB", "discharge")], Delta_C), op_idx(e[("PVB", "production")], -1))
        return battery_discharge_production

    def create_battery_charge_consumption(Delta_C = 1):
        def battery_charge_consumption(s, e, u=None, op_add=operator.add, op_sub=operator.sub, op_mul=operator.mul, op_div=operator.truediv, op_idx = lambda l, i: l[i]):
            if u is None:
                raise MissingArgument("Action needs to be provided for the battery charge consumption")
            return op_mul(u[("PVB", "charge")], Delta_C)
        return battery_charge_consumption

    production_function = {
        "C" : null,
        "PVB" : create_battery_discharge_and_exogenous_production(Delta_C=Delta_C)
    }

    consumption_function = {
        "PVB" : create_battery_charge_consumption(Delta_C=Delta_C),
        "C" : exogenous_only_consumption
    }
    
    costs = {

    }

    rec_import_fees = round(max(
        abs(p[0] - p[1]) for p in product(*list(exogenous_variable_members_buying_prices.values()))
    ) + 0.01, 2)
    rec_export_fees = round(max(
        abs(p[0] - p[1]) for p in product(*list(exogenous_variable_members_selling_prices.values()))
    ) + 0.01, 2)
    infos = {
        "members_with_controllable_assets": ["PVB"],
        "battery_specs": {"PVB":
        [{
            "charge_as": "charge",
            "discharge_as": "discharge",
            "soc_as": "soc",
            "minsoc": 0,
            "maxsoc": 1,
            "discharge_efficiency": 1.0,
            "charge_efficiency": 1.0
        }]},
        "rec_import_fees": rec_import_fees,
        "rec_export_fees": rec_export_fees
    }

    return members, Delta_C, T, states_controllable_assets, exogenous_variable_members, exogenous_variable_members_buying_prices, exogenous_variable_members_selling_prices, costs, feasible_actions_controllable_assets, consumption_function, production_function, actions_controllable_assets, offtake_peak_cost, injection_peak_cost, offtake_peak_cost/3, injection_peak_cost/3, 1e-3, infos


def create_rec_2_stochastic(Delta_C=1, Delta_M=4, offtake_peak_cost=None, injection_peak_cost=None, T = 101, multiprice=False, **kwargs):
    members, Delta_C, T, states_controllable_assets, exogenous_variable_members, exogenous_variable_members_buying_prices, exogenous_variable_members_selling_prices, costs, feasible_actions_controllable_assets, consumption_function, production_function, actions_controllable_assets, current_offtake_peak_cost, current_injection_peak_cost, historical_offtake_peak_cost, historical_injection_peak_cost, precision_2, infos = (
        create_rec_2(
            Delta_C=Delta_C,
            Delta_M=Delta_M,
            current_offtake_peak_cost=offtake_peak_cost,
            current_injection_peak_cost=injection_peak_cost,
            T=T,
            multiprice=multiprice
        )
    )
    exogenous_variable_members["PVB"]["production"] = TimeSerieBetaAdditiveNoiser(exogenous_variable_members["PVB"]["production"], 0.5)
    exogenous_variable_members["C"]["consumption"] = TimeSerieBetaAdditiveNoiser(exogenous_variable_members["C"]["consumption"], 0.5)
    return members, Delta_C, T, states_controllable_assets, exogenous_variable_members, exogenous_variable_members_buying_prices, exogenous_variable_members_selling_prices, costs, feasible_actions_controllable_assets, consumption_function, production_function, actions_controllable_assets, current_offtake_peak_cost, current_injection_peak_cost, historical_offtake_peak_cost, historical_injection_peak_cost, precision_2, infos

def create_rec_2_red_stochastic(Delta_C=1, Delta_M=4, offtake_peak_cost=None, injection_peak_cost=None, T = 101, multiprice=False, r=0.5, scale=1.0, seed=None, **kwargs):
    members, Delta_C, T, states_controllable_assets, exogenous_variable_members, exogenous_variable_members_buying_prices, exogenous_variable_members_selling_prices, costs, feasible_actions_controllable_assets, consumption_function, production_function, actions_controllable_assets, current_offtake_peak_cost, current_injection_peak_cost, historical_offtake_peak_cost, historical_injection_peak_cost, precision_2, infos = (
        create_rec_2(
            Delta_C=Delta_C,
            Delta_M=Delta_M,
            current_offtake_peak_cost=offtake_peak_cost,
            current_injection_peak_cost=injection_peak_cost,
            T=T,
            multiprice=multiprice
        )
    )
    if seed is None:
        seed = np.random.randint(1, 1000000)
    old_prod = exogenous_variable_members["PVB"]["production"]
    exogenous_variable_members["PVB"]["production"] = TimeSerieRedGaussianScalingAndAdditiveNoiser(exogenous_variable_members["PVB"]["production"], max_error_scale=1.0, max_error_additive=1.0, r=r, scale=scale, max_error_scale_support=0.33, np_random_state=np.random.RandomState(np.random.randint(1, seed)))
    exogenous_variable_members["C"]["consumption"] = TimeSerieRedGaussianScalingAndAdditiveNoiser(exogenous_variable_members["C"]["consumption"], max_error_scale=1.0, max_error_additive=1.0, r=r, scale=scale, max_error_scale_support=0.33, np_random_state=np.random.RandomState(np.random.randint(1, seed)))
    """
    ts = np.asarray([exogenous_variable_members["PVB"]["production"].sample(101)])
    N = 200
    for _ in range(N-1):
        ts = np.vstack([ts, [exogenous_variable_members["PVB"]["production"].sample(101)]])
    ts_average = np.mean(ts, axis=0)
    ts_std_ddof1 = np.std(ts, axis=0, ddof=1)
    ts_percentile_95 = np.percentile(ts, 95, axis=0)
    ts_percentile_5 = np.percentile(ts, 5, axis=0)
    idxs = np.random.choice(list(range(N)), 4, replace=False)
    sample_ts = ts[idxs]
    dict_data = {
        "ts_average": ts_average,
        "ts_std": ts_std_ddof1,
        "ts_percentile_95": ts_percentile_95,
        "ts_percentile_5": ts_percentile_5,
        "ts_true" : np.asarray(old_prod)
    }
    for i,j in enumerate(idxs):
        dict_data[f"ts_sample_{j}"] = ts[i]
    df = pd.DataFrame.from_dict(dict_data)
    pd.options.plotting.backend = "plotly"
    fig = df.plot()
    fig.show()
    exit()
    """
    return members, Delta_C, T, states_controllable_assets, exogenous_variable_members, exogenous_variable_members_buying_prices, exogenous_variable_members_selling_prices, costs, feasible_actions_controllable_assets, consumption_function, production_function, actions_controllable_assets, current_offtake_peak_cost, current_injection_peak_cost, historical_offtake_peak_cost, historical_injection_peak_cost, precision_2, infos

def create_rec_2_red_stochastic_25(Delta_C=1, Delta_M=4, offtake_peak_cost=None, injection_peak_cost=None, T = 101, multiprice=False, **kwargs):
    return create_rec_2_red_stochastic(
        Delta_C=Delta_C,
        Delta_M=Delta_M,
        current_offtake_peak_cost=offtake_peak_cost,
        current_injection_peak_cost=injection_peak_cost,
        T=T,
        multiprice=multiprice,
        r=0.25
    )

def create_rec_2_red_stochastic_50(Delta_C=1, Delta_M=4, offtake_peak_cost=None, injection_peak_cost=None, T = 101, multiprice=False, **kwargs):
    return create_rec_2_red_stochastic(
        Delta_C=Delta_C,
        Delta_M=Delta_M,
        current_offtake_peak_cost=offtake_peak_cost,
        current_injection_peak_cost=injection_peak_cost,
        T=T,
        multiprice=multiprice,
        r=0.5
    )

def create_rec_2_red_stochastic_75(Delta_C=1, Delta_M=4, offtake_peak_cost=None, injection_peak_cost=None, T = 101, multiprice=False, **kwargs):
    return create_rec_2_red_stochastic(
        Delta_C=Delta_C,
        Delta_M=Delta_M,
        current_offtake_peak_cost=offtake_peak_cost,
        current_injection_peak_cost=injection_peak_cost,
        T=T,
        multiprice=multiprice,
        r=0.75
    )

def create_rec_2_noisy_provider(Delta_C=1, Delta_M=4, offtake_peak_cost=None, injection_peak_cost=None, T = 101, multiprice=False, stochastic=False, seed=18333, **kwargs):
    members, Delta_C, T, states_controllable_assets, exogenous_variable_members, exogenous_variable_members_buying_prices, exogenous_variable_members_selling_prices, costs, feasible_actions_controllable_assets, consumption_function, production_function, actions_controllable_assets, current_offtake_peak_cost, current_injection_peak_cost, historical_offtake_peak_cost, historical_injection_peak_cost, precision_2, infos = (
        create_rec_2(
            Delta_C=Delta_C,
            Delta_M=Delta_M,
            current_offtake_peak_cost=offtake_peak_cost,
            current_injection_peak_cost=injection_peak_cost,
            T=T,
            multiprice=multiprice
        )
    )
    #seed = 18333#np.random.randint(0,100000)
    np.random.seed(seed)
    #print(seed)
    #seed candidates: 58071
    #noise_volts = np.random.normal(mean_noise, np.sqrt(target_noise_watts), len(exogenous_variable_members["PVB"]["production"]))
    scale=10.0
    loc=0.0
    myclip_a=-0.33
    myclip_b=0.33
    a, b = (myclip_a - loc) / scale, (myclip_b - loc) / scale
    prod_temp = list(exogenous_variable_members["PVB"]["production"])
    #print(prod_temp)
    #print(max(prod_temp) - min(prod_temp))
    white_noise_scaler = (np.random.uniform(1 - myclip_b, 1 + myclip_b, size=len(prod_temp)))#1+truncnorm.rvs(a, b, size=len(prod_temp), loc=loc, scale=scale)      
    exogenous_variable_members["PVB"]["production"] = list(np.asarray(prod_temp) * white_noise_scaler)
    #list(random_walk(prod_temp, start_value=prod_temp[0], threshold=0.5, max_step_size=0.5*((max(prod_temp) - min(prod_temp))**2), min_value=0, max_value=max(prod_temp)*0.75, max_deviation_ratio=0.3))
    #max_elem = float(max(exogenous_variable_members["PVB"]["production"]))
    #prod_temp_2 = exogenous_variable_members["PVB"]["production"]
    #idx_max = prod_temp_2.index(max_elem)
    #exogenous_variable_members["PVB"]["production"][idx_max] = max_elem*0.5
    #for i in range(len(exogenous_variable_members["PVB"]["production"])):
    #    exogenous_variable_members["PVB"]["production"][i] *= np.random.uniform(0.5,0.8)
    #print(exogenous_variable_members["PVB"]["production"])
    #import pandas as pd
    #import plotly.express as px
    #df = pd.DataFrame.from_dict({
    #    "noised_production": exogenous_variable_members["PVB"]["production"],
    #    "true_production": prod_temp
    #})
    #pd.options.plotting.backend = "plotly"
    #fig = df.plot()
    #fig.show()
    #exit()
    
    #from tsaug.visualization import plot
    #fig, axes = plot(np.vstack([prod_temp, exogenous_variable_members["PVB"]["production"]]))
    #fig.suptitle("Perfect Foresight PVB Production vs Random error max 33%")
    #plt.show()
    

    return members, Delta_C, T, states_controllable_assets, exogenous_variable_members, exogenous_variable_members_buying_prices, exogenous_variable_members_selling_prices, costs, feasible_actions_controllable_assets, consumption_function, production_function, actions_controllable_assets, current_offtake_peak_cost, current_injection_peak_cost, historical_offtake_peak_cost, historical_injection_peak_cost, precision_2, infos