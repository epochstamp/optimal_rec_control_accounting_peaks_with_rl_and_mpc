from env.rec_env import RecEnv, InFeasibleActionProcess
from exceptions import MissingArgument
import numpy as np
from scipy.stats import norm, truncnorm
from base import IneqType
from utils.utils import epsilonify, merge_dicts

def create_long_simple_env_3(Delta_C=1, Delta_M=8, offtake_peak_cost=None, injection_peak_cost=None, T = 321, multiprice=False, **kwargs):
    members = ["C1", "C2", "PVB"]
    if offtake_peak_cost is None:
        offtake_peak_cost = 2
    if injection_peak_cost is None:
        injection_peak_cost = 2
    if T is None:
        T = 321
    else:
        T = min(321, T)

    min_soc = 0
    max_soc = 100
    init_soc = 60
    charge_efficiency = 0.8
    discharge_efficiency = 0.88
    min_charge, max_charge=0, 20
    min_discharge, max_discharge=0, 40
    
    def create_soc_dynamics(Delta_C = 1, charge_efficiency=1.0, discharge_efficiency=1.0):
        def soc_dynamics(soc, s, e, u):
            next_soc = soc + Delta_C * ((charge_efficiency * u[("PVB", "charge")]) - (u[("PVB", "discharge")] / discharge_efficiency))
            if type(next_soc) == float:
                next_soc = min(max(next_soc, min_soc), max_soc)
            return next_soc
        return soc_dynamics
    
    states_controllable_assets = {
        "PVB": {
            "soc": (init_soc, min_soc, max_soc, create_soc_dynamics(Delta_C, charge_efficiency=charge_efficiency, discharge_efficiency=discharge_efficiency))
        }
    }
    actions_controllable_assets = {
        "PVB": {
            "charge": (min_charge, max_charge),
            "discharge": (min_discharge, max_discharge)
        }
    }
    np.random.seed(448)
    x = np.arange(-51, 50, 1)

    #create range of y-values that correspond to normal pdf with mean=0 and sd=1 
    PVB_profile = ((norm.pdf(x,2,2)*2 + norm.pdf(x,-38,1.5)*1.8 + norm.pdf(x,-16,1.7)*1.5 + norm.pdf(x,24,1.7)*1.3 + norm.pdf(x,30,1.7)*0.9)*2)*100
    C_profile = ((norm.pdf(x,5,2)*2.5 + norm.pdf(x,-7,2)*1.5 + norm.pdf(x,-36,2)*1.9 + norm.pdf(x,-21,2)*1.5 + norm.pdf(x,27,2)*0.7 + norm.pdf(x,19,2)*1.1 + norm.pdf(x,44,1.7)*1.5)*2)*100
    PVB_profile = np.hstack([PVB_profile]*10)[:T]/1.3
    C_profile = np.hstack([C_profile] * 10)[:T]
    lower, upper, mu, sigma = 0, 2, 1, 1
    noise1=truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).rvs(T)/2
    noise2 = 1 - noise1
    PVB_profile=PVB_profile
    C_profile = C_profile
    C2_profile = C_profile*noise1#(1-noise)*C_profile*0.4
    C_profile = C_profile*noise2
    nb_metering_periods = int(np.ceil(T/Delta_M))
    exogenous_variable_members = {
        "C1": {
            "consumption": (list(C_profile)*4)[:T]
        },
        "C2": {
            "consumption": (list(C2_profile)*4)[:T]
        },
        "PVB": {
            "production": (list(PVB_profile)*4)[:T]
        }
    }
    if multiprice:
        exogenous_variable_members_buying_prices = {
            "C1": [float(np.random.uniform(0.02, 0.2)) for _ in range(nb_metering_periods)],
            "C2": [float(np.random.uniform(0.05, 0.25)) for _ in range(nb_metering_periods)],
            "PVB": [(2.0 if i%2 == 0 else 1.5) for i in range(nb_metering_periods)]
        }

    else:
        C1_price = float(np.random.uniform(0.02, 0.2))
        C2_price = float(np.random.uniform(0.05, 0.25))
        exogenous_variable_members_buying_prices = {
            "C1": [C1_price]*nb_metering_periods,
            "C2": [C2_price]*nb_metering_periods,
            "PVB": [2.0]*nb_metering_periods
        }

    exogenous_variable_members_selling_prices = {
        "PVB": [0.0]*nb_metering_periods,
        "C2": [0.0]*nb_metering_periods,
        "C1": [0.0]*nb_metering_periods
    }

    def create_soc_charge_dynamics_constraint(Delta_C = 1, charge_efficiency=1):
        def soc_charge_dynamics(s, _, u):
            value = s[("PVB", "soc")] + Delta_C * charge_efficiency * u[("PVB", "charge")]
            if type(value) == float:
                value = round(value, 4)
            return value, max_soc + 1e-4, IneqType.LOWER_OR_EQUALS
        return soc_charge_dynamics

    def create_soc_discharge_dynamics_constraint(Delta_C = 1, discharge_efficiency=1):
        def soc_discharge_dynamics(s, _, u):
            value = s[("PVB", "soc")] - Delta_C * (u[("PVB", "discharge")] / discharge_efficiency)
            if type(value) == float:
                value = round(value, 4)
            return value, min_soc-1e-4, IneqType.GREATER_OR_EQUALS
        return soc_discharge_dynamics

    def create_discharge_fees_cost_function(fees=10e-6):
        def discharge_fees(s, _, u, sprime):
            return fees * u[("PVB", "discharge")]
        return discharge_fees

    feasible_actions_controllable_assets = {
        "soc_charge_dynamics_member_B": create_soc_charge_dynamics_constraint(charge_efficiency=1.0),
        "soc_discharge_dynamics_member_B": create_soc_discharge_dynamics_constraint(discharge_efficiency=1.0),
    }

    def null(s, e, u=None):
        return 0.0

    def create_exogenous_only_consumption(member):
        def exogenous_only_consumption(s, e, u=None):
            return e[(member, "consumption")][-1]
        return exogenous_only_consumption

    def create_battery_discharge_and_exogenous_production(Delta_C = 1):
        def battery_discharge_production(s, e, u=None):
            if u is None:
                raise MissingArgument("Action needs to be provided for the battery discharge production")
            return u[("PVB", "discharge")] * Delta_C + e[("PVB", "production")][-1]
        return battery_discharge_production

    def create_battery_charge_consumption(Delta_C = 1):
        def battery_charge_consumption(s, e, u=None):
            if u is None:
                raise MissingArgument("Action needs to be provided for the battery charge consumption")
            return u[("PVB", "charge")] * Delta_C
        return battery_charge_consumption

    production_function = {
        "C1" : null,
        "C2": null,
        "PVB" : create_battery_discharge_and_exogenous_production(Delta_C=Delta_C)
    }

    consumption_function = {
        "PVB" : create_battery_charge_consumption(Delta_C=Delta_C),
        "C1" : create_exogenous_only_consumption("C1"),
        "C2": create_exogenous_only_consumption("C2")
    }

    return members, Delta_C, T, states_controllable_assets, exogenous_variable_members, exogenous_variable_members_buying_prices, exogenous_variable_members_selling_prices, {}, feasible_actions_controllable_assets, consumption_function, production_function, actions_controllable_assets, offtake_peak_cost, injection_peak_cost, offtake_peak_cost/3, injection_peak_cost/3