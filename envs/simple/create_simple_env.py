from env.rec_env import RecEnv, InFeasibleActionProcess
from exceptions import MissingArgument
import numpy as np
from scipy.stats import norm
from base import IneqType
from utils.utils import epsilonify, merge_dicts

def create_simple_env(Delta_C=1, Delta_M=4, Delta_P=1, Delta_P_prime=1, offtake_peak_cost=None, injection_peak_cost=None, surrogate=False, T = 25, multiprice=False, locally_minimize_repartition_keys=False, disable_warnings=True, force_return_previous_costs=False, **kwargs):
    members = ["C", "PVB"]
    if offtake_peak_cost is None:
        offtake_peak_cost = 1
    if injection_peak_cost is None:
        injection_peak_cost = 1
    T = max(min(25, T), 1)
    def create_soc_dynamics(Delta_C = 1, charge_efficiency=1.0, discharge_efficiency=1.0):
        def soc_dynamics(soc, s, e, u):
            next_soc = soc + Delta_C * ((charge_efficiency * u[("PVB", "charge")]) - (u[("PVB", "discharge")] / discharge_efficiency))
            if type(next_soc) == float:
                next_soc = min(max(next_soc, 0.0), 1.0)
            return next_soc
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

    x = np.arange(-12, 13, 1)
    PVB_profile = list(norm.pdf(x,0,2)*3)
    C_profile = list(norm.pdf(x,5,2)*4 + norm.pdf(x,-7,2)*1.5)
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
            "C": [(1.0 if i%2 == 0 else 2.0) for i in range(nb_metering_periods)],
            "PVB": [(2.0 if i%2 == 0 else 1.5) for i in range(nb_metering_periods)]
        }

        exogenous_variable_members_selling_prices = {
            "PVB": [0.0]*nb_metering_periods,
            "C": [0.0]*nb_metering_periods
        }

    else:
        exogenous_variable_members_buying_prices = {
            "C": [1.0]*nb_metering_periods,
            "PVB": [2.0]*nb_metering_periods
        }

        exogenous_variable_members_selling_prices = {
            "PVB": [0.0]*nb_metering_periods,
            "C": [0.0]*nb_metering_periods
        }

    def create_soc_charge_dynamics_constraint(Delta_C = 1, charge_efficiency=1):
        def soc_charge_dynamics(s, _, u):
            value = s[("PVB", "soc")] + Delta_C * charge_efficiency * u[("PVB", "charge")]
            if type(value) == float:
                value = round(value, 4)
            return value, 1 + 1e-4, IneqType.LOWER_OR_EQUALS
        return soc_charge_dynamics

    def create_soc_discharge_dynamics_constraint(Delta_C = 1, discharge_efficiency=1):
        def soc_discharge_dynamics(s, _, u):
            value = s[("PVB", "soc")] - Delta_C * (u[("PVB", "discharge")] / discharge_efficiency)
            if type(value) == float:
                value = round(value, 4)
            return value, -1e-4, IneqType.GREATER_OR_EQUALS
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

    def exogenous_only_consumption(s, e, u=None):
        return e[("C", "consumption")][-1]

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
        "C" : null,
        "PVB" : create_battery_discharge_and_exogenous_production(Delta_C=Delta_C)
    }

    consumption_function = {
        "PVB" : create_battery_charge_consumption(Delta_C=Delta_C),
        "C" : exogenous_only_consumption
    }

    return members, Delta_C, T, states_controllable_assets, exogenous_variable_members, exogenous_variable_members_buying_prices, exogenous_variable_members_selling_prices, {}, feasible_actions_controllable_assets, consumption_function, production_function, actions_controllable_assets, offtake_peak_cost, injection_peak_cost, offtake_peak_cost/3, injection_peak_cost/3