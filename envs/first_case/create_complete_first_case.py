from copy import deepcopy
from env.rec_env import RecEnv, InFeasibleActionProcess
from exceptions import MissingArgument
import numpy as np
from scipy.stats import norm
from base import IneqType
from utils.utils import epsilonify, merge_dicts, flatten
import pandas as pd
from .first_env_data.profiles import profiles_C1, profiles_C2, profiles_C3, profiles_C4, profiles_P

def create_complete_first_case(Delta_C=None, Delta_M=4, Delta_P=1, Delta_P_prime=1, offtake_peak_cost=None, injection_peak_cost=None, multiprice=True, surrogate=False, T = None, projector=None, disable_warnings=True, locally_minimize_repartition_keys=False, force_return_previous_costs=False, **kwargs):
    members = ["C1", "C2", "C3", "C4", "PV", "B"]
    if offtake_peak_cost is None:
        offtake_peak_cost = 1
    if injection_peak_cost is None:
        injection_peak_cost = 1
    if Delta_C is None:
        Delta_C = 0.25
    if T is None:
        T = 2871
    max_soc = kwargs.get("max_soc", 300.0)
    initial_soc = kwargs.get("initial_soc", 160.0)
    max_charge = kwargs.get("max_charge", 176.0)
    max_discharge = kwargs.get("max_discharge", 352.0)
    charge_efficiency = kwargs.get("charge_efficiency", 0.88)
    discharge_efficiency = kwargs.get("discharge_efficiency", 0.88)
    def create_soc_dynamics(Delta_C = Delta_C, charge_efficiency=charge_efficiency, discharge_efficiency=discharge_efficiency):
        def soc_dynamics(soc, s, e, u):
            next_soc = soc + Delta_C * ((charge_efficiency * u[("B", "charge")]) - (u[("B", "discharge")] / discharge_efficiency))
            return next_soc
        return soc_dynamics
    states_controllable_assets = {
        "B": {
            "soc": (initial_soc, .0, max_soc, create_soc_dynamics(Delta_C))
        }
    }
    
    actions_controllable_assets = {
        "B": {
            "charge": (0.0, max_charge),
            "discharge": (0.0, max_discharge)
        }
    }
    
    PV_profile = kwargs.get("PV_profile", [float(np.round(x, 2)) for x in list(flatten(profiles_P))])
    C1_profile = kwargs.get("C1_profile", [float(np.round(x, 2)) for x in list(flatten(profiles_C1))])
    C2_profile = kwargs.get("C2_profile", [float(np.round(x, 2)) for x in list(flatten(profiles_C2))])
    C3_profile = kwargs.get("C3_profile", [float(np.round(x, 2)) for x in list(flatten(profiles_C3))])
    C4_profile = kwargs.get("C4_profile", [float(np.round(x, 2)) for x in list(flatten(profiles_C4))])
    if T is None:
        T = len(PV_profile) - 1
    else:
        T = max(min(len(PV_profile) - 1, T), 1)
    nb_metering_periods = int(np.ceil(int(np.ceil(T/Delta_M))))*2
    exogenous_variable_members = {
        "C1": {
            "consumption": C1_profile
        },
        "C2": {
            "consumption": C2_profile
        },
        "C3": {
            "consumption": C3_profile
        },
        "C4": {
            "consumption": C4_profile
        },
        "PV": {
            "production": PV_profile
        }
    }
    if multiprice:
        exogenous_variable_members_buying_prices = {
            'C1':[0.1653455 if (t % 24 >= 6 and t % 24 <= 21) else 0.1334302 for t in range(nb_metering_periods)],
            'C2':[0.1630755 if (t % 24 >= 6 and t % 24 <= 21) else 0.1295502 for t in range(nb_metering_periods)],
            'C3':[0.1608245 if (t % 24 >= 6 and t % 24 <= 21) else 0.1455692 for t in range(nb_metering_periods)],
            'C4':[0.2097361 if (t % 24 >= 6 and t % 24 <= 21) else 0.134957 for t in range(nb_metering_periods)],
            'PV':[0.325 if (t % 24 >= 6 and t % 24 <= 21) else 0.272 for t in range(nb_metering_periods)],
            'B':[0.335 if (t % 24 >= 6 and t % 24 <= 21) else 0.277 for t in range(nb_metering_periods)]
        }
        exogenous_variable_members_selling_prices = {
            'C1':[0 for t in range(nb_metering_periods)],
            'C2':[0 for t in range(nb_metering_periods)],
            'C3':[0 for t in range(nb_metering_periods)],
            'C4':[0 for t in range(nb_metering_periods)],
            'PV':[0.046 if (t % 24 >= 6 and t % 24 <= 21) else 0.028 for t in range(nb_metering_periods)],
            'B':[0.002 if (t % 24 >= 6 and t % 24 <= 21) else 0.001 for t in range(nb_metering_periods)]
        }
    else:
        exogenous_variable_members_buying_prices = {
            'C1':[(0.1653455+0.1334302)/2 for t in range(nb_metering_periods)],
            'C2':[(0.1630755+0.1295502)/2 for t in range(nb_metering_periods)],
            'C3':[(0.1608245+0.1455692)/2 for t in range(nb_metering_periods)],
            'C4':[(0.2097361+0.134957)/2 for t in range(nb_metering_periods)],
            'PV':[(0.325+ 0.272)/2 for t in range(nb_metering_periods)],
            'B':[(0.335+0.277)/2 for t in range(nb_metering_periods)]
        }
        exogenous_variable_members_selling_prices = {
            'C1':[0 for t in range(nb_metering_periods)],
            'C2':[0 for t in range(nb_metering_periods)],
            'C3':[0 for t in range(nb_metering_periods)],
            'C4':[0 for t in range(nb_metering_periods)],
            'PV':[(0.046+0.028)/2 for t in range(nb_metering_periods)],
            'B':[(0.002+0.001)/2 for t in range(nb_metering_periods)]
        }
    for member in members:
        exogenous_variable_members_buying_prices[member] = kwargs.get(f"buying_prices", {}).get(member, exogenous_variable_members_buying_prices[member])
        exogenous_variable_members_selling_prices[member] = kwargs.get(f"selling_prices", {}).get(member, exogenous_variable_members_selling_prices[member])

    def create_soc_charge_dynamics_constraint(Delta_C = Delta_C, charge_efficiency=charge_efficiency, discharge_efficiency=discharge_efficiency):
        def soc_charge_dynamics(s, _, u):
            value = s[("B", "soc")] + Delta_C * charge_efficiency * u[("B", "charge")]
            return value, states_controllable_assets["B"]["soc"][2], IneqType.LOWER_OR_EQUALS
        return soc_charge_dynamics

    def create_soc_discharge_dynamics_constraint(Delta_C = Delta_C, charge_efficiency=charge_efficiency, discharge_efficiency=discharge_efficiency):
        def soc_discharge_dynamics(s, _, u):
            value = s[("B", "soc")]  - Delta_C * (u[("B", "discharge")] / discharge_efficiency)
            return value, 0, IneqType.GREATER_OR_EQUALS
        return soc_discharge_dynamics
    
    def create_soc_charge_discharge_dynamics_constraint(Delta_C = Delta_C, charge_efficiency=charge_efficiency, discharge_efficiency=discharge_efficiency):
        def soc_charge_discharge_dynamics(s, _, u):
            value = s[("B", "soc")]  - Delta_C * (u[("B", "discharge")] / discharge_efficiency) + Delta_C * charge_efficiency * u[("B", "charge")]
            return value, (0, states_controllable_assets["B"]["soc"][2]), IneqType.BOUNDS
        return soc_charge_discharge_dynamics

    def create_discharge_fees_cost_function(fees=1e-6):
        def discharge_fees(s, _, u, sprime):
            return fees * u[("B", "discharge")] * Delta_C
        return discharge_fees
    
    def charge_discharge_mutex_constraint(s, _, u):
        return u[("B", "charge")], u[("B", "discharge")], IneqType.MUTEX

    feasible_actions_controllable_assets = {
        "soc_charge_discharge_dynamics_member_B": create_soc_charge_discharge_dynamics_constraint(),
        "soc_charge_discharge_mutex_B": charge_discharge_mutex_constraint
    }

    def null(s, e, u=None):
        return 0.0

    def create_exogenous_only_consumption(member, Delta_C = Delta_C):
        def exogenous_only_consumption(s, e, u=None):
            return e[(member, "consumption")][-1]*Delta_C
        return exogenous_only_consumption

    def create_exogenous_only_production(member, Delta_C = Delta_C):
        def exogenous_only_production(s, e, u=None):
            return e[(member, "production")][-1]*Delta_C
        return exogenous_only_production

    def create_battery_charge_consumption(Delta_C = Delta_C):
        def battery_charge_consumption(s, e, u=None):
            if u is None:
                raise MissingArgument("Action needs to be provided for the battery charge consumption")
            return u[("B", "charge")] * Delta_C
        return battery_charge_consumption
    
    

    def create_battery_discharge_production(Delta_C = Delta_C):
        def battery_discharge_production(s, e, u=None):
            if u is None:
                raise MissingArgument("Action needs to be provided for the battery discharge production")
            return u[("B", "discharge")] * Delta_C
        return battery_discharge_production

    production_function = merge_dicts([{
        member: null for member in members if "C" in member
    }, {"PV": create_exogenous_only_production("PV")}, {"B": create_battery_discharge_production()}])

    consumption_function = merge_dicts([{
        member: create_exogenous_only_consumption(member) for member in members if "C" in member
    }, {"PV": null}, {"B": create_battery_charge_consumption()}])

    #costs = {
    #    "B": [create_discharge_fees_cost_function(fees=kwargs.get("battery_fees", 0.0))]
    #}

    return members, Delta_C, T, states_controllable_assets, exogenous_variable_members, exogenous_variable_members_buying_prices, exogenous_variable_members_selling_prices, costs, feasible_actions_controllable_assets, consumption_function, production_function, actions_controllable_assets, offtake_peak_cost, injection_peak_cost, offtake_peak_cost/3, injection_peak_cost/3, kwargs.get("precision", 1e-3), {}

def rec_6_from_rec_28_data(Delta_C=None, Delta_M=4, Delta_P=1, Delta_P_prime=1, offtake_peak_cost=None, injection_peak_cost=None, multiprice=True, surrogate=False, T = None, projector=None, disable_warnings=True, locally_minimize_repartition_keys=False, force_return_previous_costs=False, **kwargs):
    from envs import create_big_rec, create_big_rec_summer_begin
    
    members_28, Delta_C_28, T_28, states_controllable_assets_28, exogenous_variable_members_28, exogenous_variable_members_buying_prices_28, exogenous_variable_members_selling_prices_28, costs_28, feasible_actions_controllable_assets_28, consumption_function_28, production_function_28, actions_controllable_assets_28, current_offtake_peak_cost_28, current_injection_peak_cost_28, historical_offtake_peak_cost_28, historical_injection_peak_cost_28, precision_28, infos_28 = create_big_rec_summer_begin(
        Delta_C=Delta_C,
        Delta_M=Delta_M,
        Delta_P=Delta_P,
        Delta_P_prime=Delta_P_prime,
        offtake_peak_cost=offtake_peak_cost,
        injection_peak_cost=injection_peak_cost,
        multiprice=multiprice,
        disable_warnings=disable_warnings
    )
    rng = np.random.default_rng(seed=123)
    members_to_sample = set(members_28).difference(set(infos_28["members_with_batteries"]))
    members_to_sample = sorted(list(members_28))
    exogenous_variable_members_6 = deepcopy(exogenous_variable_members_28)
    pure_consumers = [
        member for member in members_to_sample if epsilonify(sum(exogenous_variable_members_6[member]["production"]), epsilon=precision_28) == 0.0
    ]
    producers = sorted(list(set(members_to_sample).difference(pure_consumers)))
    pure_consumers_idxs = rng.choice(range(len(pure_consumers)), size=4, replace=False)
    
    pure_consumers = [pure_consumers[i] for i in pure_consumers_idxs]
    members = pure_consumers + [producers[0]] + infos_28["members_with_batteries"]


    exogenous_variable_members_6 = {
        member: exogenous_data for member, exogenous_data in exogenous_variable_members_6.items() if member in members
    }
    exogenous_variable_members_buying_prices_6 = {
        member: list(exogenous_data) for member, exogenous_data in exogenous_variable_members_buying_prices_28.items() if member in members
    }
    exogenous_variable_members_selling_prices_6 = {
        member: list(exogenous_data) for member, exogenous_data in exogenous_variable_members_selling_prices_28.items() if member in members
    }
    for member_with_battery in infos_28["members_with_batteries"]:
        exogenous_variable_members_6[member_with_battery]["consumption"] = [0.0] * len(exogenous_variable_members_6[infos_28["members_with_batteries"][0]]["consumption"])
        exogenous_variable_members_6[member_with_battery]["production"] = [0.0] * len(exogenous_variable_members_6[infos_28["members_with_batteries"][0]]["production"])
    total_prod = np.asarray([0.0] * len(exogenous_variable_members_6[member_with_battery]["production"]))
    for member_prod in producers:
        total_prod += (
            np.asarray(exogenous_variable_members_28[member_prod]["production"])
            - np.asarray(exogenous_variable_members_28[member_prod]["consumption"])
        )
    total_prod[total_prod < 0] = 0.0
    total_prod *= 8
    
    exogenous_variable_members_6[producers[0]]["production"] = list(total_prod)
    exogenous_variable_members_6[producers[0]]["consumption"] = [0.0] * len(total_prod)
    total_cons = np.asarray([0.0] * len(exogenous_variable_members_6[member_with_battery]["consumption"]))
    for member_cons in pure_consumers:
        total_cons += np.asarray(exogenous_variable_members_6[member_cons]["consumption"])
    #print(members)
    #print(sum(exogenous_variable_members_6[infos_28["members_with_batteries"][0]]["consumption"]))
    #print([v[0] for v in exogenous_variable_members_selling_prices_6.values()])
    consumption_function_6 = {
        member: consumption_function for member, consumption_function in consumption_function_28.items() if member in members
    }
    production_function_6 = {
        member: production_function for member, production_function in production_function_28.items() if member in members
    }
    #print(members, Delta_C_28, T_28, current_offtake_peak_cost_28, actions_controllable_assets_28, 1e-3)
    #exit()
    return members, Delta_C_28, T_28, states_controllable_assets_28, exogenous_variable_members_6, exogenous_variable_members_buying_prices_6, exogenous_variable_members_selling_prices_6, costs_28, feasible_actions_controllable_assets_28, consumption_function_6, production_function_6, actions_controllable_assets_28, current_offtake_peak_cost_28, current_injection_peak_cost_28, historical_offtake_peak_cost_28, historical_injection_peak_cost_28, 1e-3, infos_28

def rec_6_from_rec_28_data_hourly(Delta_C=1.0, Delta_M=4, Delta_P=360, Delta_P_prime=0, offtake_peak_cost=None, injection_peak_cost=None, multiprice=True, surrogate=False, T = None, projector=None, disable_warnings=True, locally_minimize_repartition_keys=False, force_return_previous_costs=False, **kwargs):
    pass