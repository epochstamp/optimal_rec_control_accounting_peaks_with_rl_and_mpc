from distributions.time_serie_scaling.time_serie_red_beta_scaling import TimeSerieRedBetaScaling
from distributions.time_serie_scaling_and_additive.time_serie_red_gaussian_scaling_and_additive_noiser import TimeSerieRedGaussianScalingAndAdditiveNoiser
from env.rec_env import RecEnv
from exceptions import MissingArgument
import numpy as np
from scipy.stats import norm
from base import IneqType
from utils.utils import chunks, epsilonify, flatten, merge_dicts
import pandas as pd
from experiment_scripts.generic.rec_generator import REC, RECMember
import json
import os
from pprint import pprint
from itertools import product
import operator
def create_big_rec(Delta_C=None, Delta_M=41, Delta_P=14, Delta_P_prime=1, offtake_peak_cost=None, injection_peak_cost=None, multiprice=True, surrogate=False, T = None, projector=None, disable_warnings=True, locally_minimize_repartition_keys=False, force_return_previous_costs=False, offset=0, freq_reduce=1, prod_mult=10, cons_mult=1, **kwargs):
    f = open(os.path.expanduser('~') + "/rec_paper_peaks_code/envs/big_rec_data/big_rec.json")
    rec_data_structure = json.load(f)
    from pprint import pprint
    f.close()
    if Delta_C is None:
        Delta_C = 0.25
    if T is None:
        T = 34561
    peak_cost = max([member["peak_month_price"] for member in rec_data_structure["members"]])
    if offtake_peak_cost is None:
        offtake_peak_cost = peak_cost
    if injection_peak_cost is None:
        injection_peak_cost = peak_cost
    T = min(len(rec_data_structure["members"][0]["demand"][offset:offset+T]), T)
    members_dicts = rec_data_structure["members"]
    for member in rec_data_structure["members"]:
        member["name"] = member["name"].replace(" ", "-")
    members = [member["name"] for member in rec_data_structure["members"]]
    exogenous_variable_members = {
        member["name"]:
        {
            "consumption": [round(max(abs(member["demand"][i]*cons_mult) - abs(member["base_injection"][i]*prod_mult), 0), 1) for i in range(offset, offset+T)],
            "production": [round(max(abs(member["base_injection"][i]*prod_mult) - abs(member["demand"][i]*cons_mult), 0), 1) for i in range(offset, offset+T)]
        } for member in members_dicts
    }
    dropped_members = set()
    for member in members_dicts:
        dropped_consumption=epsilonify(sum(exogenous_variable_members[member["name"]]["consumption"])) == 0.0 or max(exogenous_variable_members[member["name"]]["consumption"]) < 1.0
        if dropped_consumption:
            exogenous_variable_members[member["name"]].pop("consumption")
        dropped_production=epsilonify(sum(exogenous_variable_members[member["name"]]["production"])) == 0.0 or max(exogenous_variable_members[member["name"]]["production"]) < 1.0
        if dropped_production:
            exogenous_variable_members[member["name"]].pop("production")
        if dropped_consumption and dropped_production:
            dropped_members.add(member["name"])
            exogenous_variable_members.pop(member["name"])
    if len(dropped_members) > 0:
        members = [member for member in members if member not in dropped_members]
        members_dicts = [
            member_data for member_data in members_dicts if member_data["name"] not in dropped_members
        ]
    
    def create_exogenous_only_consumption(member, Delta_C = 0.25):
        def exogenous_only_consumption(s, e, u=None, op_add=operator.add, op_sub=operator.sub, op_mul=operator.mul, op_div=operator.truediv, op_idx = lambda l, i: l[i]):
            return op_idx(e.get((member, "consumption"), [0.0]), -1)
        return exogenous_only_consumption

    def create_exogenous_only_production(member, Delta_C = 0.25):
        def exogenous_only_production(s, e, u=None, op_add=operator.add, op_sub=operator.sub, op_mul=operator.mul, op_div=operator.truediv, op_idx = lambda l, i: l[i]):
            return op_idx(e.get((member, "production"), [0.0]), -1)
        return exogenous_only_production
    
    def create_battery_charge_with_exogenous_consumption(member, Delta_C = 0.25):
        def battery_charge_with_exogenous_consumption(s, e, u=None, op_add=operator.add, op_sub=operator.sub, op_mul=operator.mul, op_div=operator.truediv, op_idx = lambda l, i: l[i]):
            if u is None:
                raise MissingArgument("Action needs to be provided for the battery charge consumption")
            value = op_add(op_mul(u[(member, "charge")], Delta_C), op_idx(e.get((member, "consumption"), [0.0]), -1))
            return value
        return battery_charge_with_exogenous_consumption

    def create_battery_discharge_with_exogenous_production(member, Delta_C = 0.25):
        def battery_discharge_with_exogenous_production(s, e, u=None, op_add=operator.add, op_sub=operator.sub, op_mul=operator.mul, op_div=operator.truediv, op_idx = lambda l, i: l[i]):
            if u is None:
                raise MissingArgument("Action needs to be provided for the battery discharge production")
            value = op_add(op_mul(u[(member, "discharge")], Delta_C), op_idx(e.get((member, "production"), [0.0]), -1))
            return value
        return battery_discharge_with_exogenous_production

    
    can_have_battery = [
        (member["name"], member["battery_maximum_capacity"]) for member in members_dicts if member["battery_maximum_capacity"] > 0
    ]
    #pprint(can_have_battery)
    member_sorted_per_max_capacity = list(sorted(can_have_battery, key=lambda k: -k[1]))
    max_capacity_member_name = member_sorted_per_max_capacity[0][0]
    max_capacity_member_capacity = member_sorted_per_max_capacity[0][1]
    members_with_batteries = [max_capacity_member_name]
    consumption_function = {
        member["name"]: (
            create_exogenous_only_consumption(member["name"], Delta_C=Delta_C) if member["name"] not in members_with_batteries
            else create_battery_charge_with_exogenous_consumption(member["name"], Delta_C=Delta_C)
        ) for member in members_dicts
    }
    production_function = {
        member["name"]: (
            create_exogenous_only_production(member["name"], Delta_C=Delta_C) if member["name"] not in members_with_batteries
            else create_battery_discharge_with_exogenous_production(member["name"], Delta_C=Delta_C)
        ) for member in members_dicts
    }

    def create_soc_dynamics(member, Delta_C = 0.25, charge_efficiency=1.0, discharge_efficiency=1.0):
        def soc_dynamics(soc, s, e, u, op_add=operator.add, op_sub=operator.sub, op_mul=operator.mul, op_div=operator.truediv):
            return op_add(soc, op_mul(Delta_C, op_sub(op_mul(charge_efficiency, u[(member, "charge")]), op_div(u[(member, "discharge")], discharge_efficiency))))
        return soc_dynamics
    
    member_batteries_efficiencies = {
        member["name"]:
        {
            "charge_efficiency": round(max(min((max_capacity_member_capacity / member["battery_maximum_capacity"])*0.88, 1.0), 0.33), 2),
            "discharge_efficiency": round(max(min(max_capacity_member_capacity / member["battery_maximum_capacity"] - 0.13, 1.0)*0.82, 0.33), 2)
        } for member in members_dicts if member["name"] in members_with_batteries
    }
    states_controllable_assets = {
        member["name"]: {
            "soc": (member["battery_maximum_capacity"]/2.0, .0, member["battery_maximum_capacity"] , create_soc_dynamics(member["name"], Delta_C=Delta_C, charge_efficiency=member_batteries_efficiencies[member["name"]]["charge_efficiency"], discharge_efficiency=member_batteries_efficiencies[member["name"]]["discharge_efficiency"]))
        } for member in members_dicts if member["name"] in members_with_batteries
    }
    #print(states_controllable_assets)
    
    actions_controllable_assets = {
        member["name"]: {
            "charge": (0, member["battery_maximum_capacity"]/10.0),
            "discharge": (0, member["battery_maximum_capacity"]/5.0)
        } for member in members_dicts if member["name"] in members_with_batteries
    }
    #print(
    #    states_controllable_assets,
    #    actions_controllable_assets,
    #)
    #exit()
    def create_soc_charge_dynamics_constraint(member, Delta_C = 0.25, charge_efficiency=0.88):
        def soc_charge_dynamics(s, _, u):
            value = s[(member, "soc")] + Delta_C * charge_efficiency * u[(member, "charge")]
            return value, states_controllable_assets[member]["soc"][2], IneqType.LOWER_OR_EQUALS
        return soc_charge_dynamics

    def create_soc_discharge_dynamics_constraint(member, Delta_C = 0.25, discharge_efficiency=0.88):
        def soc_discharge_dynamics(s, _, u):
            value = s[(member, "soc")] - Delta_C * (u[(member, "discharge")] / discharge_efficiency)
            return value, 0, IneqType.GREATER_OR_EQUALS
        return soc_discharge_dynamics
    
    def create_soc_charge_discharge_dynamics_constraint(member, Delta_C = 0.25, charge_efficiency=0.88, discharge_efficiency=0.88):
        def soc_discharge_dynamics(s, _, u, op_add=operator.add, op_sub=operator.sub, op_mul=operator.mul, op_div=operator.truediv, op_idx = lambda l, i: l[i]):
            value = op_add(s[(member, "soc")], op_mul(Delta_C, op_sub(op_mul(charge_efficiency, u[(member, "charge")]), op_div(u[(member, "discharge")], discharge_efficiency))))
            return value, (0, states_controllable_assets[member]["soc"][2]), IneqType.BOUNDS
        return soc_discharge_dynamics
    
    def create_charge_discharge_mutex_constraint(member):
        def charge_discharge_mutex_constraint(s, _, u, **kwargs):
            return u[(member, "charge")], u[(member, "discharge")], IneqType.MUTEX
        return charge_discharge_mutex_constraint
    
    feasible_actions_controllable_assets = merge_dicts([

            {
                f"soc_charge_discharge_dynamics_member_{member['name']}": create_soc_charge_discharge_dynamics_constraint(member['name'], Delta_C=Delta_C, charge_efficiency=member_batteries_efficiencies[member["name"]]["charge_efficiency"], discharge_efficiency=member_batteries_efficiencies[member["name"]]["discharge_efficiency"]) for member in members_dicts if member["name"] in members_with_batteries
            },
            {
                f"charge_discharge_mutex_{member['name']}": create_charge_discharge_mutex_constraint(member['name']) for member in members_dicts if member["name"] in members_with_batteries
            }
        ]
    )
    
    
    exogenous_variable_members_buying_prices = dict()
    exogenous_variable_members_selling_prices = dict()
    nb_metering_periods = int(np.ceil(int(np.ceil(T/Delta_M))))
    price_duration_in_quarters = int(8760*4 // 12)
    price_duration_in_number_of_metering_periods = int(np.floor(price_duration_in_quarters/Delta_M))
    for member in rec_data_structure["members"]:
        grid_import_prices = list(chunks(list(dict.fromkeys(member["grid_import_price"])), 2))
        grid_export_prices = list(chunks(list(dict.fromkeys(member["grid_export_price"])), 2))
        H = int(np.ceil(T / len(member["grid_import_price"])))
        grid_import_prices = grid_import_prices[:int(np.ceil(T/price_duration_in_quarters))]
        Y = nb_metering_periods / len(grid_import_prices)
        monoprice_grid_import = float(np.mean(member["grid_import_price"]))
        monoprice_grid_export = float(np.mean(member["grid_export_price"]))
        exogenous_variable_members_buying_prices[member["name"]] = [
            round(((grid_import_prices[int(np.floor((i) / (price_duration_in_number_of_metering_periods)))][int(i%2)]) if multiprice else monoprice_grid_import), 6) for i in range(nb_metering_periods-1)
        ]
        exogenous_variable_members_selling_prices[member["name"]] = [
            round(((grid_export_prices[int(np.floor((i) / (price_duration_in_number_of_metering_periods)))][int(i%2)]) if multiprice else monoprice_grid_export), 6) for i in range(nb_metering_periods-1)
        ]

    def create_discharge_fees_cost_function(member, fees=1e-3, Delta_C=0.25):
        def discharge_fees(s, _, u, sprime):
            return fees * u[(member, "discharge")] * Delta_C
        return discharge_fees

    costs = {}
    
    product_buying_prices = list(flatten(exogenous_variable_members_buying_prices.values()))
    product_selling_prices = list(flatten(exogenous_variable_members_selling_prices.values()))
    rec_import_fees = max(product_buying_prices) - min(product_buying_prices) + 0.1
    rec_export_fees = max(product_selling_prices) - min(product_selling_prices) + 0.1

    infos = {
        "members_with_controllable_assets": members_with_batteries,
        "members_with_batteries": members_with_batteries,
        "battery_specs": {member_with_battery:
        [{
            "charge_as": "charge",
            "discharge_as": "discharge",
            "soc_as": "soc",
            "minsoc": states_controllable_assets[member_with_battery]["soc"][1],
            "maxsoc": states_controllable_assets[member_with_battery]["soc"][2],
            "discharge_efficiency": member_batteries_efficiencies[member_with_battery]["discharge_efficiency"],
            "charge_efficiency": member_batteries_efficiencies[member_with_battery]["charge_efficiency"],
        }] for member_with_battery in members_with_batteries},
        "rec_import_fees": rec_import_fees,
        "rec_export_fees": rec_export_fees
    }
    return members, Delta_C, T, states_controllable_assets, exogenous_variable_members, exogenous_variable_members_buying_prices, exogenous_variable_members_selling_prices, {}, feasible_actions_controllable_assets, consumption_function, production_function, actions_controllable_assets, offtake_peak_cost, injection_peak_cost, offtake_peak_cost/3, injection_peak_cost/3, 1e-3, infos

def create_big_rec_summer_end(Delta_C=None, Delta_M=4, Delta_P=120, Delta_P_prime=1, offtake_peak_cost=None, injection_peak_cost=None, multiprice=True, surrogate=False, T = None, projector=None, disable_warnings=True, locally_minimize_repartition_keys=False, force_return_previous_costs=False, **kwargs):
    return create_big_rec(
        Delta_C=Delta_C,
        Delta_M=Delta_M,
        Delta_P=Delta_P,
        Delta_P_prime=Delta_P_prime,
        offtake_peak_cost=offtake_peak_cost,
        injection_peak_cost=injection_peak_cost,
        multiprice=multiprice,
        T = 2881,
        disable_warnings=disable_warnings,
        offset=20440,
        **kwargs)

def rec_28_summer_end_data_biweekly_peak(Delta_C=None, Delta_M=4, Delta_P=360, Delta_P_prime=0, offtake_peak_cost=None, injection_peak_cost=None, multiprice=True, surrogate=False, T = None, projector=None, disable_warnings=True, locally_minimize_repartition_keys=False, force_return_previous_costs=False, **kwargs):
    return create_big_rec(
        Delta_C=Delta_C,
        Delta_M=Delta_M,
        Delta_P=Delta_P,
        Delta_P_prime=Delta_P_prime,
        offtake_peak_cost=offtake_peak_cost,
        injection_peak_cost=injection_peak_cost,
        multiprice=multiprice,
        T = 8641,
        disable_warnings=disable_warnings,
        offset=20440,
        **kwargs)
    

def create_big_rec_summer_begin(Delta_C=None, Delta_M=41, Delta_P=14, Delta_P_prime=1, offtake_peak_cost=None, injection_peak_cost=None, multiprice=True, surrogate=False, T = None, projector=None, disable_warnings=True, locally_minimize_repartition_keys=False, force_return_previous_costs=False, **kwargs):
    return create_big_rec(
        Delta_C=Delta_C,
        Delta_M=Delta_M,
        Delta_P=Delta_P,
        Delta_P_prime=Delta_P_prime,
        offtake_peak_cost=offtake_peak_cost,
        injection_peak_cost=injection_peak_cost,
        multiprice=multiprice,
        T = 2881,
        disable_warnings=disable_warnings,
        offset=20440 - 2881,
        **kwargs)

def create_rec_28_summer_end_red_stochastic(Delta_C=None, Delta_M=41, Delta_P=14, Delta_P_prime=1, offtake_peak_cost=None, injection_peak_cost=None, multiprice=True, surrogate=False, T = None, projector=None, disable_warnings=True, locally_minimize_repartition_keys=False, force_return_previous_costs=False, r=0.5, **kwargs):
    members, Delta_C, T, states_controllable_assets, exogenous_variable_members, exogenous_variable_members_buying_prices, exogenous_variable_members_selling_prices, costs, feasible_actions_controllable_assets, consumption_function, production_function, actions_controllable_assets, current_offtake_peak_cost, current_injection_peak_cost, historical_offtake_peak_cost, historical_injection_peak_cost, precision, infos = (
        create_big_rec_summer_begin(
            Delta_C=Delta_C,
            Delta_M=Delta_M,
            Delta_P=Delta_P,
            Delta_P_prime=Delta_P_prime,
            offtake_peak_cost=offtake_peak_cost,
            injection_peak_cost=injection_peak_cost,
            multiprice=multiprice
        )
    )
    for member in members:
        if "production" in exogenous_variable_members[member]:
            exogenous_variable_members[member]["production"] = TimeSerieRedGaussianScalingAndAdditiveNoiser(exogenous_variable_members[member]["production"], max_error_scale=1.0, max_error_additive=0.75, r=r)
        if "consumption" in exogenous_variable_members[member]:
            exogenous_variable_members[member]["consumption"] = TimeSerieRedGaussianScalingAndAdditiveNoiser(exogenous_variable_members[member]["consumption"], max_error_scale=1.0, max_error_additive=0.75, r=r)
    return members, Delta_C, T, states_controllable_assets, exogenous_variable_members, exogenous_variable_members_buying_prices, exogenous_variable_members_selling_prices, costs, feasible_actions_controllable_assets, consumption_function, production_function, actions_controllable_assets, current_offtake_peak_cost, current_injection_peak_cost, historical_offtake_peak_cost, historical_injection_peak_cost, precision, infos

def create_rec_28_summer_end_red_stochastic_25(Delta_C=None, Delta_M=41, Delta_P=14, Delta_P_prime=1, offtake_peak_cost=None, injection_peak_cost=None, multiprice=True, surrogate=False, T = None, projector=None, disable_warnings=True, locally_minimize_repartition_keys=False, force_return_previous_costs=False, **kwargs):
    return create_rec_28_summer_end_red_stochastic(
        Delta_C=Delta_C,
        Delta_M=Delta_M,
        Delta_P=Delta_P,
        Delta_P_prime=Delta_P_prime,
        offtake_peak_cost=offtake_peak_cost,
        injection_peak_cost=injection_peak_cost,
        multiprice=multiprice,
        r=0.25
    )

def create_rec_28_summer_end_red_stochastic_50(Delta_C=None, Delta_M=41, Delta_P=14, Delta_P_prime=1, offtake_peak_cost=None, injection_peak_cost=None, multiprice=True, surrogate=False, T = None, projector=None, disable_warnings=True, locally_minimize_repartition_keys=False, force_return_previous_costs=False, **kwargs):
    return create_rec_28_summer_end_red_stochastic(
        Delta_C=Delta_C,
        Delta_M=Delta_M,
        Delta_P=Delta_P,
        Delta_P_prime=Delta_P_prime,
        offtake_peak_cost=offtake_peak_cost,
        injection_peak_cost=injection_peak_cost,
        multiprice=multiprice,
        r=0.5
    )

def create_rec_28_summer_end_red_stochastic_75(Delta_C=None, Delta_M=41, Delta_P=14, Delta_P_prime=1, offtake_peak_cost=None, injection_peak_cost=None, multiprice=True, surrogate=False, T = None, projector=None, disable_warnings=True, locally_minimize_repartition_keys=False, force_return_previous_costs=False, **kwargs):
    return create_rec_28_summer_end_red_stochastic(
        Delta_C=Delta_C,
        Delta_M=Delta_M,
        Delta_P=Delta_P,
        Delta_P_prime=Delta_P_prime,
        offtake_peak_cost=offtake_peak_cost,
        injection_peak_cost=injection_peak_cost,
        multiprice=multiprice,
        r=0.75
    )


def create_short_horizon_rec_7_from_rec_28_summer_end(Delta_C=None, Delta_M=4, Delta_P=120, Delta_P_prime=1, offtake_peak_cost=None, injection_peak_cost=None, multiprice=True, surrogate=False, T = None, projector=None, disable_warnings=True, locally_minimize_repartition_keys=False, force_return_previous_costs=False, reduce_horizon_by_factor=1, **kwargs):
    T_7 = 721
    members_28, Delta_C_28, T, states_controllable_assets_28, exogenous_variable_members_28, exogenous_variable_members_buying_prices_28, exogenous_variable_members_selling_prices_28, costs_28, feasible_actions_controllable_assets_28, consumption_function_28, production_function_28, actions_controllable_assets_28, current_offtake_peak_cost_28, current_injection_peak_cost_28, historical_offtake_peak_cost_28, historical_injection_peak_cost_28, precision_28, infos_28 = create_big_rec(
        Delta_C=Delta_C,
        Delta_M=Delta_M,
        Delta_P=Delta_P,
        Delta_P_prime=Delta_P_prime,
        offtake_peak_cost=offtake_peak_cost,
        injection_peak_cost=injection_peak_cost,
        multiprice=multiprice,
        T = T_7,
        disable_warnings=disable_warnings,
        offset=20520,
        prod_mult=19,
        cons_mult=1,
        **kwargs
    )
    #members_28 = [m for m in members_28 if m != "New-Inductotherm"]
    members_7= list(infos_28["members_with_batteries"])[:1]
    members_28 = [m for m in members_28 if m not in members_7]
    exogenous_variable_members_28_numpy = {
        k:{k2:np.asarray(v)} for k,dk2 in exogenous_variable_members_28.items() for k2, v in dk2.items()
    }
    top_4_consumers = sorted(members_28, reverse=True, key=lambda k: np.sum(np.maximum(exogenous_variable_members_28_numpy[k].get("consumption", np.zeros(T_7)) - exogenous_variable_members_28_numpy[k].get("production", np.zeros(T_7)), 0.0)))[:4]
    members_28 = [m for m in members_28 if m not in top_4_consumers]
    top_2_producers = sorted(members_28, reverse=True, key=lambda k: np.sum(np.maximum(exogenous_variable_members_28_numpy[k].get("production", np.zeros(T_7)) - exogenous_variable_members_28_numpy[k].get("consumption", np.zeros(T_7)), 0.0)))[:2]
    members_7.extend(top_4_consumers)
    members_7.extend(top_2_producers)
    states_controllable_assets_7 = states_controllable_assets_28
    exogenous_variable_members_7 = {
        m:exogenous_variable_members_28[m] for m in members_7
    }
    exogenous_variable_members_buying_prices_7 = {
        m:exogenous_variable_members_buying_prices_28[m] for m in members_7
    }
    exogenous_variable_members_selling_prices_7 = {
        m:exogenous_variable_members_selling_prices_28[m] for m in members_7
    }
    consumption_function_7 = {
        m:consumption_function_28[m] for m in members_7
    }
    production_function_7 = {
        m:production_function_28[m] for m in members_7
    }
    production_function_7 = {
        m:production_function_28[m] for m in members_7
    }
    if reduce_horizon_by_factor > 1:
        T_7 = ((T_7-1)//reduce_horizon_by_factor)+1
        for member in members_7:
            for k in exogenous_variable_members_7[member].keys():
                chunked_exogenous_var = np.sum(np.asarray(np.array_split(exogenous_variable_members_7[member][k][1:], reduce_horizon_by_factor)), axis=0)
                chunked_exogenous_var[0] += exogenous_variable_members_7[member][k][0]
                exogenous_variable_members_7[member][k] = list(chunked_exogenous_var) + [0]
    exogenous_variable_members_7["NRB"].pop("consumption")
    """
    import pandas as pd
    import plotly.express as px
    d_prod = {
        "#".join((member, "production")): exogenous_variable_members_7[member]["production"] for member in members_7 if "production" in exogenous_variable_members_7[member]
    }
    d_cons = {
        "#".join((member, "consumption")): exogenous_variable_members_7[member]["consumption"] for member in members_7 if "consumption" in exogenous_variable_members_7[member]
    }
    d_data = {
        **d_prod, **d_cons
    }
    df = pd.DataFrame.from_dict(d_data)
    pd.options.plotting.backend = "plotly"
    fig = df.plot()
    fig.show()
    exit()
    """
    #from tsaug.visualization import plot
    #fig, axes = plot(np.vstack([prod_temp, exogenous_variable_members["PVB"]["production"]]))
    #fig.suptitle("Perfect Foresight PVB Production vs Random error max 33%")
    #plt.show()
    buying_price_product = np.asarray(list(exogenous_variable_members_buying_prices_7.values()))
    selling_price_product = np.asarray(list(exogenous_variable_members_selling_prices_7.values()))
    infos_28["rec_import_fees"] = (np.max(buying_price_product[:, 0]) - np.min(buying_price_product[:, 0]) + 0.01)
    infos_28["rec_export_fees"] = (np.max(selling_price_product[:, 0]) - np.min(selling_price_product[:, 0]) + 0.01)
    return members_7, Delta_C_28, T_7, states_controllable_assets_7, exogenous_variable_members_7, exogenous_variable_members_buying_prices_7, exogenous_variable_members_selling_prices_7, costs_28, feasible_actions_controllable_assets_28, consumption_function_7, production_function_7, actions_controllable_assets_28, current_offtake_peak_cost_28, current_injection_peak_cost_28, historical_offtake_peak_cost_28, historical_injection_peak_cost_28, precision_28, infos_28

def create_short_horizon_rec_2_from_rec_28_summer_end(Delta_C=None, Delta_M=4, Delta_P=120, Delta_P_prime=1, offtake_peak_cost=None, injection_peak_cost=None, multiprice=True, surrogate=False, T = None, projector=None, disable_warnings=True, locally_minimize_repartition_keys=False, force_return_previous_costs=False, reduce_horizon_by_factor=1, **kwargs):
    T_7 = 721
    members_28, Delta_C_28, T, states_controllable_assets_28, exogenous_variable_members_28, exogenous_variable_members_buying_prices_28, exogenous_variable_members_selling_prices_28, costs_28, feasible_actions_controllable_assets_28, consumption_function_28, production_function_28, actions_controllable_assets_28, current_offtake_peak_cost_28, current_injection_peak_cost_28, historical_offtake_peak_cost_28, historical_injection_peak_cost_28, precision_28, infos_28 = create_big_rec(
        Delta_C=Delta_C,
        Delta_M=Delta_M,
        Delta_P=Delta_P,
        Delta_P_prime=Delta_P_prime,
        offtake_peak_cost=offtake_peak_cost,
        injection_peak_cost=injection_peak_cost,
        multiprice=multiprice,
        T = T_7,
        disable_warnings=disable_warnings,
        offset=20640,
        **kwargs
    )
    members_28 = [m for m in members_28 if m != "New-Inductotherm"]
    members_2= list(infos_28["members_with_batteries"])[:1]
    top_2_producers = sorted(members_28, reverse=True, key=lambda k: sum(exogenous_variable_members_28[k].get("production", [0.0])) - sum(exogenous_variable_members_28[k].get("consumption", [0.0])))[:1]
    members_2.extend(top_2_producers)
    states_controllable_assets_2 = states_controllable_assets_28
    exogenous_variable_members_2 = {
        m:exogenous_variable_members_28[m] for m in members_2
    }
    exogenous_variable_members_buying_prices_2 = {
        m:exogenous_variable_members_buying_prices_28[m] for m in members_2
    }
    exogenous_variable_members_selling_prices_2 = {
        m:exogenous_variable_members_selling_prices_28[m] for m in members_2
    }
    consumption_function_2 = {
        m:consumption_function_28[m] for m in members_2
    }
    production_function_2 = {
        m:production_function_28[m] for m in members_2
    }
    
    if reduce_horizon_by_factor > 1:
        T_7 = ((T_7-1)//reduce_horizon_by_factor)+1
        for member in members_2:
            for k in exogenous_variable_members_2[member].keys():
                chunked_exogenous_var = np.sum(np.asarray(np.array_split(exogenous_variable_members_2[member][k][1:], reduce_horizon_by_factor)), axis=0)
                chunked_exogenous_var[0] += exogenous_variable_members_2[member][k][0]
                exogenous_variable_members_2[member][k] = list(chunked_exogenous_var) + [0]
    exogenous_variable_members_2["NRB"].pop("consumption")
    """
    from pyts.image import GramianAngularField
    import matplotlib.pyplot as plt
    exogenous_variable_members_2_condensed = {
       (member, type_exogenous):exogenous_variable_members_2[member][type_exogenous] for member in members_2 for type_exogenous in ("production", "consumption") if type_exogenous in exogenous_variable_members_2[member]
    }
    
    Xtrain = np.asarray(list(exogenous_variable_members_2_condensed.values())) # both N x T time series samples
    print(Xtrain)
    gaf = GramianAngularField(image_size=64)
    from time import time
    
    for i in range(Xtrain.shape[0]):
        t = time()
        im_train = gaf.fit_transform([Xtrain[i]])
        print(time() - t, "seconds")
        # plot one image
        print(np.min(im_train[0]), np.max(im_train[0]))
        im_train = im_train*0-0.5
        plt.imshow(im_train[0])
        plt.show()
        plt.close()
    exit()
    """
    
    """
    import pandas as pd
    import plotly.express as px
    d_prod = {
        "#".join((member, "production")): exogenous_variable_members_2[member]["production"] for member in members_2 if "production" in exogenous_variable_members_2[member]
    }
    d_cons = {
        "#".join((member, "consumption")): exogenous_variable_members_2[member]["consumption"] for member in members_2 if "consumption" in exogenous_variable_members_2[member]
    }
    d_data = {
        **d_prod, **d_cons
    }
    df = pd.DataFrame.from_dict(d_data)
    pd.options.plotting.backend = "plotly"
    fig = df.plot()
    fig.show()
    exit()
    """
    
    #from tsaug.visualization import plot
    #fig, axes = plot(np.vstack([prod_temp, exogenous_variable_members["PVB"]["production"]]))
    #fig.suptitle("Perfect Foresight PVB Production vs Random error max 33%")
    #plt.show()
    return members_2, Delta_C_28, T_7, states_controllable_assets_2, exogenous_variable_members_2, exogenous_variable_members_buying_prices_2, exogenous_variable_members_selling_prices_2, costs_28, feasible_actions_controllable_assets_28, consumption_function_2, production_function_2, actions_controllable_assets_28, current_offtake_peak_cost_28, current_injection_peak_cost_28, historical_offtake_peak_cost_28, historical_injection_peak_cost_28, precision_28, infos_28

def create_very_short_horizon_rec_2_from_rec_28_summer_end(Delta_C=None, Delta_M=4, Delta_P=120, Delta_P_prime=1, offtake_peak_cost=None, injection_peak_cost=None, multiprice=True, surrogate=False, T = None, projector=None, disable_warnings=True, locally_minimize_repartition_keys=False, force_return_previous_costs=False, **kwargs):
    return create_short_horizon_rec_2_from_rec_28_summer_end(
        Delta_C=1.0, Delta_M=Delta_M, Delta_P=Delta_P, Delta_P_prime=Delta_P_prime, offtake_peak_cost=offtake_peak_cost, injection_peak_cost=injection_peak_cost, multiprice=True, surrogate=surrogate, T = T, projector=projector, disable_warnings=disable_warnings, locally_minimize_repartition_keys=locally_minimize_repartition_keys, force_return_previous_costs=force_return_previous_costs, reduce_horizon_by_factor=8
    )

def create_very_short_horizon_rec_7_from_rec_28_summer_end(Delta_C=None, Delta_M=4, Delta_P=30, Delta_P_prime=1, offtake_peak_cost=None, injection_peak_cost=None, multiprice=True, surrogate=False, T = None, projector=None, disable_warnings=True, locally_minimize_repartition_keys=False, force_return_previous_costs=False, seed=None, **kwargs):
    return (
        create_short_horizon_rec_7_from_rec_28_summer_end(
            Delta_C=1.0,
            Delta_M=Delta_M,
            current_offtake_peak_cost=offtake_peak_cost,
            current_injection_peak_cost=injection_peak_cost,
            T=721,
            multiprice=multiprice,
            reduce_horizon_by_factor=4
        )
    )

def create_short_horizon_rec_7_from_rec_28_summer_end_red_stochastic(Delta_C=None, Delta_M=4, Delta_P=30, Delta_P_prime=1, offtake_peak_cost=None, injection_peak_cost=None, multiprice=True, surrogate=False, T = None, projector=None, disable_warnings=True, locally_minimize_repartition_keys=False, force_return_previous_costs=False, r=0.5, scale=1, max_error_scale=1.0, max_error_additive=1.0, max_error_scale_support=0.33, seed=None, **kwargs):
    members_7, Delta_C_28, T_7, states_controllable_assets_7, exogenous_variable_members_7, exogenous_variable_members_buying_prices_7, exogenous_variable_members_selling_prices_7, costs_28, feasible_actions_controllable_assets_28, consumption_function_7, production_function_7, actions_controllable_assets_28, current_offtake_peak_cost_28, current_injection_peak_cost_28, historical_offtake_peak_cost_28, historical_injection_peak_cost_28, precision_28, infos_28 = (
        create_short_horizon_rec_7_from_rec_28_summer_end(
            Delta_C=Delta_C,
            Delta_M=Delta_M,
            current_offtake_peak_cost=offtake_peak_cost,
            current_injection_peak_cost=injection_peak_cost,
            T=721,
            multiprice=multiprice
        )
    )
    if seed is None:
        seed = np.random.randint(1, 1000000)
    for member in members_7:
        if "production" in exogenous_variable_members_7[member]:
            exogenous_variable_members_7[member]["production"] = TimeSerieRedGaussianScalingAndAdditiveNoiser(exogenous_variable_members_7[member]["production"], max_error_scale=max_error_scale, max_error_additive=max_error_additive, r=r, scale=scale, max_error_scale_support=max_error_scale_support, np_random_state=np.random.RandomState(np.random.randint(1, seed)))
        if "consumption" in exogenous_variable_members_7[member]:
            exogenous_variable_members_7[member]["consumption"] = TimeSerieRedGaussianScalingAndAdditiveNoiser(exogenous_variable_members_7[member]["consumption"], max_error_scale=max_error_scale, max_error_additive=max_error_additive, r=r, scale=scale, max_error_scale_support=max_error_scale_support, np_random_state=np.random.RandomState(np.random.randint(1, seed)))
    return members_7, Delta_C_28, T_7, states_controllable_assets_7, exogenous_variable_members_7, exogenous_variable_members_buying_prices_7, exogenous_variable_members_selling_prices_7, costs_28, feasible_actions_controllable_assets_28, consumption_function_7, production_function_7, actions_controllable_assets_28, current_offtake_peak_cost_28, current_injection_peak_cost_28, historical_offtake_peak_cost_28, historical_injection_peak_cost_28, precision_28, infos_28