from base import Policy, ExogenousProvider, IneqType
from typing import Dict, Any, Callable, Tuple, List, Optional, Union
import numpy as np
from docplex.mp.model import Model
from docplex.mp.linear import Var
from gym.spaces import Dict as DictSpace
from itertools import product, chain
from operator import le, ge, eq
from docplex.util.status import JobSolveStatus
from env.counter_utils import future_counters
from exceptions import InfeasiblePolicy
from utils.utils import epsilonify, merge_dicts, normalize_bounds, rindex, roundify, flatten, chunks, split_list, split_list_by_number_np
from env.peaks_utils import elapsed_metering_periods_in_peak_period, elapsed_timesteps_in_peak_period, number_of_time_steps_elapsed_in_peak_period
import random
from uuid import uuid4
from time import time
from experiment_scripts.mpc.cplex_params_zoo import cplex_params_dict
from docplex.mp.constr import IndicatorConstraint

M = 10000
EPSILON = 10e-6

def create_sos1_pair_operator(model: Model):
    def sos1_pair(a, b):
        return model.add_sos1([a, b])
    return sos1_pair

def create_indicator_pair_operator(model: Model):
    def ind_pair(a, b):
        bin = model.binary_var()
        return model.add_indicator_constraints([
            IndicatorConstraint(model, bin, a == 0, active_value=0),
            IndicatorConstraint(model, bin, b == 0, active_value=1)
        ])
    return ind_pair

class ModelPredictiveControlPolicy(Policy):

    def __init__(self,
                 members: List[str],
                 controllable_assets_state_space: DictSpace,
                 controllable_assets_action_space: DictSpace,
                 constraints_controllable_assets: Dict[str, Callable[[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]], Tuple[Any, float, IneqType]]],
                 consumption_function: Dict[str, Callable[[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float], Optional[Dict[Tuple[str, str], float]]], float]],
                 production_function: Dict[str, Callable[[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float], Optional[Dict[Tuple[str, str], float]]], float]],
                 dynamics_controllable_assets: Dict[Tuple[str, str], Callable[[Union[float, Var], Dict[Tuple[str, str], float], Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]], Dict[Tuple[str, str], float]]],
                 exogenous_provider: ExogenousProvider,
                 cost_functions_controllable_assets: Dict[str, List[Callable[[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float], Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]], float]]] = dict(), 
                 T=1,
                 n_samples=1,
                 max_length_samples=1,
                 Delta_C: float= 1.0,
                 Delta_M: int = 1,
                 Delta_P: int = 1,
                 Delta_P_prime: int = 0,
                 current_offtake_peak_cost: float = 0,
                 current_injection_peak_cost: float = 0,
                 historical_offtake_peak_cost: float = 0,
                 historical_injection_peak_cost: float = 0,
                 force_last_time_step_to_global_bill=False,
                 verbose=False,
                 net_consumption_production_mutex_before=np.inf,
                 optimal_action_population_size=1,
                 n_threads=None,
                 small_penalty_control_actions=0.0,
                 rec_import_export_mutex=True,
                 gamma=1.0,
                 rescaled_gamma_mode="no_rescale"):
        super().__init__(
            members,
            controllable_assets_state_space,
            controllable_assets_action_space,
            constraints_controllable_assets,
            consumption_function,
            production_function
        )
        if max_length_samples == 0:
            raise BaseException("The performance of this policy would be equivalent to NoActionPolicy even if last timestep is forced to the end of a metering/peak period")
        
        self._members = members
        self._exogenous_provider = exogenous_provider
        self._dynamics_controllable_assets = dynamics_controllable_assets
        self._consumption_function = consumption_function
        self._production_function = production_function
        self._cost_functions_controllable_assets = cost_functions_controllable_assets
        self._n_samples = n_samples
        self._max_length_samples = max_length_samples
        self._Delta_C = Delta_C
        self._Delta_P = Delta_P
        self._Delta_P_prime = Delta_P_prime
        self._Delta_M = Delta_M
        self._current_offtake_peak_cost = current_offtake_peak_cost
        self._current_injection_peak_cost = current_injection_peak_cost
        self._historical_offtake_peak_cost = historical_offtake_peak_cost
        self._historical_injection_peak_cost = historical_injection_peak_cost
        self._involve_current_peaks = current_offtake_peak_cost > 0 or current_injection_peak_cost > 0
        self._involve_historical_peaks = self._Delta_P_prime > 0 and (historical_offtake_peak_cost > 0 or historical_injection_peak_cost > 0)
        self._involve_peaks = self._involve_current_peaks or self._involve_historical_peaks
        self._force_last_time_step_to_global_bill = force_last_time_step_to_global_bill
        self._T = T
        self._verbose = verbose
        self._n_threads = n_threads
        self._optimal_action_population_size = optimal_action_population_size
        self._small_penalty_control_actions = small_penalty_control_actions
        self._net_consumption_production_mutex_before = net_consumption_production_mutex_before
        self._rec_import_export_mutex = rec_import_export_mutex
        self._rescaled_gamma_mode = rescaled_gamma_mode
        if gamma == 1.0:
            self._gammas = [1.0]*T
        else:
            self._gammas = [gamma**(i+1) for i in range(T)]
        self.reset()

    def _create_variable_name_pattern(self, variable_letter, reversed_key=True):
        def variable_name_pattern(variable_key):
            if type(variable_key) == str or len(variable_key) == 1:
                subscript_tuple = ",".join(
                    (f"i {variable_key if type(variable_key) == str else variable_key[0]}", "m 0", "t 0")
                )
                variable_name=variable_key[0]
            elif len(variable_key) == 2:
                subscript_tuple = ",".join(
                    (f"i {variable_key[1 if reversed_key else 0]}", "m 0", "t 0")
                )
                variable_name = variable_key[(0 if reversed_key else 1)]
            else:
                if len(variable_key) == 3:
                    
                    if type(variable_key[0]) == str:
                        variable_name, member, timestep = variable_key
                        n_sample = 0
                        timestep = -(timestep+1)
                    else:
                        variable_id, n_sample, timestep = variable_key
                        member, variable_name = variable_id
                else:
                    variable_name, member, n_sample, timestep = variable_key
                subscript_tuple = ",".join(
                    (f"i {member}", f"m {n_sample}", f"t {timestep+1}")
                )
            
            variable_full_name = f"{variable_letter} ({variable_name}) ({subscript_tuple})"
            return variable_full_name
        return variable_name_pattern


    def _build_planning_model(self, state: Dict[Union[str, Tuple[str, str]], Any], exogenous_variable_members: Dict[Tuple[str, str], List[float]], exogenous_prices: Dict[Tuple[str, str], List[float]], full_sequence_of_actions=False) -> None:
        
        initial_counters = {
            "metering_period_counter": state["metering_period_counter"]
        }
        if self._involve_peaks:
            initial_counters["peak_period_counter"] = state["peak_period_counter"]

        current_timestep = len(list(exogenous_variable_members.values())[0])
        length_samples = min(self._max_length_samples, self._T - current_timestep)
        future_counters = self._future_counters(
            initial_counters,
            duration=length_samples
        )
        if not self._force_last_time_step_to_global_bill:
            if self._involve_peaks:
                length_samples = rindex(future_counters["peak_period_counter"], self._Delta_P)
            else:
                length_samples = rindex(future_counters["metering_period_counter"], self._Delta_M)
            if length_samples is not None:
                length_samples += 1
                future_counters = {
                    counter_key:counter_lst[:length_samples] for counter_key, counter_lst in future_counters.items()
                }
            else:
                length_samples=0
        timesteps = list(range(length_samples-1))
        if length_samples == 0:
            return None

        if self._involve_peaks:
            if state["metering_period_counter"] < self._Delta_M and state["peak_period_counter"] < self._Delta_P: 
                range_previous_meters = list(range(len(state[(self._members[0], "consumption_meters")][:-1])))
            else:
                range_previous_meters = list(range(len(state[(self._members[0], "consumption_meters")])))
        else:
            range_previous_meters = []
        metering_period_counter_complete_sequence = [state["metering_period_counter"]] + future_counters["metering_period_counter"]
        mapping_metering_periods_meters_indices = split_list_by_number_np(
            metering_period_counter_complete_sequence[:-1], max(self._Delta_M-1, 1), check_end=self._force_last_time_step_to_global_bill, return_indices=True, shift_indices=False
        )
        
        mapping_end_metering_periods = split_list_by_number_np(
            metering_period_counter_complete_sequence[1:], self._Delta_M, check_end=self._force_last_time_step_to_global_bill, return_indices=False
        )

        #print(metering_period_counter_complete_sequence)
        #print(mapping_metering_periods_meters_indices)
        #print(mapping_end_metering_periods)
        #print()
        
        #print(metering_period_counter_complete_sequence, max(self._Delta_M-1, 1))
        range_metering_periods_indices = list(range(len(mapping_end_metering_periods)))
        #mapping_metering_periods_meters_indices = mapping_metering_periods_meters_indices[:len(mapping_end_metering_periods)]
        if range_metering_periods_indices == []:
            return None
        if self._rescaled_gamma_mode == "no_rescale":
            gammas_tau_m = self._gammas[:len(range_metering_periods_indices)]
        elif self._rescaled_gamma_mode == "rescale_terminal":
            gammas_tau_m = [self._gammas[i] for i, tau_m in enumerate(future_counters["metering_period_counter"]) if tau_m == self._Delta_M or (self._force_last_time_step_to_global_bill and i == len(future_counters["metering_period_counter"]) - 1)]
        elif self._rescaled_gamma_mode == "rescale_delayed_terminal":
            gammas_tau_m = [(self._gammas[i] if tau_m == self._Delta_M else self._gammas[i]*(self._gammas[0]**((self._Delta_M - tau_m)))) for i, tau_m in enumerate(future_counters["metering_period_counter"]) if tau_m == self._Delta_M or (self._force_last_time_step_to_global_bill and i == len(future_counters["metering_period_counter"]) - 1)]
        gammas_tau_p = None
        if self._involve_peaks:
            peak_period_counter_complete_sequence = [state["peak_period_counter"]] + future_counters["peak_period_counter"]
            mapping_peak_periods_indices = split_list_by_number_np(
                peak_period_counter_complete_sequence, self._Delta_P, check_end=self._force_last_time_step_to_global_bill, return_indices=False, shift_indices=False
            )
            
            if not self._involve_historical_peaks:
                if len(mapping_peak_periods_indices) > 0 and len(mapping_peak_periods_indices[0]) == 1 and mapping_peak_periods_indices[0][0] == self._Delta_P:
                    mapping_peak_periods_indices = list(mapping_peak_periods_indices[1:])
                    
            range_peak_periods_indices = list(range(len(mapping_peak_periods_indices)))
            mapping_peak_periods_list_len = [
                len(set(mapping_peak_periods_indice).difference({self._Delta_P})) for mapping_peak_periods_indice in mapping_peak_periods_indices
            ]
            if self._rescaled_gamma_mode == "no_rescale":
                gammas_tau_p = self._gammas[:len(range_peak_periods_indices)]
            elif self._rescaled_gamma_mode == "rescale_terminal":
                gammas_tau_p = [self._gammas[i] for i, tau_p in enumerate(future_counters["peak_period_counter"]) if tau_p == self._Delta_P or (self._force_last_time_step_to_global_bill and i == len(future_counters["peak_period_counter"]) - 1)] 
                gammas_tau_m = list(flatten(
                    [[gammas_tau_p[i]]*nb_metering_period_in_peak_period for i,nb_metering_period_in_peak_period in enumerate(mapping_peak_periods_list_len)]
                ))
            elif self._rescaled_gamma_mode == "rescale_delayed_terminal":
                nb_of_timesteps_from_last_timestep_to_next_peak_period = (self._Delta_P * self._Delta_M) - number_of_time_steps_elapsed_in_peak_period(future_counters["metering_period_counter"][-1], future_counters["peak_period_counter"][-1], self._Delta_M, self._Delta_P)
                gammas_tau_p = [(self._gammas[i] if tau_p == self._Delta_P else self._gammas[i]*(self._gammas[0]**(nb_of_timesteps_from_last_timestep_to_next_peak_period))) for i, tau_p in enumerate(future_counters["peak_period_counter"]) if tau_p == self._Delta_P or (self._force_last_time_step_to_global_bill and i == len(future_counters["peak_period_counter"]) - 1)] 
                gammas_tau_m = list(flatten(
                    [[gammas_tau_p[i]]*nb_metering_period_in_peak_period for i,nb_metering_period_in_peak_period in enumerate(mapping_peak_periods_list_len)]
                ))
        future_exogenous_members_variables, future_exogenous_prices = self._exogenous_provider.sample_future_sequences(
            exogenous_variables_members=exogenous_variable_members,
            exogenous_prices=exogenous_prices,
            length=length_samples-1,
            n_samples=self._n_samples
        )
        
        n_samples = list(range(min(self._n_samples, len(future_exogenous_members_variables))))
        sample_timestep_product = list(product(n_samples, timesteps))
        sample_metering_period_product = list(product(n_samples, range_metering_periods_indices))
        if self._involve_peaks:
            sample_peak_period_product = list(product(n_samples, range_peak_periods_indices))
        self._model = Model()
        self._model = self._cplex_configurator(self._model)
        
        #self._model.parameters.mip.tolerances.absmipgap = 1e-2
        #self._model.parameters.simplex.tolerances.optimality = 1e-2
        #self._model.parameters.mip.tolerances.mipgap = 1e-2
        # Create variables related to controllable assets states
        
        if timesteps != []:
            controllable_assets_state = (
                self._model.continuous_var_dict(
                    product(list(self._controllable_assets_state_space_keys), n_samples, timesteps), 
                    lb = lambda k: epsilonify(round(float(self._controllable_assets_state_space[k[0]].low), 6)),
                    ub = lambda k : epsilonify(round(float(self._controllable_assets_state_space[k[0]].high), 6)),
                    name = lambda k: self._create_variable_name_pattern("s(c)")(k)
                )
            )
            
        controllable_assets_first_next_state = (
            self._model.continuous_var_dict(
                list(self._controllable_assets_state_space_keys), 
                lb = lambda k: epsilonify(round(float(self._controllable_assets_state_space[k].low), 6)),
                ub = lambda k : epsilonify(round(float(self._controllable_assets_state_space[k].high), 6)),
                name = lambda k: self._create_variable_name_pattern("s(c)")(k)
            )
        )
        
        # Create variables related to controllable assets actions
        if timesteps != []:
            controllable_assets_action = (
                self._model.continuous_var_dict(
                    product(list(self._controllable_assets_action_space_keys), n_samples, timesteps), 
                    lb = lambda k: epsilonify(round(float(self._controllable_assets_action_space[k[0]].low), 6)),
                    ub = lambda k : epsilonify(round(float(self._controllable_assets_action_space[k[0]].high), 6)),
                    name = lambda k: self._create_variable_name_pattern("u(c)")(k)
                )
            )
        
        controllable_assets_first_action = (
            self._model.continuous_var_dict(
                list(self._controllable_assets_action_space_keys), 
                lb = lambda k: epsilonify(round(float(self._controllable_assets_action_space[k].low), 6)),
                ub = lambda k : epsilonify(round(float(self._controllable_assets_action_space[k].high), 6)),
                name = lambda k: self._create_variable_name_pattern("u(c)")(k)
            )
        )
        
        truncated_exogenous_future_sequence_lst = [[{
            key: [lst_values[timestep]] for key, lst_values in future_exogenous_members_variables[sample].items()
        } for timestep in timesteps] for sample in n_samples]
        
        ctrl_asset_state_lst = [[
            {
                ctrl_asset_state_key: (
                    (controllable_assets_first_next_state[ctrl_asset_state_key]
                    if timestep == -1 else controllable_assets_state[(ctrl_asset_state_key, sample, timestep)]) 
                ) for ctrl_asset_state_key in self._controllable_assets_state_space_keys
            } for timestep in ([-1] + timesteps)
        ] for sample in n_samples]

        ctrl_asset_action_lst = [[
            {
                ctrl_asset_action_key: (
                    controllable_assets_action[(ctrl_asset_action_key, sample, timestep)] 
                ) for ctrl_asset_action_key in self._controllable_assets_action_space_keys
            } for timestep in timesteps
        ] for sample in n_samples]
        
        rec_exchanges = self._model.continuous_var_dict(
            product(["grid import", "grid export", "rec import", "rec export"], self._members, n_samples, range_metering_periods_indices),
            lb=0,
            ub=M,
            name = lambda k: self._create_variable_name_pattern("u(k)")(k)
        )
        # Create variables related to net consumption and net production
        members_with_controllable_assets = [
            member for member in self._members if 
                type(self._consumption_function[member](
                    controllable_assets_first_next_state, exogenous_variable_members, controllable_assets_first_action
                )) not in (int, float, np.float32, np.float64, np.int32, np.int64)
                and
                type(self._production_function[member](
                    controllable_assets_first_next_state, exogenous_variable_members, controllable_assets_first_action
                )) not in (int, float, np.float32, np.float64, np.int32, np.int64)
            
        ]

        
        if self._involve_peaks and mapping_peak_periods_list_len != [] and mapping_peak_periods_list_len[0] == 0:
            mapping_peak_periods_list_len = mapping_peak_periods_list_len[1:]
            range_peak_periods_indices = range_peak_periods_indices[1:]
            
        
        timesteps_binflag = None
        if self._net_consumption_production_mutex_before > 0:
            if timesteps != []:
                net_consumption_production = self._model.continuous_var_dict(
                    product(["net consumption", "net production"], members_with_controllable_assets, n_samples, timesteps),
                    lb=0,
                    ub=M,
                    name = lambda k: self._create_variable_name_pattern(f"A({('ncp')})")(k)
                )
                #if self._net_consumption_production_mutex_before - 1 > 0:
                #    timesteps_binflag = list(range(min(len(timesteps), self._net_consumption_production_mutex_before - 1)))
                #    binary_net_consumption_production = self._model.binary_var_dict(
                #        product(members_with_controllable_assets, n_samples, timesteps_binflag),
                #        name = lambda k: self._create_variable_name_pattern(f"B({('bncp')})")(k)
                #    )
            first_net_consumption_production = self._model.continuous_var_dict(
                product(["net consumption", "net production"], members_with_controllable_assets),
                lb=0,
                ub=M,
                name = lambda k: self._create_variable_name_pattern(f"A({('ncp')})")(k)
            )
            #binary_first_net_consumption_production = self._model.binary_var_dict(
            #    members_with_controllable_assets,
            #    name = lambda k: self._create_variable_name_pattern(f"B({('bncp')})")(k)
            #)
            
            
         
        
        current_peaks_state = None
        current_peaks_first_state = None
        historical_peaks_state = None
        historical_peaks_first_state = None
         # Create variables related to peaks states
        if self._involve_peaks:
            if range_peak_periods_indices != []:
                lst_current_peaks = []
                if self._current_offtake_peak_cost > 0:
                    lst_current_peaks += ["current offtake peak"]
                if self._current_injection_peak_cost > 0:
                    lst_current_peaks += ["current injection peak"]
                current_peaks_state = self._model.continuous_var_dict(
                    product(lst_current_peaks, self._members, n_samples, range_peak_periods_indices),
                    lb=0,
                    ub=M,
                    name = lambda k: self._create_variable_name_pattern("s(curr_peaks)")(k)
                )
            
            if self._involve_historical_peaks:
                if state["peak_period_counter"] == self._Delta_P or (self._force_last_time_step_to_global_bill and range_peak_periods_indices == []):
                    current_peaks_first_state = self._model.continuous_var_dict(
                        product(lst_current_peaks, self._members),
                        lb=0,
                        ub=M,
                        name = lambda k: self._create_variable_name_pattern("s(curr_peaks)")(k)
                    )
                lst_historical_peaks = []
                if self._historical_offtake_peak_cost > 0:
                    lst_historical_peaks += ["historical offtake peak"]
                if self._historical_injection_peak_cost > 0:
                    lst_historical_peaks += ["historical injection peak"]
                if lst_historical_peaks != []:
                    historical_peaks_state = self._model.continuous_var_dict(
                        product(lst_historical_peaks, self._members, n_samples, range_peak_periods_indices),
                        lb=0,
                        ub=M,
                        name = lambda k: self._create_variable_name_pattern("s(histo_peaks)")(k)
                    )
                    if state["peak_period_counter"] == self._Delta_P or (self._force_last_time_step_to_global_bill and range_peak_periods_indices == []):
                        historical_peaks_first_state = self._model.continuous_var_dict(
                            product(lst_historical_peaks, self._members),
                            lb=0,
                            ub=M,
                            name = lambda k: self._create_variable_name_pattern("s(histo_peaks)")(k)
                        )
        
        previous_rec_exchanges = None 
        if self._involve_peaks and range_previous_meters != [] and state["peak_period_counter"] < self._Delta_P:
            previous_rec_exchanges = self._model.continuous_var_dict(
                product(["grid import", "grid export", "rec import", "rec export"], self._members, range_previous_meters),
                lb=0,
                ub=M,
                name = lambda k: self._create_variable_name_pattern("u(prev_k)")(k)
            )
            #self._variables += list(binary_flag_net_consumption_production.values()) + list(first_binary_flag_net_consumption_production.values())
        
        self._variables = sum([
            (list(controllable_assets_action.values()) if timesteps != [] else []),
            list(controllable_assets_first_action.values())
        ], start=[])

        # Create objective function (sum of cost functions related to controllable assets, meters and peaks)
        a = 0
        b = 0
        lst_a = []
        lst_b = []
        lst_a_append = lst_a.append
        lst_b_append = lst_b.append

        
        #t = time()
        consumption_lst = [
            [
                {member:self._consumption_function[member](
                    ctrl_asset_state_lst[sample][timestep], truncated_exogenous_future_sequence_lst[sample][timestep], ctrl_asset_action_lst[sample][timestep]
                ) for member in self._members} 
            for timestep in timesteps
            ] for sample in n_samples
        ]
        production_lst = [
            [
                {member:self._production_function[member](
                    ctrl_asset_state_lst[sample][timestep], truncated_exogenous_future_sequence_lst[sample][timestep], ctrl_asset_action_lst[sample][timestep]
                ) for member in self._members} 
            for timestep in timesteps
            ] for sample in n_samples
        ]
        
        if self._net_consumption_production_mutex_before > 0:
            raw_net_consumption_lst = [
            [
                {member:consumption_lst[sample][timestep][member] - production_lst[sample][timestep][member] for member in self._members} 
            for timestep in timesteps
            ] for sample in n_samples
            ]
            raw_net_production_lst = [
                [
                    {member:production_lst[sample][timestep][member] - consumption_lst[sample][timestep][member] for member in self._members} 
                    for timestep in timesteps
                ] for sample in n_samples
            ]
        else:
            raw_net_consumption_lst = [
                [
                    {member:consumption_lst[sample][timestep][member] for member in self._members} 
                for timestep in timesteps
                ] for sample in n_samples
            ]
            raw_net_production_lst = [
                [
                    {member:production_lst[sample][timestep][member] for member in self._members} 
                    for timestep in timesteps
                ] for sample in n_samples
            ]
        
        first_raw_consumption_dct = {
            member:self._consumption_function[member](
                state, exogenous_variable_members, controllable_assets_first_action
            ) for member in self._members
        }
        first_raw_production_dct = {
            member:self._production_function[member](
                state, exogenous_variable_members, controllable_assets_first_action
            ) for member in self._members
        }
        first_raw_net_consumption_lst = {
            member:first_raw_consumption_dct[member] - first_raw_production_dct[member] for member in self._members
        }
        first_raw_net_production_lst = {
            member:first_raw_production_dct[member] - first_raw_consumption_dct[member] for member in self._members
        }
        first_net_consumption_lst = ({
            member:(
            (first_net_consumption_production[("net consumption", member)]) if member in members_with_controllable_assets else
            max(first_raw_net_consumption_lst[member], 0)
            ) for member in self._members
        } if self._net_consumption_production_mutex_before > 0 else first_raw_consumption_dct)
        first_net_production_lst = ({
            member:(
            (first_net_consumption_production[("net production", member)]) if member in members_with_controllable_assets else
            max(first_raw_net_production_lst[member], 0)
            ) for member in self._members
        } if self._net_consumption_production_mutex_before > 0 else first_raw_consumption_dct)
        net_consumption_lst = [
            {member:[((net_consumption_production[("net consumption", member, sample, timestep)] if self._net_consumption_production_mutex_before > 0 else raw_net_consumption_lst[sample][timestep][member]) if member in members_with_controllable_assets else max(raw_net_consumption_lst[sample][timestep][member], 0)) for timestep in timesteps] for member in self._members} 
            for sample in n_samples
        ]
        net_production_lst = [
            {member:[((net_consumption_production[("net production", member, sample, timestep)] if self._net_consumption_production_mutex_before > 0 else raw_net_production_lst[sample][timestep][member]) if member in members_with_controllable_assets else max(raw_net_production_lst[sample][timestep][member], 0)) for timestep in timesteps] for member in self._members} 
            for sample in n_samples
        ]
        #print("Time spent to build variables", time() - t, "seconds")

        """
            Controllable assets costs
        """
        #print(future_exogenous_members_variables)
        #self._small_penalty_control_actions = 1e-3
        #t = time()
        if self._small_penalty_control_actions > 0:
            lst_b_append(self._small_penalty_control_actions * sum(
                controllable_assets_first_action[(action)] for action in self._controllable_assets_action_space_keys
            ))
            if timesteps != []:
                for sample, timestep in sample_timestep_product:
                    lst_b_append(self._small_penalty_control_actions * sum(
                        controllable_assets_action[(action, sample, timestep)] for action in self._controllable_assets_action_space_keys
                    ))
                    
        for member, cost_functions in self._cost_functions_controllable_assets.items():
            for cost_function in cost_functions:
                lst_a_append(cost_function(state, exogenous_variable_members, controllable_assets_first_action, controllable_assets_first_next_state))
                #self._first_cost += cost_function(state, exogenous_sequences, controllable_assets_first_action, controllable_assets_first_next_state)
                for sample, timestep in sample_timestep_product:
                    controllable_assets_current_state = ctrl_asset_state_lst[sample][timestep]
                    controllable_assets_current_action = ctrl_asset_action_lst[sample][timestep]
                    controllable_assets_next_state = ctrl_asset_state_lst[sample][timestep+1]
                    truncated_exogenous_future_sequence = truncated_exogenous_future_sequence_lst[sample][timestep]
                    lst_a_append(self._gammas[timestep] * cost_function(controllable_assets_current_state, truncated_exogenous_future_sequence, controllable_assets_current_action, controllable_assets_next_state) * (1.0/self._n_samples))
        
        """
            Meters states costs
        """
        
        len_range_previous_meters = len(range_previous_meters)
        for member in self._members:
            
            
            if previous_rec_exchanges is not None:
                if state["metering_period_counter"] < self._Delta_M:
                    buying_prices = list(exogenous_prices[(member, "buying_price")][-len_range_previous_meters-(1):-1])
                    selling_prices = list(exogenous_prices[(member, "selling_price")][-len_range_previous_meters-(1):-1])
                else:
                    buying_prices = list(exogenous_prices[(member, "buying_price")][-len_range_previous_meters-(1):])
                    selling_prices = list(exogenous_prices[(member, "selling_price")][-len_range_previous_meters-(1):])
                for t in range_previous_meters:
                    lst_a_append(
                        previous_rec_exchanges[("grid import", member, t)] * buying_prices[t]
                        - previous_rec_exchanges[("grid export", member, t)] * selling_prices[t]
                    )
            
            for sample in n_samples:
                if state["metering_period_counter"] < self._Delta_M:
                    future_buying_prices = [exogenous_prices[(member, "buying_price")][-1]] + future_exogenous_prices[sample][(member, "buying_price")]
                    future_selling_prices = [exogenous_prices[(member, "selling_price")][-1]] + future_exogenous_prices[sample][(member, "selling_price")]
                else:
                    future_buying_prices = future_exogenous_prices[sample][(member, "buying_price")]
                    future_selling_prices = future_exogenous_prices[sample][(member, "selling_price")]
                for tau_m in range_metering_periods_indices:
                    future_buying_price = future_buying_prices[tau_m]
                    future_selling_price = future_selling_prices[tau_m]
                    lst_a_append(gammas_tau_m[tau_m] * (
                        rec_exchanges[("grid import", member, sample, tau_m)] * future_buying_price
                        - rec_exchanges[("grid export", member, sample, tau_m)] * future_selling_price
                    ) * (1.0/self._n_samples))
       
        """
            Peaks states costs
        """
        if self._involve_peaks:
            last_prorata = elapsed_timesteps_in_peak_period(
                0, 0, metering_period_counter_complete_sequence[-1], peak_period_counter_complete_sequence[-1], Delta_M=self._Delta_M, Delta_P=self._Delta_P
            )
            for member in self._members:
                if current_peaks_first_state is not None:
                    peak_term = 0
                    if self._current_offtake_peak_cost > 0:
                        peak_term += current_peaks_first_state[("current offtake peak", member)] * epsilonify(self._current_offtake_peak_cost, epsilon=1e-8)
                    if self._current_injection_peak_cost > 0:
                        peak_term += current_peaks_first_state[("current injection peak", member)] * epsilonify(self._current_injection_peak_cost, epsilon=1e-8)
                    lst_a_append(peak_term * (1.0/(self._Delta_C * self._Delta_M)))
                for sample, tau_p in sample_peak_period_product:
                    #prorata = (((future_counters["peak_period_counter"][timestep]*self._Delta_M + future_counters["nb_timesteps_elapsed_current_metering_period"][timestep]))/(self._Delta_P * self._Delta_M) if (self._force_surrogate_prorata and self._surrogate and timestep == timesteps[-1] and future_counters["peak_period_counter"][timestep] != self._Delta_P) else 1.0)
                    #prorata = normalize_bounds(np.exp(prorata**self._Delta_M), 0, 1, np.exp(0), np.exp(1))
                    #prorata = (future_counters["peak_period_counter"][timestep]/self._Delta_M) if self._force_surrogate_prorata and self._surrogate and timestep == timesteps[-1] and future_counters["peak_period_counter"][timestep] != self._Delta_P else 1.0
                    peak_term = 0
                    
                    if self._involve_current_peaks:
                        if current_peaks_state is not None:
                            if self._current_offtake_peak_cost > 0:
                                peak_term += current_peaks_state[("current offtake peak", member, sample, tau_p)] * epsilonify(self._current_offtake_peak_cost, epsilon=1e-8)
                            if self._current_injection_peak_cost > 0:
                                peak_term += current_peaks_state[("current injection peak", member, sample, tau_p)] * epsilonify(self._current_injection_peak_cost, epsilon=1e-8)
                    if self._involve_historical_peaks:
                        if historical_peaks_state is not None:
                            if self._historical_offtake_peak_cost > 0:
                                peak_term += historical_peaks_state[("historical offtake peak", member, sample, tau_p)] * epsilonify(self._historical_offtake_peak_cost, epsilon=1e-8)
                            if self._historical_injection_peak_cost > 0:
                                peak_term += historical_peaks_state[("historical injection peak", member, sample, tau_p)] * epsilonify(self._historical_injection_peak_cost, epsilon=1e-8)
                    prorata = (last_prorata if tau_p == len(range_peak_periods_indices) - 1 else 1.0)
                    lst_a_append(gammas_tau_p[tau_p] * (peak_term * (1.0/(self._Delta_C * self._Delta_M)) * (1.0/self._n_samples) * (prorata)))
        self._obj_formula = self._model.sum(lst_a) + self._model.sum(lst_b)
        #print("Time spent to build objective function", time() - t, "seconds")
        self._model.minimize(self._obj_formula)
        constraints = []
        constraints_append = constraints.append
        constraints_extend = constraints.extend
        # Create constraints related to controllable assets dynamics
        #t = time()
        
        for key, dynamics_controllable_assets_function in self._dynamics_controllable_assets.items():
            constraints_append(
                controllable_assets_first_next_state[key] == dynamics_controllable_assets_function(state[key], state, exogenous_variable_members, controllable_assets_first_action)
            )
            for sample, timestep in sample_timestep_product:
                controllable_assets_current_state = ctrl_asset_state_lst[sample][timestep]
                controllable_assets_current_action = ctrl_asset_action_lst[sample][timestep]
                
                truncated_exogenous_future_sequence = truncated_exogenous_future_sequence_lst[sample][timestep]
                controllable_assets_next_state = ctrl_asset_state_lst[sample][timestep+1]
                next_state = dynamics_controllable_assets_function(
                        controllable_assets_current_state[key], 
                        controllable_assets_current_state,
                        truncated_exogenous_future_sequence,
                        controllable_assets_current_action)
                constraints_append(
                    controllable_assets_next_state[key] == next_state
                )
                if timestep == timesteps[-1]:
                    constraints_extend(
                        [
                            next_state >= round(float(self._controllable_assets_state_space[key].low), 6),
                            next_state <= round(float(self._controllable_assets_state_space[key].high), 6)
                        ]
                    )
        # Create constraints related to controllable assets constraints
        
        #Ugly mutex patch for testing
        #print("Time spent ctrl assets dynamics", time() - t, "seconds")
        sos_1_operator = None
        #t = time()
        
        for _, constraint_func in self._constraints_controllable_assets.items():
            #TODO : solve current bug since there is no constraint for state 0
            constraint_tuple = constraint_func(state, exogenous_variable_members, controllable_assets_first_action)
            if constraint_tuple is not None:
                lhs_value, rhs_value, constraint_type = constraint_tuple
                sos_op = None
                op = None
                op_1 = None
                op_2 = None
                if constraint_type == IneqType.EQUALS:
                    op = eq
                elif constraint_type == IneqType.LOWER_OR_EQUALS:
                    op = le
                elif constraint_type == IneqType.GREATER_OR_EQUALS:
                    op = ge
                elif constraint_type == IneqType.MUTEX and timestep < self._net_consumption_production_mutex_before:
                    if sos_1_operator is None:
                        sos_1_operator = create_sos1_pair_operator(self._model)
                    sos_op = sos_1_operator
                elif constraint_type == IneqType.BOUNDS:
                    op_1 = ge
                    op_2 = le
                if op is not None:
                    constraints_append(
                        op(lhs_value, rhs_value)
                    )
                elif sos_op is not None:
                    sos_op(lhs_value, rhs_value)
                elif op_1 is not None and op_2 is not None:
                    rhs_value_1, rhs_value_2 = rhs_value
                    constraints_append(op_1(lhs_value, rhs_value_1))
                    constraints_append(op_2(lhs_value, rhs_value_2))

            for sample, timestep in sample_timestep_product:
                controllable_assets_current_state = ctrl_asset_state_lst[sample][timestep]
                controllable_assets_current_action = ctrl_asset_action_lst[sample][timestep]
                truncated_exogenous_future_sequence = truncated_exogenous_future_sequence_lst[sample][timestep]
                constraint_tuple = constraint_func(controllable_assets_current_state, truncated_exogenous_future_sequence, controllable_assets_current_action)
                if constraint_tuple is not None:
                    lhs_value, rhs_value, constraint_type = constraint_tuple
                    sos_op = None
                    op = None
                    op_1 = None
                    op_2 = None
                    """"""
                    if constraint_type == IneqType.EQUALS:
                        op = eq
                    elif constraint_type == IneqType.LOWER_OR_EQUALS:
                        op = le
                    elif constraint_type == IneqType.GREATER_OR_EQUALS:
                        op = ge
                    elif constraint_type == IneqType.MUTEX:
                        if sos_1_operator is None:
                            sos_1_operator = create_sos1_pair_operator(self._model)
                        sos_op = sos_1_operator
                    elif constraint_type == IneqType.BOUNDS:
                        op_1 = ge
                        op_2 = le
                    if op is not None:
                        constraints_append(
                            op(lhs_value, rhs_value)
                        )
                    elif sos_op is not None:
                        sos_op(lhs_value, rhs_value)
                    elif op_1 is not None and op_2 is not None:
                        rhs_value_1, rhs_value_2 = rhs_value
                        constraints_append(op_1(lhs_value, rhs_value_1))
                        constraints_append(op_2(lhs_value, rhs_value_2))
                        
        #print("Time spent ctrl assets constraints", time() - t, "seconds")
        # Create constraints to compute net production and consumption
        #t = time()
        if self._net_consumption_production_mutex_before > 0:
            for member in members_with_controllable_assets:
                consumption_function_member = self._consumption_function[member]
                production_function_member = self._production_function[member]
                consumption_member_first = consumption_function_member(
                    state, exogenous_variable_members, controllable_assets_first_action
                )
                production_member_first = production_function_member(
                    state, exogenous_variable_members, controllable_assets_first_action
                )
                first_net_consumption_member = first_net_consumption_lst[member]
                first_net_production_member = first_net_production_lst[member]
                
                constraints_extend(
                    [
                        first_net_consumption_member - first_net_production_member == consumption_member_first - production_member_first,
                        first_net_production_member <= production_member_first,
                        #first_net_consumption_member <= M*binary_first_net_consumption_production[member],
                        first_net_consumption_member <= consumption_member_first,
                        #first_net_production_member <= M*(1-binary_first_net_consumption_production[member])
                    ]
                )
                #self._model.add_indicator(binary_first_net_consumption_production[member], first_net_consumption_member == 0, active_value=1, name=None)
                #self._model.add_indicator(binary_first_net_consumption_production[member], first_net_production_member == 0, active_value=0, name=None)
                self._model.add_sos1([first_net_consumption_member, first_net_production_member])
                #self._model.add_if_then(binary_first_net_consumption_production[member] == 1, consumption_member_first >= production_member_first)
                """
                sos_cstr = self._model.add_sos1(
                    [first_net_consumption_member, first_net_production_member]
                )
                """
                #lst_prio_sos = [sos_cstr.index]
                #lst_prio_sos_append = lst_prio_sos.append
                
                
                if timesteps != []:
                    
                    constraints_extend(
                        [
                            net_consumption_lst[sample][member][timestep] - net_production_lst[sample][member][timestep] == (
                                consumption_lst[sample][timestep][member] - production_lst[sample][timestep][member]
                            )
                            for sample, timestep in sample_timestep_product
                        ] 
                    )
                    constraints_extend(
                        [
                            net_production_lst[sample][member][timestep] <= production_lst[sample][timestep][member]
                            for sample, timestep in sample_timestep_product
                        ] 
                    )
                    constraints_extend(
                        [
                            net_consumption_lst[sample][member][timestep] <= consumption_lst[sample][timestep][member]
                            for sample, timestep in sample_timestep_product
                        ] 
                    )
                    """
                    constraints_extend(
                        [
                            net_consumption_lst[sample][member][timestep] <= M*binary_net_consumption_production[(member, sample, timestep)]
                            for sample, timestep in sample_timestep_product if timestep < self._net_consumption_production_mutex_before - 1
                        ] 
                    )
                    constraints_extend(
                        [
                            net_production_lst[sample][member][timestep] <= M*(1-binary_net_consumption_production[(member, sample, timestep)])
                            for sample, timestep in sample_timestep_product if timestep < self._net_consumption_production_mutex_before - 1
                        ] 
                    )
                    """
                    
                    
                    for sample, timestep in sample_timestep_product:
                        if timestep < self._net_consumption_production_mutex_before:
                            self._model.add_sos1([net_consumption_lst[sample][member][timestep], net_production_lst[sample][member][timestep]])
                            #self._model.add_indicator(binary_net_consumption_production[(member, sample, timestep)], net_consumption_lst[sample][member][timestep] == 0, active_value=1, name=None)
                            #self._model.add_indicator(binary_net_consumption_production[(member, sample, timestep)], net_production_lst[sample][member][timestep] == 0, active_value=0, name=None)
                
                    
            reverse_timesteps = timesteps[::-1]
            reverse_timesteps = [reverse_timesteps[0] + 1] + reverse_timesteps

            #lst_prio_sos = [
            #    (lst_prio_sos[i], reverse_timesteps[i], 0) for i in range(len(reverse_timesteps))
            #]
            #self._model.cplex.order.set(lst_prio_sos)       
                        
                        
                            
        #print("Time spent net consumption production constraints", time() - t, "seconds")
        # Create constraints related to feasible rec exchanges
        #t = time()
        sum_repartition_keys_action_rec_export_lst = [[sum([rec_exchanges[("rec export", member, sample, tau_m)] for member in self._members]) for tau_m in range_metering_periods_indices] for sample in n_samples]
        sum_repartition_keys_action_rec_import_lst = [[sum([rec_exchanges[("rec import", member, sample, tau_m)] for member in self._members]) for tau_m in range_metering_periods_indices] for sample in n_samples]
        if previous_rec_exchanges is not None:
            sum_repartition_keys_previous_action_rec_export_lst = [sum([previous_rec_exchanges[("rec export", member, t)] for member in self._members]) for t in range_previous_meters]
            sum_repartition_keys_previous_action_rec_import_lst = [sum([previous_rec_exchanges[("rec import", member, t)] for member in self._members]) for t in range_previous_meters]
                    
        for member in self._members:

            if previous_rec_exchanges is not None:
                for t in range_previous_meters:
                    rec_member_energy_produced_metering_period = state[(member, "production_meters")][t]
                    rec_member_energy_consumed_metering_period = state[(member, "consumption_meters")][t]
                    constraints_extend(
                        [
                            rec_member_energy_consumed_metering_period == (previous_rec_exchanges[("grid import", member, t)] + previous_rec_exchanges[("rec import", member, t)]),
                            rec_member_energy_produced_metering_period == (previous_rec_exchanges[("grid export", member, t)] + previous_rec_exchanges[("rec export", member, t)]),
                            sum_repartition_keys_previous_action_rec_export_lst[t] == sum_repartition_keys_previous_action_rec_import_lst[t]
                        ]
                    )
            first_net_consumption = first_net_consumption_lst[member]
            first_net_production = first_net_production_lst[member]
            for sample in n_samples:
                consumption_meter_term = state[(member, "consumption_meters")][-1] if state["metering_period_counter"] < self._Delta_M else 0.0
                production_meter_term = state[(member, "production_meters")][-1] if state["metering_period_counter"] < self._Delta_M else 0.0
                lst_net_consumption = [consumption_meter_term + first_net_consumption] + net_consumption_lst[sample][member]
                lst_net_consumption = [
                    [lst_net_consumption[i] for i in mapping_metering_periods_meters_indices_lst] for mapping_metering_periods_meters_indices_lst in mapping_metering_periods_meters_indices
                ]
                lst_net_production = [production_meter_term + first_net_production] + net_production_lst[sample][member]
                lst_net_production = [
                    [lst_net_production[i] for i in mapping_metering_periods_meters_indices_lst] for mapping_metering_periods_meters_indices_lst in mapping_metering_periods_meters_indices
                ]
                
                for tau_m in range_metering_periods_indices:
                    rec_member_energy_produced_metering_period = sum(lst_net_production[tau_m])
                    rec_member_energy_consumed_metering_period = sum(lst_net_consumption[tau_m])
                    sum_repartition_keys_action_rec_export = sum_repartition_keys_action_rec_export_lst[sample][tau_m]
                    sum_repartition_keys_action_rec_import = sum_repartition_keys_action_rec_import_lst[sample][tau_m]
                    #print(rec_member_energy_consumed_metering_period)
                    constraints_extend(
                        [
                            rec_member_energy_consumed_metering_period == rec_exchanges[("grid import", member, sample, tau_m)] + rec_exchanges[("rec import", member, sample, tau_m)],
                            rec_member_energy_produced_metering_period == rec_exchanges[("grid export", member, sample, tau_m)] + rec_exchanges[("rec export", member, sample, tau_m)],
                            sum_repartition_keys_action_rec_export == sum_repartition_keys_action_rec_import
                        ]
                    )
                    
                     
        #print("Time spent rec exchange constraint", time() - t, "seconds")
        # Create constraints related to peak states dynamics
        #t = time()
        if self._involve_peaks:
            
            for member in self._members:
                
                if current_peaks_first_state is not None:
                    consumption_power_member_first = previous_rec_exchanges[("grid import", member, range_previous_meters[-1])]
                    production_power_member_first = previous_rec_exchanges[("grid export", member, range_previous_meters[-1])]
                    
                    first_current_peak_constraints = [
                        current_peaks_first_state[("current offtake peak", member)] >= consumption_power_member_first,
                        current_peaks_first_state[("current injection peak", member)] >= production_power_member_first,
                    ]
                    constraints_extend(first_current_peak_constraints)
                
                    if previous_rec_exchanges is not None:
                        for t in range_previous_meters:
                            previous_peak_constraints = [
                                current_peaks_first_state[("current offtake peak", member)] >= previous_rec_exchanges[("grid import", member, t)],
                                current_peaks_first_state[("current injection peak", member)] >= previous_rec_exchanges[("grid export", member, t)]
                            ]
                            constraints_extend(previous_peak_constraints)

                if self._involve_historical_peaks and historical_peaks_first_state is not None:
                    first_historical_peak_constraints = [
                        historical_peaks_first_state[("historical offtake peak", member)] >= max(state[("member", "historical_offtake_peaks")]),
                        historical_peaks_first_state[("historical offtake peak", member)] >= current_peaks_first_state[("current offtake peak", member)],
                        historical_peaks_first_state[("historical injection peak", member)] >= max(state[("member", "historical_injection_peaks")]),
                        historical_peaks_first_state[("historical injection peak", member)] >= current_peaks_first_state[("current injection peak", member)],
                    ]
                    constraints_extend(first_historical_peak_constraints)
                        
                if range_peak_periods_indices != []:
                    for sample in n_samples:
                        rec_grid_import_sequence = [rec_exchanges[("grid import", member, sample, tau_m)] for tau_m in range_metering_periods_indices]
                        rec_grid_export_sequence = [rec_exchanges[("grid export", member, sample, tau_m)] for tau_m in range_metering_periods_indices]
                        
                        rec_grid_import_sequence = list(split_list(rec_grid_import_sequence, mapping_peak_periods_list_len))
                        rec_grid_export_sequence = list(split_list(rec_grid_export_sequence, mapping_peak_periods_list_len))
                        #exit()
                        if previous_rec_exchanges is not None:
                            rec_grid_import_sequence[0] = [previous_rec_exchanges[("grid import", member, t)] for t in range_previous_meters] + rec_grid_import_sequence[0]
                            rec_grid_export_sequence[0] = [previous_rec_exchanges[("grid export", member, t)] for t in range_previous_meters] + rec_grid_export_sequence[0]
                        
                        if self._involve_historical_peaks:
                            trailing_current_offtake_peaks = list(state[(member, "historical_offtake_peaks")])
                            trailing_current_injection_peaks = list(state[(member, "historical_injection_peaks")])
                        for tau_p in range_peak_periods_indices:
                            current_offtake_peaks_lower_bounds = [
                                current_peaks_state[("current offtake peak", member, sample, tau_p)] >= grid_import for grid_import in rec_grid_import_sequence[tau_p]
                            ]
                            current_injection_peaks_lower_bounds = [
                                current_peaks_state[("current injection peak", member, sample, tau_p)] >= grid_export for grid_export in rec_grid_export_sequence[tau_p]
                            ]
                            constraints_extend(current_offtake_peaks_lower_bounds)
                            constraints_extend(current_injection_peaks_lower_bounds)
                            if self._involve_historical_peaks:
                                trailing_current_offtake_peaks = (trailing_current_offtake_peaks + [current_peaks_state[("current offtake peak", member, sample, tau_p)]])[-self._Delta_P_prime:]
                                trailing_current_injection_peaks = (trailing_current_injection_peaks + [current_peaks_state[("current injection peak", member, sample, tau_p)]])[-self._Delta_P_prime:]
                                historical_offtake_peaks_lower_bounds = [
                                    historical_peaks_state[("historical offtake peak", member, sample, tau_p)] >= current_offtake_peak for current_offtake_peak in trailing_current_offtake_peaks
                                ]
                                historical_injection_peaks_lower_bounds = [
                                    historical_peaks_state[("historical injection peak", member, sample, tau_p)] >= current_injection_peak for current_injection_peak in trailing_current_injection_peaks
                                ]
                                constraints_extend(historical_offtake_peaks_lower_bounds + historical_injection_peaks_lower_bounds)
         
        #print("Time spent peaks", time() - t, "seconds")   
        #t = time()         
        self._model.add_constraints(constraints)
        #print("Time spent add constraints", time() - t, "seconds") 
        if timesteps_binflag is not None:
            if self._previous_solution is None:
                warmstart = self._model.new_solution()
                for sample in n_samples:
                    for timestep in [-1]+timesteps_binflag:
                        
                        for member in members_with_controllable_assets:
                            if timestep == -1:
                                sum_first_raw_net_consumption = first_raw_net_consumption_lst[member].constant
                                #lst_order_append((binary_first_net_consumption_production[member].index, priority, 0))
                                warmstart.add_var_value(binary_first_net_consumption_production[member], (0 if sum_first_raw_net_consumption>0 else 1))
                            else:
                                sum_raw_net_consumption = raw_net_consumption_lst[sample][timestep][member].constant
                                warmstart.add_var_value(binary_net_consumption_production[(member, sample, timestep)], (0 if sum_raw_net_consumption>0 else 1))
                                #lst_order_append((binary_net_consumption_production[(member, sample, timestep)].index, priority, 0))
            else:
                warmstart = self._model.new_solution()
                previous_binary_net_consumption_production = self._previous_bin_dicts["previous_binary_net_consumption_production"]
                previous_binary_net_consumption_production_dict = self._previous_solution.get_value_dict(previous_binary_net_consumption_production)
                previous_controllable_actions_dict = self._previous_solution.get_value_dict(self._previous_bin_dicts["previous_controllable_actions"])
                for sample in n_samples:
                    for timestep in [-1]+timesteps_binflag:
                                
                        for member in members_with_controllable_assets:
                            if timestep == -1:
                                warmstart.add_var_value(binary_first_net_consumption_production[member], int(round(previous_binary_net_consumption_production_dict[(member, sample, 0)], 0)))
                            else:
                                if (member, sample, timestep+1) in previous_binary_net_consumption_production_dict:
                                    warmstart.add_var_value(binary_net_consumption_production[(member, sample, timestep)], int(round(previous_binary_net_consumption_production_dict[(member, sample, timestep+1)], 0)))
                                else:
                                    sum_raw_net_consumption = raw_net_consumption_lst[sample][timestep][member].constant
                                    
                                    warmstart.add_var_value(binary_net_consumption_production[(member, sample, timestep)], (0 if sum_raw_net_consumption>0 else 1))
            #self._model.parameters.advance = 2
            #self._model.parameters.mip.strategy.variableselect = 3
            #self._model.parameters.mip.strategy.probe = 3
            #self._model.parameters.emphasis.mip = 2
            #self._model.parameters.mip.cuts.nodecuts = 2
            #self._model.add_mip_start(warmstart)
            self._previous_bin_dicts = {
                "previous_binary_net_consumption_production": binary_net_consumption_production,
                "previous_controllable_actions": controllable_assets_action
            }
        #self._model.parameters.mip.strategy.variableselect = 3
        #self._model.parameters.mip.strategy.probe = 3
        self._model.parameters.emphasis.mip = 2
        self._model.parameters.mip.cuts.nodecuts = 2
        #self._model.parameters.mip.cuts.nodecuts = 2
        
        if full_sequence_of_actions:
            self._timesteps = timesteps
            self._future_counters_lst = future_counters
            self._controllable_assets_action = controllable_assets_action
        self._rec_exchanges = rec_exchanges
        return controllable_assets_first_action
        
    def _solve(self, state, controllable_assets_first_action, full_sequence_of_actions=False):
        if self._optimal_action_population_size > 1:
            #old_simplex_parameter = self._model.parameters.simplex.limits.iterations
            #old_solution_limit_parameter = self._model.parameters.mip.limits.solutions
            solution = self._model.solve(log_output=self._verbose)
            solve_status = self._model.solve_status
            objective_value = solution.get_objective_value()
            if type(self._obj_formula) != float:
                self._model.add_constraints([
                        self._obj_formula == objective_value
                    ]
                )
            nb_variables = len(self._variables)
            coeffs = list(range(1, nb_variables+1))
            coeffs = [c - nb_variables//2 for c in coeffs]
            coeffs = [c - (-1 if c > 0 else 1) for c in coeffs]
            
            
            solutions = [solution]
            for _ in range(self._optimal_action_population_size-1):
                random.shuffle(coeffs)
                self._model._clear_objective_expr()
                self._model.maximize(
                    np.dot(coeffs, self._variables)
                )
                solution = self._model.solve(log_output=self._verbose, clean_before_solve=True)
                if solution is not None:
                    solutions.append(solution)
            solution = None
            if solutions is not None:
                solution = random.choice(solutions)
        else:
            #print("Solving the model...")
            #t = time()
            solution = self._model.solve(log_output=self._verbose)
            #print("Time spent to solve this problem", time() - t, "seconds")
            #exit()
            solve_status = self._model.solve_status
        #d = solution.get_value_dict(self._rec_exchanges)
        #print(solution)
        print(solution.get_objective_value())
        self._previous_solution = solution
        #print(self._previous_solution)
        #exit()
        #print()
        #exit()
        #try:
        #    print("instant peak cost guess", solution.get_value(self._first_cost))
        #except:
        #    print("instant peak cost guess", self._first_cost)
        #exit()
        
        if solve_status != JobSolveStatus.OPTIMAL_SOLUTION:
            raise InfeasiblePolicy(solve_status, self._model.solve_details._solve_status)
        else:
            #print("Solved the model with solution:", solution.get_objective_value())
            output_action = dict()
            for key_controllable_assets_action in self._controllable_assets_action_space_keys:
                output_action[key_controllable_assets_action] = solution.get_value(controllable_assets_first_action[key_controllable_assets_action])
            if full_sequence_of_actions:
                controllable_assets_action_converted = [
                    {key: float(solution.get_value(self._controllable_assets_action[(key, 0, timestep)])) for key in self._controllable_assets_action_space_keys} for timestep in self._timesteps
                ]
                return [output_action] + controllable_assets_action_converted
            else:
                return output_action

    def _future_counters(self, counters_states: Dict[str, int], duration: int = 1) -> Dict[str, List[int]]:
        future_metering_period_counter, future_peak_period_counter = future_counters(
            counters_states["metering_period_counter"],
            counters_states.get("peak_period_counter", 0),
            duration=duration,
            Delta_M=self._Delta_M,
            Delta_P=self._Delta_P
        )
        return {
            "metering_period_counter": future_metering_period_counter,
            "peak_period_counter": future_peak_period_counter if self._involve_peaks else None
        }

    def _action(self, state: Dict[str, Any], exogenous_variable_members: Dict[Tuple[str, str], List[float]], exogenous_prices: Dict[Tuple[str, str], List[float]]) -> Dict[Any, float]:
        #t = time()
        controllable_assets_first_action = self._build_planning_model(state, exogenous_variable_members, exogenous_prices)
        #print("Time spent to build this problem", time() - t, "seconds")
        if controllable_assets_first_action is None:
            return {
                k:0.0 for k in self._controllable_assets_action_space_keys
            }
        return self._solve(state, controllable_assets_first_action)
    
    def sequence_of_actions(self, state: Dict[str, Any], exogenous_variable_members: Dict[Tuple[str, str], List[float]], exogenous_prices: Dict[Tuple[str, str], List[float]]) -> Dict[Any, float]:
        #t = time()
        controllable_assets_first_action = self._build_planning_model(state, exogenous_variable_members, exogenous_prices, full_sequence_of_actions=True)
        #print("Time spent to build this problem", time() - t, "seconds")
        if controllable_assets_first_action is None:
            raise BaseException("Having empty sequence of actions is not handled as the use case is for optimal policy")
        return self._solve(state, controllable_assets_first_action, full_sequence_of_actions=True)
