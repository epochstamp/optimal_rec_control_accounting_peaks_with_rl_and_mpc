from base import Policy, ExogenousProvider, IneqType
from typing import Dict, Any, Callable, Tuple, List, Optional, Union
import numpy as np
from docplex.mp.model import Model
from docplex.mp.linear import Var
from gym.spaces import Dict as DictSpace
from itertools import product
from operator import le, ge, eq
from docplex.util.status import JobSolveStatus
from env.counter_utils import future_counters
from exceptions import InfeasiblePolicy
from utils.utils import epsilonify, merge_dicts, normalize_bounds, roundify, flatten, chunks, split_list_by_number_np
from env.peaks_utils import elapsed_metering_periods_in_peak_period
import random
from uuid import uuid4
from time import time

M = 10000
EPSILON = 10e-6


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
                 net_consumption_production_mutex=True,
                 optimal_action_population_size=1,
                 n_threads=None,
                 small_penalty_control_actions=1e-6):
        super().__init__(
            members,
            controllable_assets_state_space,
            controllable_assets_action_space,
            constraints_controllable_assets,
            consumption_function,
            production_function
        )
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
        self._involve_current_peaks = current_offtake_peak_cost > 0 and current_injection_peak_cost > 0
        self._involve_historical_peaks = historical_offtake_peak_cost > 0 and historical_injection_peak_cost > 0
        self._involve_peaks = self._involve_current_peaks or self._involve_historical_peaks
        self._force_last_time_step_to_global_bill = force_last_time_step_to_global_bill
        self._T = T
        self._verbose = verbose
        self._n_threads = n_threads
        self._optimal_action_population_size = optimal_action_population_size
        self._small_penalty_control_actions = small_penalty_control_actions
        self._net_consumption_production_mutex = net_consumption_production_mutex
        self.reset()


    def _build_planning_model(self, state: Dict[Union[str, Tuple[str, str]], Any], exogenous_variable_members: Dict[Tuple[str, str], List[float]], exogenous_prices: Dict[Tuple[str, str], List[float]], full_sequence_of_actions=False) -> None:
        
        if self._n_threads is not None:
            self._model.parameters.threads = self._n_threads
        current_timestep = len(list(exogenous_variable_members.values())[0])
        length_samples = min(self._max_length_samples, self._T - current_timestep)
        timesteps = list(range(length_samples))
        initial_counters = {
            "metering_period_counter": state["metering_period_counter"]
        }
        if self._involve_peaks:
            initial_counters["peak_period_counter"] = state["peak_period_counter"]
        future_counters = self._future_counters(
            initial_counters,
            duration=length_samples
        )
        
        range_previous_meters = range(len(state[(self._members[0], "consumption_meters")][:-1]))
        mapping_metering_periods_indices = split_list_by_number_np(
            [state["metering_period_counter"]] + future_counters["metering_period_counter"], max(self._Delta_M-1, 1), check_end=self._force_last_time_step_to_global_bill, return_indices=True, shift_indices=False
        )
        range_metering_periods_indices = range(len(mapping_metering_periods_indices))
        if self._involve_peaks:
            mapping_peak_periods_indices = split_list_by_number_np(
                [state["peak_period_counter"]] + future_counters["peak_period_counter"], self._Delta_P, check_end=self._force_last_time_step_to_global_bill, return_indices=True
            )
            range_peak_periods_indices = range(len(mapping_peak_periods_indices))
        if length_samples > 0:
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
        else:
            future_exogenous_members_variables, future_exogenous_prices = None, None
            n_samples = []
            sample_timestep_product = []
            sample_metering_period_product = []
            if self._involve_peaks:
                sample_peak_period_product = []
        
        # Create variables related to controllable assets states
        controllable_assets_state = (
            self._model.continuous_var_dict(
                product(list(self._controllable_assets_state_space.keys()), n_samples, timesteps), 
                lb = lambda k: max(float(self._controllable_assets_state_space[k[0]].low), 0),
                ub = lambda k : float(self._controllable_assets_state_space[k[0]].high),
                name = lambda k: self._create_variable_name_pattern("s(c)")(k)
            )
        )
        controllable_assets_first_next_state = (
            self._model.continuous_var_dict(
                list(self._controllable_assets_state_space.keys()), 
                lb = lambda k: max(float(self._controllable_assets_state_space[k].low), 0),
                ub = lambda k : float(self._controllable_assets_state_space[k].high),
                name = lambda k: self._create_variable_name_pattern("s(c)")(k)
            )
        )
        # Create variables related to controllable assets actions
        controllable_assets_action = (
            self._model.continuous_var_dict(
                product(list(self._controllable_assets_action_space.keys()), n_samples, timesteps), 
                lb = lambda k: float(self._controllable_assets_action_space[k[0]].low),
                ub = lambda k : float(self._controllable_assets_action_space[k[0]].high),
                name = lambda k: self._create_variable_name_pattern("u(c)")(k)
            )
        )
        controllable_assets_first_action = (
            self._model.continuous_var_dict(
                list(self._controllable_assets_action_space.keys()), 
                lb = lambda k: float(self._controllable_assets_action_space[k].low),
                ub = lambda k : float(self._controllable_assets_action_space[k].high),
                name = lambda k: self._create_variable_name_pattern("u(c)")(k)
            )
        )
       

        
        # Create variable related to rec import/export
        rec_exchanges = self._model.continuous_var_dict(
            product(["grid import", "grid export", "rec import", "rec export"], self._members, n_samples, range_metering_periods_indices),
            lb=0,
            ub=M,
            name = lambda k: self._create_variable_name_pattern("u(k)")(k)
        )

        rec_exchanges_first = None
        if state["metering_period_counter"] == self._Delta_M or self._force_last_time_step_to_global_bill:
            rec_exchanges_first = self._model.continuous_var_dict(
                product(["grid import", "grid export", "rec import", "rec export"], self._members),
                lb=0,
                ub=M,
                name = lambda k: self._create_variable_name_pattern("u(k)")(k)
            )
        

        # Create variables related to net consumption and net production
        net_consumption_production = self._model.continuous_var_dict(
            product(["net consumption", "net production"], self._members, n_samples, timesteps),
            lb=0,
            ub=M,
            name = lambda k: self._create_variable_name_pattern(f"A({('nc' if k[0] == 'net consumption' else 'np')})")(k)
        )
        first_net_consumption_production = self._model.continuous_var_dict(
            product(["net consumption", "net production"], self._members),
            lb=0,
            ub=M,
            name = lambda k: self._create_variable_name_pattern(f"A({('nc' if k[0] == 'net consumption' else 'np')})")(k)
        )
        
        current_peaks_state = None
        current_peaks_first_state = None
        historical_peaks_state = None
        historical_peaks_first_state = None
        current_peaks_state = None
        current_peaks_first_state = None
        historical_peaks_state = None
        historical_peaks_first_state = None
         # Create variables related to peaks states
        if self._involve_peaks:
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
            if state["peak_period_counter"] == self._Delta_P or self._force_last_time_step_to_global_bill:
                current_peaks_first_state = self._model.continuous_var_dict(
                    product(lst_current_peaks, self._members),
                    lb=0,
                    ub=M,
                    name = lambda k: self._create_variable_name_pattern("s(curr_peaks)")(k)
                )
            if self._involve_historical_peaks:
                lst_historical_peaks = []
                if self._historical_offtake_peak_cost > 0:
                    lst_historical_peaks += ["historical offtake peak"]
                if self._historical_injection_peak_cost > 0:
                    lst_historical_peaks += ["historical injection peak"]
                historical_peaks_state = self._model.continuous_var_dict(
                    product(lst_historical_peaks, self._members, n_samples, range_peak_periods_indices),
                    lb=0,
                    ub=M,
                    name = lambda k: self._create_variable_name_pattern("s(histo_peaks)")(k)
                )
                if state["peak_period_counter"] == self._Delta_P or self._force_last_time_step_to_global_bill:
                    historical_peaks_first_state = self._model.continuous_var_dict(
                        product(lst_historical_peaks, self._members),
                        lb=0,
                        ub=M,
                        name = lambda k: self._create_variable_name_pattern("s(histo_peaks)")(k)
                    )
                
        previous_rec_exchanges = None
        if self._involve_peaks and len(range_previous_meters) > 1:
            previous_rec_exchanges = self._model.continuous_var_dict(
                product(["grid import", "grid export", "rec import", "rec export"], self._members, range_previous_meters),
                lb=0,
                ub=M,
                name = lambda k: self._create_variable_name_pattern("u(k)")(k)
            )
        if self._net_consumption_production_mutex:
            binary_flag_net_consumption_production = self._model.binary_var_dict(
                product(["net consumption/production"], self._members, n_samples, timesteps),
                name = lambda k: self._create_variable_name_pattern(f"B({('ncp')})")(k)
            )
            first_binary_flag_net_consumption_production = self._model.binary_var_dict(
                product(["net consumption/production"], self._members),
                name = lambda k: self._create_variable_name_pattern(f"B({('ncp')})")(k)
            )
            #self._variables += list(binary_flag_net_consumption_production.values()) + list(first_binary_flag_net_consumption_production.values())
        
        self._variables = sum([
            list(controllable_assets_action.values()),
            list(controllable_assets_first_action.values())
        ], start=[])

        # Create objective function (sum of cost functions related to controllable assets, meters and peaks)
        a = 0

        


        """
            Controllable assets costs
        """

        if self._small_penalty_control_actions > 0:
            a += self._small_penalty_control_actions * sum(
                controllable_assets_first_action[(action)] for action in self._controllable_assets_action_space.keys()
            )
            for sample, timestep in sample_timestep_product:
                a += self._small_penalty_control_actions * sum(
                    controllable_assets_action[(action, sample, timestep)] for action in self._controllable_assets_action_space.keys()
                )
        for member, cost_functions in self._cost_functions_controllable_assets.items():
            
            for cost_function in cost_functions:
                a += cost_function(state, exogenous_variable_members, controllable_assets_first_action, controllable_assets_first_next_state)
                #self._first_cost += cost_function(state, exogenous_sequences, controllable_assets_first_action, controllable_assets_first_next_state)
                for sample, timestep in sample_timestep_product:
                    if timestep == 0:
                        controllable_assets_current_state = controllable_assets_first_next_state
                    else:
                        controllable_assets_current_state = {
                            state: controllable_assets_state[(state, sample, timestep-1)] for state in self._controllable_assets_state_space.keys()
                        }
                    controllable_assets_current_action = {
                        action: controllable_assets_action[(action, sample, timestep)] for action in self._controllable_assets_action_space.keys()
                    }
                    controllable_assets_next_state = {
                        state: controllable_assets_state[(state, sample, timestep)] for state in self._controllable_assets_state_space.keys()
                    }
                    truncated_exogenous_future_sequence = {
                        key: lst_values[:timestep+1] for key, lst_values in future_exogenous_members_variables[sample].items()
                    }
                    a += cost_function(controllable_assets_current_state, truncated_exogenous_future_sequence, controllable_assets_current_action, controllable_assets_next_state) * (1.0/self._n_samples)
                    if self._small_penalty_control_actions > 0:
                        a += self._small_penalty_control_actions * sum(
                            controllable_assets_action[(action, sample, timestep)] for action in self._controllable_assets_action_space.keys()
                        )
        """
            Meters states costs
        """
        #end_of_metering_periods = [timestep for timesteps in timesteps if future_counters["nb_timesteps_elapsed_current_metering_period"][timestep] == self._Delta_M or (self._surrogate and timestep == timesteps[-1])]
        """
        if False and not self._net_consumption_production_mutex:
            lst_buying_prices_first = [exogenous_sequences[(member, "buying_price")][-1] for member in self._members]
            lst_selling_prices_first = [exogenous_sequences[(member, "selling_price")][-1] for member in self._members]
            delta_buying_price_first = max(lst_buying_prices_first) - min(lst_selling_prices_first) + 1e-6
        else:
            delta_buying_price_first = 0
        """
        if future_exogenous_prices is not None:
            future_exogenous_prices_length = len(list(future_exogenous_prices[0].values())[0])
        for member in self._members:
            if self._force_last_time_step_to_global_bill or state["metering_period_counter"] == self._Delta_M:
                a += (
                    rec_exchanges_first[("grid import", member)] * exogenous_prices[(member, "buying_price")][-1] 
                    - rec_exchanges_first[("grid export", member)] * exogenous_prices[(member, "selling_price")][-1]
                )
            if previous_rec_exchanges is not None:
                for t in range_previous_meters:
                    a += (
                        previous_rec_exchanges[("grid import", member, t)] * exogenous_prices[(member, "buying_price")][t] 
                        - previous_rec_exchanges[("grid export", member, t)] * exogenous_prices[(member, "selling_price")][t]
                    )
            for sample, tau_m in sample_metering_period_product:
                tau_m_prime = min(tau_m, future_exogenous_prices_length-1)
                a += (
                    rec_exchanges[("grid import", member, sample, tau_m)] * future_exogenous_prices[sample][(member, "buying_price")][tau_m_prime]
                    - rec_exchanges[("grid export", member, sample, tau_m)] * future_exogenous_prices[sample][(member, "selling_price")][tau_m_prime]
                ) * (1.0/self._n_samples)
                        
        """
            Peaks states costs
        """
        if self._involve_peaks:
            prorata_first = elapsed_metering_periods_in_peak_period(
                0, (0 if len(timesteps) == 0 else (timesteps[-1]+1)), state["metering_period_counter"], state["peak_period_counter"], Delta_M=self._Delta_M, Delta_P=self._Delta_P
            )
            last_prorata = elapsed_metering_periods_in_peak_period(
                0, 0, future_counters["metering_period_counter"][-1], future_counters["peak_period_counter"][-1], Delta_M=self._Delta_M, Delta_P=self._Delta_P
            )
            for member in self._members:
                if self._force_last_time_step_to_global_bill or state["peak_period_counter"] == self._Delta_P:
                    peak_term = 0
                    if self._involve_current_peaks:
                        if current_peaks_first_state is not None:
                            if self._current_offtake_peak_cost > 0:
                                peak_term += current_peaks_first_state[("current offtake peak", member)] * self._current_offtake_peak_cost
                            if self._current_injection_peak_cost > 0:
                                peak_term += current_peaks_first_state[("current injection peak", member)] * self._current_injection_peak_cost
                    if self._involve_historical_peaks:
                        if historical_peaks_first_state is not None:
                            if self._current_offtake_peak_cost > 0:
                                peak_term += historical_peaks_first_state[("historical offtake peak", member)] * self._historical_offtake_peak_cost
                            if self._current_injection_peak_cost > 0:
                                peak_term += historical_peaks_first_state[("historical injection peak", member)] * self._historical_injection_peak_cost
                    a += ((peak_term/(self._Delta_M*self._Delta_C)) * prorata_first)
                for sample, tau_p in sample_peak_period_product:
                    #prorata = (((future_counters["peak_period_counter"][timestep]*self._Delta_M + future_counters["nb_timesteps_elapsed_current_metering_period"][timestep]))/(self._Delta_P * self._Delta_M) if (self._force_surrogate_prorata and self._surrogate and timestep == timesteps[-1] and future_counters["peak_period_counter"][timestep] != self._Delta_P) else 1.0)
                    #prorata = normalize_bounds(np.exp(prorata**self._Delta_M), 0, 1, np.exp(0), np.exp(1))
                    #prorata = (future_counters["peak_period_counter"][timestep]/self._Delta_M) if self._force_surrogate_prorata and self._surrogate and timestep == timesteps[-1] and future_counters["peak_period_counter"][timestep] != self._Delta_P else 1.0
                    peak_term = 0
                    if self._involve_current_peaks:
                        if current_peaks_state is not None:
                            if self._current_offtake_peak_cost > 0:
                                peak_term += current_peaks_state[("current offtake peak", member, sample, tau_p)] * self._current_offtake_peak_cost
                            if self._current_injection_peak_cost > 0:
                                peak_term += current_peaks_state[("current injection peak", member, sample, tau_p)] * self._current_injection_peak_cost
                    if self._involve_historical_peaks:
                        if historical_peaks_state is not None:
                            if self._current_offtake_peak_cost > 0:
                                peak_term += historical_peaks_state[("historical offtake peak", member, sample, tau_p)] * self._historical_offtake_peak_cost
                            if self._current_injection_peak_cost > 0:
                                peak_term += historical_peaks_state[("historical injection peak", member, sample, tau_p)] * self._historical_injection_peak_cost
                    a += (peak_term * (1.0/self._n_samples) * (last_prorata if tau_p == range_peak_periods_indices[-1] else 1))

        self._obj_formula = a
        self._model.minimize(self._obj_formula)
        # Create constraints related to controllable assets dynamics
        for key, dynamics_controllable_assets_function in self._dynamics_controllable_assets.items():
            self._model.add_constraint(
                controllable_assets_first_next_state[key] == dynamics_controllable_assets_function(state[key], state, exogenous_variable_members, controllable_assets_first_action)
            )
            for sample, timestep in sample_timestep_product:
                if timestep == 0:
                    controllable_assets_current_state = controllable_assets_first_next_state
                else:
                    controllable_assets_current_state = {
                        state: controllable_assets_state[(state, sample, timestep-1)] for state in self._controllable_assets_state_space.keys()
                    }
                controllable_assets_current_action = {
                    action: controllable_assets_action[(action, sample, timestep)] for action in self._controllable_assets_action_space.keys()
                }
                truncated_exogenous_future_sequence = {
                    key: lst_values[:timestep+1] for key, lst_values in future_exogenous_members_variables[sample].items()
                }
                self._model.add_constraint(
                    controllable_assets_state[(key, sample, timestep)] == dynamics_controllable_assets_function(
                        controllable_assets_current_state[key], 
                        controllable_assets_current_state,
                        truncated_exogenous_future_sequence,
                        controllable_assets_current_action)
                )
        # Create constraints related to controllable assets constraints
        for _, constraint_func in self._constraints_controllable_assets.items():
            #TODO : solve current bug since there is no constraint for state 0
            constraint_tuple = constraint_func(state, exogenous_variable_members, controllable_assets_first_action)
            if constraint_tuple is not None:
                lhs_value, rhs_value, constraint_type = constraint_tuple
                
                if constraint_type == IneqType.EQUALS:
                    op = eq
                elif constraint_type == IneqType.LOWER_OR_EQUALS:
                    op = le
                elif constraint_type == IneqType.GREATER_OR_EQUALS:
                    if type(rhs_value) == float:
                        rhs_value = max(rhs_value, 0)
                    op = ge
                else:
                    op = None
                self._model.add_constraint(
                    op(lhs_value, rhs_value)
                )
            for sample, timestep in sample_timestep_product:
                if timestep == 0:
                    controllable_assets_current_state = controllable_assets_first_next_state
                else:
                    controllable_assets_current_state = {
                        state: controllable_assets_state[(state, sample, timestep-1)] for state in self._controllable_assets_state_space.keys()
                    }
                controllable_assets_current_action = {
                    action: controllable_assets_action[(action, sample, timestep)] for action in self._controllable_assets_action_space.keys()
                }
                truncated_exogenous_future_sequence = {
                    key: lst_values[:timestep+1] for key, lst_values in future_exogenous_members_variables[sample].items()
                }
                constraint_tuple = constraint_func(controllable_assets_current_state, truncated_exogenous_future_sequence, controllable_assets_current_action)
                if constraint_tuple is not None:
                    lhs_value, rhs_value, constraint_type = constraint_tuple
                    if constraint_type == IneqType.EQUALS:
                        op = eq
                    elif constraint_type == IneqType.LOWER_OR_EQUALS:
                        op = le
                    elif constraint_type == IneqType.GREATER_OR_EQUALS:
                        op = ge
                    self._model.add_constraint(
                        op(lhs_value, rhs_value)
                    )

        # Create constraints to compute net production and consumption
        for member in self._members:
            consumption_member = self._consumption_function[member](
                state, exogenous_variable_members, controllable_assets_first_action
            )
            production_member = self._production_function[member](
                state, exogenous_variable_members, controllable_assets_first_action
            )
            if self._net_consumption_production_mutex:
                self._model.add_constraints(
                    [
                        first_net_consumption_production[("net consumption", member)] - first_net_consumption_production[("net production", member)] == consumption_member - production_member,
                        first_net_consumption_production[("net production", member)] - first_net_consumption_production[("net consumption", member)] == production_member - consumption_member,
                        first_net_consumption_production[("net consumption", member)] <=  M * first_binary_flag_net_consumption_production[("net consumption/production", member)],
                        first_net_consumption_production[("net production", member)] <= M * (1-first_binary_flag_net_consumption_production[("net consumption/production", member)])
                    ]
                )
            else:
                self._model.add_constraints(
                    [
                        first_net_consumption_production[("net consumption", member)] - first_net_consumption_production[("net production", member)] == consumption_member - production_member,
                        first_net_consumption_production[("net consumption", member)] <= consumption_member,
                        first_net_consumption_production[("net production", member)] - first_net_consumption_production[("net consumption", member)] == production_member - consumption_member,
                        first_net_consumption_production[("net production", member)] <= production_member
                    ]
                )
            
            
            
            
            
            for sample, timestep in sample_timestep_product:
                if timestep == 0:
                    controllable_assets_current_state = controllable_assets_first_next_state
                else:
                    controllable_assets_current_state = {
                        state: controllable_assets_state[(state, sample, timestep-1)] for state in self._controllable_assets_state_space.keys()
                    }
                controllable_assets_current_action = {
                        action: controllable_assets_action[(action, sample, timestep)] for action in self._controllable_assets_action_space.keys()
                }
                truncated_exogenous_future_sequence = {
                    key: lst_values[:timestep+1] for key, lst_values in future_exogenous_members_variables[sample].items()
                }
                consumption_member = self._consumption_function[member](
                    controllable_assets_current_state, truncated_exogenous_future_sequence, controllable_assets_current_action
                )
                production_member = self._production_function[member](
                    controllable_assets_current_state, truncated_exogenous_future_sequence, controllable_assets_current_action
                )
                if self._net_consumption_production_mutex:
                    self._model.add_constraints(
                        [
                            net_consumption_production[("net consumption", member, sample, timestep)] - net_consumption_production[("net production", member, sample, timestep)] == consumption_member - production_member,
                            net_consumption_production[("net production", member, sample, timestep)] - net_consumption_production[("net consumption", member, sample, timestep)] == production_member - consumption_member,
                            net_consumption_production[("net consumption", member, sample, timestep)] <= M * binary_flag_net_consumption_production[("net consumption/production", member, sample, timestep)],
                            net_consumption_production[("net production", member, sample, timestep)] <= M * (1-binary_flag_net_consumption_production[("net consumption/production", member, sample, timestep)])
                        ]
                    )
                else:
                    self._model.add_constraints(
                        [
                            net_consumption_production[("net consumption", member, sample, timestep)] - net_consumption_production[("net production", member, sample, timestep)] == consumption_member - production_member,
                            net_consumption_production[("net consumption", member, sample, timestep)] <= consumption_member,
                            net_consumption_production[("net production", member, sample, timestep)] - net_consumption_production[("net consumption", member, sample, timestep)] == production_member - consumption_member,   
                            net_consumption_production[("net production", member, sample, timestep)] <= production_member                        
                        ]
                    )
        # Create constraints related to feasible rec exchanges
        for member in self._members:
            if self._force_last_time_step_to_global_bill or state["metering_period_counter"] == self._Delta_M:
                #rec_member_energy_produced = first_net_consumption_production[("net production", member)]
                #rec_member_energy_consumed = first_net_consumption_production[("net consumption", member)]
                rec_member_energy_produced_metering_period = state[(member, "production_meters")][-1]
                rec_member_energy_consumed_metering_period = state[(member, "consumption_meters")][-1]
                sum_repartition_keys_first_action_rec_export = sum([rec_exchanges_first[("rec export", member)] for member in self._members])
                sum_repartition_keys_first_action_rec_import = sum([rec_exchanges_first[("rec import", member)] for member in self._members])
                self._model.add_constraints(
                    [
                        rec_member_energy_produced_metering_period == rec_exchanges_first[("grid export", member)] + rec_exchanges_first[("rec export", member)],
                        rec_member_energy_consumed_metering_period == rec_exchanges_first[("grid import", member)] + rec_exchanges_first[("rec import", member)],
                        sum_repartition_keys_first_action_rec_export == sum_repartition_keys_first_action_rec_import
                    ]
                )
            if previous_rec_exchanges is not None:
                for t in range_previous_meters:
                    rec_member_energy_produced_metering_period = state[(member, "production_meters")][t]
                    rec_member_energy_consumed_metering_period = state[(member, "consumption_meters")][t]
                    sum_repartition_keys_previous_action_rec_export = sum([previous_rec_exchanges[("rec export", member, t)] for member in self._members])
                    sum_repartition_keys_previous_action_rec_import = sum([previous_rec_exchanges[("rec import", member, t)] for member in self._members])
                    self._model.add_constraints(
                        [
                            rec_member_energy_produced_metering_period == previous_rec_exchanges[("grid export", member, t)] + previous_rec_exchanges[("rec export", member, t)],
                            rec_member_energy_consumed_metering_period == previous_rec_exchanges[("grid import", member, t)] + previous_rec_exchanges[("rec import", member, t)],
                            sum_repartition_keys_previous_action_rec_export == sum_repartition_keys_previous_action_rec_import
                        ]
                    )
            
            for n_samples in sample_timestep_product:
                lst_net_consumption = [state[(member, "consumption_meters")][-1] + first_net_consumption_production[("net consumption", member)]] + [net_consumption_production[("net consumption", member, sample, timestep)] for timestep in timesteps]
                lst_net_consumption = [
                   [lst_net_consumption[i] for i in mapping_metering_periods_indices_lst] for mapping_metering_periods_indices_lst in mapping_metering_periods_indices
                ]
                lst_net_production = [state[(member, "production_meters")][-1] +first_net_consumption_production[("net production", member)]] + [net_consumption_production[("net production", member, sample, timestep)] for timestep in timesteps]
                lst_net_production = [
                   [lst_net_production[i] for i in mapping_metering_periods_indices_lst] for mapping_metering_periods_indices_lst in mapping_metering_periods_indices
                ]
                for tau_m in range_metering_periods_indices:
                    rec_member_energy_produced_metering_period = sum(lst_net_production[tau_m])
                    rec_member_energy_consumed_metering_period = sum(lst_net_consumption[tau_m])
                    sum_repartition_keys_action_rec_export = sum([rec_exchanges[("rec export", member, sample, tau_m)] for member in self._members])
                    sum_repartition_keys_action_rec_import = sum([rec_exchanges[("rec import", member, sample, tau_m)] for member in self._members])
                    self._model.add_constraints(
                        [
                            rec_member_energy_produced_metering_period == rec_exchanges[("grid export", member, sample, tau_m)] + rec_exchanges[("rec export", member, sample, tau_m)],
                            rec_member_energy_consumed_metering_period == rec_exchanges[("grid import", member, sample, tau_m)] + rec_exchanges[("rec import", member, sample, tau_m)],
                            sum_repartition_keys_action_rec_export == sum_repartition_keys_action_rec_import
                        ]
                    )
                     
        
        # Create constraints related to peak states dynamics
        if self._involve_peaks:
            for member in self._members:
                
                if self._force_last_time_step_to_global_bill or state["peak_period_counter"] == self._Delta_P:
                    consumption_power_member_first = repartition_keys_first_action[("grid import", member)] / (self._Delta_C * self._Delta_M)
                    production_power_member_first = repartition_keys_first_action[("grid export", member)] / (self._Delta_C * self._Delta_M)
                    first_peak_constraints = [
                        peaks_first_state[("offtake peaks", member)] >= consumption_power_member_first,
                        peaks_first_state[("injection peaks", member)] >= production_power_member_first,
                    ]
                    
                    if self._activate_first_step_lower_bound_current_peaks:
                        first_peak_constraints += [
                            peaks_first_state[("offtake peaks", member)] >= state[(member, "current_offtake_peak")],
                            peaks_first_state[("injection peaks", member)] >= state[(member, "current_injection_peak")]
                        ]
                    if self._activate_first_step_lower_bound_history_peaks:
                        for offtake_peaks, injection_peaks in zip(state[(member, "offtake_peaks")], state[(member, "injection_peaks")]):
                            first_peak_constraints += [
                                peaks_first_state[("offtake peaks", member)] >= offtake_peaks,
                                peaks_first_state[("injection peaks", member)] >= injection_peaks
                            ]
                    self._model.add_constraints(first_peak_constraints)
                    
                        
                for sample, timestep in sample_timestep_product:
                    if future_counters["peak_period_counter"][timestep] == self._Delta_P or (self._surrogate and timestep == timesteps[-1]):
                        consumption_power_member_action = repartition_keys_action[("grid import", member, sample, timestep)] / (self._Delta_M * self._Delta_C)
                        production_power_member_action = repartition_keys_action[("grid export", member, sample, timestep)] / (self._Delta_M * self._Delta_C)
                        #TODO : 
                        #Fetch all metering periods of that current peak period, including the first state if end of metering period but not of peak period
                        #Get the timestep of the last peak period if it exists
                        #If so also takes them if Delta_P_prime > 1
                        offtake_peaks = []
                        injection_peaks = []
                        cur_timestep = timestep - 1
                        while cur_timestep >= 0 and future_counters["peak_period_counter"][cur_timestep] != self._Delta_P:
                            if future_counters["nb_timesteps_elapsed_current_metering_period"][cur_timestep] == self._Delta_M:
                                offtake_peaks.insert(0, repartition_keys_action[("grid import", member, sample, cur_timestep)] / (self._Delta_M * self._Delta_C))
                                injection_peaks.insert(0, repartition_keys_action[("grid export", member, sample, cur_timestep)] / (self._Delta_M * self._Delta_C))
                            cur_timestep -= 1
                        if cur_timestep == -1:
                            if state["peak_period_counter"] != self._Delta_P and state["nb_timesteps_elapsed_current_metering_period"] == self._Delta_M:
                                offtake_peaks.insert(0, repartition_keys_first_action[("grid import", member)] / (self._Delta_C * self._Delta_M))
                                injection_peaks.insert(0, repartition_keys_first_action[("grid export", member)] / (self._Delta_C * self._Delta_M))
                                if self._activate_first_step_lower_bound_current_peaks:
                                    offtake_peaks.insert(0, state[(member, "current_offtake_peak")])
                                    injection_peaks.insert(0, state[(member, "current_injection_peak")])
                            #TODO : Not sure about this case but atm we consider that Delta_P_prime = 1
                            if self._Delta_P_prime > 1 and self._activate_first_step_lower_bound_history_peaks:
                                offtake_peaks = list(flatten((state[(member, "offtake_peaks")] + ([[repartition_keys_first_action[("grid import", member)] / (self._Delta_C * self._Delta_M), state[(member, "current_offtake_peak")]]] if self._activate_first_step_lower_bound_current_peaks else [repartition_keys_first_action[("grid import", member)] / (self._Delta_C * self._Delta_M)]) if state["peak_period_counter"] == self._Delta_P else [])[-(self._Delta_P_prime - 1):] + offtake_peaks))
                                injection_peaks = list(flatten((state[(member, "injection_peaks")] + ([[repartition_keys_first_action[("grid export", member)] / (self._Delta_C * self._Delta_M), state[(member, "current_injection_peak")]]] if self._activate_first_step_lower_bound_current_peaks else [repartition_keys_first_action[("grid export", member)] / (self._Delta_C * self._Delta_M)]) if state["peak_period_counter"] == self._Delta_P else [])[-(self._Delta_P_prime - 1):] + injection_peaks))
                        else:
                            #TODO : Not sure about this case but atm we consider that Delta_P_prime = 1
                            if self._Delta_P_prime > 1:
                                offtake_peaks_behind = (
                                    list(np.repeat(state[(member, "offtake_peaks")], self._Delta_P)) +
                                    ([repartition_keys_first_action[("grid import", member)] / (self._Delta_C * self._Delta_M)] if (state["nb_timesteps_elapsed_current_metering_period"] == self._Delta_M) else []) + 
                                    list([repartition_keys_action[("grid import", member, sample, t)] / (self._Delta_M * self._Delta_C) for t in range(cur_timestep) if future_counters["nb_timesteps_elapsed_current_metering_period"][t] == self._Delta_M]) +
                                    [repartition_keys_action[("grid import", member, sample, cur_timestep)] / (self._Delta_M * self._Delta_C) ]
                                )[-(((self._Delta_P_prime-1)*self._Delta_P)):]
                                injection_peaks_behind = (
                                        (list(np.repeat(state[(member, "injection_peaks")], self._Delta_P))) +
                                        ([repartition_keys_first_action[("grid export", member)] / (self._Delta_C * self._Delta_M)] if (state["nb_timesteps_elapsed_current_metering_period"] == self._Delta_M) else []) + 
                                        list([repartition_keys_action[("grid export", member, sample, t)] / (self._Delta_M * self._Delta_C) for t in range(cur_timestep) if future_counters["nb_timesteps_elapsed_current_metering_period"][t] == self._Delta_M]) +
                                        [repartition_keys_action[("grid export", member, sample, cur_timestep)] / (self._Delta_M * self._Delta_C) ]
                                    )[-(((self._Delta_P_prime-1)*self._Delta_P)):]
                            else:
                                offtake_peaks_behind = []
                                injection_peaks_behind = []
                            offtake_peaks = offtake_peaks_behind + offtake_peaks
                            injection_peaks = injection_peaks_behind + injection_peaks
                        offtake_peaks.append(consumption_power_member_action)
                        injection_peaks.append(production_power_member_action)
                        for offtake_peak, injection_peak in zip(offtake_peaks, injection_peaks):
                            self._model.add_constraints([
                                peaks_state[("offtake peaks", member, sample, timestep)] >= offtake_peak,
                                peaks_state[("injection peaks", member, sample, timestep)] >= injection_peak
                            ]) 
                        """
                        offtake_peaks = (
                            list(np.repeat(state[(member, "offtake_peaks")], self._Delta_P*self._Delta_M)) +
                            [repartition_keys_first_action[("grid import", member)] / (self._Delta_C * self._Delta_M)] + 
                            list([repartition_keys_action[("grid import", member, sample, t)] / (self._Delta_M * self._Delta_C) for t in range(timestep)]) +
                            [consumption_power_member_action]
                        )[-((self._Delta_P_prime*self._Delta_P*self._Delta_M)):]
                        injection_peaks = (
                            list(np.repeat(state[(member, "injection_peaks")], self._Delta_P*self._Delta_M)) +
                            [repartition_keys_first_action[("grid export", member)] / (self._Delta_C * self._Delta_M)] + 
                            list([repartition_keys_action[("grid export", member, sample, t)] / (self._Delta_M * self._Delta_C) for t in range(timestep)]) +
                            [production_power_member_action]
                        )[-((self._Delta_P_prime*self._Delta_P*self._Delta_M)):]
                        #if member == "C":
                            #print(timestep, offtake_peaks)
                        for offtake_peak, injection_peak in zip(offtake_peaks, injection_peaks):
                            self._model.add_constraints([
                                peaks_state[("offtake peaks", member, sample, timestep)] >= offtake_peak,
                                peaks_state[("injection peaks", member, sample, timestep)] >= injection_peak
                            ]) 
                        
                        """

        
        if full_sequence_of_actions:
            self._timesteps = timesteps
            self._future_counters_lst = future_counters
            self._controllable_assets_action = controllable_assets_action
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
                        self._obj_formula <= objective_value + 1e-7,
                        self._obj_formula >= objective_value - 1e-7 
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
            solution = self._model.solve(log_output=self._verbose)
            solve_status = self._model.solve_status
        
        #print(solution)
        #print(solution.get_objective_value())
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
            for key_controllable_assets_action in self._controllable_assets_action_space.keys():
                output_action[key_controllable_assets_action] = solution.get_value(controllable_assets_first_action[key_controllable_assets_action])
            if full_sequence_of_actions:
                controllable_assets_action_converted = [
                    {key: epsilonify(solution.get_value(self._controllable_assets_action[(key, 0, timestep)])) for key in self._controllable_assets_action_space.keys()} for timestep in self._timesteps
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
            "peak_period_counter": future_peak_period_counter
        }

    def _action(self, state: Dict[str, Any], exogenous_variable_members: Dict[Tuple[str, str], List[float]], exogenous_prices: Dict[Tuple[str, str], List[float]]) -> Dict[Any, float]:
        controllable_assets_first_action = self._build_planning_model(state, exogenous_variable_members, exogenous_prices)
        return self._solve(state, controllable_assets_first_action)
    
    def sequence_of_actions(self, state: Dict[str, Any], exogenous_variable_members: Dict[Tuple[str, str], List[float]], exogenous_prices: Dict[Tuple[str, str], List[float]]) -> Dict[Any, float]:
        controllable_assets_first_action = self._build_planning_model(state, exogenous_variable_members, exogenous_prices, full_sequence_of_actions=True)
        return self._solve(state, controllable_assets_first_action, full_sequence_of_actions=True)
