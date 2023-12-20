from typing import Callable, List, Optional, Tuple, Union, Dict, Any
import uuid
from gym import Env
from gym.core import ObsType, ActType, RenderFrame
from gym.spaces import Dict as DictSpace, Box, Discrete, Sequence
import numpy as np
from enum import Enum
from base import ConditionalProbabilityDistribution, IneqType
import warnings
from env.global_bill_adaptative_optimiser import GlobalBillAdaptativeOptimiser
from utils.utils import epsilonify, merge_dicts, net_value
from exceptions import *
from copy import deepcopy
from time import time


COST_BREAKING_CONSTRAINT = 10000



np_hstack = np.hstack
np_append = np.append

def get_support_min(v: Union[List[float], ConditionalProbabilityDistribution]) -> float:
    if isinstance(v, list):
        return min(v)
    else:
        return v.support()[0]

def get_support_max(v: Union[List[float], ConditionalProbabilityDistribution]) -> float:
    if isinstance(v, list):
        return max(v)
    else:
        return v.support()[1]

def sample_initial_state(v: Union[float, ConditionalProbabilityDistribution]):
    if type(v) in [float, int, np.float32, np.float64]:
        value = v
    else:
        value = v.sample()
    return value

def sample_exogenous_variable(v: Union[List[float], ConditionalProbabilityDistribution], max_length=1):
    if isinstance(v, list):
        values = v[:max_length]
    else:
        values = v.sample(length=max_length)
    return list([value for value in values])




"""
    TODO:
    - Think about what to do when actions are non-feasible
"""
class InFeasibleActionProcess(Enum):
    STOP_WITH_BIG_PENALTY = 0
    PROJECT_TO_CLOSEST_FEASIBLE_ACTION = 1





class RecEnv(Env):

    def clone(self, **kwargs):
        members = kwargs.get("members", self._members)
        states_controllable_assets_with_dynamics = kwargs.get("states_controllable_assets_with_dynamics", self._states_controllable_assets)
        exogenous_variables_members = kwargs.get("exogenous_variables_members", self._exogenous_variables_members_initialiser)
        exogenous_variables_members_buying_prices = kwargs.get("exogenous_variables_members_buying_prices", self._exogenous_variables_members_buying_prices_initialiser)
        exogenous_variables_members_selling_prices = kwargs.get("exogenous_variables_members_selling_prices", self._exogenous_variables_members_selling_prices_initialiser)
        actions_controllable_assets = kwargs.get("actions_controllable_assets", self._actions_controllable_assets)
        feasible_actions_controllable_assets = kwargs.get("feasible_actions_controllable_assets", self._feasible_actions_controllable_assets)
        consumption_function = kwargs.get("consumption_function", self._consumption_function)
        production_function = kwargs.get("production_function", self._production_function)
        Delta_C = kwargs.get("Delta_C", self._Delta_C)
        Delta_M = kwargs.get("Delta_M", self._Delta_M)
        Delta_P = kwargs.get("Delta_P", self._Delta_P)
        Delta_P_prime = kwargs.get("Delta_P_prime", self._Delta_P_prime)
        T = kwargs.get("T", self._T)
        current_offtake_peak_cost = kwargs.get("current_offtake_peak_cost", self._current_offtake_peak_cost)
        current_injection_peak_cost = kwargs.get("current_injection_peak_cost", self._current_injection_peak_cost)
        historical_offtake_peak_cost = kwargs.get("historical_offtake_peak_cost", self._historical_offtake_peak_cost)
        historical_injection_peak_cost = kwargs.get("historical_injection_peak_cost", self._historical_injection_peak_cost)
        cost_function_controllable_assets = kwargs.get("cost_function_controllable_assets", self._cost_function_controllable_assets)
        disable_warnings = kwargs.get("disable_warnings", self._disable_warnings)
        env_name = kwargs.get("env_name", self._env_name)
        global_bill_optimiser_enable_greedy_init = kwargs.get("global_bill_optimiser_enable_greedy_init", self._global_bill_optimiser_enable_greedy_init)
        incremental_build_flag = kwargs.get("incremental_build_flag", self._incremental_build_flag)
        n_cpus_global_bill_optimiser = kwargs.get("n_cpus_global_bill_optimiser", self._n_cpus_global_bill_optimiser)
        precision = kwargs.get("precision", self._precision)
        rec_import_fees = kwargs.get("rec_import_fees", self._rec_import_fees)
        rec_export_fees = kwargs.get("rec_export_fees", self._rec_export_fees)
        disable_global_bill_trigger = kwargs.get("disable_global_bill_trigger", self._disable_global_bill_trigger)
        compute_global_bill_on_next_observ = kwargs.get("compute_global_bill_on_next_observ", self._compute_global_bill_on_next_observ)
        type_solver=kwargs.get("type_solver", self._type_solver)
        force_optim_no_peak_costs=kwargs.get("force_optim_no_peak_costs", False)
        return RecEnv(
            members,
            states_controllable_assets_with_dynamics,
            exogenous_variables_members,
            exogenous_variables_members_buying_prices,
            exogenous_variables_members_selling_prices,
            actions_controllable_assets,
            feasible_actions_controllable_assets,
            consumption_function,
            production_function,
            Delta_C = Delta_C,
            Delta_M = Delta_M,
            Delta_P = Delta_P,
            Delta_P_prime = Delta_P_prime,
            T = T,
            current_offtake_peak_cost = current_offtake_peak_cost,
            current_injection_peak_cost = current_injection_peak_cost,
            historical_offtake_peak_cost = historical_offtake_peak_cost,
            historical_injection_peak_cost = historical_injection_peak_cost, 
            cost_function_controllable_assets = cost_function_controllable_assets,
            disable_warnings = disable_warnings,
            env_name = env_name,
            global_bill_optimiser_enable_greedy_init = global_bill_optimiser_enable_greedy_init,
            incremental_build_flag = incremental_build_flag,
            n_cpus_global_bill_optimiser=n_cpus_global_bill_optimiser,
            precision=precision,
            rec_import_fees=rec_import_fees,
            rec_export_fees=rec_export_fees,
            disable_global_bill_trigger=disable_global_bill_trigger,
            compute_global_bill_on_next_observ=compute_global_bill_on_next_observ,
            type_solver=type_solver,
            force_optim_no_peak_costs=force_optim_no_peak_costs
        )

    def __init__(self,
                 members: List[str],
                 states_controllable_assets_with_dynamics: Dict[str, Dict[str, Tuple[Union[Tuple[int, int, int], Tuple[float, float, float], ConditionalProbabilityDistribution], Callable[[float, Dict[Tuple[str, str], float], Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]], Dict[Tuple[str, str], float]]]]], 
                 exogenous_variables_members: Dict[str, Dict[str, Union[List[float], ConditionalProbabilityDistribution]]],
                 exogenous_variables_members_buying_prices: Dict[str, Union[List[float], ConditionalProbabilityDistribution]],
                 exogenous_variables_members_selling_prices: Dict[str, Union[List[float], ConditionalProbabilityDistribution]],
                 actions_controllable_assets: Dict[str, Dict[str, Tuple[float, float]]],
                 feasible_actions_controllable_assets: Dict[str, Callable[[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]], Tuple[Any, float, IneqType]]],
                 consumption_function: Dict[str, Callable[[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float], Optional[Dict[Tuple[str, str], float]]], float]],
                 production_function: Dict[str, Callable[[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float], Optional[Dict[Tuple[str, str], float]]], float]],
                 Delta_C: float= 1.0,
                 Delta_M: int = 1,
                 Delta_P: int = 1,
                 Delta_P_prime: int = 0,
                 T: int = 1,
                 current_offtake_peak_cost: float = 0.0,
                 current_injection_peak_cost: float = 0.0,
                 historical_offtake_peak_cost: float = 0.0,
                 historical_injection_peak_cost: float = 0.0, 
                 cost_function_controllable_assets: Dict[str, List[Callable[[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float], Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]], float]]] = dict(),
                 disable_warnings=True,
                 env_name=None,
                 global_bill_optimiser_enable_greedy_init=False,
                 incremental_build_flag=False,
                 n_cpus_global_bill_optimiser=None,
                 precision=1e-6,
                 rec_import_fees=0.0,
                 rec_export_fees=0.0,
                 disable_global_bill_trigger=False,
                 compute_global_bill_on_next_observ=False,
                 type_solver="mosek",
                 force_optim_no_peak_costs=False

                 
        ):
        self._disable_global_bill_trigger = disable_global_bill_trigger
        self._just_initialized = False
        self._type_solver = type_solver
        """
            Checking phase
            TODO : check for list lengths for exogenous variables
            check that initial state correspond to support
        """
        if not set(states_controllable_assets_with_dynamics.keys()).issubset(set(members)):
            raise MismatchMembersWithStateAndExogenousDicts(
                f"Members in controllable states dict should be subset of list of members. List of members:{members}, Members in controllable assets:{list(states_controllable_assets_with_dynamics.keys())}"
            )

        for state in states_controllable_assets_with_dynamics.values():
            for state_name, (initial_state, min_value, max_value, _) in state.items():
                if not isinstance(initial_state, ConditionalProbabilityDistribution):
                    if initial_state < min_value:
                        raise OutOfBounds(f"State {state_name} initial value is lower than its minimum value (current value = {initial_state}, minimum value = {min_value})")
                    elif initial_state > max_value:
                        raise OutOfBounds(f"State {state_name} initial value is greater than its maximum value (current value = {initial_state}, maximum value = {max_value})")

        if not set(exogenous_variables_members.keys()).issubset(set(members)):
            raise MismatchMembersWithStateAndExogenousDicts(
                f"Members in controllable exogenous variable dict should be subset of list of members. List of members:{members}, Members in controllable assets:{list(exogenous_variables_members.keys())}"
            )

        for exogenous_value in exogenous_variables_members.values():
            for exogenous_name, exogenous_generator in exogenous_value.items():
                if isinstance(exogenous_generator, list):
                    if len(exogenous_generator) < T:
                        raise NotEnoughExogenousData(
                            f"The length of the list {exogenous_name} is lower than the time horizon T={T} (length = {len(exogenous_generator)})"
                        )


        if not set(exogenous_variables_members_buying_prices.keys()) == set(members):
            raise MismatchMembersWithStateAndExogenousDicts(
                f"Buying price sequence or cpd should be provided for each member. List of members:{members}, Members in controllable assets:{list(exogenous_variables_members_buying_prices.keys())}"
            )

        for member, exogenous_price_generator in exogenous_variables_members_buying_prices.items():
            if not isinstance(exogenous_price_generator, ConditionalProbabilityDistribution):
                if len(exogenous_price_generator) < np.floor(T/Delta_M):
                    raise NotEnoughExogenousData(
                        f"The length of the buying price list of member {member} is lower than the time horizon T={T} (length = {len(exogenous_price_generator)})"
                    )
                

        if not set(exogenous_variables_members_selling_prices.keys()) == set(members):
            raise MismatchMembersWithStateAndExogenousDicts(
                f"Selling price sequence or cpd should be provided for each member. List of members:{members}, Members in controllable assets:{list(exogenous_variables_members_buying_prices.keys())}"
            )

        for member, exogenous_price_generator in exogenous_variables_members_selling_prices.items():
            if not isinstance(exogenous_price_generator, ConditionalProbabilityDistribution):
                if len(exogenous_price_generator) < np.floor(T/Delta_M):
                    raise NotEnoughExogenousData(
                        f"The length of the selling price list of member {member} is lower than the time horizon of metering periods T={T} Delta_M={Delta_M} (length = {len(exogenous_price_generator)})"
                    )

        if not set(consumption_function.keys()) == set(members):
            raise MismatchMembersWithStateAndExogenousDicts(
                f"Consumption function should be provided for each member. List of members:{members}, Members for which consumption function is provided:{list(consumption_function.keys())}"
            )

        if not set(production_function.keys()) == set(members):
            raise MismatchMembersWithStateAndExogenousDicts(
                f"Production function should be provided for each member. List of members:{members}, Members for which consumption function is provided:{list(production_function.keys())}"
            )

        if T < 1:
            print(f"Time horizon T needs to be strictly positive, current value: {T}")

        if Delta_C <= 0:
            print(f"Delta_C needs to be strictly positive, current value: {Delta_C}")

        if Delta_C > 1:
            print(f"Delta_C needs to be in ]0, 1], current value: {Delta_C}")

        if Delta_M <= 0:
            print(f"Delta_M needs to be strictly positive, current value: {Delta_M}")

        if Delta_P <= 0:
            print(f"Delta_P needs to be strictly positive, current value: {Delta_P}")

        if Delta_P_prime < 0:
            print(f"Delta_P_prime needs to be positive, current value: {Delta_P_prime}")

        if current_offtake_peak_cost < 0:
            print(f"Offtake peak cost needs to be be non-negative. Current value: {current_offtake_peak_cost}")

        if current_injection_peak_cost < 0:
            print(f"Injection peak cost needs to be be non-negative. Current value: {current_injection_peak_cost}")

        if historical_offtake_peak_cost < 0:
            print(f"Offtake peak cost needs to be be non-negative. Current value: {current_offtake_peak_cost}")

        if historical_injection_peak_cost < 0:
            print(f"Injection peak cost needs to be be non-negative. Current value: {current_injection_peak_cost}")


        """
            Building the observation(state+exogenous) space and the action (controllable assets) stage
        """
        self._disable_warnings = disable_warnings
        self._T = T
        self._t = None
        self._Delta_P_prime = Delta_P_prime
        self._Delta_P = Delta_P
        self._Delta_M = Delta_M
        self._Delta_C = Delta_C
        self._n_members = len(members)
        self._members = members
        self._states_controllable_assets = states_controllable_assets_with_dynamics
        
        self._consumption_function = consumption_function
        self._production_function = production_function
        self._exogenous_variables_members_initialiser = exogenous_variables_members
        self._exogenous_variables_members_buying_prices_initialiser = exogenous_variables_members_buying_prices
        self._exogenous_variables_members_selling_prices_initialiser = exogenous_variables_members_selling_prices
        self._exogenous_variables_members_raw = None
        self._exogenous_variables_members_buying_prices_raw = None
        self._exogenous_variables_members_selling_prices_raw = None
        self._actions_controllable_assets = actions_controllable_assets
        self._feasible_actions_controllable_assets = feasible_actions_controllable_assets
        self._feasible_actions = dict(feasible_actions_controllable_assets)
        
        self._current_offtake_peak_cost = current_offtake_peak_cost
        self._current_injection_peak_cost = current_injection_peak_cost
        self._historical_offtake_peak_cost = historical_offtake_peak_cost
        self._historical_injection_peak_cost = historical_injection_peak_cost
        self._env_name = env_name if env_name is not None else str(uuid.uuid4())
        self._global_bill_optimiser_enable_greedy_init = global_bill_optimiser_enable_greedy_init
        self._incremental_build_flag = incremental_build_flag
        self._n_cpus_global_bill_optimiser = n_cpus_global_bill_optimiser
        self._precision = precision
        self._rec_import_fees = rec_import_fees
        self._rec_export_fees = rec_export_fees
        self._compute_global_bill_on_next_observ = compute_global_bill_on_next_observ
        
        self._controllable_assets_state_space = {
            (member, state_name): Box(x[1], x[2], dtype=np.float32, shape=())
            for member, states in states_controllable_assets_with_dynamics.items()
            for state_name, x in states.items()
        }
        constant_price_per_member = False
        if type(exogenous_variables_members_buying_prices) == dict and type(exogenous_variables_members_selling_prices) == dict:
            constant_price_per_member = list(exogenous_variables_members_buying_prices.values()) + list(exogenous_variables_members_selling_prices.values())
            constant_price_per_member = list(set([len(set(p)) for p in constant_price_per_member]))
            constant_price_per_member = len(constant_price_per_member) == 1 and constant_price_per_member[0] == 1
        self._global_bill_adaptative_optimiser = GlobalBillAdaptativeOptimiser(
            members=self._members,
            current_offtake_peak_cost=self._current_offtake_peak_cost,
            current_injection_peak_cost=self._current_injection_peak_cost,
            historical_offtake_peak_cost=self._historical_offtake_peak_cost,
            historical_injection_peak_cost=self._historical_injection_peak_cost,
            Delta_M=Delta_M,
            Delta_C=Delta_C,
            Delta_P=Delta_P,
            Delta_P_prime=Delta_P_prime,
            id_optimiser=None,
            incremental_build=self._incremental_build_flag,
            greedy_init=self._global_bill_optimiser_enable_greedy_init,
            n_cpus=n_cpus_global_bill_optimiser,
            rec_import_fees=rec_import_fees,
            rec_export_fees=rec_export_fees,
            constant_price_per_member=constant_price_per_member,
            type_solve=self._type_solver,
            activate_optim_no_peak_costs=force_optim_no_peak_costs,
            force_optim_no_peak_costs=force_optim_no_peak_costs
        )
        self._involve_current_peaks = self._global_bill_adaptative_optimiser.involve_current_peaks
        self._involve_historical_peaks = self._global_bill_adaptative_optimiser.involve_historical_peaks
        self._involve_peaks = self._global_bill_adaptative_optimiser.involve_peaks

        lst_states = [
            # Controllable assets states
            self._controllable_assets_state_space,
            # Counter for number of time steps elapsed in the current metering period
            {
                "metering_period_counter": Discrete(Delta_M+1),
            },
            # Electricity consumption metering period meter for each member (stores consumption to be satisfied inside REC when surrogate)
            {
                (member, "consumption_meters"): Sequence(Box(0, 1000000, dtype=np.float32, shape=()))
                for member in members
            },
            # Electricity produced in the current metering period for each member (stores production to be reallocated inside REC when surrogate)
            {
                (member, "production_meters"): Sequence(Box(0, 1000000, dtype=np.float32, shape=()))
                for member in members
            }
            #Previous metering period cost
            #{
            #    "previous_metering_period_cost": Box(0, 1000000, dtype=np.float32, shape=())
            #}
        ]
        if self._involve_peaks:
            # Counter for number of metering periods in the current peak period
            lst_states += [
                {
                    "peak_period_counter": Discrete(Delta_P+1)
                    #"previous_peak_period_cost": Box(0, 1000000, dtype=np.float32, shape=())
                }
            ]
            if self._involve_historical_peaks:
                if self._historical_offtake_peak_cost > 0:
                    # Historical offtake-peak state for each member
                    lst_states += [
                        {
                            (member, "historical_offtake_peaks"): Sequence(Box(0, 1000000, dtype=np.float32))
                            for member in members
                        }
                    ]
                if self._historical_injection_peak_cost > 0:
                    # Historical injection-peak state for each member
                    lst_states += [
                        {
                            (member, "historical_injection_peaks"): Sequence(Box(0, 1000000, dtype=np.float32))
                            for member in members
                        }
                    ]
        """
        if self._current_offtake_peak_cost > 0:
            # Current offtake-peak state for each member
            lst_states += [
                {
                    (member, "current_offtake_peak"): Box(0, 1000000, dtype=np.float32, shape=())
                    for member in members
                }
            ]
        if self._current_injection_peak_cost > 0:
            # Current injection-peak state for each member
            lst_states += [
                {
                    (member, "current_injection_peak"): Box(0, 1000000, dtype=np.float32, shape=())
                    for member in members
                }
            ]
        if self._historical_offtake_peak_cost > 0:
            # Historical offtake-peak state for each member
            lst_states += [
                {
                    (member, "historical_offtake_peaks"): Box(0, 1000000, shape=(self._Delta_P_prime-1,))
                    for member in members
                }
            ]
        if self._historical_injection_peak_cost > 0:
            # Historical injection-peak state for each member
            lst_states += [
                {
                    (member, "historical_injection_peaks"): Box(0, 1000000, shape=(self._Delta_P_prime-1,))
                    for member in members
                }
            ]
        """
        states = merge_dicts(lst_states)
        exogenous_variables = merge_dicts([
            #Exogenous variables related to members
            {
                (member, exogenous_variable_name): Sequence(Box(get_support_min(x), get_support_max(x), dtype=np.float32))
                for member, exogenous_variables in self._exogenous_variables_members_initialiser.items()
                for exogenous_variable_name, x in exogenous_variables.items()
            },
            # Exogenous variable related to buying prices
            {
                (member, "buying_price"): Sequence(Box(0, 1000000, dtype=np.float32, shape=()))
                for member in members
            },
            # Exogenous variable related to selling prices
            {
                (member, "selling_price"): Sequence(Box(0, 1000000, dtype=np.float32, shape=()))
                for member in members
            }
        ])
        self._controllable_assets_action_space = {
            (member, action_name): Box(x[0], x[1], dtype=np.float32, shape=())
            for member, actions in actions_controllable_assets.items()
            for action_name, x in actions.items()
        }
        self.action_space = DictSpace(self._controllable_assets_action_space)
        """
        self.action_space = DictSpace(
            # Actions related to controllable assets
            merge_dicts(
                [
                    self._controllable_assets_action_space,
                    ({
                        "metering_period_bill_trigger": Discrete(1),
                        "peak_period_bill_trigger": Discrete(1)
                    } if self._enable_cost_triggers_as_actions else {})
                ]
            )
        )
        """

        self._controllable_assets_dynamics = {
            (member, state_name): self._states_controllable_assets[member][state_name][-1]
            for member in self._states_controllable_assets.keys()
            for state_name in self._states_controllable_assets[member].keys()
        }

        self._cost_function_controllable_assets = cost_function_controllable_assets
        
        self._state_space = DictSpace(states)
        self._exogenous_space = DictSpace(exogenous_variables)
        self._observation_space = DictSpace({**states, **exogenous_variables})
        self.observation_space = self._observation_space
        self._reward_range = Box(-1000000, 1000000, dtype=np.float32)
        #todo : set global dynamics and global cost functions (see paper)
        self._metering_period_counter = None
        self._peak_period_counter = None
        self._consumption_meters = {
            member:None for member in self._members
        }
        self._production_meters = {
            member:None for member in self._members
        }
        self._counters_states = None
        self._meters_states = None
        self._peaks_states = None
        self._controllable_assets_states = None
        self._exogenous_variables_members = None
        self._exogenous_variables_prices = None
        self._projector = None
        self._t = None
        self._Tm = None
        self._len_cost_functions = len(list(self._cost_function_controllable_assets.values()))

    @property
    def global_bill_adaptative_optimiser(self):
        return self._global_bill_adaptative_optimiser
    
    @property
    def type_solver(self):
        return self._type_solver

    @property
    def involve_peaks(self):
        return self._involve_peaks
    
    @property
    def involve_current_peaks(self):
        return self._involve_current_peaks
    
    @property
    def involve_historical_peaks(self):
        return self._involve_historical_peaks

    @property
    def controllable_assets_dynamics(self):
        return dict(self._controllable_assets_dynamics)

    @property
    def controllable_assets_state_space(self):
        return dict(self._controllable_assets_state_space)

    @property
    def controllable_assets_action_space(self):
        return dict(self._controllable_assets_action_space)

    @property
    def Delta_C(self):
        return self._Delta_C

    @property
    def Delta_M(self):
        return self._Delta_M
    
    @property
    def Delta_P(self):
        return self._Delta_P

    @property
    def Delta_P_prime(self):
        return self._Delta_P_prime

    @property
    def current_offtake_peak_cost(self):
        return self._current_offtake_peak_cost

    @property
    def current_injection_peak_cost(self):
        return self._current_injection_peak_cost
    
    @property
    def historical_offtake_peak_cost(self):
        return self._historical_offtake_peak_cost

    @property
    def historical_injection_peak_cost(self):
        return self._historical_injection_peak_cost

    @property
    def T(self):
        return self._T
    
    @T.setter
    def T(self, T=None):
        if not self._just_initialized:
            raise BaseException("Cannot modify time horizon after stepping in resetted env")
        if T is not None:
            self._T = T

    @property
    def disable_global_bill_trigger(self):
        return self._disable_global_bill_trigger
    
    @disable_global_bill_trigger.setter
    def disable_global_bill_trigger(self, flag):
        self._disable_global_bill_trigger = flag

    @property
    def members(self):
        return self._members

    @property
    def env_name(self):
        return self._env_name
    
    @property
    def projector(self):
        return self._projector
    
    @property
    def counters_states(self):
        return self._counters_states
    
    @property
    def feasible_actions_controllable_assets(self):
        return self._feasible_actions_controllable_assets

    @property
    def cost_function_controllable_assets(self):
        return self._cost_function_controllable_assets
    
    @property
    def consumption_function(self):
        return self._consumption_function
    
    @property
    def production_function(self):
        return self._production_function
    
    @projector.setter
    def projector(self, projector):
        self._projector = projector

    @property
    def exogenous_space(self):
        return self._exogenous_space
    
    @property
    def rec_import_fees(self):
        return self._rec_import_fees
    
    @property
    def rec_export_fees(self):
        return self._rec_export_fees
    
    @property
    def compute_global_bill_on_next_observ(self):
        return self._compute_global_bill_on_next_observ
    
    def set_exogenous_variables_members(self, exogenous_variables_members: Dict[str, List[float]]):
        if not self._just_initialized:
            raise BaseException("Cannot modify exogenous values after stepping in resetted env")
        for exogenous_variable_member_key, exogenous_variable_member_key_values in exogenous_variables_members.items():
            self._exogenous_variables_members_raw[exogenous_variable_member_key] = exogenous_variable_member_key_values

    def set_buying_prices(self, buying_prices_members: Dict[str, List[float]]):
        if not self._just_initialized:
            raise BaseException("Cannot modify exogenous values after stepping in resetted env")
        for member, buying_prices_values in buying_prices_members.items():
            self._exogenous_variables_members_prices_raw[(member, "buying_price")] = buying_prices_values

    def set_selling_prices(self, selling_prices_members: Dict[str, List[float]]):
        if not self._just_initialized:
            raise BaseException("Cannot modify exogenous values after stepping in resetted env")
        for member, selling_prices_values in selling_prices_members.items():
            self._exogenous_variables_members_prices_raw[(member, "selling_price")] = selling_prices_values

            

    def step(self, action: ActType):
        self._just_initialized=False
        return self._step(
            action,
            project_action=True,
            projected_action=False,
            previous_action=dict()
        )

    def _step(self, action: ActType, project_action: bool = True, projected_action: bool = False, previous_action: dict = dict()) -> Tuple[ObsType, float, bool, bool, dict]:
        if self._t is None:
            raise NotInitializedEnv("Env not initialized. Please reset the environment.")
        if self._t == self._T:
            raise ReachedTimeLimitEnv("Time limit reached in current env instance. Please reset the environment.")
        """
            Checking that the action is actually feasible
        """
        for action_key, action_value in action.items():
            action[action_key] = np.clip(
                float(epsilonify(action_value, epsilon=self._precision)),
                np.round(self._controllable_assets_action_space[action_key].low, 6),
                np.round(self._controllable_assets_action_space[action_key].high, 6)
            )
        is_action_feasible, id_constraint_break, lhs_value, rhs_value = self._is_action_feasible(action)
        if is_action_feasible:
            current_observation = self._current_observation_dict
            current_state = self.compute_current_state()
            current_exogenous_sequences = self.compute_current_exogenous_sequences()
            t = self._t
            if not self._compute_global_bill_on_next_observ:
                is_metering_period_cost_triggered = (not self._involve_peaks and self.is_end_of_metering_period(current_observation["counters_states"])) or self.is_end_of_peak_period(current_observation["counters_states"])
                is_peak_period_cost_triggered = self.is_end_of_peak_period(current_observation["counters_states"])
                offtake_peaks = None
                injection_peaks = None
                
                if not self._disable_global_bill_trigger and (is_metering_period_cost_triggered or is_peak_period_cost_triggered):
                    metering_period_cost, peak_period_cost, offtake_peaks, injection_peaks = self._global_bill_adaptative_optimiser.optimise_global_bill(
                        current_state,
                        current_exogenous_sequences
                    )
                else:
                    metering_period_cost = 0
                    peak_period_cost = 0
                metering_period_cost *= is_metering_period_cost_triggered
                if peak_period_cost is not None:
                    peak_period_cost *= is_peak_period_cost_triggered
                else:
                    peak_period_cost = 0
            #print(current_observation["meters_states"][("PVB", "consumption_meters")][-1])
            
            self._update_controllable_assets(current_observation["controllable_assets_states"], current_observation["exogenous_variables_members"], action)
            self._update_counters(current_observation["counters_states"])

            self._update_meters(current_observation["counters_states"], current_observation["controllable_assets_states"], current_observation["exogenous_variables_members"],action)
            #t = time()
            self._update_members_exogenous_variables(self._t)
            self._update_exogenous_variables_prices(current_observation["counters_states"])
            #print("Time spent to update exogenous stuff",time() - t, flush=True)
            if self._compute_global_bill_on_next_observ:
                current_observation = self._compute_current_observation()
                current_state = self.compute_current_state()
                current_exogenous_sequences = self.compute_current_exogenous_sequences()
                is_metering_period_cost_triggered = (not self._involve_peaks and self.is_end_of_metering_period(current_observation["counters_states"])) or self.is_end_of_peak_period(current_observation["counters_states"])
                is_peak_period_cost_triggered = self.is_end_of_peak_period(current_observation["counters_states"])
                offtake_peaks = None
                injection_peaks = None
                
                if not self._disable_global_bill_trigger and (is_metering_period_cost_triggered or is_peak_period_cost_triggered):
                    metering_period_cost, peak_period_cost, offtake_peaks, injection_peaks = self._global_bill_adaptative_optimiser.optimise_global_bill(
                        current_state,
                        current_exogenous_sequences
                    )
                else:
                    metering_period_cost = 0
                    peak_period_cost = 0
                metering_period_cost *= is_metering_period_cost_triggered
                if peak_period_cost is not None:
                    peak_period_cost *= is_peak_period_cost_triggered
                else:
                    peak_period_cost = 0

            
            current_peaks = None
            if self._involve_historical_peaks:
                if offtake_peaks is not None and injection_peaks is not None:
                    current_peaks = dict()
                    if offtake_peaks is not None:
                        current_peaks = {
                            **current_peaks,
                            **{
                                (member, "offtake_peaks"): offtake_peaks[member] for member in self._members
                            }
                        }
                    if injection_peaks is not None:
                        current_peaks = {
                            **current_peaks,
                            **{
                                (member, "injection_peaks"): injection_peaks[member] for member in self._members
                            }
                        }
                self._update_historical_peaks(current_observation["historical_peaks_states"], current_peaks)

            
            self._current_observation_dict = self._compute_current_observation_dict()
            self._current_observation = self._compute_current_observation()


            #self._update_previous_costs(current_observation["counters_states"], metering_period_cost, peak_period_cost)
            if self._len_cost_functions > 0:
                next_state = self.compute_current_state()
                operational_cost = sum([
                sum([cost_function(next_state,current_exogenous_sequences,action, next_state) for cost_function in cost_functions]) for cost_functions in self._cost_function_controllable_assets.values()
                ])
            else:
                operational_cost = 0.0

            next_is_metering_period_cost_triggered = (not self._involve_peaks and self.is_end_of_metering_period(self._counters_states)) or self.is_end_of_peak_period(self._counters_states)
            next_is_peak_period_cost_triggered = self.is_end_of_peak_period(self._counters_states)
            info = {
                "is_metering_period_cost_triggered":is_metering_period_cost_triggered,
                "is_peak_period_cost_triggered":is_metering_period_cost_triggered,
                "costs": {"controllable_assets_cost": operational_cost,
                     "metering_period_cost": metering_period_cost,
                     "peak_period_cost": peak_period_cost},
                "is_action_projected": projected_action,
                "previous_action_projected": previous_action,
                "time_horizon": self._T,
                "current_t": self._t,
                "next_step_cost_triggered":next_is_peak_period_cost_triggered or next_is_metering_period_cost_triggered
            }
            metering_period_cost = metering_period_cost if metering_period_cost is not None else 0.0
            peak_period_cost = peak_period_cost if peak_period_cost is not None else 0.0
            
            cost = operational_cost + metering_period_cost + peak_period_cost
            #print(is_metering_period_cost_triggered, is_peak_period_cost_triggered, cost)
            #cost = ((metering_period_cost + peak_period_cost) - ((current_observation["previous_costs"]["previous_metering_period_cost"] + current_observation["previous_costs"]["previous_peak_period_cost"]) if self._surrogate else 0.0))
            #print(cost)
            return (self._current_observation,
                    cost,
                    False,
                    self._t == self._T,
                    info)
        else:
            obs = self._current_observation
            text = f"Infeasible action detected (id constraint broken: {id_constraint_break}, lhs_value={lhs_value}, rhs_value={rhs_value})"
            if not project_action or self.projector is None:
                if not self._disable_warnings:
                    warnings.warn(f"{text} => Terminal state with big penalty.")
                info = {
                    "is_metering_period_cost_triggered":False,
                    "is_peak_period_cost_triggered":False,
                    "costs": {"controllable_assets_cost": np.nan,
                        "metering_period_cost": np.nan,
                        "peak_period_cost": np.nan},
                    "is_action_projected": projected_action,
                    "previous_action_projected": previous_action,
                    "infeasible_action": action,
                    "id_constraint_break": id_constraint_break,
                    "projector": self.projector
                }
                return obs, COST_BREAKING_CONSTRAINT, True, True, info
            else:
                if not self._disable_warnings:
                    pass#warnings.warn(f"{text} => {self._projector.project_type()}. Action : {action}")
                current_state = self.compute_current_state()
                current_exogenous_sequences = self.compute_current_exogenous_sequences()
                projected_action = self._projector.project_action(current_state, current_exogenous_sequences, action)
                return self._step(projected_action, project_action=False, projected_action=True, previous_action=action)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[ObsType, dict]:
        super().reset(seed=seed, options=options)
        self._just_initialized=True
        self._t = 0
        self._nb_metering_periods_elapsed=0
        self._global_bill_adaptative_optimiser.reset()
        self._controllable_assets_states = self._reset_controllable_assets_states()
        self._counters_states = self._reset_counters()
        self._meters_states = self._reset_meters()
        if self._involve_historical_peaks:
            self._historical_peaks_states = self._reset_historical_peaks()
        self._exogenous_variables_members = self._reset_members_exogenous_variables()
        self._exogenous_variables_prices = self._reset_exogenous_variables_prices()
        #self._previous_costs = self._reset_previous_costs()
        self._current_observation_dict = self._compute_current_observation_dict()
        self._current_observation = self._compute_current_observation()
        return self._current_observation

    def _compute_current_observation(self):
        return {**self._counters_states,
                **self._meters_states,
                **(self._historical_peaks_states if self._involve_historical_peaks else {}),
                **self._controllable_assets_states,
                **self._observe_members_exogenous_variables(),
                **self._observe_prices_exogenous_variables()}

    def compute_current_state(self):
        return {
            **self._counters_states,
            **self._meters_states,
            **(self._historical_peaks_states if self._involve_historical_peaks else {}),
            **self._controllable_assets_states,
        }

    def compute_current_exogenous_sequences(self):
        return {
            **self._observe_members_exogenous_variables(),
            **self._observe_prices_exogenous_variables()
        }

    def _compute_current_observation_dict(self):
        obs_dict = {"counters_states": self._counters_states,
                "meters_states": self._meters_states,
                "controllable_assets_states": self._controllable_assets_states,
                "exogenous_variables_members": self._observe_members_exogenous_variables(),
                "exogenous_variables_prices": self._observe_prices_exogenous_variables()
        }
        if self._involve_historical_peaks:
            obs_dict["historical_peaks_states"] = self._historical_peaks_states
        return obs_dict

    def _compute_ctrl_assets_costs(self, current_observation_dict, next_observation_dict, action):
        #todo : compute costs
        
        return (
            {member: cost_function_controllable_assets(
                current_observation_dict["controllable_assets_states"],
                current_observation_dict["exogenous_variables_members"],
                action,
                next_observation_dict["controllable_assets_states"]
            ) for member in self._cost_function_controllable_assets.keys() for cost_function_controllable_assets in self._cost_function_controllable_assets[member]}
        )
    """
    def _reset_previous_costs(self):
        previous_costs = {
            "previous_metering_period_cost": 0
        }
        return previous_costs if "previous_peak_period_cost" not in list(self._observation_space.keys()) else {
            **previous_costs,
            **{
                "previous_peak_period_cost": 0.0
            }
        }
    """
        
    def _reset_controllable_assets_states(self):
        self._controllable_assets_states = (
            {
                (member, state_name): sample_initial_state(x[0])
                for member, states in self._states_controllable_assets.items()
                for state_name, x in states.items()
            }
        )
        return self._controllable_assets_states
    
    def _reset_counters(self):
        #todo : fix the representation of the counter observation
        counters = {
            "metering_period_counter": 0
        }
        if self._involve_peaks:
            counters["peak_period_counter"] = 0
        return counters

    def _reset_meters(self):
        return merge_dicts([
            {
                (member, "consumption_meters"): np.zeros(1)
                for member in self._members
            },
            {
                (member, "production_meters"): np.zeros(1)
                for member in self._members
            }
        ])

    def _reset_historical_peaks(self):
        return merge_dicts([
            ({
                (member, "historical_offtake_peaks"): []
                for member in self._members
            } if self._historical_offtake_peak_cost > 0 else {}),
            ({
                (member, "historical_injection_peaks"): []
                for member in self._members
            } if self._historical_injection_peak_cost > 0 else {})
        ])
    
    def _observe_members_exogenous_variables(self):
        """
        if max_length is None:
            max_length = self._t+1
        return {
            member_exogenous_variable_key: (x[:max(max_length,1)])
            for member_exogenous_variable_key, x in self._exogenous_variables_members_raw.items()
        }
        """
        return self._exogenous_variables_members
    
    def observe_all_members_exogenous_variables(self):
        return dict(self._exogenous_variables_members_raw)
    
    def observe_all_exogenous_variables(self):
        return merge_dicts(
            [
                self.observe_all_members_exogenous_variables(),
                self.observe_all_raw_prices_exogenous_variables()
            ]
        )

    def _reset_members_exogenous_variables(self):
        self._exogenous_variables_members_raw = {
            (member, exogenous_variable_name): np.asarray(sample_exogenous_variable(x, max_length=self._T))
            for member, exogenous_variables in self._exogenous_variables_members_initialiser.items()
            for exogenous_variable_name, x in exogenous_variables.items()
        }
        self._exogenous_variables_members = {
            exogenous_variable_key:exogenous_variable_value[:1] for exogenous_variable_key, exogenous_variable_value in self._exogenous_variables_members_raw.items()
        }
        
        return self._observe_members_exogenous_variables()
    
    def sample_members_exogenous_variables(self):
        return {
            (member, exogenous_variable_name): np.asarray(sample_exogenous_variable(x, max_length=self._T))
            for member, exogenous_variables in self._exogenous_variables_members_initialiser.items()
            for exogenous_variable_name, x in exogenous_variables.items()
        }
    
    def _update_members_exogenous_variables(self, t:int):
        t_prime = min(t, self._T-1)
        self._exogenous_variables_members = {
            exogenous_variable_key:np_append(exogenous_variable_value, self._exogenous_variables_members_raw[exogenous_variable_key][t_prime]) for exogenous_variable_key, exogenous_variable_value in self._exogenous_variables_members.items()
        }

    def _observe_prices_exogenous_variables(self):
        """
        if max_length is None:
            max_length=self._nb_metering_periods_elapsed+1
        return {
           member_price_key: member_price_lst[:max_length] for member_price_key, member_price_lst in self._exogenous_variables_members_prices_raw.items()
        }
        """
        return self._exogenous_variables_members_prices
    
    def observe_all_raw_prices_exogenous_variables(self):
        return self._exogenous_variables_members_prices_raw

    def _reset_exogenous_variables_prices(self):
        self._exogenous_variables_members_prices_raw = merge_dicts([{
            (member, "buying_price"): np.asarray(sample_exogenous_variable(x, max_length=int(np.floor(self._T/self._Delta_M))))
            for member, x in self._exogenous_variables_members_buying_prices_initialiser.items()
        },
        {
            (member, "selling_price"): np.asarray(sample_exogenous_variable(x, max_length=int(np.floor(self._T/self._Delta_M))))
            for member, x in self._exogenous_variables_members_selling_prices_initialiser.items()
        }])
        self._exogenous_variables_members_prices = {
            (member, price): self._exogenous_variables_members_prices_raw[(member, price)][:1] for member in self._members for price in ("buying_price", "selling_price")
        }
        if self._Tm is None:
            self._Tm = len(list(self._exogenous_variables_members_prices_raw.values())[0])
        return self._observe_prices_exogenous_variables()

    def _update_exogenous_variables_prices(self, counter_states):
        if self.is_end_of_metering_period(counter_states):
            nb_metering_periods_elapsed = min(self._nb_metering_periods_elapsed, self._Tm-1)
            self._exogenous_variables_members_prices = {
                (member, price): np_append(v, self._exogenous_variables_members_prices_raw[(member, price)][nb_metering_periods_elapsed]) for (member, price), v in self._exogenous_variables_members_prices.items()
            }
    
    def _update_controllable_assets(self, controllable_assets_states, members_exogenous_variables, action):
        for state_key in controllable_assets_states:
            current_state = controllable_assets_states[state_key]
            transition_function = self._controllable_assets_dynamics[state_key]
            self._controllable_assets_states[state_key] = float(np.clip(
                epsilonify(transition_function(current_state, controllable_assets_states, members_exogenous_variables, action), epsilon=self._precision),
                self._controllable_assets_state_space[state_key].low,
                self._controllable_assets_state_space[state_key].high
            ))


    def _update_counters(self, counter_states):
        self._t += 1
        metering_period_counter = counter_states["metering_period_counter"]
        if not self.is_end_of_metering_period(counter_states):
            metering_period_counter += 1
        else:
            metering_period_counter = 1
            self._nb_metering_periods_elapsed += 1
        new_counter_states = dict()
        new_counter_states["metering_period_counter"] = metering_period_counter
    
        if self._involve_peaks:
            peak_period_counter = counter_states["peak_period_counter"]
            new_counter_states["peak_period_counter"] = peak_period_counter
            if self.is_end_of_peak_period(counter_states=new_counter_states):
                peak_period_counter = 0
            elif self.is_end_of_metering_period(counter_states=new_counter_states):
                peak_period_counter += 1
            new_counter_states["peak_period_counter"] = peak_period_counter
            
        self._counters_states = new_counter_states

    """
    def _update_previous_costs(self, counter_states, metering_period_cost, peak_period_cost):
        if metering_period_cost is not None:
            if ("peak_period_counter" not in counter_states and self.is_end_of_metering_period(counter_states)) or self.is_end_of_peak_period(counter_states):
                self._previous_costs["previous_metering_period_cost"] = 0.0
            else:
                self._previous_costs["previous_metering_period_cost"] = epsilonify(metering_period_cost)

        if peak_period_cost is not None:
            if self.is_end_of_peak_period(counter_states):
                self._previous_costs["previous_peak_period_cost"] = 0.0
            else:
                self._previous_costs["previous_peak_period_cost"] = epsilonify(peak_period_cost)
    """

    def _update_meters(self, counter_states, controllable_assets_states, exogenous_variables_members, u):
        for member in self._members:
            consumption_member = (
                self._consumption_function[member](controllable_assets_states,
                                                    exogenous_variables_members,
                                                    u)
            )
            production_member = (
                self._production_function[member](controllable_assets_states,
                                                    exogenous_variables_members,
                                                    u)
            )
            net_consumption = epsilonify(net_value(consumption_member, production_member), epsilon=self._precision)
            net_production = epsilonify(net_value(production_member, consumption_member), epsilon=self._precision)
            if self.is_end_of_peak_period(counter_states) or (not self._involve_peaks and self.is_end_of_metering_period(counter_states)):
                self._meters_states[(member, "consumption_meters")] = np.zeros(1)
                self._meters_states[(member, "production_meters")] = np.zeros(1)
            elif self.is_end_of_metering_period(counter_states):
                self._meters_states[(member, "consumption_meters")] = np.append(self._meters_states[(member, "consumption_meters")], 0.0)
                self._meters_states[(member, "production_meters")] = np.append(self._meters_states[(member, "production_meters")], 0.0)
            self._meters_states[(member, "consumption_meters")][-1] += net_consumption
            self._meters_states[(member, "production_meters")][-1] += net_production

    
    def _update_historical_peaks(self, historical_peak_states, current_peaks):
        if self._involve_historical_peaks:
            for member in self._members:
                if self._historical_offtake_peak_cost > 0:
                    current_offtake_peak = max(
                        historical_peak_states[(member, "historical_offtake_peaks")] +
                        [current_peaks[(member, "offtake_peak")] if current_peaks is not None else [0]]
                    )
                    self._historical_peaks_states[(member, "historical_offtake_peaks")] = (
                        historical_peak_states[(member, "historical_offtake_peaks")] + [current_offtake_peak]
                    )[-self._Delta_P_prime:]
                if self._historical_injection_peak_cost > 0:
                    current_injection_peak = max(
                        historical_peak_states[(member, "historical_injection_peaks")] +
                        [current_peaks[(member, "injection_peak")] if current_peaks is not None else [0]]
                    )
                    self._historical_peaks_states[(member, "historical_injection_peaks")] = (
                        historical_peak_states[(member, "historical_injection_peaks")] + [current_injection_peak]
                    )[-self._Delta_P_prime:]

    def _is_action_feasible(self, action: ActType) -> bool:
        action = dict(action)
        current_observation = self._compute_current_observation()
        for constraint_id, constraint_funct in self._feasible_actions.items():
            
            constraint_tuple = constraint_funct(current_observation, current_observation, action)
            if constraint_tuple is not None:
                lhs_value, rhs_value, constraint_type = constraint_tuple
                lhs_value = epsilonify(lhs_value, epsilon=self._precision)
                if type(rhs_value) not in (list, tuple):
                    rhs_value = epsilonify(rhs_value, epsilon=self._precision)
                else:
                    rhs_value = tuple(epsilonify(r, epsilon=self._precision) for r in rhs_value)
                if constraint_type == IneqType.EQUALS:
                    if epsilonify(abs(lhs_value - rhs_value), epsilon=self._precision) != 0:
                        return False, constraint_id, lhs_value, rhs_value
                elif constraint_type == IneqType.LOWER_OR_EQUALS:
                    if lhs_value > rhs_value + self._precision:
                        return False, constraint_id, lhs_value, rhs_value
                elif constraint_type == IneqType.GREATER_OR_EQUALS:
                    if lhs_value < rhs_value - self._precision:
                        return False, constraint_id, lhs_value, rhs_value
                elif constraint_type == IneqType.MUTEX:
                    if epsilonify(abs(lhs_value)*abs(rhs_value), epsilon=self._precision) > 0:
                        return False, constraint_id, lhs_value, rhs_value
                elif constraint_type == IneqType.BOUNDS:
                    rhs_value_1, rhs_value_2 = rhs_value
                    if lhs_value < rhs_value_1 - self._precision or lhs_value > rhs_value_2 + self._precision:
                        return False, constraint_id, lhs_value, rhs_value
        return True, None, None, None

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass

    def is_end_of_metering_period(self, counter_states):
        return counter_states["metering_period_counter"] == self._Delta_M 

    def is_end_of_peak_period(self, counter_states):
        return self._involve_peaks and (counter_states["peak_period_counter"] == self._Delta_P)

    def _net_consumption(self, consumption: float, production: float) -> float:
        return max(consumption - production, 0.0)

    def _net_production(self, production: float, consumption: float) -> float:
        return max(production - consumption, 0.0)    
