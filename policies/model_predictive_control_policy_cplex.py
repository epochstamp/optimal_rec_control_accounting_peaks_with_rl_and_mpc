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
from policies.model_predictive_control_policy_solver_agnostic import ModelPredictiveControlPolicySolverAgnostic
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

class ModelPredictiveControlPolicyCplex(ModelPredictiveControlPolicySolverAgnostic):

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
                 n_threads=None,
                 small_penalty_control_actions=0.0,
                 gamma=1.0,
                 rescaled_gamma_mode="no_rescale",
                 solver_config="none",
                 solution_chained_optimisation=False,
                 disable_env_ctrl_assets_constraints=False,
                 rec_import_fees=0.0,
                 rec_export_fees=0.0,
                 members_with_controllable_assets=[],
                 disable_sos=False):
        super().__init__(members,
                         controllable_assets_state_space,
                         controllable_assets_action_space,
                         constraints_controllable_assets,
                         consumption_function,
                         production_function,
                         dynamics_controllable_assets,
                         exogenous_provider,
                         cost_functions_controllable_assets,
                         T,
                         n_samples,
                         max_length_samples,
                         Delta_C,
                         Delta_M,
                         Delta_P,
                         Delta_P_prime,
                         current_offtake_peak_cost,
                         current_injection_peak_cost,
                         historical_offtake_peak_cost,
                         historical_injection_peak_cost,
                         force_last_time_step_to_global_bill,
                         verbose,
                         net_consumption_production_mutex_before,
                         n_threads,
                         small_penalty_control_actions,
                         gamma,
                         rescaled_gamma_mode,
                         solver_config=solver_config,
                         solution_chained_optimisation=solution_chained_optimisation,
                         disable_env_ctrl_assets_constraints=disable_env_ctrl_assets_constraints,
                         rec_import_fees=rec_import_fees,
                         rec_export_fees=rec_export_fees,
                         members_with_controllable_assets=members_with_controllable_assets,
                         disable_sos=disable_sos)
        self._counter = 1

    def _create_or_get_model(self):
        self._model = Model()
        if self._n_threads is not None:
            self._model.parameters.threads = self._n_threads
        self._model.parameters.mip.tolerances.mipgap = 1e-6
        self._model.parameters.mip.tolerances.absmipgap = 0.0001
        self._solver_configurator(self._model)
        return self._model
    
    def _value(self, var):
        return float(var)
    
    def _internal_prepare(self):
        pass
    
    def _sum(self, lst_exprs):
        return self._model.sum(lst_exprs)

    def _add_sos_1(self, expr_1, expr_2, w1=0, w2=1):
        return self._model.add_sos1([expr_1, expr_2])
    
    def _max_over_zero(self, expr):
        return None
    
    def _create_variables(self, lst_keys, lb=-np.inf, ub=np.inf, key_pattern = lambda k: str(k), boolean=False):
        keys = list(product(*lst_keys)) if len(lst_keys) > 1 else lst_keys[0]
        if boolean:

            return self._model.binary_var_dict(
                keys
            )
        else:
            return self._model.continuous_var_dict(
                keys,
                lb=lb,
                ub=ub
            )

    def _set_minimize_objective(self, obj_expr):
        self._model.minimize(obj_expr)
    
    def _solve_by_solver(self, controllable_assets_actions):
        #self._model.parameters.mip.tolerances.mipgap=1e-5
        #self._model.parameters.mip.tolerances.absmipgap=0.001
        #self._model.parameters.mip.limits.gomorypass=1
        #self._model.parameters.mip.strategy.heuristiceffort = 0
        #self._model.parameters.mip.strategy.variableselect = 3
        #self._model.parameters.mip.strategy.probe = 3
        #self._model.parameters.benders.strategy = 3
        #self._model.parameters.emphasis.mip = 2
        #self._model.parameters.simplex.pgradient = 1
        #self._model.parameters.mip.limits.cutpasses = 3
        #self._model.parameters.mip.cuts.nodecuts = 3
        #self._model.parameters.mip.cuts.mircut = 2
        solution = self._model.solve(log_output=self._verbose)
        if self._model.solve_status == JobSolveStatus.OPTIMAL_SOLUTION:
            
            controllable_assets_action_dict = solution.get_value_dict(controllable_assets_actions, keep_zeros=True)
           
            #d = dict()
            
            """
            if self._rec_exchanges is not None:
                for k,v in self._rec_exchanges.items():
                    k2 = (k[0], k[1])
                    if k[0] not in ("total import", "total export", "rec import", "rec export"):
                        if k2 not in d.keys():
                            d[k2] = []
                        d[k2] += [float(v)]
                d = {
                    k:np.asarray(v) for k,v in d.items()
                }
                from pprint import pprint
                pprint(d)
            d = dict()
            for k,v in self._net_consumption_production_vars.items():
                k2 = (k[0], k[1])
                if k2 not in d.keys():
                    d[k2] = 0
                d[k2] += float(v)
            print(d)
            from pprint import pprint
            pprint(
                {
                    k:float(v) for k,v in self._rec_exchanges.items()
                }
            )
            
            exit()
            d = dict()
            for k,v in self._net_consumption_production_vars.items():
                k2 = (k[0], k[1])
                if k2 not in d.keys():
                    d[k2] = 0
                d[k2] += float(v)
            print(d)
            
            """
            #if self._previous_net_production_meters is not None:
            #    print(self._previous_net_production_meters["PVB"])
            return solution, controllable_assets_action_dict, 0
        else:
            return None, None, self._model.solve_status
    
    def _commit_constraints(self):
        self._model.add_constraints(self._constraints)