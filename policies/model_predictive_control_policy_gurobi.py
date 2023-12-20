from itertools import product
from base import Policy, ExogenousProvider, IneqType
from typing import Dict, Any, Callable, Tuple, List, Optional, Union
import numpy as np
from gurobipy import Model, GRB, max_, Var
from gym.spaces import Dict as DictSpace
from policies.model_predictive_control_policy_solver_agnostic import ModelPredictiveControlPolicySolverAgnostic
from experiment_scripts.mpc.gurobi_params_zoo import gurobi_params_dict
from utils.utils import flatten

M = 10000
EPSILON = 10e-6

class ModelPredictiveControlPolicyGurobi(ModelPredictiveControlPolicySolverAgnostic):

    def __init__(self,
                 members: List[str],
                 controllable_assets_state_space: Dict,
                 controllable_assets_action_space: Dict,
                 constraints_controllable_assets: Dict[str, Callable[[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]], Tuple[Any, float, IneqType]]],
                 consumption_function: Dict[str, Callable[[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float], Dict[Tuple[str, str], float] | None], float]],
                 production_function: Dict[str, Callable[[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float], Dict[Tuple[str, str], float] | None], float]],
                 dynamics_controllable_assets: Dict[Tuple[str, str], Callable[[float | Var, Dict[Tuple[str, str], float], Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]], Dict[Tuple[str, str], float]]],
                 exogenous_provider: ExogenousProvider,
                 cost_functions_controllable_assets: Dict[str, List[Callable[[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float], Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]], float]]] = dict(),
                 T=1,
                 n_samples=1,
                 max_length_samples=1,
                 Delta_C: float = 1,
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
                 small_penalty_control_actions=0,
                 gamma=1,
                 rescaled_gamma_mode="no_rescale",
                 solver_config="none",
                 solution_chained_optimisation=False,
                 disable_env_ctrl_assets_constraints=False,
                 rec_import_fees=0.0,
                 rec_export_fees=0.0,
                 members_with_controllable_assets=[]):
        super().__init__(members,
                         controllable_assets_state_space,
                         controllable_assets_action_space,
                         constraints_controllable_assets,
                         consumption_function,
                         production_function,
                         dynamics_controllable_assets,
                         exogenous_provider,
                         cost_functions_controllable_assets,
                         T=T,
                         n_samples=n_samples,
                         max_length_samples=max_length_samples,
                         Delta_C=Delta_C,
                         Delta_M=Delta_M,
                         Delta_P=Delta_P,
                         Delta_P_prime=Delta_P_prime,
                         current_offtake_peak_cost=current_offtake_peak_cost,
                         current_injection_peak_cost=current_injection_peak_cost,
                         historical_offtake_peak_cost=historical_offtake_peak_cost,
                         historical_injection_peak_cost=historical_injection_peak_cost,
                         force_last_time_step_to_global_bill=force_last_time_step_to_global_bill,
                         verbose=verbose,
                         net_consumption_production_mutex_before=net_consumption_production_mutex_before,
                         n_threads=n_threads,
                         small_penalty_control_actions=small_penalty_control_actions,
                         gamma=gamma,
                         rescaled_gamma_mode=rescaled_gamma_mode,
                         solver_config=solver_config,
                         solution_chained_optimisation=solution_chained_optimisation,
                         disable_env_ctrl_assets_constraints=disable_env_ctrl_assets_constraints,
                         rec_import_fees=rec_import_fees,
                         rec_export_fees=rec_export_fees,
                         members_with_controllable_assets=members_with_controllable_assets)

        
    def _create_or_get_model(self):
        self._model = Model()
        self._model.Params.OutputFlag = self._verbose
        #self._model.Params.NonConvex = 2
        self._model.Params.MIPGap = 1e-6
        self._model.Params.MIPGapAbs = 0.001
        self._model.Params.Presolve = 2
        self._model.Params.MIPFocus = 2
        self._model.Params.SimplexPricing = 0
        self._model.Params.OBBT = 0
        self._model.Params.CutPasses = 3
        self._model.Params.MIRCuts = 0
        if self._n_threads is not None:
            self._model.Params.Threads = self._n_threads
        self._solver_configurator(self._model)
        return self._model
    
    def _internal_prepare(self):
        pass
    
    def _sum(self, lst_exprs):
        return sum(lst_exprs)
    
    def _max_over_zero(self, expr):
        if type(expr) != Var:
            aux_var = self._model.addVar(lb=-np.inf, ub=np.inf)
            self._store_constraint_lst(
                [
                    aux_var == expr
                ]
            )
            return max_(
                aux_var, constant=0.0
            )
        return max_(
            expr, constant=0.0
        )
    
    def _value(self, var):
        return var.x

    def _add_sos_1(self, expr_1, expr_2, w1=0, w2=1):
        return self._model.addSOS(GRB.SOS_TYPE1, [expr_1, expr_2], [w1, w2])
    
    def _create_variables(self, lst_keys, lb=-np.inf, ub=np.inf, key_pattern = lambda k: str(k), boolean=False):
        keys = list(product(*lst_keys)) if len(lst_keys) > 1 else lst_keys[0]
        
        if boolean:

            vars = self._model.addVars(
                *lst_keys,
                vtype=GRB.BINARY
            )
        else:
            if type(lb) in (int, float) and type(ub) in (int, float):
                vars = self._model.addVars(
                    *lst_keys,
                    lb=lb,
                    ub=ub,
                    vtype=GRB.CONTINUOUS
                )
            else:
                
                vars = self._model.addVars(
                    *lst_keys,
                    vtype=GRB.CONTINUOUS
                )
                
                for kk in keys:
                    k = tuple(flatten(kk))
                    vars[k].lb = lb(kk)
                    vars[k].ub = ub(kk)
        vars_dict = dict()
        for k in keys:
            vars_dict[k] = vars[tuple(flatten(k))]
        return vars_dict

    def _set_minimize_objective(self, obj_expr):
        self._model.setObjective(obj_expr, GRB.MINIMIZE)
    
    def _solve_by_solver(self, controllable_assets_actions):
        solution = self._model.optimize()
        
        if self._model.status == GRB.Status.OPTIMAL:
            print(self._model.ObjVal)
            controllable_assets_action_dict = {
                k: v.x for k,v in 
                controllable_assets_actions.items()
            }
            return solution, controllable_assets_action_dict, 0
        else:
            return None, None, self._model.status
    
    def _commit_constraints(self):
        self._model.addConstrs((c for c in self._constraints))