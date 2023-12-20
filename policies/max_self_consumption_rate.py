from base import IneqType
from typing import Dict, Any, Callable, Tuple, List, Optional
from gym.spaces import Dict as DictSpace
from policies.no_action_policy import NoActionPolicy
from policies.simple_policy import SimplePolicy
from utils.utils import epsilonify, merge_dicts, roundify
import cvxpy as cp
import numpy as np
import mosek.fusion as mk
M = 100000

TYPE="mosek"

class MaxSelfConsumptionRate(SimplePolicy):

    def __init__(self,
                 members: List[str],
                 controllable_assets_state_space: DictSpace,
                 controllable_assets_action_space: DictSpace,
                 constraints_controllable_assets: Dict[str, Callable[[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]], Tuple[Any, float, IneqType]]],
                 consumption_function: Dict[str, Callable[[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float], Optional[Dict[Tuple[str, str], float]]], float]],
                 production_function: Dict[str, Callable[[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float], Optional[Dict[Tuple[str, str], float]]], float]],
                 exogenous_space: DictSpace,
                 sum_self_consumption=False,
                 members_with_controllable_assets=[],
                 use_meters=True,
                 Delta_M=1):
        super().__init__(
            members,
            controllable_assets_state_space,
            controllable_assets_action_space,
            constraints_controllable_assets,
            consumption_function,
            production_function,
            exogenous_space,
            members_with_controllable_assets=members_with_controllable_assets,
            Delta_M=Delta_M
        )
        self._sum_self_consumption=sum_self_consumption
        self._prob = None
        self._no_action_policy = None
        self._use_meters = use_meters
        self._len_members = len(members)

    def _configure_solver_mosek(self, M):
        M.setSolverParam("presolveUse", "on")
        M.setSolverParam("numThreads", 1)
        M.setSolverParam("log", 0)
        M.setSolverParam("simPrimalRestrictSelection", 0)
        M.setSolverParam("simPrimalSelection", "ase")
        M.setSolverParam("simDualRestrictSelection", 0)
        M.setSolverParam("simDualSelection", "ase")
        M.setSolverParam("simExploitDupvec", "off")
        M.setSolverParam("simMaxNumSetbacks", 0)
        M.setSolverParam("simNonSingular", "off")
        M.setSolverParam("writeLpFullObj", "off")
        M.setSolverParam("simHotstart", "free")

    def _create_prob_mosek(self, exogenous_member_variables:dict):
        self._prob = mk.Model()
        self._configure_solver_mosek(self._prob)

        if self._sum_self_consumption:
            self._self_consumption_var = self._prob.variable()
        else:
            self._self_consumption_var = self._prob.variable(len(self._members_with_controllable_assets))
        len_controllable_assets_action_space_keys = len(list(self._controllable_assets_action_space_keys))
        len_controllable_assets_state_space_keys = len(list(self._controllable_assets_state_space_keys))
        self._ctrl_assets_action_vars = {
            k:self._prob.variable(mk.Domain.inRange(float(self._controllable_assets_action_space[k].low), float(self._controllable_assets_action_space[k].high))) for k in self._controllable_assets_action_space_keys
        }
        self._objective_expr = self._self_consumption_var if self._sum_self_consumption else mk.Expr.sum(self._self_consumption_var)
        self._ctrl_assets_state_parameter = self._prob.parameter(len_controllable_assets_state_space_keys)
        self._ctrl_assets_state_parameter_dict = {
            key: self._ctrl_assets_state_parameter.index(i) for i, key in enumerate(self._controllable_assets_state_space.keys())
        }
        self._exogenous_space_keys = exogenous_member_variables.keys()
        self._exogenous_variables_parameter = self._prob.parameter(len(self._exogenous_space_keys))
        self._exogenous_variables_dict = {
            key: [self._exogenous_variables_parameter.index(i)] for i, key in enumerate(self._exogenous_space_keys)
        }
        self._ctrl_assets_action_variable_dict = {
            key: self._ctrl_assets_action_vars[key] for i, key in enumerate(self._controllable_assets_action_space.keys())
        }
        if self._use_meters:
            self._net_production_meters = self._prob.parameter(self._len_members)
            self._net_consumption_meters = self._prob.parameter(self._len_members)
            self._net_meters_dict = {
                **{
                    (member, "net_production_meters"): self._net_production_meters.index(i)
                    for i, member in enumerate(self._members)
                },
                **{
                    (member, "net_consumption_meters"): self._net_consumption_meters.index(i)
                    for i, member in enumerate(self._members)
                }
            }
        else:
            self._net_meters_dict = {
                **{
                    (member, "net_production_meters"): mk.Expr.constTerm(0.0)
                    for member in self._members
                },
                **{
                    (member, "net_consumption_meters"): mk.Expr.constTerm(0.0)
                    for member in self._members
                }
            }
        member_test = 'FRIANDA-SA'
        consumption_production_pairs = [
            (member, mk.Expr.add(self._net_meters_dict[(member, "net_consumption_meters")], self._consumption_function[member](
                self._ctrl_assets_state_parameter_dict, self._exogenous_variables_dict, self._ctrl_assets_action_variable_dict,
                op_add=mk.Expr.add,
                op_mul=mk.Expr.mul,
                op_div=lambda n,m: mk.Expr.mul(n, 1.0/m),
                op_sub=mk.Expr.sub
            )),
            mk.Expr.add(self._net_meters_dict[(member, "net_production_meters")], self._production_function[member](
                self._ctrl_assets_state_parameter_dict, self._exogenous_variables_dict, self._ctrl_assets_action_variable_dict,
                op_add=mk.Expr.add,
                op_mul=mk.Expr.mul,
                op_div=lambda n,m: mk.Expr.mul(n, 1.0/m),
                op_sub=mk.Expr.sub
            )))
            for member in self._members

        ]
        
        if self._sum_self_consumption:
            consumption_production_pairs_1_stack = mk.Expr.hstack([
                (c[1] if type(c[1]) not in (float, np.float32, np.float64, int, np.int32, np.int64) else mk.Expr.constTerm(c[1])) for c in consumption_production_pairs
            ])
            consumption_production_pairs_2_stack = mk.Expr.hstack([
                (c[2] if type(c[2]) not in (float, np.float32, np.float64, int, np.int32, np.int64) else mk.Expr.constTerm(c[2])) for c in consumption_production_pairs
            ])
            sum_consumptions = mk.Expr.sum(consumption_production_pairs_1_stack)
            sum_productions = mk.Expr.sum(consumption_production_pairs_2_stack)
            self._prob.constraint(
                mk.Expr.sub(self._self_consumption_var, mk.Expr.sub(sum_consumptions, sum_productions)), mk.Domain.greaterThan(0.0)
            )
            self._prob.constraint(
                mk.Expr.sub(self._self_consumption_var, mk.Expr.sub(sum_productions, sum_consumptions)), mk.Domain.greaterThan(0.0)
            )
        else:
            consumption_production_difference_1 = [
                    mk.Expr.sub(cp_pair[1], cp_pair[2]) for cp_pair in consumption_production_pairs if cp_pair[0] in self._members_with_controllable_assets
                ]
            consumption_production_difference_2 = [
                mk.Expr.sub(cp_pair[2], cp_pair[1]) for cp_pair in consumption_production_pairs if cp_pair[0] in self._members_with_controllable_assets
            ]
            for i, _ in enumerate(consumption_production_difference_1):
                self._prob.constraint(
                    mk.Expr.sub(self._self_consumption_var.index(i), consumption_production_difference_1[i]), mk.Domain.greaterThan(0.0)
                )
                self._prob.constraint(
                    mk.Expr.sub(self._self_consumption_var.index(i), consumption_production_difference_2[i]), mk.Domain.greaterThan(0.0)
                )

        for constraint_id, constraint_funct in self._constraints_controllable_assets.items():
            constraint_tuple = constraint_funct(
                self._ctrl_assets_state_parameter_dict,
                self._exogenous_variables_dict,
                self._ctrl_assets_action_variable_dict,
                op_add=mk.Expr.add,
                op_mul=mk.Expr.mul,
                op_div=lambda n,m: mk.Expr.mul(n, 1.0/m),
                op_sub=mk.Expr.sub
            )
            if constraint_tuple is not None:
                lhs_value, rhs_value, constraint_type = constraint_tuple
                if constraint_type == IneqType.EQUALS:
                    self._prob.constraint(
                        mk.Expr.sub(lhs_value, rhs_value), mk.Domain.equalsTo(0.0)
                    )
                elif constraint_type == IneqType.LOWER_OR_EQUALS:
                    self._prob.constraint(
                        mk.Expr.sub(lhs_value, rhs_value), mk.Domain.lessThan(0.0)
                    )
                elif constraint_type == IneqType.GREATER_OR_EQUALS:
                    self._prob.constraint(
                        mk.Expr.sub(lhs_value, rhs_value), mk.Domain.greaterThan(0.0)
                    )
                elif constraint_type == IneqType.BOUNDS:
                    rhs_value_1, rhs_value_2 = rhs_value
                    self._prob.constraint(
                        mk.Expr.sub(lhs_value, rhs_value_1), mk.Domain.greaterThan(0.0)
                    )
                    self._prob.constraint(
                        mk.Expr.sub(lhs_value, rhs_value_2), mk.Domain.lessThan(0.0)
                    )
                elif constraint_type == IneqType.MUTEX:
                    self._prob.disjunction(mk.DJC.term(lhs_value, mk.Domain.equalsTo(0.0)), mk.DJC.term(rhs_value, mk.Domain.equalsTo(0.0)))
        ctrl_actions_vars = mk.Expr.stack(0, [self._ctrl_assets_action_variable_dict[k].asExpr() for k in self._controllable_assets_action_space_keys])
        objective_expr = mk.Expr.add(self._objective_expr, mk.Expr.mul(mk.Expr.sum(ctrl_actions_vars),1e-4))
        self._prob.objective(mk.ObjectiveSense.Minimize, objective_expr)
        return True


    def _create_prob(self, exogenous_member_variables):
        if self._sum_self_consumption:
            self._self_consumption_var = cp.Variable()
        else:
            self._self_consumption_var = cp.Variable(len(self._members_with_controllable_assets))
        self._controllable_assets_action_space_keys = self._controllable_assets_action_space.keys()
        if len(self._controllable_assets_action_space_keys) == 0:
            return False
        self._controllable_assets_state_space_keys = self._controllable_assets_state_space.keys()
        len_controllable_assets_action_space_keys = len(list(self._controllable_assets_action_space_keys))
        len_controllable_assets_state_space_keys = len(list(self._controllable_assets_state_space_keys))
        self._ctrl_assets_vars = cp.Variable(len(list(self._controllable_assets_action_space_keys)))
        self._ctrl_assets_lb = cp.Parameter(len_controllable_assets_action_space_keys, value=[float(self._controllable_assets_action_space[k].low) for k in self._controllable_assets_action_space_keys])
        self._ctrl_assets_ub = cp.Parameter(len_controllable_assets_action_space_keys, value=[float(self._controllable_assets_action_space[k].high) for k in self._controllable_assets_action_space_keys])
        self._objective_expr = self._self_consumption_var if self._sum_self_consumption else cp.sum(self._self_consumption_var)
        self._ctrl_assets_state_parameter = cp.Parameter(len_controllable_assets_state_space_keys)
        self._ctrl_assets_state_parameter_dict = {
            key: self._ctrl_assets_state_parameter[i] for i, key in enumerate(self._controllable_assets_state_space.keys())
        }
        self._exogenous_space_keys = exogenous_member_variables.keys()
        self._exogenous_variables_parameter = cp.Parameter((len(self._exogenous_space_keys), 1))
        self._exogenous_variables_dict = {
            key: self._exogenous_variables_parameter[i] for i, key in enumerate(self._exogenous_space_keys)
        }
        self._ctrl_assets_action_variable_dict = {
            key: self._ctrl_assets_vars[i] for i, key in enumerate(self._controllable_assets_action_space.keys())
        }
        self._basic_constraints = [
            self._ctrl_assets_vars >= self._ctrl_assets_lb,
            self._ctrl_assets_vars <= self._ctrl_assets_ub
        ]
        consumption_production_pairs = [
            (member, self._consumption_function[member](
                self._ctrl_assets_state_parameter_dict, self._exogenous_variables_dict, self._ctrl_assets_action_variable_dict
            ),
            self._production_function[member](
                self._ctrl_assets_state_parameter_dict, self._exogenous_variables_dict, self._ctrl_assets_action_variable_dict
            ))
            for member in self._members

        ]
        
        if self._sum_self_consumption:
            sum_consumptions = cp.sum([consumption_production_pair[1] for consumption_production_pair in consumption_production_pairs])
            sum_productions = cp.sum([consumption_production_pair[2] for consumption_production_pair in consumption_production_pairs])
            self._basic_constraints += [
                self._self_consumption_var >= sum_consumptions - sum_productions,
                self._self_consumption_var >= sum_productions - sum_consumptions
            ]
        else:
            consumption_production_difference_1 = [
                cp_pair[1] - cp_pair[2] for cp_pair in consumption_production_pairs if cp_pair[0] in self._members_with_controllable_assets
            ]
            consumption_production_difference_2 = [
                cp_pair[2] - cp_pair[1] for cp_pair in consumption_production_pairs if cp_pair[0] in self._members_with_controllable_assets
            ]
            self._basic_constraints += [
                self._self_consumption_var >= cp.hstack(consumption_production_difference_1),
                self._self_consumption_var >= cp.hstack(consumption_production_difference_2)
            ]
        #ctrl assets constraints
        equals_group_lhs = []
        equals_group_rhs = []
        le_group_lhs = []
        le_group_rhs = []
        ge_group_lhs = []
        ge_group_rhs = []
        mutex_group_rhs = []
        mutex_group_lhs = []
        for constraint_id, constraint_funct in self._constraints_controllable_assets.items():
            constraint_tuple = constraint_funct(self._ctrl_assets_state_parameter_dict, self._exogenous_variables_dict, self._ctrl_assets_action_variable_dict)
            if constraint_tuple is not None:
                lhs_value, rhs_value, constraint_type = constraint_tuple
                if constraint_type == IneqType.EQUALS:
                    equals_group_lhs.append(lhs_value)
                    equals_group_rhs.append(rhs_value)
                elif constraint_type == IneqType.LOWER_OR_EQUALS:
                    le_group_lhs.append(lhs_value)
                    le_group_rhs.append(rhs_value)
                elif constraint_type == IneqType.GREATER_OR_EQUALS:
                    ge_group_lhs.append(lhs_value)
                    ge_group_rhs.append(rhs_value)
                elif constraint_type == IneqType.BOUNDS:
                    rhs_value_1, rhs_value_2 = rhs_value
                    ge_group_lhs.append(lhs_value)
                    ge_group_rhs.append(rhs_value_1)
                    le_group_lhs.append(lhs_value)
                    le_group_rhs.append(rhs_value_2)
                elif constraint_type == IneqType.MUTEX:
                    mutex_group_rhs.append(rhs_value)
                    mutex_group_lhs.append(lhs_value)
        if len(equals_group_lhs) > 0:
            self._basic_constraints += [
                cp.hstack(equals_group_lhs) == cp.hstack(equals_group_rhs)
            ]
        if len(le_group_lhs) > 0:
            self._basic_constraints += [
                cp.hstack(le_group_lhs) <= cp.hstack(le_group_rhs)
            ]
        if len(ge_group_lhs) > 0:
            self._basic_constraints += [
                cp.hstack(ge_group_lhs) >= cp.hstack(ge_group_rhs)
            ]
        if len(mutex_group_lhs) > 0:
            mutex_bin_var = cp.Variable(len(mutex_group_lhs), boolean=True)
            self._basic_constraints += [
                cp.hstack(mutex_group_lhs) <= mutex_bin_var * M,
                cp.hstack(mutex_group_rhs) <= (1-mutex_bin_var) * M,
                
            ]

        objective_expr = self._objective_expr + cp.sum(self._ctrl_assets_vars)*1e-4
        self._prob = cp.Problem(
            cp.Minimize(objective_expr), self._basic_constraints
        )
        return True

    def _action(self, state: Dict[str, Any], exogenous_variable_members: Dict[Tuple[str, str], List[float]], exogenous_prices: Dict[Tuple[str, str], List[float]]):
        if self._prob is None:
            if TYPE == "cvxpy":
                method = self._create_prob
            else:
                method = self._create_prob_mosek
            if not method(exogenous_variable_members):
                self._no_action_policy = NoActionPolicy(
                    self._members,
                    self._controllable_assets_state_space,
                    self._controllable_assets_action_space,
                    self._constraints_controllable_assets,
                    self._consumption_function,
                    self._production_function,
                    self._exogenous_space
                )
        
        if self._no_action_policy is not None:
            return self._no_action_policy(state, exogenous_variable_members, exogenous_prices)
        else:
            if TYPE == "mosek":
                self._ctrl_assets_state_parameter.setValue([state[k] for k in self._controllable_assets_state_space_keys])
                exogenous_values = np.asarray([exogenous_variable_members[k][-1]  for k in exogenous_variable_members.keys()])
                self._exogenous_variables_parameter.setValue(exogenous_values)
                if self._use_meters:
                    if state["metering_period_counter"] == self._Delta_M:
                        net_consumption_meters = np.zeros(self._len_members)
                        net_production_meters = np.zeros(self._len_members)
                    else:
                        net_consumption_meters = np.asarray([max(state[(m, "consumption_meters")][-1] - state[(m, "production_meters")][-1], 0.0) for i, m in enumerate(self._members)])
                        net_production_meters = np.asarray([max(state[(m, "production_meters")][-1] - state[(m, "consumption_meters")][-1], 0.0) for i, m in enumerate(self._members)])
                    self._net_consumption_meters.setValue(net_consumption_meters)
                    self._net_production_meters.setValue(net_production_meters)
                self._prob.solve()
                ctrl_action = {
                    ctrl_asset_key:round(float(self._ctrl_assets_action_vars[ctrl_asset_key].level()), 3) for i,ctrl_asset_key in enumerate(self._controllable_assets_action_space_keys)
                }
            else:
                self._ctrl_assets_state_parameter.value = [state[k] for k in self._controllable_assets_state_space_keys]
                self._exogenous_variables_parameter.value = np.asarray([exogenous_variable_members[k][-1]  for k in exogenous_variable_members.keys()]).reshape((len(exogenous_variable_members.keys()), 1))
                self._prob.solve()
                ctrl_action = {
                    ctrl_asset_key:round(float(self._ctrl_assets_vars.value[i]), 3) for i,ctrl_asset_key in enumerate(self._controllable_assets_action_space_keys)
                }
            return ctrl_action