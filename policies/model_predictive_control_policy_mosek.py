from base import Policy, ExogenousProvider, IneqType
from typing import Dict, Any, Callable, Tuple, List, Optional, Union
import numpy as np
from gym.spaces import Dict as DictSpace
from itertools import product, chain
from operator import le, ge, eq
from env.counter_utils import future_counters
from exceptions import InfeasiblePolicy
from policies.model_predictive_control_policy_solver_agnostic import ModelPredictiveControlPolicySolverAgnostic
from utils.utils import epsilonify, merge_dicts, normalize_bounds, rec_gamma_sequence, rindex, roundify, flatten, chunks, split_list, split_list_by_number_np, find_indices
from env.peaks_utils import elapsed_metering_periods_in_peak_period, elapsed_timesteps_in_peak_period, number_of_time_steps_elapsed_in_peak_period
import random
from uuid import uuid4
from time import time
import mosek.fusion as mk
import docplex

M = 10000
EPSILON = 10e-6

def replace_last_one(matrix):
    # Trouver l'indice du premier 1 dans chaque ligne de la matrice inversée
    matrix[np.arange(matrix.shape[0]),(matrix!=0).cumsum(1).argmax(1)] = 0
    return matrix

def generate_matrix_from_binary_vector(vector):
    # Trouver les indices des 1 dans le vecteur
    indices = np.where(vector == 1)[0]

    # Créer une matrice de zéros de la taille nécessaire
    result = np.zeros((len(indices), len(vector)))

    # Mettre un 1 à l'indice correspondant dans chaque ligne
    result[np.arange(len(indices)), indices] = 1

    return result
    

def add_rows(matrix):
    # Ajouter chaque ligne à la précédente
    added_matrix = np.empty(matrix.shape)
    added_matrix[0] = matrix[0]
    added_matrix[1:] = matrix[:-1] + matrix[1:]
    
    return added_matrix


def fill_zeros_between_ones(matrix):
    # Trouver les indices du premier et du dernier "1" dans chaque ligne
    first_one = np.argmax(matrix, axis=1)
    last_one = matrix.shape[1] - np.argmax(np.flip(matrix, axis=1), axis=1) - 1

    # Créer une matrice d'index pour comparer avec first_one et last_one
    idx_matrix = np.indices(matrix.shape)[1]

    # Remplacer les zéros entre les "1" par des "1"
    matrix[(idx_matrix >= np.expand_dims(first_one, axis=1)) & (idx_matrix <= np.expand_dims(last_one, axis=1))] = 1
    
    # Pour la première ligne, remplir les zéros avant le premier "1"
    matrix[0, :first_one[0]] = 1
    
    return matrix


def count_unique_values(vector, value):
    # Obtenir les indices de toutes les occurrences de 'value'
    indices = np.where(vector == value)[0]

    # Inclure le premier indice dans la liste des indices de début
    start_indices = np.concatenate(([0], indices[:-1] + 1))
    end_indices = indices

    # Calculer le nombre de valeurs uniques (sauf 'value') dans chaque sous-vecteur
    unique_counts = [np.unique(vector[start:end+1][vector[start:end+1] != value]).size for start, end in zip(start_indices, end_indices)]
    
    return unique_counts

def create_matrix_2(vector):
    # Créer une grande matrice identité
    matrix = np.eye(len(vector) + np.max(vector))

    # Décaler les rangées en fonction de chaque nombre dans le vecteur
    matrix = np.roll(matrix, vector.cumsum(), axis=1)

    # Tronquer la matrice pour obtenir la taille finale voulue
    matrix = matrix[:len(vector), :len(vector)]
    
    return matrix

def generate_matrix_vectorized(pairs):
    # Calculer la longueur de chaque ligne
    row_lengths = np.max(pairs[:, 1]) + pairs[:, 0]

    # Calculer la longueur totale de la matrice
    total_length = np.max(row_lengths)

    # Créer la matrice initiale de zéros
    matrix = np.zeros((len(pairs), total_length))

    # Calculer les indices de début et de fin pour chaque paire
    starts = pairs[:, 1]
    ends = starts + pairs[:, 0]

    # Créer une matrice d'indices pour la comparaison
    indices = np.arange(total_length)

    # Mettre les "1" aux bonnes positions
    matrix[(indices >= starts[:, None]) & (indices < ends[:, None])] = 1

    return matrix


class ModelPredictiveControlPolicyMosek(ModelPredictiveControlPolicySolverAgnostic):

    def __init__(self,
                 members: List[str],
                 controllable_assets_state_space: DictSpace,
                 controllable_assets_action_space: DictSpace,
                 constraints_controllable_assets: Dict[str, Callable[[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]], Tuple[Any, float, IneqType]]],
                 consumption_function: Dict[str, Callable[[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float], Optional[Dict[Tuple[str, str], float]]], float]],
                 production_function: Dict[str, Callable[[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float], Optional[Dict[Tuple[str, str], float]]], float]],
                 dynamics_controllable_assets: Dict[Tuple[str, str], Callable[[Union[float, mk.Variable], Dict[Tuple[str, str], float], Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]], Dict[Tuple[str, str], float]]],
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
                         members_with_controllable_assets=members_with_controllable_assets)
        self._counter = 1
        self._model = None

    def _internal_prepare(self):
        pass

    def _create_or_get_model(self):
        pass
    
    def _sum(self, lst_exprs):
        pass
    
    def _max_over_zero(self, expr):
        pass

    def _add_sos_1(self, expr_1, expr_2, w1 = 0, w2 = 1):
        pass
    
    def _create_variables(self, lst_keys, lb=-np.inf, ub=np.inf, key_pattern = lambda k: str(k), boolean=False):
        pass

    def _value(self, var):
        return var.level()

    def _set_minimize_objective(self, obj_expr):
        pass
    
    def _solve_by_solver(self, controllable_assets_actions):
        pass

    def _commit_constraints(self):
        pass

    def _build_planning_model(self, state: Dict[Union[str, Tuple[str, str]], Any], exogenous_variable_members: Dict[Tuple[str, str], List[float]], exogenous_prices: Dict[Tuple[str, str], List[float]], full_sequence_of_actions=False) -> None:
        if self._model is None:
            if self._max_length_samples == 0:
                return None
            self._model = mk.Model()
            self._model.setSolverParam("mioTolRelGap", 1.0e-6)
            self._model.setSolverParam("mioTolAbsGap", 0.000001)
            self._model.setSolverParam("presolveUse", "on")
            #self._model.setSolverParam("presolveLindepUse", "on")
            #self._model.setSolverParam("simSolveForm", "primal")
            #self._model.setSolverParam("simSwitchOptimizer", "on")
            #self._model.setSolverParam("simReformulation", "on")
            #self._model.setSolverParam("simScaling", "none")
            self._model.setSolverParam("numThreads", self._n_threads)
            self._model.setSolverParam("log", 10)
            self._first_pass = True
            import sys
            self._model.setLogHandler(sys.stdout)
            #self._model.setSolverParam("simPrimalRestrictSelection", 0)
            #self._model.setSolverParam("simPrimalSelection", "ase")
            #self._model.setSolverParam("simDualRestrictSelection", 0)
            #self._model.setSolverParam("simDualSelection", "ase")
            #self._model.setSolverParam("simExploitDupvec", "on")
            #self._model.setSolverParam("simMaxNumSetbacks", 0)
            #self._model.setSolverParam("simNonSingular", "on")
            #self._model.setSolverParam("simSaveLu", "on")
            self._model.setSolverParam("mioConstructSol", "off")
            #self._model.setSolverParam("mioFeaspumpLevel", "1")
            #self._model.setSolverParam("mioRootOptimizer", "free")
            #self._model.setSolverParam("mioSymmetryLevel", "4")
            #self._model.setSolverParam("mioRootRepeatPresolveLevel", "1")
            #self._model.setSolverParam("mioVbDetectionLevel", "2")
            #self._model.setSolverParam("mioRinsMaxNodes", "0")
            #self._model.setSolverParam("mioHeuristicLevel", "0")

            self._max_length_samples = min(self._max_length_samples, self._T-1)
            self._members_without_controllable_assets = [
                member for member in self._members if member not in self._members_with_controllable_assets
            ]
            full_future_counters_sliding_window_metering_period = np.lib.stride_tricks.sliding_window_view(
                self._future_counters(
                    {"metering_period_counter": state["metering_period_counter"]},
                    duration=self._T-1
                )["metering_period_counter"],
                self._max_length_samples
            )
            self._max_nb_metering_periods = np.max(np.count_nonzero(full_future_counters_sliding_window_metering_period == self._Delta_M, axis=1)) + (
                1 if (self._force_last_time_step_to_global_bill and full_future_counters_sliding_window_metering_period[0][-1] != self._Delta_M)  else 0
            )
            
            if self._involve_peaks:
                full_future_counters_sliding_window_peak_period = np.lib.stride_tricks.sliding_window_view(
                    self._future_counters(
                        {
                            "metering_period_counter": state["metering_period_counter"],
                            "peak_period_counter": state["peak_period_counter"]
                        },
                        duration=self._T-1
                    )["peak_period_counter"],
                    self._max_length_samples
                )
                self._max_nb_peak_periods = np.max(np.count_nonzero(full_future_counters_sliding_window_peak_period == self._Delta_P, axis=1)) + (
                    1 if (self._force_last_time_step_to_global_bill and full_future_counters_sliding_window_peak_period[0][-1] != self._Delta_P) else 0
                )

            #Variables
            self._controllable_assets_actions = {
                k: self._model.variable(self._max_length_samples, mk.Domain.inRange(epsilonify(round(float(self._controllable_assets_action_space[k].low), 6)), epsilonify(round(float(self._controllable_assets_action_space[k].high), 6)))) for k in self._controllable_assets_action_space_keys
            }
            self._controllable_assets_states = {
                k: self._model.variable(self._max_length_samples, mk.Domain.inRange(epsilonify(round(float(self._controllable_assets_state_space[k].low), 6)), epsilonify(round(float(self._controllable_assets_state_space[k].high), 6)))) for k in self._controllable_assets_state_space_keys
            }
            self._rec_exchanges = {
                (k, member): self._model.variable(self._max_nb_metering_periods, mk.Domain.greaterThan(0.0)) for k in [
                    "rec_import", "rec_export", "grid_import", "grid_export", "total_import", "total_export"
                ] for member in self._members
            }
            self._rec_exchanges_bin = {
                member: self._model.variable(self._max_nb_metering_periods, mk.Domain.binary()) for member in self._members
            }
            self._net_consumption_production = {
                (k, member): self._model.variable(self._max_length_samples, mk.Domain.greaterThan(0.0)) for k in [
                    "net_consumption", "net_production"
                ] for member in self._members_with_controllable_assets
            }
            self._net_consumption_production_bin = {
                member: self._model.variable(self._max_length_samples, mk.Domain.binary()) for member in self._members_with_controllable_assets
            }
            self._current_peaks_state = None
            self._previous_rec_exchanges = None
            
            if self._involve_peaks:
                self._lst_current_peaks = []
                if self._current_offtake_peak_cost > 0:
                    self._lst_current_peaks += ["current_offtake_peak"]
                if self._current_injection_peak_cost > 0:
                    self._lst_current_peaks += ["current_injection_peak"]
                self._current_peaks_state = {
                    (k, member): self._model.variable(self._max_nb_peak_periods, mk.Domain.greaterThan(0.0)) for k in self._lst_current_peaks for member in self._members
                }
            
                #TODO if needed : implement for historical peaks
                if self._Delta_P > 1:
                    self._previous_rec_exchanges = {
                        (k, member): self._model.variable(self._Delta_P-1, mk.Domain.greaterThan(0.0)) for k in [
                            "rec_import", "rec_export", "grid_import", "grid_export"
                        ] for member in self._members
                    }
            
            #Parameters
            

            self._net_consumption_param = self._model.parameter(self._max_length_samples, len(self._members_without_controllable_assets))
            self._net_production_param = self._model.parameter(self._max_length_samples, len(self._members_without_controllable_assets))

            self._max_consumption_param = self._model.parameter(self._max_length_samples, len(self._members_without_controllable_assets))
            self._max_production_param = self._model.parameter(self._max_length_samples, len(self._members_without_controllable_assets))
            self._max_cons_meter_param = self._model.parameter(self._max_nb_metering_periods, len(self._members_without_controllable_assets))
            self._max_prod_meter_param = self._model.parameter(self._max_nb_metering_periods, len(self._members_without_controllable_assets))

            self._exogenous_variable_members_keys = exogenous_variable_members.keys()
            
            self._enable_timesteps = self._model.parameter(self._max_length_samples)
            
            self._enable_metering_periods = self._model.parameter(self._max_nb_metering_periods)
            
            self._gammas_tau_m=self._model.parameter(self._max_nb_metering_periods)
            
            self._initial_parametrized_state_param = self._model.parameter(len(self._controllable_assets_state_space_keys))
            self._initial_parametrized_state = {
                k: self._initial_parametrized_state_param.index(i) for i, k in enumerate(self._controllable_assets_state_space_keys)
            }
            
            self._rec_exchange_meters_mask = self._model.parameter(
                self._max_length_samples, self._max_nb_metering_periods
            )
            
            
            self._exogenous_variable_member = {
                k:self._model.parameter() for k,v in self._exogenous_variable_members_keys
            }
            self._buying_exogenous_prices = self._model.parameter(self._max_nb_metering_periods, len(self._members))
            self._selling_exogenous_prices = self._model.parameter(self._max_nb_metering_periods, len(self._members))
            
            self._consumption_meter_terms = self._model.parameter(len(self._members_with_controllable_assets))
            self._production_meter_terms = self._model.parameter(len(self._members_with_controllable_assets))
            self._consumption_meter_terms_dict = {
                member:self._consumption_meter_terms.index(m) #self._model.parameter(len(self._members_with_controllable_assets))
                for m, member in enumerate(self._members_with_controllable_assets)
            }
            self._production_meter_terms_dict = {
                member:self._production_meter_terms.index(m) #self._model.parameter(len(self._members_with_controllable_assets))
                for m, member in enumerate(self._members_with_controllable_assets)
            }

            self._truncated_exogenous_future_sequence = self._model.parameter(self._max_length_samples, len(self._exogenous_variable_members_keys))
            self._truncated_exogenous_future_sequence_lst = [{
                key: self._truncated_exogenous_future_sequence.slice([t, i], [t+1, i+1]).reshape([1]) for i, key in enumerate(self._exogenous_variable_members_keys)
            } for t in range(self._max_length_samples)]
            if self._involve_peaks:
                if self._previous_rec_exchanges is not None:
                    self._enable_previous_rec_exchange = self._model.parameter(self._Delta_P-1)
                    self._previous_net_consumption_meters = self._model.parameter(self._Delta_P-1, len(self._members))
                    self._previous_net_production_meters = self._model.parameter(self._Delta_P-1, len(self._members))
                    self._exogenous_previous_buying_prices = self._model.parameter(self._Delta_P-1, len(self._members))
                    self._exogenous_previous_selling_prices = self._model.parameter(self._Delta_P-1, len(self._members))
                self._rec_exchange_peaks_mask = self._model.parameter(
                    self._max_nb_metering_periods, self._max_nb_peak_periods
                )
                self._gammas_tau_p=self._model.parameter(self._max_nb_peak_periods)
                self._enable_peak_periods = self._model.parameter(self._max_nb_peak_periods)
                self._proratas = self._model.parameter(self._max_nb_peak_periods)
                
            #Objective function
            """
                Meters states costs
            """
            objective_expr = 0
            for m, member in enumerate(self._members):
                if self._involve_peaks and self._previous_rec_exchanges is not None:
                    for tau_m in range(self._Delta_P-1):
                        grid_fees = mk.Expr.sub(
                            mk.Expr.mul(self._exogenous_previous_buying_prices.slice([tau_m, m], [tau_m+1, m+1]), self._previous_rec_exchanges[("grid_import", member)].index(tau_m)),
                            mk.Expr.mul(self._exogenous_previous_selling_prices.slice([tau_m, m], [tau_m+1, m+1]), self._previous_rec_exchanges[("grid_export", member)].index(tau_m))
                        )

                        rec_fees = mk.Expr.add(
                            mk.Expr.mul(self._rec_import_fees, self._previous_rec_exchanges[("rec_import", member)].index(tau_m)),
                            mk.Expr.mul(self._rec_export_fees, self._previous_rec_exchanges[("rec_export", member)].index(tau_m))
                        )
                        objective_expr = mk.Expr.add(objective_expr, mk.Expr.mul(self._enable_previous_rec_exchange.index(tau_m), mk.Expr.add(grid_fees, rec_fees)))

                for tau_m in range(self._max_nb_metering_periods):
                    grid_fees = mk.Expr.mul(mk.Expr.sub(
                            mk.Expr.mul(self._buying_exogenous_prices.slice([tau_m, m], [tau_m+1, m+1]), self._rec_exchanges[("grid_import", member)].index(tau_m)),
                            mk.Expr.mul(self._selling_exogenous_prices.slice([tau_m,m],[tau_m+1,m+1]), self._rec_exchanges[("grid_export", member)].index(tau_m))
                        ), self._enable_metering_periods.index(tau_m))

                    rec_fees = mk.Expr.mul(mk.Expr.add(
                            mk.Expr.mul(self._rec_import_fees, self._rec_exchanges[("rec_import", member)].index(tau_m)),
                            mk.Expr.mul(self._rec_export_fees, self._rec_exchanges[("rec_export", member)].index(tau_m))
                        ), self._enable_metering_periods.index(tau_m))
                    objective_expr = mk.Expr.add(objective_expr, mk.Expr.mul(self._gammas_tau_m.index(tau_m), mk.Expr.add(grid_fees, rec_fees)))
                
       
            """
                Peaks states costs
            """
            
            if self._involve_peaks:
                for member in self._members:
                    for tau_p in range(self._max_nb_peak_periods):
                        peak_term = 0
                        if self._involve_current_peaks:
                            if self._current_offtake_peak_cost > 0:
                                peak_term = mk.Expr.add(peak_term, mk.Expr.mul(
                                    self._current_peaks_state[("current_offtake_peak", member)].index(tau_p), epsilonify(self._current_offtake_peak_cost, epsilon=1e-8)
                                ))
                            if self._current_injection_peak_cost > 0:
                                peak_term = mk.Expr.add(peak_term, mk.Expr.mul(
                                    self._current_peaks_state[("current_injection_peak", member)].index(tau_p), epsilonify(self._current_injection_peak_cost, epsilon=1e-8)
                                ))
                        constant_term = 1.0/(self._Delta_C * self._Delta_M)
                        peak_term = mk.Expr.mul(
                            constant_term,
                            peak_term 
                        )
                        peak_term = mk.Expr.mul(
                            self._proratas.index(tau_p),
                            peak_term  
                        )
                        peak_term = mk.Expr.mul(
                            self._enable_peak_periods.index(tau_p),
                            peak_term  
                        )
                        objective_expr = mk.Expr.add(objective_expr, mk.Expr.mul(self._gammas_tau_p.index(tau_p), peak_term))

            ctrl_asset_action_lst = [
                {
                    k: self._controllable_assets_actions[k].index(t)  for k in self._controllable_assets_action_space_keys
                } for t in range(self._max_length_samples)
            ]
            #for actions in ctrl_asset_action_lst:
                #for action in actions.values():
                    #objective_expr = mk.Expr.add(objective_expr, mk.Expr.mul(action, 1e-6))
            self._model.objective(mk.ObjectiveSense.Minimize, objective_expr)

            ctrl_asset_state_lst = [self._initial_parametrized_state] + [
                {
                    k: self._controllable_assets_states[k].index(t)  for k in self._controllable_assets_state_space_keys
                } for t in range(self._max_length_samples)
            ]
            
            for key, dynamics_controllable_assets_function in self._dynamics_controllable_assets.items():
            
                for t in range(self._max_length_samples):
                    controllable_assets_current_state = ctrl_asset_state_lst[t]
                    controllable_assets_current_action = ctrl_asset_action_lst[t]
                    #action_values = list(controllable_assets_current_action.values())
                    #self._model.disjunction(mk.DJC.term(action_values[0], mk.Domain.equalsTo(0.0)), mk.DJC.term(action_values[1], mk.Domain.equalsTo(0.0)))
                    #truncated_exogenous_future_sequence = truncated_exogenous_future_sequence_lst[sample][timestep]
                    controllable_assets_next_state = ctrl_asset_state_lst[t+1]
                    next_state = dynamics_controllable_assets_function(
                        controllable_assets_current_state[key], 
                        controllable_assets_current_state,
                        None,
                        controllable_assets_current_action,
                        op_add=mk.Expr.add,
                        op_mul=mk.Expr.mul,
                        op_div=lambda n,m: mk.Expr.mul(n, 1.0/m),
                        op_sub=mk.Expr.sub
                    )
                    self._model.constraint(mk.Expr.sub(controllable_assets_next_state[key], next_state), mk.Domain.equalsTo(0.0))
                    #self._model.disjunction(mk.DJC.term(action_values[0], mk.Domain.equalsTo(0.0)), mk.DJC.term(action_values[1], mk.Domain.equalsTo(0.0)))
                    if t == self._max_length_samples-1:
                        self._model.constraint(next_state, mk.Domain.greaterThan(epsilonify(round(float(self._controllable_assets_state_space[key].low), 6))))
                        self._model.constraint(next_state, mk.Domain.lessThan(epsilonify(round(float(self._controllable_assets_state_space[key].high), 6))))

            
            timesteps = range(self._max_length_samples)
            
            consumption_lst = [{member:self._consumption_function[member](
                    ctrl_asset_state_lst[timestep], self._truncated_exogenous_future_sequence_lst[timestep], ctrl_asset_action_lst[timestep],
                    op_add=mk.Expr.add,
                    op_mul=mk.Expr.mul,
                    op_div=lambda n,m: mk.Expr.mul(n, 1.0/m),
                    op_sub=mk.Expr.sub,
                    op_idx=lambda l, i: (l[i] if type(l) in (list, tuple, np.ndarray) else l.index(i if i!=-1 else l.getShape()[0]-1))
                ) for member in self._members_with_controllable_assets} 
                for timestep in timesteps]
            
            production_lst = [{member:self._production_function[member](
                        ctrl_asset_state_lst[timestep], self._truncated_exogenous_future_sequence_lst[timestep], ctrl_asset_action_lst[timestep],
                        op_add=mk.Expr.add,
                        op_mul=mk.Expr.mul,
                        op_div=lambda n,m: mk.Expr.mul(n, 1.0/m),
                        op_sub=mk.Expr.sub,
                        op_idx=lambda l, i: (l[i] if type(l) in (list, tuple, np.ndarray) else l.index(i if i!=-1 else l.getShape()[0]-1))
                    ) for member in self._members_with_controllable_assets} 
                    for timestep in timesteps]

            max_consumption_lst = [
                {member:
                    self._max_consumption_param.slice([t, m], [t+1, m+1]).reshape([1]) for m, member in enumerate(self._members_with_controllable_assets)
                } for t in timesteps
            ]
            max_production_lst = [
                {member:
                    self._max_production_param.slice([t, m], [t+1, m+1]).reshape([1]) for m, member in enumerate(self._members_with_controllable_assets)
                } for t in timesteps
            ]
            max_cons_meter_lst = {member:
                self._max_cons_meter_param.slice([0, m], [self._max_nb_metering_periods, m+1]).reshape([self._max_nb_metering_periods]) for m, member in enumerate(self._members_with_controllable_assets)
            }
            
            max_prod_meter_lst = {member:
                self._max_prod_meter_param.slice([0, m], [self._max_nb_metering_periods, m+1]).reshape([self._max_nb_metering_periods]) for m, member in enumerate(self._members_with_controllable_assets)
            }
            
            max_consumption_lst_sequence = {
                member: self._max_consumption_param.slice([0, m], [self._max_length_samples, m+1]).reshape([self._max_length_samples]) for m, member in enumerate(self._members_with_controllable_assets)
            }
            max_production_lst_sequence = {
                member: self._max_production_param.slice([0, m], [self._max_length_samples, m+1]).reshape([self._max_length_samples]) for m, member in enumerate(self._members_with_controllable_assets)
            }
            
            net_consumption_lst = {
                member:[
                    (
                        self._net_consumption_production[("net_consumption", member)].index(timestep)
                    ) for timestep in timesteps
                ] for member in self._members_with_controllable_assets
            }
            net_consumption_lst = {
                **net_consumption_lst,
                **{
                    member:[
                        (
                            self._net_consumption_param.slice([timestep, m], [timestep+1, m+1])
                        ) for timestep in timesteps
                    ] for m,member in enumerate(self._members_without_controllable_assets)
                }
            }
            net_production_lst = {
                member:[
                    (
                        self._net_consumption_production[("net_production", member)].index(timestep)
                    ) for timestep in timesteps
                ] for member in self._members_with_controllable_assets
            }
            net_production_consumption_lst_bin = {
                member:[
                    (
                        self._net_consumption_production_bin[member].index(timestep)
                    ) for timestep in timesteps
                ] for member in self._members_with_controllable_assets
            } 
            net_production_lst = {
                **net_production_lst,
                **{
                    member:[
                        (
                            self._net_production_param.slice([timestep, m], [timestep+1, m+1])
                        ) for timestep in timesteps
                    ] for m, member in enumerate(self._members_without_controllable_assets)
                }
            }

            net_consumption_lst_sequence = {
                member: self._net_consumption_production[("net_consumption", member)].reshape([self._max_length_samples])
                for member in self._members_with_controllable_assets
            }
            net_consumption_lst_sequence = {
                **net_consumption_lst_sequence,
                **{
                    member: self._net_consumption_param.slice([0, m], [self._max_length_samples, m+1]).reshape([self._max_length_samples])
                    for m,member in enumerate(self._members_without_controllable_assets)
                }
            }
            net_production_lst_sequence = {
                member: self._net_consumption_production[("net_production", member)].reshape([self._max_length_samples])
                for member in self._members_with_controllable_assets
            } 
            net_production_lst_sequence = {
                **net_production_lst_sequence,
                **{
                    member: self._net_production_param.slice([0, m], [self._max_length_samples, m+1]).reshape([self._max_length_samples])
                    for m,member in enumerate(self._members_without_controllable_assets)
                }
            }

            for member in self._members_with_controllable_assets:
                for t in range(self._max_length_samples):
                    self._model.constraint(
                        mk.Expr.sub(
                            mk.Expr.sub(net_consumption_lst[member][t], net_production_lst[member][t]),
                            mk.Expr.mul(mk.Expr.sub(consumption_lst[t][member], production_lst[t][member]), self._enable_timesteps.index(t))
                        ),
                        mk.Domain.equalsTo(0.0)
                    )
                    
                    self._model.constraint(
                        mk.Expr.sub(net_consumption_lst[member][t], mk.Expr.mul(consumption_lst[t][member], self._enable_timesteps.index(t))), mk.Domain.lessThan(0.0)
                    )
                    self._model.constraint(
                        mk.Expr.sub(net_production_lst[member][t], mk.Expr.mul(production_lst[t][member], self._enable_timesteps.index(t))), mk.Domain.lessThan(0.0)
                    )
                    
                    self._model.constraint(
                        mk.Expr.sub(net_consumption_lst[member][t], mk.Expr.mul(mk.Expr.mul(max_consumption_lst[t][member], mk.Expr.sub(1.0, net_production_consumption_lst_bin[member][t])), self._enable_timesteps.index(t))), mk.Domain.lessThan(0.0)
                    )
                    self._model.constraint(
                        mk.Expr.sub(net_production_lst[member][t], mk.Expr.mul(mk.Expr.mul(max_production_lst[t][member], net_production_consumption_lst_bin[member][t]), self._enable_timesteps.index(t))), mk.Domain.lessThan(0.0)
                    )
                    #self._model.disjunction(mk.DJC.term(net_production_lst[member][t], mk.Domain.equalsTo(0.0)), mk.DJC.term(net_consumption_lst[member][t], mk.Domain.equalsTo(0.0)))

            if self._involve_peaks:
                #previous rec exchanges
                if self._previous_rec_exchanges is not None:
                    for tau_m in range(self._Delta_P-1):
                        
                        self._model.constraint(
                            mk.Expr.mul(
                                mk.Expr.sub(
                                    mk.Expr.sum(
                                        mk.Expr.hstack(
                                            [self._previous_rec_exchanges[("rec_import", member)].index(tau_m) for member in self._members]
                                        )
                                    ),
                                    mk.Expr.sum(
                                        mk.Expr.hstack(
                                            [self._previous_rec_exchanges[("rec_export", member)].index(tau_m) for member in self._members]
                                        )
                                    )
                                ), self._enable_previous_rec_exchange.index(tau_m)
                            ),
                            mk.Domain.equalsTo(0.0)
                        )
                        for m, member in enumerate(self._members):
                            net_consumption = self._previous_net_consumption_meters.slice([tau_m, m], [tau_m+1, m+1])
                            net_production = self._previous_net_production_meters.slice([tau_m, m], [tau_m+1, m+1])
                            self._model.constraint(
                                mk.Expr.sub(mk.Expr.mul(net_consumption, self._enable_previous_rec_exchange.index(tau_m)), mk.Expr.add(self._previous_rec_exchanges[("rec_import", member)].index(tau_m), self._previous_rec_exchanges[("grid_import", member)].index(tau_m))),
                                mk.Domain.equalsTo(0.0)
                            )
                            self._model.constraint(
                                mk.Expr.sub(mk.Expr.mul(net_production, self._enable_previous_rec_exchange.index(tau_m)), mk.Expr.add(self._previous_rec_exchanges[("rec_export", member)].index(tau_m), self._previous_rec_exchanges[("grid_export", member)].index(tau_m))),
                                mk.Domain.equalsTo(0.0)
                            )
                            """
                            self._model.constraint(
                                mk.Expr.sub(
                                    self._previous_rec_exchanges[("rec_import", member)].index(tau_m), mk.Expr.mul(net_consumption, self._enable_previous_rec_exchange.index(tau_m))
                                ),
                                mk.Domain.lessThan(0.0)
                            )
                            self._model.constraint(
                                mk.Expr.sub(
                                    self._previous_rec_exchanges[("rec_export", member)].index(tau_m), mk.Expr.mul(net_production, self._enable_previous_rec_exchange.index(tau_m))
                                ),
                                mk.Domain.lessThan(0.0)
                            )
                            self._model.constraint(
                                mk.Expr.sub(
                                    self._previous_rec_exchanges[("grid_import", member)].index(tau_m), mk.Expr.mul(net_consumption, self._enable_previous_rec_exchange.index(tau_m))
                                ),
                                mk.Domain.lessThan(0.0)
                            )
                            self._model.constraint(
                                mk.Expr.sub(
                                    self._previous_rec_exchanges[("grid_export", member)].index(tau_m), mk.Expr.mul(net_production, self._enable_previous_rec_exchange.index(tau_m))
                                ),
                                mk.Domain.lessThan(0.0)
                            )
                            """
                                
                            """
                            grid_fees = mk.Expr.sub(
                                mk.Expr.mul(self._exogenous_previous_prices[(member, "buying_price")].index(tau_m), self._previous_rec_exchanges[("grid_import", member)].index(tau_m)),
                                mk.Expr.mul(self._exogenous_previous_prices[(member, "selling_price")].index(tau_m), self._previous_rec_exchanges[("grid_export", member)].index(tau_m))
                            )

                            rec_fees = mk.Expr.add(
                                mk.Expr.mul(self._rec_import_fees, self._previous_rec_exchanges[("rec_import", member)].index(tau_m)),
                                mk.Expr.mul(self._rec_export_fees, self._previous_rec_exchanges[("rec_export", member)].index(tau_m))
                            )
                            objective_expr = mk.Expr.add(objective_expr, mk.Expr.mul(self._enable_previous_rec_exchange.index(tau_m), mk.Expr.add(grid_fees, rec_fees)))

                            constraints_extend(
                                net_consumption_meter == (previous_rec_exchanges[("grid import", member, t)] + previous_rec_exchanges[("rec import", member, t)]),
                                net_production_meter == (previous_rec_exchanges[("grid export", member, t)] + previous_rec_exchanges[("rec export", member, t)]),
                                sum_repartition_keys_previous_action_rec_export_lst[t] == sum_repartition_keys_previous_action_rec_import_lst[t]
                            )
                            """

            #future rec exchanges
            
            for tau_m in range(self._max_nb_metering_periods):
                metering_period_mask = self._rec_exchange_meters_mask.slice([0, tau_m], [self._max_length_samples, tau_m+1]).reshape([self._max_length_samples])
                self._model.constraint(
                    mk.Expr.mul(
                        mk.Expr.sub(
                            mk.Expr.sum(
                                mk.Expr.hstack(
                                    [self._rec_exchanges[("rec_import", member)].index(tau_m) for member in self._members]
                                )
                            ),
                            mk.Expr.sum(
                                mk.Expr.hstack(
                                    [self._rec_exchanges[("rec_export", member)].index(tau_m) for member in self._members]
                                )
                            )
                        ), self._enable_metering_periods.index(tau_m)
                    ),
                    mk.Domain.equalsTo(0.0)
                )
                for member in self._members:
                    net_consumption = mk.Expr.dot(net_consumption_lst_sequence[member], metering_period_mask)
                    net_production = mk.Expr.dot(net_production_lst_sequence[member], metering_period_mask)
                    if member in self._members_with_controllable_assets:
                        if tau_m == 0:
                            net_consumption = mk.Expr.add(net_consumption, self._consumption_meter_terms_dict[member])
                            net_production = mk.Expr.add(net_production, self._production_meter_terms_dict[member])
                        
                    
                    self._model.constraint(
                        mk.Expr.sub(
                            self._rec_exchanges[("total_import", member)].index(tau_m),
                            mk.Expr.mul(
                            (
                                mk.Expr.add(
                                    self._rec_exchanges[("rec_import", member)].index(tau_m),
                                    self._rec_exchanges[("grid_import", member)].index(tau_m)
                                )
                            ), self._enable_metering_periods.index(tau_m))
                        ),
                        mk.Domain.equalsTo(0.0)
                    )
                    self._model.constraint(
                        mk.Expr.sub(
                            self._rec_exchanges[("total_export", member)].index(tau_m),
                            mk.Expr.mul(
                            (
                                mk.Expr.add(
                                    self._rec_exchanges[("rec_export", member)].index(tau_m),
                                    self._rec_exchanges[("grid_export", member)].index(tau_m)
                                )
                            ), self._enable_metering_periods.index(tau_m))
                        ),
                        mk.Domain.equalsTo(0.0)
                    )
                    self._model.constraint(
                        mk.Expr.sub(
                            self._rec_exchanges[("rec_import", member)].index(tau_m), mk.Expr.mul(net_consumption, self._enable_metering_periods.index(tau_m))
                        ),
                        mk.Domain.lessThan(0.0)
                    )
                    self._model.constraint(
                        mk.Expr.sub(
                            self._rec_exchanges[("rec_export", member)].index(tau_m), mk.Expr.mul(net_production, self._enable_metering_periods.index(tau_m))
                        ),
                        mk.Domain.lessThan(0.0)
                    )
                    
                    if member in self._members_with_controllable_assets:
                        #self._model.disjunction(
                        #    mk.DJC.term(self._rec_exchanges[("total_import", member)].index(tau_m), mk.Domain.equalsTo(0.0)),
                        #    mk.DJC.term(self._rec_exchanges[("total_export", member)].index(tau_m), mk.Domain.equalsTo(0.0))
                        #)
                        max_consumption = max_cons_meter_lst[member].index(tau_m)
                        max_production = max_prod_meter_lst[member].index(tau_m)
                        self._model.constraint(
                            mk.Expr.sub(self._rec_exchanges[("total_import", member)].index(tau_m), mk.Expr.mul(mk.Expr.mul(max_consumption, mk.Expr.sub(1, self._rec_exchanges_bin[member].index(tau_m))), self._enable_metering_periods.index(tau_m))), mk.Domain.lessThan(0.0)
                        )
                        self._model.constraint(
                            mk.Expr.sub(self._rec_exchanges[("total_export", member)].index(tau_m), mk.Expr.mul(mk.Expr.mul(max_production, self._rec_exchanges_bin[member].index(tau_m)), self._enable_metering_periods.index(tau_m))), mk.Domain.lessThan(0.0)
                        )
                        self._model.constraint(
                            mk.Expr.mul(mk.Expr.sub(
                                mk.Expr.sub(
                                    net_consumption,
                                    net_production
                                ),
                                mk.Expr.sub(
                                    self._rec_exchanges[("total_import", member)].index(tau_m),
                                    self._rec_exchanges[("total_export", member)].index(tau_m)
                                )
                            ), self._enable_metering_periods.index(tau_m)),
                            mk.Domain.equalsTo(0.0)
                        )
                        
                    else:
                        self._model.constraint(
                            mk.Expr.sub(
                                mk.Expr.mul(net_consumption, self._enable_metering_periods.index(tau_m)),
                                self._rec_exchanges[("total_import", member)].index(tau_m)
                            ),
                            mk.Domain.equalsTo(0.0)
                        )
                        self._model.constraint(
                            mk.Expr.sub(
                                mk.Expr.mul(net_production, self._enable_metering_periods.index(tau_m)),
                                self._rec_exchanges[("total_export", member)].index(tau_m)
                            ),
                            mk.Domain.equalsTo(0.0)
                        )
                    """
                        constraints_append(
                            sum_repartition_keys_action_rec_export == sum_repartition_keys_action_rec_import
                        )
                        constraints_extend_lst(
                        [
                            rec_exchanges[("total import", member, sample, tau_m)] == rec_exchanges[("grid import", member, sample, tau_m)] + rec_exchanges[("rec import", member, sample, tau_m)],
                            rec_exchanges[("total export", member, sample, tau_m)] == rec_exchanges[("grid export", member, sample, tau_m)] + rec_exchanges[("rec export", member, sample, tau_m)],
                            rec_exchanges[("rec export", member, sample, tau_m)] <= rec_member_energy_produced_metering_period,
                            rec_exchanges[("rec import", member, sample, tau_m)] <= rec_member_energy_consumed_metering_period
                        ]
                    )
                    constraints_append(
                            rec_member_energy_consumed_metering_period - rec_member_energy_produced_metering_period == (
                                rec_exchanges[("total import", member, sample, tau_m)] - rec_exchanges[("total export", member, sample, tau_m)]
                            )
                        )
                        self._add_sos_1(
                            rec_exchanges[("total import", member, sample, tau_m)], rec_exchanges[("total export", member, sample, tau_m)]
                        )
                    """
            if self._involve_peaks:
                if self._previous_rec_exchanges is not None:
                    for member in self._members:
                        for tau_m_previous in range(self._Delta_P-1):
                            self._model.constraint(
                                mk.Expr.sub(
                                    mk.Expr.mul(
                                        self._current_peaks_state[("current_offtake_peak", member)].index(0),
                                        self._enable_peak_periods.index(0)
                                    ),
                                    mk.Expr.mul(
                                        self._previous_rec_exchanges[("grid_import", member)].index(tau_m_previous),
                                        self._enable_previous_rec_exchange.index(tau_m_previous)
                                        
                                    )
                                ),
                                mk.Domain.greaterThan(0.0)
                            )
                            self._model.constraint(
                                mk.Expr.sub(
                                    mk.Expr.mul(
                                        self._current_peaks_state[("current_injection_peak", member)].index(0),
                                        self._enable_peak_periods.index(0)
                                    ),
                                    mk.Expr.mul(
                                        self._previous_rec_exchanges[("grid_export", member)].index(tau_m_previous),
                                        self._enable_previous_rec_exchange.index(tau_m_previous)
                                    )
                                ),
                                mk.Domain.greaterThan(0.0)
                            )
                for member in self._members:
                    for tau_p in range(self._max_nb_peak_periods):
                        for tau_m in range(self._max_nb_metering_periods):
                            self._model.constraint(
                                mk.Expr.sub(
                                    mk.Expr.mul(
                                        self._current_peaks_state[("current_offtake_peak", member)].index(tau_p),
                                        self._enable_peak_periods.index(tau_p)
                                    ),
                                    mk.Expr.mul(
                                        mk.Expr.mul(
                                            self._rec_exchanges[("grid_import", member)].index(tau_m),
                                            self._enable_metering_periods.index(tau_m),
                                        ),
                                        self._rec_exchange_peaks_mask.slice([tau_m, tau_p], [tau_m+1, tau_p+1]).reshape([1]) 
                                    )
                                ),
                                mk.Domain.greaterThan(0.0)
                            )
                            self._model.constraint(
                                mk.Expr.sub(
                                    mk.Expr.mul(
                                        self._current_peaks_state[("current_injection_peak", member)].index(tau_p),
                                        self._enable_peak_periods.index(tau_p)
                                    ),
                                    mk.Expr.mul(
                                        mk.Expr.mul(
                                            self._rec_exchanges[("grid_export", member)].index(tau_m),
                                            self._enable_metering_periods.index(tau_m),
                                        ),
                                        self._rec_exchange_peaks_mask.slice([tau_m, tau_p], [tau_m+1, tau_p+1]).reshape([1]) 
                                    )
                                ),
                                mk.Domain.greaterThan(0.0)
                            )
        else:
            #self._model.setSolverParam("presolveUse", "off")
            self._model.setSolverParam("mioConstructSol", "on")
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
        
        if length_samples == 0:
            return None
        timesteps = list(range(length_samples))
        self._timesteps=timesteps
        enable_timesteps = np.zeros(self._max_length_samples)
        enable_timesteps[:length_samples] = 1
        self._enable_timesteps.setValue(enable_timesteps)
        nb_previous_meters = 0
        if self._involve_peaks:
            enable_previous_rec_exchange = np.zeros(self._Delta_P-1)
            if self._previous_rec_exchanges is not None and state["peak_period_counter"] != self._Delta_P:
                if state["metering_period_counter"] < self._Delta_M: 
                    previous_consumption_meters = state[(self._members[0], "consumption_meters")][:-1]
                else:
                    previous_consumption_meters = state[(self._members[0], "consumption_meters")]
                nb_previous_meters = len(previous_consumption_meters)
                if nb_previous_meters > 0:
                    enable_previous_rec_exchange[:nb_previous_meters] = 1
                #print(state["metering_period_counter"], state["peak_period_counter"])
            self._enable_previous_rec_exchange.setValue(enable_previous_rec_exchange)
        #print(nb_previous_meters)
        enable_metering_periods = np.zeros(self._max_nb_metering_periods)
        if initial_counters["metering_period_counter"] == self._Delta_M:
            future_counters_metering_period = np.asarray([0] + future_counters["metering_period_counter"])[:-1]
        else:
            future_counters_metering_period = np.asarray([initial_counters["metering_period_counter"]] + future_counters["metering_period_counter"])[:-1]
        number_of_metering_periods = sum(np.asarray(future_counters_metering_period) == self._Delta_M)
        if self._force_last_time_step_to_global_bill and future_counters_metering_period[-1] != self._Delta_M:
            number_of_metering_periods += 1
        number_of_metering_periods = min(number_of_metering_periods, self._max_nb_metering_periods)
        enable_metering_periods[:number_of_metering_periods] = 1
        self._enable_metering_periods.setValue(enable_metering_periods)
        
        number_of_peak_periods = 0
        if self._involve_peaks:
            enable_peak_periods = np.zeros(self._max_nb_peak_periods)
            future_counters_peak_period = np.asarray([initial_counters["peak_period_counter"]] + future_counters["peak_period_counter"])
            number_of_peak_periods = sum(np.asarray(future_counters_peak_period) == self._Delta_P)
            if self._force_last_time_step_to_global_bill and future_counters_peak_period[-1] != self._Delta_P:
                number_of_peak_periods += 1
            enable_peak_periods[:number_of_peak_periods] = 1
            self._enable_peak_periods.setValue(enable_peak_periods)

        
    
        controllable_states = {
            k:state[k] for k in self._controllable_assets_state_space_keys
        }
        controllable_states = np.asarray(list(controllable_states.values())).reshape(self._initial_parametrized_state_param.getShape())
        self._initial_parametrized_state_param.setValue(controllable_states)

        
        future_counters_metering_period_vector = future_counters_metering_period.copy()
        future_counters_metering_period_vector[future_counters_metering_period_vector != self._Delta_M] = 0
        future_counters_metering_period_vector[future_counters_metering_period_vector == self._Delta_M] = 1
        if self._force_last_time_step_to_global_bill:
            future_counters_metering_period_vector[-1] = 1
        future_counters_metering_period_matrix = replace_last_one(fill_zeros_between_ones(add_rows(generate_matrix_from_binary_vector(future_counters_metering_period_vector.copy())))).T
        future_counters_metering_period = np.zeros(self._rec_exchange_meters_mask.getShape())
        if future_counters_metering_period_vector[-1] < self._Delta_M:
            future_counters_metering_period_matrix = future_counters_metering_period_matrix.T
            future_counters_metering_period_matrix[-1][-1] = 1
            future_counters_metering_period_matrix = future_counters_metering_period_matrix.T

        #print(future_counters_metering_period_matrix.T)
        #print(future_counters_metering_period_matrix.T.shape)
        
        #print(future_counters_metering_period_matrix.T)
        #print(future_counters_metering_period_matrix.T[-1])
        #print(future_counters_metering_period_matrix.shape)
        #print()
        future_counters_metering_period[:length_samples, :number_of_metering_periods] = future_counters_metering_period_matrix
        
        #future_counters_metering_period_matrix = create_matrix(future_counters_metering_period_vector)
        #belongs_to_metering_period_matrix = np.zeros(self._belongs_to_metering_period.getShape())
        #belongs_to_metering_period_matrix[:future_counters_metering_period_matrix.shape[0],max(0, self._Delta_M - length_samples):future_counters_metering_period_matrix.shape[1] + max(0, self._Delta_M - length_samples)] = future_counters_metering_period_matrix
        #print(future_counters_metering_period.T)
        self._rec_exchange_meters_mask.setValue(future_counters_metering_period)
        #for tau_m in range(self._max_nb_metering_periods):
            #print(self._rec_exchange_meters_mask.slice([0, tau_m], [self._max_length_samples, tau_m+1]).reshape([self._max_length_samples]).getValue())
        #print(future_counters_metering_period[0])

        truncated_exogenous_future_sequence = np.zeros((self._max_length_samples, len(self._exogenous_variable_members_keys)))
        future_exogenous_members_variables, future_exogenous_prices = self._exogenous_provider.sample_future_sequences(
            exogenous_variables_members=exogenous_variable_members,
            exogenous_prices=exogenous_prices,
            length=length_samples-1,
            n_samples=self._n_samples
        )
        
        future_exogenous_members_variables_dict = future_exogenous_members_variables[0]
        future_exogenous_members_variables = np.asarray(list({
            k:([exogenous_variable_members[k][-1]]+v.tolist())[:self._max_length_samples] for k,v in future_exogenous_members_variables[0].items()
        }.values())).T
        
        truncated_exogenous_future_sequence[:length_samples, :] = future_exogenous_members_variables[:length_samples, :]
        #print(truncated_exogenous_future_sequence.T)
        self._truncated_exogenous_future_sequence.setValue(truncated_exogenous_future_sequence)

        exogenous_buying_prices_matrix = np.zeros(
            self._buying_exogenous_prices.getShape()
        )
        exogenous_selling_prices_matrix = np.zeros(
            self._selling_exogenous_prices.getShape()
        )
        
        if state["metering_period_counter"] < self._Delta_M:
            exogenous_buying_prices_values = np.asarray(list({
            k:[exogenous_prices[k][-1]] + v for k,v in future_exogenous_prices[0].items() if "buying_price" in k
            }.values()))
            exogenous_selling_prices_values = np.asarray(list({
                k:[exogenous_prices[k][-1]] + v for k,v in future_exogenous_prices[0].items() if "selling_price" in k
            }.values()))
            
        else:
            exogenous_buying_prices_values = np.asarray(list({
            k:v for k,v in future_exogenous_prices[0].items() if "buying_price" in k
            }.values()))
            exogenous_selling_prices_values = np.asarray(list({
                k:v for k,v in future_exogenous_prices[0].items() if "selling_price" in k
            }.values()))
        exogenous_buying_prices_matrix[:number_of_metering_periods, :] = exogenous_buying_prices_values.T[:number_of_metering_periods, :]
        exogenous_selling_prices_matrix[:number_of_metering_periods, :] = exogenous_selling_prices_values.T[:number_of_metering_periods, :]
        
        self._buying_exogenous_prices.setValue(exogenous_buying_prices_matrix)
        self._selling_exogenous_prices.setValue(exogenous_selling_prices_matrix)
        
        if self._involve_peaks:
            nb_previous_prices = nb_previous_meters
            exogenous_previous_buying_prices = np.zeros((
                self._Delta_P-1, len(self._members)
            ))
            exogenous_previous_selling_prices = np.zeros((
                self._Delta_P-1, len(self._members)
            ))
            if nb_previous_prices > 0:
                exogenous_previous_buying_prices[:nb_previous_prices, :] = np.asarray([
                    exogenous_prices[(m, "buying_price")][-(nb_previous_prices):] for m in self._members
                ]).T
                exogenous_previous_selling_prices[:nb_previous_prices, :] = np.asarray([
                    exogenous_prices[(m, "selling_price")][-(nb_previous_prices):] for m in self._members
                ]).T
                self._exogenous_previous_buying_prices.setValue(exogenous_previous_buying_prices)
                self._exogenous_previous_selling_prices.setValue(exogenous_previous_selling_prices)

        consumption_meter_terms = [
            0.0
        ] * len(self._members_with_controllable_assets)
        production_meter_terms = [
            0.0
        ] * len(self._members_with_controllable_assets)
        if state["metering_period_counter"] < self._Delta_M:
            consumption_meter_terms = [
                state[(member, "consumption_meters")][-1] for member in self._members_with_controllable_assets
            ]
            production_meter_terms = [
                state[(member, "production_meters")][-1] for member in self._members_with_controllable_assets
            ]
        self._consumption_meter_terms.setValue(np.asarray(consumption_meter_terms))
        self._production_meter_terms.setValue(np.asarray(production_meter_terms))

        net_consumption_param = np.zeros(
            self._net_consumption_param.getShape()
        )
        net_production_param = np.zeros(
            self._net_production_param.getShape()
        )
        truncated_exogenous_future_sequence_lst = [exogenous_variable_members] + [{
            key: [lst_values[timestep]] for key, lst_values in future_exogenous_members_variables_dict.items()
        } for timestep in timesteps]
        consumption_vec = np.asarray(list({
            member:[self._consumption_function[member](
                None, truncated_exogenous_future_sequence_lst[timestep], None
            ) for timestep in range(length_samples)] for member in self._members_without_controllable_assets
        }.values()))
        production_vec = np.asarray(list({
            member:[self._production_function[member](
                None, truncated_exogenous_future_sequence_lst[timestep], None
            ) for timestep in range(length_samples)] for member in self._members_without_controllable_assets
        }.values()))
        if state["metering_period_counter"] < self._Delta_M:
            consumption_vec[:, 0] += np.asarray([
                state[(member, "consumption_meters")][-1] for member in self._members_without_controllable_assets
            ])
            production_vec[:, 0] += np.asarray([
                state[(member, "production_meters")][-1] for member in self._members_without_controllable_assets
            ])
        net_consumption_vec = np.maximum(consumption_vec - production_vec, 0.0)
        net_production_vec = np.maximum(production_vec - consumption_vec, 0.0)
        net_consumption_param[:length_samples, :] = net_consumption_vec.T
        net_production_param[:length_samples, :] = net_production_vec.T
        self._net_consumption_param.setValue(net_consumption_param)
        self._net_production_param.setValue(net_production_param)

        max_consumption_param = np.zeros(
            self._max_consumption_param.getShape()
        )
        max_production_param = np.zeros(
            self._max_production_param.getShape()
        )
        max_action_values = {
            action_key: self._controllable_assets_action_space[action_key].high for action_key in self._controllable_assets_action_space_keys
        }
        max_consumption_vec = np.asarray(list({
            member:[self._consumption_function[member](
                None, truncated_exogenous_future_sequence_lst[timestep], max_action_values
            ) for timestep in range(length_samples)] for member in self._members_with_controllable_assets
        }.values()))
        max_production_vec = np.asarray(list({
            member:[self._production_function[member](
                None, truncated_exogenous_future_sequence_lst[timestep], max_action_values
            ) for timestep in range(length_samples)] for member in self._members_with_controllable_assets
        }.values()))
        
        max_consumption_param[:length_samples, :] = max_consumption_vec.T
        max_production_param[:length_samples, :] = max_production_vec.T
        self._max_consumption_param.setValue(max_consumption_param)
        self._max_production_param.setValue(max_production_param)
        max_cons_meter_param = max_consumption_param.T.dot(future_counters_metering_period)
        max_prod_meter_param = max_production_param.T.dot(future_counters_metering_period)
        self._max_cons_meter_param.setValue(max_cons_meter_param.T*10)
        self._max_prod_meter_param.setValue(max_prod_meter_param.T*10)
        
            

        if self._involve_peaks:
            proratas = np.ones(self._max_nb_peak_periods)
            last_prorata = elapsed_timesteps_in_peak_period(
                0, 0, future_counters["metering_period_counter"][-1], future_counters["peak_period_counter"][-1], Delta_M=self._Delta_M, Delta_P=self._Delta_P
            )
            proratas[-1] = last_prorata
            self._proratas.setValue(proratas)

        if self._involve_peaks:
            previous_net_consumption_meters = np.zeros((
                len(self._members), self._Delta_P-1
            ))
            previous_net_production_meters = np.zeros((
                len(self._members), self._Delta_P-1
            ))
            if nb_previous_meters > 0:
                #TOFIX
                
                if state["metering_period_counter"] < self._Delta_M: 
                    previous_net_consumption_meters_values = np.asarray([
                        state[(member, "consumption_meters")][:-1] for member in self._members
                    ])
                    previous_net_production_meters_values = np.asarray([
                        state[(member, "production_meters")][:-1] for member in self._members
                    ])
                else:
                    previous_net_consumption_meters_values = np.asarray([
                        state[(member, "consumption_meters")] for member in self._members
                    ])
                    previous_net_production_meters_values = np.asarray([
                        state[(member, "production_meters")] for member in self._members
                    ])
                
                previous_net_consumption_meters_values, previous_net_production_meters_values = (
                    np.maximum(previous_net_consumption_meters_values - previous_net_production_meters_values, 0.0),
                    np.maximum(previous_net_production_meters_values - previous_net_consumption_meters_values, 0.0)
                )
                previous_net_consumption_meters[:, :nb_previous_meters] = previous_net_consumption_meters_values
                previous_net_production_meters[:, :nb_previous_meters] = previous_net_production_meters_values
                #print(previous_net_consumption_meters_values)
            self._previous_net_consumption_meters.setValue(previous_net_consumption_meters.T)
            self._previous_net_production_meters.setValue(previous_net_production_meters.T)
            """
            self._rec_exchange_peaks_mask = self._model.parameter(
                    self._max_nb_metering_periods, self._max_nb_peak_periods
                )
            """
        if self._involve_peaks:
            rec_exchange_peaks_mask = np.zeros(
                (self._max_nb_metering_periods, self._max_nb_peak_periods)
            )
            future_counters_peak_period_vector = future_counters_peak_period.copy()
            #future_counters_peak_period_vector[future_counters_peak_period_vector != self._Delta_P] = 0
            #future_counters_peak_period_vector[future_counters_peak_period_vector == self._Delta_P] = 1

            if self._force_last_time_step_to_global_bill:
                future_counters_peak_period_vector[-1] = self._Delta_P
            if np.count_nonzero(future_counters_peak_period_vector == self._Delta_P) > 0:
                future_counters_peak_period_unique_values = np.hstack([0, count_unique_values(future_counters_peak_period_vector, self._Delta_P)])
                future_counters_peak_period_unique_values_window = np.lib.stride_tricks.sliding_window_view(future_counters_peak_period_unique_values, 2).copy()
                future_counters_peak_period_unique_values_window_first_elems_cumsum = np.cumsum(
                    future_counters_peak_period_unique_values_window[:, 0]
                )
                future_counters_peak_period_unique_values_window[:, 0] = future_counters_peak_period_unique_values_window_first_elems_cumsum
                future_counters_peak_period_matrix = generate_matrix_vectorized(np.fliplr(future_counters_peak_period_unique_values_window))
                rec_exchange_peaks_mask[:number_of_metering_periods, :number_of_peak_periods] = future_counters_peak_period_matrix.T[:number_of_metering_periods, :number_of_peak_periods]
                
                self._rec_exchange_peaks_mask.setValue(rec_exchange_peaks_mask)

        gammas_tau_m = np.zeros(self._max_nb_metering_periods)
        gammas_tau_m_vec = [self._gammas[i] for i, tau_m in enumerate(future_counters["metering_period_counter"]) if tau_m == self._Delta_M or (self._force_last_time_step_to_global_bill and i == len(future_counters["metering_period_counter"]) - 1)][:number_of_metering_periods]
        gammas_tau_m[:number_of_metering_periods] = np.asarray(gammas_tau_m_vec)
        self._gammas_tau_m.setValue(gammas_tau_m)
        gammas_tau_p = None
        if self._involve_peaks:
            rec_exchange_peaks_mask = self._rec_exchange_peaks_mask.getValue().reshape(self._rec_exchange_peaks_mask.getShape()).T
            gammas_tau_p = np.zeros(self._max_nb_peak_periods)
            gammas_tau_p[:number_of_peak_periods] = [self._gammas[i] for i, tau_p in enumerate(future_counters_peak_period) if tau_p == self._Delta_P or (self._force_last_time_step_to_global_bill and i == len(future_counters_peak_period) - 1)][:number_of_peak_periods]
            gammas_tau_m = np.zeros(self._max_nb_metering_periods)
            gammas_tau_m_mat = (rec_exchange_peaks_mask * gammas_tau_p[:, None]).flatten()
            gammas_tau_m_mat = gammas_tau_m_mat[gammas_tau_m_mat>0]
            gammas_tau_m[:number_of_metering_periods] = gammas_tau_m_mat
            self._gammas_tau_m.setValue(gammas_tau_m)
            self._gammas_tau_p.setValue(gammas_tau_p)

        #Initialise solution
        if not self._first_pass:
            for member in self._members_with_controllable_assets:
                #self._net_consumption_production[("net_consumption", member)].setLevel(
                #    np.hstack([self._net_consumption_production[("net_consumption", member)].level()[1:], [0]])
                #)
                #self._net_consumption_production[("net_production", member)].setLevel(
                #    np.hstack([self._net_consumption_production[("net_production", member)].level()[1:], [0]])
                #)
                self._net_consumption_production_bin[member].setLevel(
                    np.hstack([self._net_consumption_production_bin[member].level()[1:], [0]])
                )
                
            """
            for ctrl_asset_key in self._controllable_assets_action_space_keys:
                self._controllable_assets_actions[ctrl_asset_key].setLevel(
                    np.hstack([self._controllable_assets_actions[ctrl_asset_key].level()[1:], [0]])
                )

            for ctrl_asset_key in self._controllable_assets_state_space_keys:
                self._controllable_assets_states[ctrl_asset_key].setLevel(
                    np.hstack([self._controllable_assets_states[ctrl_asset_key].level()[1:], [0]])
                )
            """
            
            for member in self._members:
                for exchange_type in ["rec_import", "rec_export", "grid_import", "grid_export", "total_import", "total_export"]:
                    if initial_counters["metering_period_counter"] == self._Delta_M:
                        #self._rec_exchanges[(exchange_type, member)].setLevel(
                        #    np.hstack([self._rec_exchanges[(exchange_type, member)].level()[1:], [0]])
                        #)
                        self._rec_exchanges_bin[member].setLevel(
                            np.hstack([self._rec_exchanges_bin[member].level()[1:], [0]])
                        )
                    else:
                        #self._rec_exchanges[(exchange_type, member)].setLevel(
                        #    self._rec_exchanges[(exchange_type, member)].level()
                        #)
                        self._rec_exchanges_bin[member].setLevel(
                            self._rec_exchanges[(exchange_type, member)].level()
                        )
                    #if exchange_type not in ("total_import", "total_export"):
                    #    self._previous_rec_exchanges[(exchange_type, member)].setLevel(
                    #        self._previous_rec_exchanges[(exchange_type, member)].level()
                    #    )

            if self._involve_peaks:
                if initial_counters["peak_period_counter"] == self._Delta_P:
                    for member in self._members:
                        for peak_type in self._lst_current_peaks:
                            self._current_peaks_state[(peak_type, member)].setLevel(
                                np.hstack([self._current_peaks_state[(peak_type, member)].level()[1:], [0]])
                            )
                        for exchange_type in ["rec_import", "rec_export", "grid_import", "grid_export"]:
                            self._previous_rec_exchanges[(exchange_type, member)].setLevel(
                                np.hstack([self._previous_rec_exchanges[(exchange_type, member)].level()[1:], [0]])
                            )
            """
            self._current_peaks_state = {
                    (k, member): self._model.variable(self._max_nb_peak_periods, mk.Domain.greaterThan(0.0)) for k in lst_current_peaks for member in self._members
                }
            
            #TODO if needed : implement for historical peaks
            if self._Delta_P > 1:
                self._previous_rec_exchanges = {
                    (k, member): self._model.variable(self._Delta_P-1, mk.Domain.greaterThan(0.0)) for k in [
                        "rec_import", "rec_export", "grid_import", "grid_export"
                    ] for member in self._members
                }
            self._controllable_assets_actions = {
                k: self._model.variable(self._max_length_samples, mk.Domain.inRange(epsilonify(round(float(self._controllable_assets_action_space[k].low), 6)), epsilonify(round(float(self._controllable_assets_action_space[k].high), 6)))) for k in self._controllable_assets_action_space_keys
            }
            self._controllable_assets_states = {
                k: self._model.variable(self._max_length_samples, mk.Domain.inRange(epsilonify(round(float(self._controllable_assets_state_space[k].low), 6)), epsilonify(round(float(self._controllable_assets_state_space[k].high), 6)))) for k in self._controllable_assets_state_space_keys
            }
            self._rec_exchanges = {
                (k, member): self._model.variable(self._max_nb_metering_periods, mk.Domain.greaterThan(0.0)) for k in [
                    "rec_import", "rec_export", "grid_import", "grid_export", "total_import", "total_export"
                ] for member in self._members
            }
            self._net_consumption_production = {
                (k, member): self._model.variable(self._max_length_samples, mk.Domain.greaterThan(0.0)) for k in [
                    "net_consumption", "net_production"
                ] for member in self._members_with_controllable_assets
            }
            """
        self._init=False
        self._first_pass = False
        return self._controllable_assets_actions

    def _solve_by_solver(self, controllable_assets_actions):

        t = time()
        self._model.solve()
        print("Solved in ", time() - t)
        from pprint import pprint
        #pprint({
        #    k:v.level() for k,v in self._previous_rec_exchanges.items() if k[0] not in ("total_import", "total_export")
        #})
        #print("***")
        #print("FUTURE REC EXCHANGES")
        #pprint({
        #    k:v.level() for k,v in self._rec_exchanges.items() if k[0] not in ("total_import", "total_export")
        #})
        #print(np.sum(self._truncated_exogenous_future_sequence.getValue().reshape((101, 2)), axis=0))
        #print("PREVIOUS REC EXCHANGES")
        #pprint({
        #    k:v.level() for k,v in self._previous_rec_exchanges.items() if k[0]
        #})
        #print("***")
        #pprint({
        #    k:v.level() for k,v in self._rec_exchanges.items() if k[0] not in ("total_import", "total_export")
        #})
        #pprint({
        #    k:v.level() for k,v in self._current_peaks_state.items()
        #})
        #pprint({
        #    k:v.level() for k,v in self._current_peaks_state.items()
        #})
        #print(
        #    self._previous_net_production_meters.getValue()
        #)
        #print(
        #    self._previous_net_consumption_meters.getValue()
        #)
        #print()
        #print()
        if self._model.getPrimalSolutionStatus() != mk.SolutionStatus.Optimal:
            return None, None, self._model.getPrimalSolutionStatus()
        d_results = {
            (k,0):v.level() for k,v in controllable_assets_actions.items()
        }
        d_results_1 = {
            k:np.hstack([v[1:], v[0]*0]) for k,v in d_results.items()
        }
        d_results_2 = {
            k[0]:v[0] for k,v in d_results.items()
        }
        d_3 = {
            k[0]:np.round(v, 4) for k,v in d_results.items()
        }
        #pprint({
        #    (member, meter_type): np.hstack([np.dot(self._rec_exchange_meters_mask.getValue().reshape(self._rec_exchange_meters_mask.getShape())[:, tau_m:tau_m+1].T, self._net_consumption_production[(meter_type, member)].level()) for tau_m in range(self._max_nb_metering_periods)])
        #    for member in self._members_with_controllable_assets for meter_type in ("net_consumption", "net_production")
        #})
        
        #print(
        #    "ACTION = ", d_3[("PVB", "charge")][0] - d_3[("PVB", "discharge")][0]
        #)
        #print(
        #    "FUTURE_ACTIONS = ", d_3[("PVB", "charge")][1:] - d_3[("PVB", "discharge")][1:]
        #)
        #print(d_results_2)
        #print({**d_results_2, **d_results_1})
        #print()
        #print(self._model.primalObjValue())
        #print()
        return dict(), {**d_results_2, **d_results_1}, 0