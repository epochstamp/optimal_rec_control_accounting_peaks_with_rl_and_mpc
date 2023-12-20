#from env.expost_global_bill_optimisation import expost_global_bill_optimisation
from typing import Tuple
import mosek.fusion as mk
from env.peaks_utils import elapsed_timesteps_in_peak_period, number_of_time_steps_elapsed_in_peak_period
import cvxpy as cp
import numpy as np
from utils.utils import epsilonify
from time import time
import itertools

TYPE_SOLVE="mosek"

def sort_matrix(M1, M2, M3, return_indices=False):
    # Vérifier si les matrices ont la même forme
    if M1.shape != M2.shape or M1.shape != M3.shape:
        raise ValueError("Les matrices doivent avoir la même dimension")

    # Créer une matrice de rang 3 en empilant M2 et M3 le long d'un nouvel axe
    combined = np.stack((M2, M3), axis=-1)

    # Utiliser argsort pour obtenir les indices de tri pour chaque colonne
    sorted_indices = np.lexsort((combined[:, :, 1], combined[:, :, 0]), axis=0)

    # Utiliser ces indices pour trier chaque colonne de M1 en utilisant une indexation avancée
    col_indices = np.arange(M1.shape[1])
    M1_sorted = M1[sorted_indices, col_indices]
    return (M1_sorted, sorted_indices) if return_indices else M1_sorted

def index_matrix(M1, M2):
    rows, cols = M2.shape
    M3 = np.zeros_like(M1)
    print(M1)
    print(M2)
    for i in range(rows):
        for j in range(cols):
            row_index = M2[i, j]
            print(row_index, j, M1[i, j])
            M3[i, j] = M1[row_index, j]
    return M3



def find_indices(M1, M2):
    # Vérifier si les matrices ont la même forme et contiennent les mêmes valeurs
    if M1.shape != M2.shape or not np.all(np.sort(M1.ravel()) == np.sort(M2.ravel())):
        raise ValueError("Les matrices doivent avoir la même dimension et les mêmes valeurs")

    # Obtenir les indices uniques des valeurs dans M2
    unique_values, indices_M2 = np.unique(M2, return_inverse=True)

    # Créer un dictionnaire pour stocker les indices de chaque valeur dans M2
    M2_dict = {value: index for value, index in zip(unique_values, indices_M2)}

    # Obtenir les indices linéaires de chaque valeur dans M1 en utilisant le dictionnaire M2
    linear_indices_M1 = np.array([M2_dict[value] for value in M1.ravel()])

    # Convertir les indices linéaires en indices bidimensionnels
    result_indices = np.column_stack(np.unravel_index(linear_indices_M1, M2.shape))

    # Remodeler le résultat pour correspondre à la forme originale de M1
    result = result_indices.reshape(*M1.shape, 2)

    return result


def iterative_vec_sub_reverse(matrix, vector, verbose=False):
    if len(matrix) == 0:
        if verbose:
            print("here")
        return matrix
    
    row = matrix[0]
    new_row = np.maximum(0, row - vector)
    new_vector = np.maximum(0, vector - row)
    if verbose:
        print("###")
        print("ROW", row)
        print("VECTOR", vector)
        print("**")
        print("NEW ROW", new_row)
        print("NEW VECTOR", new_vector)
        print("###")
        print()
    
    if np.any(new_vector > 0):
        matrix_remain = iterative_vec_sub_reverse(matrix[1:], new_vector, verbose=verbose)
        return np.vstack((new_row, matrix_remain))
    else:
        return np.vstack((new_row, matrix[1:]))

def iterative_vec_sub(matrix, vector, verbose=False):
    if len(matrix) == 0:
        return matrix
    
    row = matrix[0]
    new_row = np.maximum(0, row - vector)
    new_vector = np.maximum(0, vector - row)
    if np.any(new_vector > 0):
        matrix_remain = iterative_vec_sub(matrix[1:], new_vector, verbose=verbose)
        return np.vstack((new_row, matrix_remain))
    else:
        return np.vstack((new_row, matrix[1:]))

class GlobalBillAdaptativeOptimiser:


    def __init__(self, members=[], current_offtake_peak_cost=0, current_injection_peak_cost=0, historical_offtake_peak_cost=0, historical_injection_peak_cost=0, Delta_M=1, Delta_C=1, Delta_P=1, Delta_P_prime=0, id_optimiser=None, incremental_build=False, greedy_init=False, rec_import_export_enforce_mutex=True, n_cpus=None, time_optim = False, rec_import_fees=0.0, rec_export_fees=0.0, constant_price_per_member=True, activate_optim_no_peak_costs=False, type_solve=TYPE_SOLVE, dpp_compile=True, key_sorting_member_import=None, key_sorting_member_export=None, force_optim_no_peak_costs=False):
        self._members = members
        self._current_offtake_peak_cost = current_offtake_peak_cost
        self._current_injection_peak_cost = current_injection_peak_cost
        self._historical_offtake_peak_cost = historical_offtake_peak_cost
        self._historical_injection_peak_cost = historical_injection_peak_cost
        self._Delta_M = Delta_M
        self._Delta_C = Delta_C
        self._Delta_P = Delta_P
        self._Delta_P_prime = Delta_P_prime
        self._id_optimiser = id_optimiser
        self._len_members = len(members)
        self._incremental_build_flag = incremental_build
        self._greedy_init = greedy_init
        self._involve_current_peaks = self._current_offtake_peak_cost > 0 or self._current_injection_peak_cost > 0
        self._involve_historical_peaks = (self._Delta_P_prime > 0 and (self._historical_offtake_peak_cost > 0 or self._historical_injection_peak_cost))
        self._involve_peaks = self._involve_current_peaks or self._involve_historical_peaks
        self._rec_import_export_enforce_mutex=rec_import_export_enforce_mutex
        self._n_cpus = n_cpus
        self._time_optim = time_optim
        self._rec_import_fees = rec_import_fees
        self._rec_export_fees = rec_export_fees
        self._complete_problem = None
        self._Delta_prod = self._Delta_M*self._Delta_C
        self._previous_buying_prices = None
        self._previous_selling_prices = None
        self._member_index_buying_price_ordered = None
        self._member_index_selling_price_ordered = None
        self._constant_price_per_member = constant_price_per_member
        self._activate_optim_no_peak_costs = activate_optim_no_peak_costs
        self._force_optim_no_peak_costs = force_optim_no_peak_costs
        self._type_solve = type_solve
        self._dpp_compile = dpp_compile
        self._key_sorting_member_import_fct = key_sorting_member_import
        self._key_sorting_member_export_fct = key_sorting_member_export
        self._key_sorting_member_import = None
        self._key_sorting_member_export = None
        self._members_uniquely_sorted_import = None
        self._members_uniquely_sorted_export = None
        self._previous_no_peaks_solution = None
        self._previous_price = None

    @property
    def involve_peaks(self):
        return self._involve_peaks
    
    @property
    def rec_import_export_enforce_mutex(self):
        return self._rec_import_export_enforce_mutex
    
    @property
    def involve_current_peaks(self):
        return self._involve_current_peaks
    
    @property
    def involve_historical_peaks(self):
        return self._involve_historical_peaks

    @property
    def incremental_build_flag(self):
        return self._incremental_build_flag
    
    @incremental_build_flag.setter
    def incremental_build_flag(self, flag: bool):
        self._incremental_build_flag = flag
    
    @property
    def greedy_init(self):
        return self._greedy_init
    
    @greedy_init.setter
    def greedy_init(self, new_greedy_init: bool):
        self._greedy_init = new_greedy_init

    @property
    def n_cpus(self):
        return self._n_cpus
    
    @n_cpus.setter
    def n_cpus(self, n_cpus: int):
        self._n_cpus = n_cpus

    @property
    def time_optim(self):
        return self._time_optim
    
    @time_optim.setter
    def time_optim(self, time_optim: bool):
        self._time_optim = time_optim

    @rec_import_export_enforce_mutex.setter
    def rec_import_export_enforce_mutex(self, flag:True):
        self._rec_import_export_enforce_mutex = flag

    def _zeroing_complete_problem(self):
        for v in self._complete_problem.variables():
            v.value = np.zeros(v.shape)
        for p in self._complete_problem.parameters():
            p.value = np.zeros(p.shape)

    def _zeroing_complete_problem_mosek(self):
        self._rec_imports.setLevel(np.zeros_like(self._rec_imports.level()))
        self._rec_exports.setLevel(np.zeros_like(self._rec_exports.level()))
        self._grid_imports.setLevel(np.zeros_like(self._grid_imports.level()))
        self._grid_exports.setLevel(np.zeros_like(self._grid_exports.level()))


    def _build_complete_problem_mosek(self):
        sequences_shape=[self._len_members, self._Delta_P]
        M = mk.Model()
        M.setSolverParam("presolveUse", "off")
        M.setSolverParam("presolveLindepUse", "off")
        M.setSolverParam("presolveMaxNumPass", "1")
        M.setSolverParam("optimizer", "primalSimplex")
        M.setSolverParam("simSolveForm", "primal")
        M.setSolverParam("simSwitchOptimizer", "on")
        M.setSolverParam("simReformulation", "off")
        M.setSolverParam("simScaling", "none")
        M.setSolverParam("numThreads", 1)
        M.setSolverParam("log", 0)
        M.setSolverParam("simPrimalRestrictSelection", 0)
        M.setSolverParam("simPrimalSelection", "ase")
        M.setSolverParam("simDualRestrictSelection", 0)
        M.setSolverParam("simDualSelection", "ase")
        M.setSolverParam("simExploitDupvec", "off")
        M.setSolverParam("simMaxNumSetbacks", 0)
        M.setSolverParam("simNonSingular", "off")
        M.setSolverParam("simSaveLu", "on")
        M.setSolverParam("writeLpFullObj", "off")






        #import sys
        #M.setLogHandler(sys.stdout)
        




        current_offtake_peak_to_bill = None
        current_injection_peak_to_bill = None
        net_consumption_meter:mk.Parameter = M.parameter(sequences_shape)
        net_production_meter:mk.Parameter = M.parameter(sequences_shape)
        buying_prices:mk.Parameter = M.parameter(sequences_shape)
        selling_prices:mk.Parameter = M.parameter(sequences_shape)

        complete_problem_parameters = {
            "net_consumption_meter": net_consumption_meter,
            "net_production_meter": net_production_meter,
            "buying_prices": buying_prices,
            "selling_prices": selling_prices
        }

        rec_import_variables = M.variable(sequences_shape, mk.Domain.greaterThan(0))
        rec_export_variables = M.variable(sequences_shape, mk.Domain.greaterThan(0))
        grid_import_variables = M.variable(sequences_shape, mk.Domain.greaterThan(0))
        grid_export_variables = M.variable(sequences_shape, mk.Domain.greaterThan(0))

        if self._involve_peaks:
            prorata:mk.Parameter = M.parameter()
            prorata.setValue(1.0)
            complete_problem_parameters["prorata"] = prorata

        if self._involve_current_peaks and (epsilonify(self._current_offtake_peak_cost, epsilon=1e-8) > 0 or epsilonify(self._current_injection_peak_cost, epsilon=1e-8) > 0):
            if epsilonify(self._current_offtake_peak_cost, epsilon=1e-8) > 0:
                current_offtake_peak_to_bill = M.variable(self._len_members, mk.Domain.greaterThan(0))
            if epsilonify(self._current_injection_peak_cost, epsilon=1e-8) > 0:
                current_injection_peak_to_bill = M.variable(self._len_members, mk.Domain.greaterThan(0))
        if self._involve_historical_peaks:
            if self._historical_offtake_peak_cost > 0:
                historical_offtake_peak_to_bill = M.variable(self._len_members, mk.Domain.greaterThan(0))
            if self._historical_injection_peak_cost > 0:
                historical_injection_peak_to_bill = M.variable(self._len_members, mk.Domain.greaterThan(0))
        metering_period_objective_expr = (
            mk.Expr.sub(mk.Expr.sum(mk.Expr.mulElm(buying_prices, grid_import_variables)), mk.Expr.sum(mk.Expr.mulElm(selling_prices, grid_export_variables)))
        )
        metering_period_objective_expr = mk.Expr.add(metering_period_objective_expr, (
            mk.Expr.add(mk.Expr.mul(mk.Expr.sum(rec_import_variables), self._rec_import_fees), mk.Expr.mul(mk.Expr.sum(rec_export_variables), self._rec_export_fees))
        ))
        peak_period_objective_expr = 0
        current_peak_cost = 0
        if self._involve_current_peaks and (epsilonify(self._current_offtake_peak_cost, epsilon=1e-8) > 0 or epsilonify(self._current_injection_peak_cost, epsilon=1e-8) > 0):
            if epsilonify(self._current_offtake_peak_cost, epsilon=1e-8) > 0:
                current_peak_cost = mk.Expr.add(current_peak_cost, mk.Expr.mul(mk.Expr.sum(current_offtake_peak_to_bill), epsilonify(self._current_offtake_peak_cost, epsilon=1e-8)))
            if epsilonify(self._current_injection_peak_cost, epsilon=1e-8) > 0:
                current_peak_cost = mk.Expr.add(current_peak_cost, mk.Expr.mul(mk.Expr.sum(current_injection_peak_to_bill), epsilonify(self._current_injection_peak_cost, epsilon=1e-8)))
            peak_period_objective_expr = mk.Expr.mul(mk.Expr.mul(prorata, current_peak_cost), 1.0/(self._Delta_prod))

        proratized_historical_peak_cost = None
        historical_peak_cost = 0.0
        if self._involve_historical_peaks:

            if self._historical_offtake_peak_cost > 0:
                historical_peak_cost = mk.Expr.add(historical_peak_cost, mk.Expr.mul(mk.Expr.sum(historical_offtake_peak_to_bill), epsilonify(self._historical_offtake_peak_cost, epsilon=1e-8)))
            if self._historical_injection_peak_cost > 0:
                historical_peak_cost = mk.Expr.add(historical_peak_cost, mk.Expr.mul(mk.Expr.sum(historical_injection_peak_to_bill), epsilonify(self._historical_injection_peak_cost, epsilon=1e-8)))
            if type(historical_peak_cost) not in (int, float):
                peak_period_objective_expr = mk.Expr.mul(mk.Expr.mul(prorata, historical_peak_cost), 1.0/(self._Delta_prod))
        """
        constraints = [
            rec_import_variables >= 0,
            grid_import_variables >= 0,
            rec_export_variables >= 0,
            grid_export_variables >= 0,
            net_consumption_meter == rec_import_variables + grid_import_variables,
            net_production_meter == rec_export_variables + grid_export_variables,
            rec_import_variables <= net_consumption_meter,
            rec_export_variables <= net_production_meter,
            cp.sum(rec_import_variables, axis=0) == cp.sum(rec_export_variables, axis=0)
        ]
        """
        objective_expr = mk.Expr.add(metering_period_objective_expr, peak_period_objective_expr)
        M.objective(mk.ObjectiveSense.Minimize, objective_expr)
        M.constraint(mk.Expr.sub(net_consumption_meter, mk.Expr.add(rec_import_variables, grid_import_variables)), mk.Domain.equalsTo(0.0))
        M.constraint(mk.Expr.sub(net_production_meter, mk.Expr.add(rec_export_variables, grid_export_variables)), mk.Domain.equalsTo(0.0))
        #M.constraint(mk.Expr.sub(rec_import_variables, net_consumption_meter), mk.Domain.lessThan(0.0))
        #M.constraint(mk.Expr.sub(rec_export_variables, net_production_meter), mk.Domain.lessThan(0.0))
        #M.constraint(mk.Expr.sub(grid_import_variables, net_consumption_meter), mk.Domain.lessThan(0.0))
        #M.constraint(mk.Expr.sub(grid_export_variables, net_production_meter), mk.Domain.lessThan(0.0))
        M.constraint(mk.Expr.sub(mk.Expr.sum(rec_import_variables, 0), mk.Expr.sum(rec_export_variables, 0)), mk.Domain.equalsTo(0.0))
        if self._involve_current_peaks and (epsilonify(self._current_offtake_peak_cost, epsilon=1e-8) > 0 or epsilonify(self._current_injection_peak_cost, epsilon=1e-8) > 0):
            if  epsilonify(self._current_offtake_peak_cost, epsilon=1e-8) > 0:
                for i in range(self._Delta_P):
                    M.constraint(mk.Expr.sub(current_offtake_peak_to_bill, grid_import_variables.slice([0, i], [self._len_members, i+1])), mk.Domain.greaterThan(0.0))
            if  epsilonify(self._current_injection_peak_cost, epsilon=1e-8) > 0:
                for i in range(self._Delta_P):
                    M.constraint(mk.Expr.sub(current_injection_peak_to_bill, grid_export_variables.slice([0, i], [self._len_members, i+1])), mk.Domain.greaterThan(0.0))
        if self._involve_historical_peaks:
    
            
            if self._historical_offtake_peak_cost > 0:
                historical_offtake_peak_max = mk.parameter(self._len_members)
                historical_offtake_peak_max.setValue(value=np.zeros(self._len_members))
                complete_problem_parameters["historical_offtake_peak_max"] = historical_offtake_peak_max
                M.constraint(mk.Expr.sub(historical_offtake_peak_to_bill, current_offtake_peak_to_bill), mk.Domain.greaterThan(0.0))
                M.constraint(mk.Expr.sub(historical_offtake_peak_to_bill, historical_offtake_peak_max), mk.Domain.greaterThan(0.0))
            if self._historical_injection_peak_cost > 0:
                historical_injection_peak_max = mk.parameter(self._len_members)
                historical_injection_peak_max.setValue(value=np.zeros(self._len_members))
                complete_problem_parameters["historical_injection_peak_max"] = historical_injection_peak_max
                M.constraint(mk.Expr.sub(historical_injection_peak_to_bill, current_injection_peak_to_bill), mk.Domain.greaterThan(0.0))
                M.constraint(mk.Expr.sub(historical_injection_peak_to_bill, historical_injection_peak_max), mk.Domain.greaterThan(0.0))
        
        self._complete_problem = M
        self._complete_problem_parameters = complete_problem_parameters
        self._complete_problem_objectives = (metering_period_objective_expr, peak_period_objective_expr)
        self._complete_problem_current_peaks = (current_offtake_peak_to_bill, current_injection_peak_to_bill)
        self._previous_number_meters = None
        self._rec_imports = rec_import_variables
        self._rec_exports = rec_export_variables
        self._grid_imports = grid_import_variables
        self._grid_exports = grid_export_variables
        if self._involve_peaks:
            self._prorata = prorata
        if self._involve_historical_peaks:
            self._historical_offtake_peak_to_bill = historical_offtake_peak_to_bill
            self._historical_injection_peak_to_bill = historical_injection_peak_to_bill

    def _incremental_fill_complete_problem_mosek(
        self,
        consumption_meter_states,
        production_meter_states,
        buying_prices,
        selling_prices,
        current_metering_period_counter,
        current_peak_period_counter,
        historical_offtake_peaks = None,
        historical_injection_peaks = None
    ):
        metering_period_objective_expr, peak_period_objective_expr = self._complete_problem_objectives
        
        if not self._incremental_build_flag:
            len_T = len(consumption_meter_states[self._members[0]])
            consumption_meter_states_parameter = np.zeros((self._len_members, self._Delta_P))
            production_meter_states_parameter = np.zeros((self._len_members, self._Delta_P))
            buying_prices_parameter = np.zeros((self._len_members, self._Delta_P))
            selling_prices_parameter = np.zeros((self._len_members, self._Delta_P))
            consumption_meter_states_parameter[:, :len_T] = np.asarray([
               consumption_meter_states[member] for member in self._members
            ])
            production_meter_states_parameter[:, :len_T] = np.asarray([
               production_meter_states[member] for member in self._members
            ])
            buying_prices_parameter[:, :len_T] = np.asarray([
                buying_prices[member] for member in self._members
            ])
            selling_prices_parameter[:, :len_T] = np.asarray([
                selling_prices[member] for member in self._members
            ])
            net_consumption_meter_states_parameter = np.maximum(consumption_meter_states_parameter - production_meter_states_parameter, 0.0)
            net_production_meter_states_parameter = np.maximum(production_meter_states_parameter - consumption_meter_states_parameter, 0.0)
            self._complete_problem_parameters["net_consumption_meter"].setValue(net_consumption_meter_states_parameter)
            self._complete_problem_parameters["net_production_meter"].setValue(net_production_meter_states_parameter)
            self._complete_problem_parameters["buying_prices"].setValue(buying_prices_parameter)
            self._complete_problem_parameters["selling_prices"].setValue(selling_prices_parameter)
            self._buying_prices = buying_prices_parameter.flatten()
            self._selling_prices = selling_prices_parameter.flatten()
            if self._involve_peaks and (epsilonify(self._current_offtake_peak_cost, epsilon=1e-8) > 0 or epsilonify(self._current_injection_peak_cost, epsilon=1e-8) > 0):
                self._prorata = elapsed_timesteps_in_peak_period(0, 0, current_metering_period_counter, current_peak_period_counter, Delta_M=self._Delta_M, Delta_P=self._Delta_P)
                self._complete_problem_parameters["prorata"].setValue(self._prorata)
            if self._involve_historical_peaks:
                if historical_offtake_peaks is not None:
                    self._complete_problem_parameters["historical_offtake_peak_max"].setValue(np.max(np.asarray(list(historical_offtake_peaks.values())), axis=1)[:, 0])
                if historical_injection_peaks is not None:
                    self._complete_problem_parameters["historical_injection_peak_max"].setValue(np.max(np.asarray(list(historical_injection_peaks.values())), axis=1)[:, 0])
            current_offtake_peak_to_bill, current_injection_peak_to_bill = None, None
            if self._involve_current_peaks:
                current_offtake_peak_to_bill, current_injection_peak_to_bill = self._complete_problem_current_peaks

            if self._previous_number_meters is not None:
                nb_new_meters = len_T - self._previous_number_meters
            else:
                nb_new_meters = len_T
            if nb_new_meters > 0 and self._init:
                self._complete_problem.setSolverParam("simHotstart", "none")
            else:
                self._complete_problem.setSolverParam("simHotstart", "free")
            self._previous_number_meters = len_T
            return self._complete_problem, metering_period_objective_expr, peak_period_objective_expr, current_offtake_peak_to_bill, current_injection_peak_to_bill
        else:
            raise NotImplementedError()

    def _build_complete_problem(self):
        sequences_shape=(self._len_members, self._Delta_P)
        net_consumption_meter = cp.Parameter(sequences_shape, nonneg=True, value=np.zeros(sequences_shape))
        net_production_meter = cp.Parameter(sequences_shape, nonneg=True, value=np.zeros(sequences_shape)) 
        buying_prices = cp.Parameter(self._len_members, nonneg=True, value=np.zeros(self._len_members))
        selling_prices = cp.Parameter(self._len_members, nonneg=True, value=np.zeros(self._len_members))

        current_offtake_peak_to_bill = None
        current_injection_peak_to_bill = None

        rec_import_variables = cp.Variable(sequences_shape)
        rec_export_variables = cp.Variable(sequences_shape)
        grid_import_variables = cp.Variable(sequences_shape)
        grid_export_variables = cp.Variable(sequences_shape)
        if self._involve_peaks:
            prorata = 1.0
            prorata = cp.Parameter(value=prorata)

        if self._involve_current_peaks and (epsilonify(self._current_offtake_peak_cost, epsilon=1e-8) > 0 or epsilonify(self._current_injection_peak_cost, epsilon=1e-8) > 0):
            if epsilonify(self._current_offtake_peak_cost, epsilon=1e-8) > 0:
                current_offtake_peak_to_bill = cp.Variable(self._len_members)
            if epsilonify(self._current_injection_peak_cost, epsilon=1e-8) > 0:
                current_injection_peak_to_bill = cp.Variable(self._len_members)
        if self._involve_historical_peaks:
            if self._historical_offtake_peak_cost > 0:
                historical_offtake_peak_to_bill = cp.Variable(self._len_members)
            if self._historical_injection_peak_cost > 0:
                historical_injection_peak_to_bill = cp.Variable(self._len_members)

        #metering_period_objective_expr = (
        #    cp.sum(cp.multiply(buying_prices, grid_import_variables)) - cp.sum(cp.multiply(selling_prices, grid_export_variables))
        #)
        metering_period_objective_expr = (
            cp.sum(buying_prices@grid_import_variables) - cp.sum(selling_prices@grid_export_variables)
        )
        metering_period_objective_expr += (
            cp.sum(rec_import_variables) * self._rec_import_fees + cp.sum(rec_export_variables) * self._rec_export_fees
        )

        peak_period_objective_expr = 0
        current_peak_cost = 0
        proratized_current_peak_cost = None
        if self._involve_current_peaks and (epsilonify(self._current_offtake_peak_cost, epsilon=1e-8) > 0 or epsilonify(self._current_injection_peak_cost, epsilon=1e-8) > 0):
            proratized_current_peak_cost = cp.Variable()
            if epsilonify(self._current_offtake_peak_cost, epsilon=1e-8) > 0:
                current_peak_cost += (cp.sum(current_offtake_peak_to_bill) * epsilonify(self._current_offtake_peak_cost, epsilon=1e-8))
            if epsilonify(self._current_injection_peak_cost, epsilon=1e-8) > 0:
                current_peak_cost += (cp.sum(current_injection_peak_to_bill) * epsilonify(self._current_injection_peak_cost, epsilon=1e-8))
            peak_period_objective_expr = proratized_current_peak_cost/(self._Delta_prod)

        proratized_historical_peak_cost = None
        historical_peak_cost = 0.0
        if self._involve_historical_peaks:
            proratized_historical_peak_cost = cp.Variable()
            if self._historical_offtake_peak_cost > 0:
                historical_peak_cost += cp.sum(historical_offtake_peak_to_bill) * epsilonify(self._historical_offtake_peak_cost, epsilon=1e-8)
            if self._historical_injection_peak_cost > 0:
                historical_peak_cost += cp.sum(historical_injection_peak_to_bill) * epsilonify(self._historical_injection_peak_cost, epsilon=1e-8)
            if type(historical_peak_cost) not in (int, float):
                peak_period_objective_expr += proratized_historical_peak_cost/(self._Delta_prod)

        constraints = [
            rec_import_variables >= 0,
            grid_import_variables >= 0,
            rec_export_variables >= 0,
            grid_export_variables >= 0,
            net_consumption_meter == rec_import_variables + grid_import_variables,
            net_production_meter == rec_export_variables + grid_export_variables,
            rec_import_variables <= net_consumption_meter,
            rec_export_variables <= net_production_meter,
            cp.sum(rec_import_variables, axis=0) == cp.sum(rec_export_variables, axis=0)
        ]
        complete_problem_parameters = {
            "net_consumption_meter": net_consumption_meter,
            "net_production_meter": net_production_meter,
            "buying_prices": buying_prices,
            "selling_prices": selling_prices
        }
        if self._involve_peaks and (epsilonify(self._current_offtake_peak_cost, epsilon=1e-8) > 0 or epsilonify(self._current_injection_peak_cost, epsilon=1e-8) > 0):
            complete_problem_parameters["prorata"] = prorata
        if self._involve_current_peaks and (epsilonify(self._current_offtake_peak_cost, epsilon=1e-8) > 0 or epsilonify(self._current_injection_peak_cost, epsilon=1e-8) > 0):
            constraints += [
                proratized_current_peak_cost == prorata*current_peak_cost
            ]
            if self._current_offtake_peak_cost > 0:
                constraints += [
                    cp.reshape(current_offtake_peak_to_bill, (self._len_members, 1)) >= grid_import_variables
                ]
            if self._current_injection_peak_cost > 0:
                constraints += [
                    cp.reshape(current_injection_peak_to_bill, (self._len_members, 1)) >= grid_export_variables
                ]
        if self._involve_historical_peaks:
            constraints += [
                proratized_historical_peak_cost == prorata*historical_peak_cost
            ]
            
            
            if self._historical_offtake_peak_cost > 0:
                historical_offtake_peak_max = cp.Parameter(self._len_members, value=np.zeros(self._len_members))
                complete_problem_parameters["historical_offtake_peak_max"] = historical_offtake_peak_max
                constraints += [
                    historical_offtake_peak_to_bill >= current_offtake_peak_to_bill
                ]
                constraints += [
                    historical_offtake_peak_to_bill >= historical_offtake_peak_max
                ]
            if self._historical_injection_peak_cost > 0:
                historical_injection_peak_max = cp.Parameter(self._len_members, value=np.zeros(self._len_members))
                complete_problem_parameters["historical_injection_peak_max"] = historical_injection_peak_max
                constraints += [
                    historical_injection_peak_to_bill >= current_injection_peak_to_bill
                ]
                constraints += [
                    historical_injection_peak_to_bill >= historical_injection_peak_max
                ]
        objective_expr = metering_period_objective_expr + peak_period_objective_expr
        self._complete_problem = cp.Problem(cp.Minimize(objective_expr), constraints)
        self._complete_problem_parameters = complete_problem_parameters
        self._complete_problem_objectives = (metering_period_objective_expr, peak_period_objective_expr)
        self._complete_problem_current_peaks = (current_offtake_peak_to_bill, current_injection_peak_to_bill)
        self._rec_imports = rec_import_variables
        self._rec_exports = rec_export_variables

    def reset(self):
        self._prob_data = None
        self._metering_period_cost = 0
        self._peak_period_cost = 0
        self._init = True
        if not self._force_optim_no_peak_costs and (not self._activate_optim_no_peak_costs or (epsilonify(self._current_offtake_peak_cost, epsilon=1e-8) > 0 or epsilonify(self._current_injection_peak_cost, epsilon=1e-8) > 0)) and self._complete_problem is None and len(self._members)>2:
            if self._type_solve == "mosek":
                build_complete_problem_method = self._build_complete_problem_mosek
            else:
                build_complete_problem_method = self._build_complete_problem
            build_complete_problem_method()
        else:
            pass
            #if self._type_solve == "mosek":
            #    zeroing_complete_problem_method = self._zeroing_complete_problem_mosek
            #else:
            #    zeroing_complete_problem_method = self._zeroing_complete_problem
            #zeroing_complete_problem_method()
        self._previous_no_peaks_solution = None
        self._previous_number_meters = 0

    def _greedy_optimisation(
            self,
            current_metering_period_counter,
            current_peak_period_counter,
            consumption_meter_states,
            production_meter_states,
            buying_prices,
            selling_prices
        ):
        len_T = len(consumption_meter_states[self._members[0]])
        if self._involve_peaks:
            prorata = elapsed_timesteps_in_peak_period(0, 0, current_metering_period_counter, current_peak_period_counter, Delta_M=self._Delta_M, Delta_P=self._Delta_P)
        else:
            prorata = 1.0
        initial_rec_imports = {
            member:[] for member in self._members
        }
        initial_rec_exports = {
            member:[] for member in self._members
        }
        initial_grid_imports = {
            member:[] for member in self._members
        }
        initial_grid_exports = {
            member:[] for member in self._members
        }
        peak_states = dict()
        if self._current_offtake_peak_cost > 0:
            max_grid_imports = {
                member:0.0 for member in self._members
            }
            peak_states = {
                **peak_states,
                **{
                    (member, "current_offtake_peak"): max_grid_imports[member] for member in max_grid_imports.keys()
                }
            }
        if self._current_injection_peak_cost > 0:
            max_grid_exports = {
                member:0.0 for member in self._members
            }
            peak_states = {
                **peak_states,
                **{
                    (member, "current_injection_peak"): max_grid_exports[member] for member in max_grid_exports.keys()
                }
            }
        initial_metering_period_cost = 0
        initial_peak_period_cost = 0
        for i in range(len_T):
            rec_exchanges = self._greedy_optimiser.optimise_repartition_keys(
                self._members,
                (self._Delta_M if i != len_T - 1 else current_metering_period_counter),
                i,
                {
                    member:consumption_meters[i] for member, consumption_meters in consumption_meter_states.items()
                },
                {
                    member:production_meters[i] for member, production_meters in production_meter_states.items()
                },
                {
                    member:buying_price[i] for member, buying_price in buying_prices.items()
                },
                {
                    member:selling_price[i] for member, selling_price in selling_prices.items()
                },
                Delta_C=self._Delta_C,
                Delta_M=self._Delta_M,
                Delta_P=self._Delta_P,
                Delta_P_prime=self._Delta_P_prime,
                current_offtake_peak_cost=self._current_offtake_peak_cost,
                current_injection_peak_cost=self._current_injection_peak_cost,
                historical_offtake_peak_cost=self._historical_offtake_peak_cost,
                historical_injection_peak_cost=self._historical_injection_peak_cost,
                peak_states=peak_states
            )
            initial_rec_imports = {
                member: initial_rec_imports[member] + [rec_exchanges[(member, "rec_import")]]
                for member in self._members
            }
            initial_rec_exports = {
                member: initial_rec_exports[member] + [rec_exchanges[(member, "rec_export")]]
                for member in self._members
            }
            initial_grid_imports = {
                member: initial_grid_imports[member] + [consumption_meter_states[member][i] - rec_exchanges[(member, "rec_import")]]
                for member in self._members
            }
            initial_grid_exports = {
                member: initial_grid_exports[member] + [production_meter_states[member][i] - rec_exchanges[(member, "rec_export")]]
                for member in self._members
            }
            buying_cost_part = sum(
                [buying_prices[member][i] * (consumption_meter_states[member][i] - rec_exchanges[(member, "rec_import")]) for member in self._members]
            )
            selling_cost_part = sum(
                [selling_prices[member][i] * (production_meter_states[member][i] - rec_exchanges[(member, "rec_export")]) for member in self._members]
            )

            if self._current_offtake_peak_cost > 0:
                max_grid_imports = {
                    member: max([max_grid_imports[member], consumption_meter_states[member][i] - rec_exchanges[(member, "rec_import")]])
                    for member in self._members
                }
            if self._current_injection_peak_cost > 0:
                max_grid_exports = {
                    member: max([max_grid_exports[member], production_meter_states[member][i] - rec_exchanges[(member, "rec_export")]])
                    for member in self._members
                }
            initial_metering_period_cost += (buying_cost_part - selling_cost_part)
        # If peaks are not activated at all then it is the optimal electricity bill
        if not self._involve_peaks:
            self._prob_data = dict()
            return initial_metering_period_cost, None, None, None, None, None, None, None
        if self._involve_current_peaks:
            if self._current_offtake_peak_cost > 0:
                initial_peak_period_cost += epsilonify(self._current_offtake_peak_cost, epsilon=1e-8) * sum(max_grid_imports.values())
            if self._current_injection_peak_cost > 0:
                initial_peak_period_cost += epsilonify(self._current_injection_peak_cost, epsilon=1e-8) * sum(max_grid_exports.values())
        if initial_peak_period_cost > 0:
            initial_peak_period_cost *= prorata
            initial_peak_period_cost /= (self._Delta_prod)
        
        if self._involve_historical_peaks:
            initial_historical_peak_period_cost = 0
            if self._Delta_P_prime > 0:
                if self._historical_offtake_peak_cost > 0:
                    historical_offtake_peaks = {
                        member:max(historical_offtake_peaks[member] + [max_grid_imports[member]]) for member in self._members
                    }
                    initial_historical_peak_period_cost += self._historical_offtake_peak_cost * sum(historical_offtake_peaks.values())
                if self._historical_injection_peak_cost > 0:
                    historical_injection_peaks = {
                        member:max(historical_injection_peaks[member] + [max_grid_exports[member]]) for member in self._members
                    }
                    initial_historical_peak_period_cost += self._historical_injection_peak_cost * sum(historical_injection_peaks.values())
            if initial_historical_peak_period_cost > 0:
                initial_historical_peak_period_cost *= prorata
                initial_historical_peak_period_cost /= (self._Delta_prod)
                initial_peak_period_cost += initial_historical_peak_period_cost
        return initial_metering_period_cost, initial_peak_period_cost, initial_rec_imports, initial_rec_exports, initial_grid_imports, initial_grid_exports, max_grid_imports, max_grid_exports
    
    def _incremental_fill_complete_problem(
        self,
        consumption_meter_states,
        production_meter_states,
        buying_prices,
        selling_prices,
        current_metering_period_counter,
        current_peak_period_counter,
        historical_offtake_peaks = None,
        historical_injection_peaks = None
    ):
        metering_period_objective_expr, peak_period_objective_expr = self._complete_problem_objectives
        if not self._incremental_build_flag:
            len_T = len(consumption_meter_states[self._members[0]])
            consumption_meter_states_parameter = np.zeros((self._len_members, self._Delta_P))
            production_meter_states_parameter = np.zeros((self._len_members, self._Delta_P))
            buying_prices_parameter = np.zeros((self._len_members, self._Delta_P))
            selling_prices_parameter = np.zeros((self._len_members, self._Delta_P))
            consumption_meter_states_parameter = np.asarray([
               consumption_meter_states[member] for member in self._members
            ])
            production_meter_states_parameter = np.asarray([
               production_meter_states[member] for member in self._members
            ])
            buying_prices_parameter = np.asarray([
                buying_prices[member][-1] for member in self._members
            ])
            selling_prices_parameter = np.asarray([
                selling_prices[member][-1] for member in self._members
            ])
            net_consumption_meter_states_parameter = consumption_meter_states_parameter - production_meter_states_parameter
            net_consumption_meter_states_parameter[net_consumption_meter_states_parameter<0.0] = 0.0
            net_production_meter_states_parameter = production_meter_states_parameter - consumption_meter_states_parameter
            net_production_meter_states_parameter[net_production_meter_states_parameter<0.0] = 0.0
            self._complete_problem_parameters["net_consumption_meter"].value[:, :len_T] = net_consumption_meter_states_parameter
            self._complete_problem_parameters["net_production_meter"].value[:, :len_T] = net_production_meter_states_parameter
            self._complete_problem_parameters["buying_prices"].value[:] = buying_prices_parameter
            self._complete_problem_parameters["selling_prices"].value[:] = selling_prices_parameter
            self._buying_prices = buying_prices_parameter
            self._selling_prices = selling_prices_parameter
            if self._involve_peaks and (epsilonify(self._current_offtake_peak_cost, epsilon=1e-8) > 0 or epsilonify(self._current_injection_peak_cost, epsilon=1e-8) > 0):
                self._prorata = elapsed_timesteps_in_peak_period(0, 0, current_metering_period_counter, current_peak_period_counter, Delta_M=self._Delta_M, Delta_P=self._Delta_P)
                self._complete_problem_parameters["prorata"].value = self._prorata
            if self._involve_historical_peaks:
                if historical_offtake_peaks is not None:
                    self._complete_problem_parameters["historical_offtake_peak_max"].value = np.max(np.asarray(list(historical_offtake_peaks.values())), axis=1)[:, 0]
                if historical_injection_peaks is not None:
                    self._complete_problem_parameters["historical_injection_peak_max"].value = np.max(np.asarray(list(historical_injection_peaks.values())), axis=1)[:, 0]
            current_offtake_peak_to_bill, current_injection_peak_to_bill = None, None
            if self._involve_current_peaks and (epsilonify(self._current_offtake_peak_cost, epsilon=1e-8) > 0 or epsilonify(self._current_injection_peak_cost, epsilon=1e-8) > 0):
                current_offtake_peak_to_bill, current_injection_peak_to_bill = self._complete_problem_current_peaks
            
            return self._complete_problem, metering_period_objective_expr, peak_period_objective_expr, current_offtake_peak_to_bill, current_injection_peak_to_bill
        else:
            raise NotImplementedError()

    def _incremental_build(
            self,
            consumption_meter_states,
            production_meter_states,
            buying_prices,
            selling_prices,
            current_metering_period_counter,
            current_peak_period_counter,
            historical_offtake_peaks = None,
            historical_injection_peaks = None
        ) -> Tuple[cp.Problem, cp.Variable, cp.Variable]:
        len_T = len(consumption_meter_states[self._members[0]])
        current_offtake_peak_to_bill = None
        current_injection_peak_to_bill = None
        initial_metering_period_cost = None
        initial_peak_period_cost = None
        prorata = None
        involve_current_peaks = self._involve_current_peaks
        involve_historical_peaks = self._involve_historical_peaks
        involve_peaks = self._involve_peaks
        if involve_peaks:
            prorata = elapsed_timesteps_in_peak_period(0, 0, current_metering_period_counter, current_peak_period_counter, Delta_M=self._Delta_M, Delta_P=self._Delta_P)
        greedy_metering_period_cost, greedy_peak_period_cost, initial_rec_imports, initial_rec_exports, initial_grid_imports, initial_grid_exports, offtake_peaks, injection_peaks = None, None, None, None, None, None, None, None
        if not self._incremental_build_flag or self._prob_data is None:
            if (self._greedy_init or not self._involve_peaks) and False:
                greedy_metering_period_cost, greedy_peak_period_cost, initial_rec_imports, initial_rec_exports, initial_grid_imports, initial_grid_exports, offtake_peaks, injection_peaks = self._greedy_optimisation(
                    current_metering_period_counter,
                    current_peak_period_counter,
                    consumption_meter_states,
                    production_meter_states,
                    buying_prices,
                    selling_prices
                )
                if not self._involve_peaks:
                    return None, greedy_metering_period_cost, greedy_metering_period_cost, offtake_peaks, injection_peaks
            sequences_shape = (self._len_members, len_T)
            consumption_meter_parameter = cp.Parameter(sequences_shape, nonneg=True, value=np.asarray(list(consumption_meter_states.values())))
            production_meter_parameter = cp.Parameter(sequences_shape, nonneg=True, value=np.asarray(list(production_meter_states.values())))
            
            buying_prices_parameter = cp.Parameter(sequences_shape, nonneg=True, value=np.asarray(list(buying_prices.values())))
            selling_prices_parameter = cp.Parameter(sequences_shape, nonneg=True, value=np.asarray(list(selling_prices.values())))
            if involve_peaks:
                prorata = cp.Parameter(value=prorata)
            if involve_current_peaks:
                if self._current_offtake_peak_cost > 0:
                    current_offtake_peak_cost = cp.Parameter(value=epsilonify(self._current_offtake_peak_cost, epsilon=1e-8))
                if self._current_injection_peak_cost > 0:
                    current_injection_peak_cost = cp.Parameter(value=epsilonify(self._current_injection_peak_cost, epsilon=1e-8))
            if involve_historical_peaks:
                if self._historical_offtake_peak_cost > 0:
                    historical_offtake_peak_cost = cp.Parameter(value=self._historical_offtake_peak_cost)
                    if historical_offtake_peaks is not None:
                        historical_offtake_peak_max = cp.Parameter(self._len_members, value=np.max(np.asarray(list(historical_offtake_peaks.values())), axis=1)[:, 0])
                    
                if self._historical_injection_peak_cost > 0:
                    historical_injection_peak_cost = cp.Parameter(value=self._historical_injection_peak_cost)
                    if historical_injection_peaks is not None:
                        historical_injection_peak_max = cp.Parameter(self._len_members, value=np.max(np.asarray(list(historical_injection_peaks.values())), axis=1)[:, 0])
            
            rec_import_variables = cp.Variable(sequences_shape)
            rec_export_variables = cp.Variable(sequences_shape)
            grid_import_variables = cp.Variable(sequences_shape)
            grid_export_variables = cp.Variable(sequences_shape)

            if involve_current_peaks:
                if self._current_offtake_peak_cost > 0:
                    current_offtake_peak_to_bill = cp.Variable(self._len_members)
                if self._current_injection_peak_cost > 0:
                    current_injection_peak_to_bill = cp.Variable(self._len_members)
            if involve_historical_peaks:
                if self._historical_offtake_peak_cost > 0:
                    historical_offtake_peak_to_bill = cp.Variable(self._len_members)
                if self._historical_injection_peak_cost > 0:
                    historical_injection_peak_to_bill = cp.Variable(self._len_members)
            
            if self._greedy_init:
                
                if self._current_offtake_peak_cost > 0:
                    current_offtake_peak_to_bill.value = np.asarray(list(offtake_peaks.values()))
                if self._current_injection_peak_cost > 0:
                    current_injection_peak_to_bill.value = np.asarray(list(injection_peaks.values()))
                if involve_historical_peaks:
                    if self._historical_offtake_peak_cost > 0:
                        historical_offtake_peak_to_bill.value = np.asarray(list(historical_offtake_peaks.values()))
                    if self._historical_injection_peak_cost > 0:
                        historical_injection_peak_to_bill.value = np.asarray(list(historical_injection_peaks.values()))

            metering_period_objective_expr = (
                cp.sum(cp.multiply(buying_prices_parameter, grid_import_variables)) - cp.sum(cp.multiply(selling_prices_parameter, grid_export_variables))
            )
            metering_period_objective_expr += (
                cp.sum(rec_import_variables) * self._rec_import_fees + cp.sum(rec_export_variables) * self._rec_export_fees
            )
            peak_period_objective_expr = 0
            current_peak_cost = 0
            proratized_current_peak_cost = None
            if involve_current_peaks:
                proratized_current_peak_cost = cp.Variable()
                if self._current_offtake_peak_cost > 0:
                    current_peak_cost += (cp.sum(current_offtake_peak_to_bill) * epsilonify(self._current_offtake_peak_cost, epsilon=1e-8))
                if self._current_injection_peak_cost > 0:
                    current_peak_cost += (cp.sum(current_injection_peak_to_bill) * epsilonify(self._current_injection_peak_cost, epsilon=1e-8))
                peak_period_objective_expr = proratized_current_peak_cost/(self._Delta_prod)

            proratized_historical_peak_cost = None
            historical_peak_cost = 0.0
            if involve_historical_peaks:
                proratized_historical_peak_cost = cp.Variable()
                if self._historical_offtake_peak_cost > 0:
                    historical_peak_cost += cp.sum(historical_offtake_peak_to_bill) * epsilonify(self._historical_offtake_peak_cost, epsilon=1e-8)
                if self._historical_injection_peak_cost > 0:
                    historical_peak_cost += cp.sum(historical_injection_peak_to_bill) * epsilonify(self._historical_injection_peak_cost, epsilon=1e-8)
                if type(historical_peak_cost) not in (int, float):
                    peak_period_objective_expr += proratized_historical_peak_cost/(self._Delta_prod)
            objective_expr = metering_period_objective_expr + peak_period_objective_expr
            net_consumption_meter = consumption_meter_parameter.value - production_meter_parameter.value
            net_production_meter = production_meter_parameter.value - consumption_meter_parameter.value
            net_consumption_meter[net_consumption_meter < 0] = 0.0
            net_production_meter[net_production_meter < 0] = 0.0
            net_consumption_meter = cp.Parameter(value=net_consumption_meter, shape=net_consumption_meter.shape)
            net_production_meter = cp.Parameter(value=net_production_meter, shape=net_production_meter.shape)
            #print("Net consumption meter", net_consumption_meter.value)
            #print("Net production meter", net_production_meter.value)
            constraints = [
                rec_import_variables >= 0,
                grid_import_variables >= 0,
                rec_export_variables >= 0,
                grid_export_variables >= 0,
                net_consumption_meter == rec_import_variables + grid_import_variables,
                net_production_meter == rec_export_variables + grid_export_variables,
                rec_import_variables <= net_consumption_meter,
                rec_export_variables <= net_production_meter,
                grid_import_variables <= net_consumption_meter,
                grid_export_variables <= net_production_meter,
                cp.sum(rec_import_variables, axis=0) == cp.sum(rec_export_variables, axis=0)
            ]  
            if involve_current_peaks:
                constraints += [
                    proratized_current_peak_cost == prorata*current_peak_cost
                ]
                if self._current_offtake_peak_cost > 0:
                    constraints += [
                        cp.reshape(current_offtake_peak_to_bill, (self._len_members, 1)) >= grid_import_variables
                    ]
                if self._current_injection_peak_cost > 0:
                    constraints += [
                        cp.reshape(current_injection_peak_to_bill, (self._len_members, 1)) >= grid_export_variables
                    ]
            if involve_historical_peaks:
                constraints += [
                    proratized_historical_peak_cost == prorata*historical_peak_cost
                ]
                if self._historical_offtake_peak_cost > 0:
                    constraints += [
                        historical_offtake_peak_to_bill >= current_offtake_peak_to_bill
                    ]
                    if historical_offtake_peaks is not None:
                        constraints += [
                            historical_offtake_peak_to_bill >= historical_offtake_peak_max
                        ]
                if self._historical_injection_peak_cost > 0:
                    constraints += [
                        historical_injection_peak_to_bill >= current_injection_peak_to_bill
                    ]
                    if historical_injection_peaks is not None:
                        constraints += [
                            historical_injection_peak_to_bill >= historical_injection_peak_max
                        ]
            prob = cp.Problem(cp.Minimize(objective_expr), constraints)
            if self._incremental_build_flag and current_peak_period_counter < self._Delta_P:
                self._prob_data = {
                    "constraints": constraints,
                    "n_meters": consumption_meter_parameter.shape[1],
                    "consumption_meter_parameter": consumption_meter_parameter,
                    "production_meter_parameter": production_meter_parameter,
                    "metering_period_objective_expr": metering_period_objective_expr,
                    "peak_period_objective_expr": peak_period_objective_expr
                }
                if involve_peaks:
                    self._prob_data["prorata"] = prorata
                if involve_current_peaks:
                    if self._current_offtake_peak_cost > 0:
                        self._prob_data["current_offtake_peak_cost"] = current_offtake_peak_cost
                        self._prob_data["current_offtake_peak_to_bill"] = current_offtake_peak_to_bill
                    if self._current_injection_peak_cost > 0:
                        self._prob_data["current_injection_peak_cost"] = current_injection_peak_cost
                        self._prob_data["current_injection_peak_to_bill"] = current_injection_peak_to_bill
                if involve_historical_peaks:
                    if self._historical_offtake_peak_cost > 0:
                        self._prob_data["historical_offtake_peak_cost"] = historical_offtake_peak_cost
                        if historical_offtake_peaks is not None:
                            self._prob_data["historical_offtake_peak_max"] = historical_offtake_peak_max
                        
                    if self._historical_injection_peak_cost > 0:
                        self._prob_data["historical_injection_peak_cost"] = historical_injection_peak_cost
                        if historical_injection_peaks is not None:
                            self._prob_data["historical_injection_peak_max"] = historical_injection_peak_max
            else:
                self._prob_data = None
        else:
            
            constraints = self._prob_data["constraints"]
            n_meters_old = self._prob_data["n_meters"]
            metering_period_objective_expr = self._prob_data["metering_period_objective_expr"]
            peak_period_objective_expr = self._prob_data["peak_period_objective_expr"]
            prorata:cp.Parameter = self._prob_data.get("prorata", prorata)
            current_offtake_peak_to_bill = self._prob_data.get("current_offtake_peak_to_bill", None)
            current_injection_peak_to_bill = self._prob_data.get("current_injection_peak_to_bill", None)
            prorata.value = elapsed_timesteps_in_peak_period(0, 0, current_metering_period_counter, current_peak_period_counter, Delta_M=self._Delta_M, Delta_P=self._Delta_P)

            len_T = len(consumption_meter_states[self._members[0]]) - n_meters_old
            
            sequences_shape = (self._len_members, len_T)
            if len_T > 0:
                delta_consumption_meter_states = np.asarray([
                    consumption_meter_states[member][-len_T:] for member in self._members
                ])
                delta_production_meter_states = np.asarray([
                    production_meter_states[member][-len_T:] for member in self._members
                ])
                delta_buying_prices= np.asarray([
                    buying_prices[member][-len_T:] for member in self._members
                ])
                delta_selling_prices = np.asarray([
                    selling_prices[member][-len_T:] for member in self._members
                ])
                consumption_meter_parameter = cp.Parameter(sequences_shape, nonneg=True, value=delta_consumption_meter_states)
                production_meter_parameter = cp.Parameter(sequences_shape, nonneg=True, value=delta_production_meter_states)
                buying_prices_parameter = cp.Parameter(sequences_shape, nonneg=True, value=delta_buying_prices)
                selling_prices_parameter = cp.Parameter(sequences_shape, nonneg=True, value=delta_selling_prices)
                rec_import_variables = cp.Variable(sequences_shape)
                rec_export_variables = cp.Variable(sequences_shape)
                grid_import_variables = cp.Variable(sequences_shape)
                grid_export_variables = cp.Variable(sequences_shape)
                net_consumption_meter = consumption_meter_parameter.value - production_meter_parameter.value
                net_production_meter = production_meter_parameter.value - consumption_meter_parameter.value
                net_consumption_meter[net_consumption_meter < 0] = 0.0
                net_production_meter[net_production_meter < 0] = 0.0
                net_consumption_meter = cp.Parameter(value=net_consumption_meter, shape=net_consumption_meter.shape)
                net_production_meter = cp.Parameter(value=net_production_meter, shape=net_production_meter.shape)
                #print("Net consumption meter", net_consumption_meter.value)
                #print("Net production meter", net_production_meter.value)
                constraints += [
                    rec_import_variables >= 0,
                    grid_import_variables >= 0,
                    rec_export_variables >= 0,
                    grid_export_variables >= 0,
                    net_consumption_meter == rec_import_variables + grid_import_variables,
                    net_production_meter == rec_export_variables + grid_export_variables,
                    rec_import_variables <= net_consumption_meter,
                    rec_export_variables <= net_production_meter,
                    cp.sum(rec_import_variables, axis=0) == cp.sum(rec_export_variables, axis=0)
                ]
                if self._current_offtake_peak_cost > 0:
                    constraints += [
                        cp.reshape(current_offtake_peak_to_bill, (self._len_members, 1)) >= grid_import_variables
                    ]
                if self._current_injection_peak_cost > 0:
                    constraints += [
                        cp.reshape(current_injection_peak_to_bill, (self._len_members, 1)) >= grid_export_variables
                    ]
                metering_period_objective_expr += (
                    cp.sum(cp.multiply(buying_prices_parameter, grid_import_variables)) - cp.sum(cp.multiply(selling_prices_parameter, grid_export_variables))
                )
                metering_period_objective_expr += (
                    cp.sum(rec_import_variables) * self._rec_import_fees + cp.sum(rec_export_variables) * self._rec_export_fees
                )
            else:
                consumption_meter_parameter:cp.Parameter = self._prob_data["consumption_meter_parameter"]
                production_meter_parameter:cp.Parameter = self._prob_data["production_meter_parameter"]
                consumption_meter_parameter[:, -1].value = np.asarray([consumption_meter_states[member][-1] for member in self._members])
                production_meter_parameter[:, -1].value = np.asarray([production_meter_states[member][-1] for member in self._members])
            objective_expr = metering_period_objective_expr + peak_period_objective_expr
            prob = cp.Problem(cp.Minimize(objective_expr), constraints)
            self._prob_data["constraints"] = constraints
            self._prob_data["n_meters"] = len(consumption_meter_states[self._members[0]])
            self._prob_data["consumption_meter_parameter"] = consumption_meter_parameter
            self._prob_data["production_meter_parameter"] = production_meter_parameter
            self._prob_data["metering_period_objective_expr"] = metering_period_objective_expr
            self._prob_data["peak_period_objective_expr"] = peak_period_objective_expr
            

            #raise BaseException("Incremental build not implemented yet")
        if current_peak_period_counter == self._Delta_P:
            self._prob_data = None
        self._rec_imports = rec_import_variables
        self._rec_exports = rec_export_variables
        return prob, metering_period_objective_expr, peak_period_objective_expr, current_offtake_peak_to_bill, current_injection_peak_to_bill
    
    def _solve(self, prob, metering_period_objective_expr, peak_period_objective_expr, current_offtake_peak_to_bill, current_injection_peak_to_bill, type_solve="cvxpy"):
        if type_solve == "cvxpy":
            #mosek
            prob.solve("CPLEX", ignore_dpp=not self._dpp_compile, warm_start=True, verbose=False)
            if prob.value is None or abs(float(prob.value)) == float("inf"):
                raise BaseException(f"Global bill could not be computed, problem status: {prob.status}")
            offtake_peaks = None
            injection_peaks = None
            if self._involve_peaks:
                if current_offtake_peak_to_bill is not None:
                    offtake_peaks = {
                        member: current_offtake_peak_to_bill[i].value for i, member in enumerate(self._members)
                    }
                if current_injection_peak_to_bill is not None:
                    injection_peaks = {
                        member: current_injection_peak_to_bill[i].value for i, member in enumerate(self._members)
                    }
            metering_period_cost = float(metering_period_objective_expr.value) if type(metering_period_objective_expr) not in (float, int) else float(metering_period_objective_expr)
            peak_period_cost = (float(peak_period_objective_expr.value) if type(peak_period_objective_expr) not in (float, int) else peak_period_objective_expr) if self._involve_peaks else 0
        elif type_solve == "mosek":
            prob.solve()
            if prob.getPrimalSolutionStatus() != mk.SolutionStatus.Optimal:
                raise BaseException(f"Global bill could not be computed, problem status: {prob.getPrimalSolutionStatus()}")
            """
            metering_period_objective_expr = (
                mk.Expr.sub(mk.ExprDotParam(buying_prices, grid_import_variables), mk.ExprDotParam(selling_prices, grid_export_variables))
            )
            metering_period_objective_expr = mk.Expr.add(metering_period_objective_expr, (
                mk.Expr.add(mk.Expr.mul(mk.Expr.sum(rec_import_variables), self._rec_import_fees), mk.Expr.mul(mk.Expr.sum(rec_export_variables), self._rec_export_fees))
            ))
            """
            metering_period_cost = np.sum(np.multiply(self._buying_prices, self._grid_imports.level())) - np.sum(np.multiply(self._selling_prices, self._grid_exports.level()))
            metering_period_cost += np.sum(self._rec_imports.level()*self._rec_import_fees) + np.sum(self._rec_exports.level()*self._rec_export_fees)
            metering_period_cost = float(metering_period_cost)
            """
                peak_period_objective_expr = 0
                current_peak_cost = 0
                proratized_current_peak_cost = None
                if self._involve_current_peaks:
                    proratized_current_peak_cost = M.variable()
                    if self._current_offtake_peak_cost > 0:
                        current_peak_cost = mk.Expr.add(current_peak_cost, mk.Expr.mul(mk.Expr.sum(current_offtake_peak_to_bill), epsilonify(self._current_offtake_peak_cost, epsilon=1e-8)))
                    if self._current_injection_peak_cost > 0:
                        current_peak_cost = mk.Expr.add(current_peak_cost, mk.Expr.mul(mk.Expr.sum(current_injection_peak_to_bill), epsilonify(self._current_injection_peak_cost, epsilon=1e-8)))
                    peak_period_objective_expr = mk.Expr.mul(proratized_current_peak_cost, 1/(self._Delta_M*self._Delta_C))

                proratized_historical_peak_cost = None
                historical_peak_cost = 0.0
                if self._involve_historical_peaks:
                    proratized_historical_peak_cost = M.variable()
                    if self._historical_offtake_peak_cost > 0:
                        historical_peak_cost += mk.Expr.mul(mk.Expr.sum(historical_offtake_peak_to_bill), epsilonify(self._historical_offtake_peak_cost, epsilon=1e-8))
                    if self._historical_injection_peak_cost > 0:
                        historical_peak_cost += mk.Expr.mul(mk.Expr.sum(historical_injection_peak_to_bill), epsilonify(self._historical_injection_peak_cost, epsilon=1e-8))
                    if type(historical_peak_cost) not in (int, float):
                        peak_period_objective_expr = mk.Expr.add(peak_period_objective_expr, mk.Expr.mul(proratized_historical_peak_cost, 1.0/(self._Delta_M*self._Delta_C)))
            """
            peak_period_cost = 0
            if self._involve_current_peaks and (epsilonify(self._current_offtake_peak_cost, epsilon=1e-8) > 0 or epsilonify(self._current_injection_peak_cost, epsilon=1e-8) > 0):
                current_peak_cost = 0
                if self._current_offtake_peak_cost > 0 and epsilonify(self._current_offtake_peak_cost, epsilon=1e-8) > 0:
                    current_peak_cost += np.sum(current_offtake_peak_to_bill.level() * epsilonify(self._current_offtake_peak_cost, epsilon=1e-8))
                if self._current_injection_peak_cost > 0 and epsilonify(self._current_injection_peak_cost, epsilon=1e-8) > 0:
                    current_peak_cost += np.sum(current_injection_peak_to_bill.level() * epsilonify(self._current_injection_peak_cost, epsilon=1e-8))
                peak_period_cost = float((self._prorata / (self._Delta_prod)) * current_peak_cost)
            if self._involve_historical_peaks:
                historical_peak_cost = 0
                if self._historical_offtake_peak_cost > 0:
                    historical_peak_cost += np.sum(self._historical_offtake_peak_to_bill.level() * epsilonify(self._historical_offtake_peak_cost, epsilon=1e-8))
                if self._historical_injection_peak_cost > 0:
                    historical_peak_cost += np.sum(self._historical_injection_peak_to_bill.level() * epsilonify(self._historical_injection_peak_cost, epsilon=1e-8))
                peak_period_cost += (self._prorata / (self._Delta_prod)) * historical_peak_cost
            offtake_peaks = None
            injection_peaks = None
            if self._involve_peaks and (epsilonify(self._current_offtake_peak_cost, epsilon=1e-8) > 0 or epsilonify(self._current_injection_peak_cost, epsilon=1e-8) > 0):
                if current_offtake_peak_to_bill is not None:
                    offtake_peaks = current_offtake_peak_to_bill.level()
                    offtake_peaks = {
                        member: float(offtake_peaks[i]) for i, member in enumerate(self._members)
                    }
                if current_injection_peak_to_bill is not None:
                    injection_peaks = current_injection_peak_to_bill.level()
                    injection_peaks = {
                        member: float(injection_peaks[i]) for i, member in enumerate(self._members)
                    }
        return metering_period_cost, peak_period_cost, offtake_peaks, injection_peaks

    def optimise_global_bill(self, state, exogenous_prices, detailed_solution=False):
        if self._complete_problem is None and ((self._len_members>2 or not self._involve_current_peaks or epsilonify(self._current_offtake_peak_cost, epsilon=1e-8) > 0 or epsilonify(self._current_injection_peak_cost, epsilon=1e-8) > 0)):
            if self._type_solve == "mosek":
                self._build_complete_problem_mosek()
            else:
                self._build_complete_problem()
        consumption_meter_states = {
            member: state[(member, "consumption_meters")][-(1 if not self._involve_peaks else 0):] for member in self._members
        }
        production_meter_states = {
            member: state[(member, "production_meters")][-(1 if not self._involve_peaks else 0):] for member in self._members
        }
        
        len_T = len(consumption_meter_states[self._members[0]])
        #print(consumption_meter_states["PVB"])
        buying_prices = {
            member: exogenous_prices[(member, "buying_price")][-(1 if not self._involve_peaks else len_T):] for member in self._members
        }
        selling_prices = {
            member: exogenous_prices[(member, "selling_price")][-(1 if not self._involve_peaks else len_T):] for member in self._members
        }
        if self._len_members == 1:
            return self._special_one_member_case(
                consumption_meter_states,
                production_meter_states,
                buying_prices,
                selling_prices,
                tau_m=state["metering_period_counter"],
                tau_p=state.get("peak_period_counter", None)
            )
        elif self._len_members == 2:
            return self._special_two_members_case(
                consumption_meter_states,
                production_meter_states,
                buying_prices,
                selling_prices,
                tau_m=state["metering_period_counter"],
                tau_p=state.get("peak_period_counter", None),
                detailed_solution=detailed_solution
            )
        elif self._force_optim_no_peak_costs or (self._activate_optim_no_peak_costs and (not self._involve_current_peaks or epsilonify(self._current_offtake_peak_cost, epsilon=1e-8) == 0 or epsilonify(self._current_injection_peak_cost, epsilon=1e-8) == 0)):
            return self._special_no_peak_costs_case(
                consumption_meter_states,
                production_meter_states,
                buying_prices,
                selling_prices,
                tau_m=state["metering_period_counter"],
                tau_p=state.get("peak_period_counter", None),
                detailed_solution=detailed_solution
            )
        historical_offtake_peaks=dict()
        historical_injection_peaks=dict()
        if self._involve_historical_peaks:
            if self._historical_offtake_peak_cost > 0:
                historical_offtake_peaks = {
                    member: state[(member, "historical_offtake_peaks")] for member in self._members
                }
            if self._historical_injection_peak_cost > 0:
                historical_injection_peaks = {
                    member: state[(member, "historical_injection_peaks")] for member in self._members
                }
        if self.time_optim:
            print("Build global bill optimiser model")
            t_build = time()
        if self._type_solve == "mosek":
            fill_complete_problem_method = self._incremental_fill_complete_problem_mosek
        else:
            fill_complete_problem_method = self._incremental_fill_complete_problem
            
        prob, metering_period_cost, peak_period_cost, offtake_peaks, injection_peaks = fill_complete_problem_method(
            consumption_meter_states,
            production_meter_states,
            buying_prices,
            selling_prices,
            state["metering_period_counter"],
            state.get("peak_period_counter", None),
            historical_offtake_peaks,
            historical_injection_peaks
        )
        if self.time_optim:
            print("Optimiser model built in ", time() - t_build, "seconds")
        
        if prob is None:
            self._metering_period_cost += metering_period_cost
            return metering_period_cost, None, None, None
        if self.time_optim:
            print("Solve global bill optimiser model")
            t_solve = time()
        metering_period_cost, peak_period_cost, offtake_peaks, injection_peaks = self._solve(prob, metering_period_cost, peak_period_cost, offtake_peaks, injection_peaks, type_solve=self._type_solve)
        if self.time_optim:
            print("Optimiser model solved in ", time() - t_solve, "seconds")
        if state.get("peak_period_counter", None) == self._Delta_P:
            self.reset()
        else:
            self._init = False
        #prob.solve("CPLEX", cplex_params=cplex_params, ignore_dpp=False, warm_start=True, verbose=False)
        
        
        self._metering_period_cost += metering_period_cost
        self._peak_period_cost += peak_period_cost
        
        #print("here", consumption_meter_states["PVB"], production_meter_states["PVB"])
        #rec_imports = self._rec_imports.level().reshape(self._len_members, self._Delta_P)
        #grid_imports = self._grid_imports.level().reshape(self._len_members, self._Delta_P)
        #rec_exports = self._rec_exports.level().reshape(self._len_members, self._Delta_P)
        #grid_exports = self._grid_exports.level().reshape(self._len_members, self._Delta_P)
        #net_consumption_meters = np.round(self._complete_problem_parameters["net_consumption_meter"].getValue().reshape(self._len_members, self._Delta_P), 2)[:, np.sum(rec_imports, axis=0)>0]
        #net_production_meters = np.round(self._complete_problem_parameters["net_production_meter"].getValue().reshape(self._len_members, self._Delta_P), 2)[:, np.sum(rec_imports, axis=0)>0]
        #net_consumption_meters = self._complete_problem_parameters["net_consumption_meter"].getValue().reshape(self._len_members, self._Delta_P)
        #net_production_meters = self._complete_problem_parameters["net_production_meter"].getValue().reshape(self._len_members, self._Delta_P)
        #buying_prices_per_member = self._complete_problem_parameters["buying_prices"].getValue().reshape(self._len_members, self._Delta_P)
        #selling_prices_per_member = self._complete_problem_parameters["selling_prices"].getValue().reshape(self._len_members, self._Delta_P)
        #print(net_consumption_meters)
        #print(rec_exports[:, np.sum(rec_imports, axis=0)>0])
        
        #print(np.asarray(list(buying_prices.values()))[:, 0].T)
        #print(np.asarray(list(selling_prices.values()))[:, 0].T)
        #print(net_consumption_meters)
        #print(net_consumption_meters - np.sum(net_production_meters, axis=0))
        #print("****************************************************")
        #print(rec_imports[:, np.sum(rec_imports, axis=0)>0])
        #net_consumption_diff = np.maximum(net_consumption_meters - np.sum(net_production_meters, axis=0), 0.0)
        #print(np.max(net_consumption_diff), np.argwhere(net_consumption_diff >= np.max(net_consumption_diff)))
        #print(
        #    np.maximum(np.round(net_consumption_meters.reshape(self._len_members, self._Delta_P), 2)[:, np.sum(rec_imports, axis=0)>0]
        #    - np.round(np.sum(net_production_meters.reshape(self._len_members, self._Delta_P), axis=0), 2)[:, np.sum(rec_imports, axis=0)>0], 0.0)
        #)
        #print("Rec export", self._rec_exports.level())
        #print()
        #print()
        #print("#########################")
        #print()

        """
        mini_global_bill, argmin_metering_period_cost, argmin_peak_period_cost, argmin_infos = (
            self.greedy_optimisation_try(
                net_consumption_meters,
                net_production_meters,
                buying_prices_per_member,
                selling_prices_per_member,
                tau_m=state["metering_period_counter"],
                tau_p=state.get("peak_period_counter", None)
            )
        )
        net_consumption_meters_copy = np.copy(net_consumption_meters) 
        net_consumption_meters_copy[net_consumption_meters_copy == 0] = -1
        if net_consumption_meters_copy[net_consumption_meters_copy==-1].shape[0] > 0:
            
            print(net_consumption_meters_copy[net_consumption_meters_copy==-1].shape)
        if np.round(mini_global_bill, 2) != np.round(metering_period_cost + peak_period_cost, 2):
            #print(np.round(mini_global_bill, 6) - np.round(metering_period_cost + peak_period_cost, 6))
            print(argmin_metering_period_cost, metering_period_cost, argmin_peak_period_cost, peak_period_cost)
            #print(argmin_infos["net_cons_meter_sorted"].shape)
            #print(argmin_infos["net_cons_meter_sorted"][find_indices(net_consumption_meters, argmin_infos["net_cons_meter_sorted"])].shape)
            #print(np.asarray(list(buying_prices.values()))[:, 0].T)
            #print(argmin_infos["bin_vec_buy"])
            #print()
            #print(grid_imports)
            
            #print(argmin_infos["grid_imports"])
            #print()
            #print(np.round(np.sort(np.asarray(list(offtake_peaks.values()))), 2))
            #print(np.sort(argmin_infos["offtake_peaks"]))
            #print()
            #print(np.round(np.sort(np.asarray(list(injection_peaks.values()))), 2))
            #print(np.sort(argmin_infos["injection_peaks"]))
            #print(np.round(np.asarray(list(offtake_peaks.values())),2))
            #print(argmin_infos["offtake_peaks"])
            
            print(buying_prices_per_member)
            print(net_consumption_meters)
            print(rec_imports)
            print(argmin_infos["rec_imports"])
            print()
            print(rec_exports)
            print(argmin_infos["rec_exports"])
            print(np.round(np.asarray(list(offtake_peaks.values())),2))
            print(np.round(argmin_infos["offtake_peaks"],2))
            print(argmin_infos["bin_vec_buy"])
            
            print("NOT PASS", argmin_infos["bin_vec_buy"], argmin_infos["bin_vec_sell"])
            if sum(argmin_infos["bin_vec_buy"]) == 2 or True:
                #print(buying_prices_per_member)
                
                
                #print(grid_imports)
                #print()
                print(grid_imports)
                print(argmin_infos["grid_imports"])
                print()
                print(rec_imports)
                print(argmin_infos["rec_imports"])
                print()
                print(argmin_infos["phi"])
                print(buying_prices_per_member[:, 0])
                #print(argmin_infos["grid_imports"])
                #print()
                #print(rec_exports)
                #print(argmin_infos["rec_exports"])
                print(np.round(np.asarray(list(offtake_peaks.values())),2))
                print(np.round(argmin_infos["offtake_peaks"],2))
                exit()
        else:
            print("PASS", argmin_infos["bin_vec_buy"], argmin_infos["bin_vec_sell"])
        
        print("****")
        """
        #print(np.round(self._grid_imports.level().reshape(self._len_members, self._Delta_P), 2))
        #print()
        if not detailed_solution:
            return metering_period_cost, peak_period_cost, offtake_peaks, injection_peaks
        else:
            rec_imports = self._rec_imports.level().reshape(self._len_members, self._Delta_P)
            grid_imports = self._grid_imports.level().reshape(self._len_members, self._Delta_P)
            rec_exports = self._rec_exports.level().reshape(self._len_members, self._Delta_P)
            grid_exports = self._grid_exports.level().reshape(self._len_members, self._Delta_P)
            return metering_period_cost, peak_period_cost, offtake_peaks, injection_peaks, rec_imports, rec_exports, grid_imports, grid_exports
        
    def _special_no_peak_costs_case(self, consumption_meters_per_member, production_meters_per_member, buying_price_per_member, selling_price_per_member, tau_m=1, tau_p=1, detailed_solution=False):
        rec_imports = np.zeros((self._len_members, min(tau_p+1, self._Delta_P)))
        rec_exports = np.zeros((self._len_members, min(tau_p+1, self._Delta_P)))
        buying_price_per_member_matrix = np.asarray([buying_price_per_member[member] for member in self._members])
        selling_price_per_member_matrix = np.asarray([selling_price_per_member[member] for member in self._members])

        if self._key_sorting_member_import is None:
            if self._key_sorting_member_import_fct is not None:
                self._key_sorting_member_import = self._key_sorting_member_import_fct(consumption_meters_per_member, production_meters_per_member, buying_price_per_member, selling_price_per_member)
            else:
                self._previous_buying_prices = buying_price_per_member_matrix
                self._key_sorting_member_import = np.argsort(-buying_price_per_member_matrix, axis=0)
            if self._key_sorting_member_import.ndim == 1:
                self._key_sorting_member_import = np.expand_dims(self._key_sorting_member_import, 0)
            
            
        elif self._key_sorting_member_import_fct is None:
            unique_prices = np.unique(buying_price_per_member_matrix, axis=0, return_counts=True)
            unique_prices_flag = unique_prices[1].shape[0] == 1
            old_unique_prices = np.unique(self._previous_buying_prices, axis=0, return_counts=True)
            old_unique_prices_flag = old_unique_prices[1].shape[0] == 1
            if not (unique_prices_flag and old_unique_prices_flag and epsilonify(float(np.sum(unique_prices[0] - old_unique_prices[1])), 1e-6) == 0.0):
                self._previous_buying_prices = buying_price_per_member_matrix
                self._key_sorting_member_import = np.argsort(-buying_price_per_member_matrix, axis=0)
                if self._key_sorting_member_import.ndim == 1:
                    self._key_sorting_member_import = np.expand_dims(self._key_sorting_member_import, 0)
        key_sorting_member_import = self._key_sorting_member_import


        if self._key_sorting_member_export is None:
            if self._key_sorting_member_export_fct is not None:
                self._key_sorting_member_export = self._key_sorting_member_export_fct(consumption_meters_per_member, production_meters_per_member, selling_price_per_member, selling_price_per_member)
            else:
                self._previous_selling_prices = selling_price_per_member_matrix
                self._key_sorting_member_export = np.argsort(selling_price_per_member_matrix, axis=0)
            if self._key_sorting_member_export.ndim == 1:
                self._key_sorting_member_export = np.expand_dims(self._key_sorting_member_export, 0)
            
            
        elif self._key_sorting_member_export_fct is None:
            unique_prices = np.unique(selling_price_per_member_matrix, axis=0, return_counts=True)
            unique_prices_flag = unique_prices[1].shape[0] == 1
            old_unique_prices = np.unique(self._previous_selling_prices, axis=0, return_counts=True)
            old_unique_prices_flag = old_unique_prices[1].shape[0] == 1
            if not (unique_prices_flag and old_unique_prices_flag and epsilonify(float(np.sum(unique_prices[0] - old_unique_prices[1])), 1e-6) == 0.0):
                self._previous_selling_prices = selling_price_per_member_matrix
                self._key_sorting_member_export = np.argsort(selling_price_per_member_matrix, axis=0)
                if self._key_sorting_member_export.ndim == 1:
                    self._key_sorting_member_export = np.expand_dims(self._key_sorting_member_export, 0)
        key_sorting_member_export = self._key_sorting_member_export

        if tau_p > 0 and self._previous_no_peaks_solution is not None:
            start_t = self._previous_no_peaks_solution["tau_p"]
            previous_global_rec_bill_retail = self._previous_no_peaks_solution["global_rec_bill_retail"]
            previous_offtake_peaks = self._previous_no_peaks_solution["offtake_peaks"]
            previous_injection_peaks = self._previous_no_peaks_solution["injection_peaks"]
            if tau_p == self._Delta_P:
                self._previous_no_peaks_solution = None
            else:
                self._previous_no_peaks_solution["tau_p"] = tau_p
        else:
            start_t = 0
            previous_global_rec_bill_retail = 0.0
            previous_offtake_peaks = np.zeros(self._len_members)
            previous_injection_peaks = np.zeros(self._len_members)

        consumption_meters = np.asarray(
            [consumption_meters_per_member[m][start_t:] for m in self._members]
        )
        production_meters = np.asarray(
            [production_meters_per_member[m][start_t:] for m in self._members]
        )
        net_consumption_meters = np.maximum(consumption_meters - production_meters, 0.0)
        net_production_meters = np.maximum(production_meters - consumption_meters, 0.0)
        #buying_price_per_member_matrix = np.take_along_axis(buying_price_per_member_matrix, key_sorting_member_import, axis=0)
        #selling_price_per_member_matrix = np.take_along_axis(selling_price_per_member_matrix, key_sorting_member_export, axis=0)
        #net_consumption_meters = np.take_along_axis(net_consumption_meters, key_sorting_member_import, axis=0)
        #net_production_meters = np.take_along_axis(net_production_meters, key_sorting_member_export, axis=0)


        total_avail_prod_per_market_period = np.sum(net_production_meters, axis=0)
        total_cons_excess_per_market_period = np.sum(net_consumption_meters, axis=0)
        #Computing rec import and export only in cases where total_avail_prod_per_market_period > 0 AND total_cons_excess_per_market_period > 0
        #(Otherwise, for others columns, no rec exchange)
        #In those case where total_avail_prod_per_market_period = total_cons_excess_per_market_period:
        #rec_import = net_consumption_meters and rec_export = net_production_meters
        non_null_prod_and_cons = np.logical_and(total_avail_prod_per_market_period >= 1e-6, total_cons_excess_per_market_period >= 1e-6)
        matching_cons_prod = np.logical_and(
            non_null_prod_and_cons,
            np.abs(total_avail_prod_per_market_period - total_cons_excess_per_market_period)<=1e-6
        )
        cons_over_prod = np.logical_and(
            non_null_prod_and_cons,
            total_cons_excess_per_market_period - total_avail_prod_per_market_period>1e-6
        )
        prod_over_cons = np.logical_and(
            non_null_prod_and_cons,
            total_avail_prod_per_market_period - total_cons_excess_per_market_period>1e-6
        )
        #print(buying_price_per_member_matrix)
        #print(selling_price_per_member_matrix)
        #print("net consumption meters")
        #print(net_consumption_meters)
        #print("net production meters")
        #print(net_production_meters)
        #print(self._rec_export_fees)
        #print(self._rec_import_fees)
        if np.any(matching_cons_prod):
            rec_imports[:, matching_cons_prod] = net_consumption_meters[:, matching_cons_prod]
            rec_exports[:, matching_cons_prod] = net_production_meters[:, matching_cons_prod]

        if np.any(prod_over_cons):
            rec_imports[:, prod_over_cons] = net_consumption_meters[:, prod_over_cons]
            # Compute greedy export when prod is above cons
            greedy_export = np.take_along_axis(net_production_meters[:, prod_over_cons], key_sorting_member_export[:, prod_over_cons], axis=0)
            
            #print()

            
            greedy_export = np.minimum(np.cumsum(greedy_export, axis=0), total_cons_excess_per_market_period[prod_over_cons])
            greedy_export_shifted = np.vstack([np.zeros(greedy_export.shape[1]), greedy_export])
            greedy_export = greedy_export - greedy_export_shifted[:-1, :]

            # Get the indices for unsorting the matrix
            reverse_sorting_export = np.argsort(key_sorting_member_export[:, prod_over_cons], axis=0)

            # Undo the sorting
            greedy_export = np.take_along_axis(greedy_export, reverse_sorting_export, axis=0)

            #print(key_sorting_member_export)
            #print(np.take_along_axis(
            #        key_sorting_member_export[:, prod_over_cons], key_sorting_member_export[:, prod_over_cons], axis=0
            #        ))
            #print(key_sorting_member_export_reverse)
            rec_exports[:, prod_over_cons] = greedy_export
        greedy_import = None
        if np.any(cons_over_prod):
            rec_exports[:, cons_over_prod] = net_production_meters[:, cons_over_prod]
            # Compute greedy import when cons is above prod
            greedy_import = np.take_along_axis(net_consumption_meters[:, cons_over_prod], key_sorting_member_import[:, cons_over_prod], axis=0)
            #print(prod_over_cons)
            #print(cons_over_prod)
            #print(net_consumption_meters[:, cons_over_prod])
            #print(greedy_import)
            #print(total_avail_prod_per_market_period[cons_over_prod])
            #print(buying_price_per_member_matrix)

            greedy_import = np.minimum(np.cumsum(greedy_import, axis=0), total_avail_prod_per_market_period[cons_over_prod])
            greedy_import_shifted = np.vstack([np.zeros(greedy_import.shape[1]), greedy_import])
            greedy_import = greedy_import - greedy_import_shifted[:-1, :]

            # Get the indices for unsorting the matrix
            reverse_sorting_import = np.argsort(key_sorting_member_import[:, cons_over_prod], axis=0)

            # Undo the sorting
            greedy_import = np.take_along_axis(greedy_import, reverse_sorting_import, axis=0)
            #print(key_sorting_member_import)
            #print(np.take_along_axis(
            #        key_sorting_member_import[:, cons_over_prod], key_sorting_member_import[:, cons_over_prod], axis=0
            #        ))
            #print(key_sorting_member_import_reverse)
            rec_imports[:, cons_over_prod] = greedy_import

        #
        #print(selling_price_per_member_matrix)
        #print(cons_over_prod, prod_over_cons)
        #Compute grid import and export
        grid_imports = net_consumption_meters - rec_imports

        grid_exports = net_production_meters - rec_exports
        metering_period_expr = (
            np.sum(np.multiply(rec_imports, self._rec_import_fees)) +
            np.sum(np.multiply(rec_exports, self._rec_export_fees)) +
            np.sum(np.multiply(buying_price_per_member_matrix, grid_imports)) -
            np.sum(np.multiply(selling_price_per_member_matrix, grid_exports))
        ) + previous_global_rec_bill_retail
        
        offtake_peaks = np.maximum(np.max(grid_imports, axis=1), previous_offtake_peaks)
        injection_peaks = np.maximum(np.max(grid_exports, axis=1), previous_injection_peaks)
        prorata = elapsed_timesteps_in_peak_period(0, 0, tau_m, tau_p, Delta_M=self._Delta_M, Delta_P=self._Delta_P)
        peak_period_expr = 0
        if epsilonify(self._current_offtake_peak_cost, epsilon=1e-8) > 0 or epsilonify(self._current_offtake_peak_cost, epsilon=1e-8) > 0:
            peak_period_expr = (
                np.sum(offtake_peaks)*self._current_offtake_peak_cost +
                np.sum(injection_peaks)*self._current_injection_peak_cost
            )*(1.0/(self._Delta_prod)) * prorata
        
        if tau_p > 0 and tau_p < self._Delta_P and tau_p - start_t >= 1:
            metering_period_expr_previous = metering_period_expr - (
                np.sum(np.multiply(rec_imports[:, -1], self._rec_import_fees)) +
                np.sum(np.multiply(rec_exports[:, -1], self._rec_export_fees)) +
                np.multiply(buying_price_per_member_matrix[:, -1], grid_imports[:, -1]) -
                np.multiply(selling_price_per_member_matrix[:, -1], grid_exports[:, -1])
            )
            if self._previous_no_peaks_solution is None:
                self._previous_no_peaks_solution = dict()
            self._previous_no_peaks_solution["tau_p"] = tau_p
            self._previous_no_peaks_solution["offtake_peaks"] = offtake_peaks
            self._previous_no_peaks_solution["injection_peaks"] = injection_peaks
            self._previous_no_peaks_solution["global_rec_bill_retail"] = metering_period_expr_previous
            
        offtake_peaks = {
            member:offtake_peaks[i] for i, member in enumerate(offtake_peaks)
        }
        injection_peaks = {
            member:injection_peaks[i] for i, member in enumerate(injection_peaks)
        }
        if not detailed_solution:
            return metering_period_expr, peak_period_expr, offtake_peaks, injection_peaks
        else:
            return metering_period_expr, peak_period_expr, offtake_peaks, injection_peaks, rec_imports, rec_exports, grid_imports, grid_exports
            

        

    
    def _special_one_member_case(self, consumption_meters, production_meters, buying_prices, selling_prices, tau_m=1, tau_p=1):
        member_1 = self._members[0]
        buying_price_member_1 = buying_prices[member_1]
        selling_price_member_1 = selling_prices[member_1]
        consumption_meters_member_1 = consumption_meters[member_1]
        production_meters_member_1 = production_meters[member_1]
        net_consumption_meters_member_1 = np.maximum(consumption_meters_member_1 - production_meters_member_1, 0.0)
        net_production_meters_member_1 = np.maximum(production_meters_member_1 - consumption_meters_member_1, 0.0)
        metering_period_expr = (
            np.dot(net_consumption_meters_member_1, buying_price_member_1)
            - np.dot(net_production_meters_member_1, selling_price_member_1)
        )
        prorata = elapsed_timesteps_in_peak_period(0, 0, tau_m, tau_p, Delta_M=self._Delta_M, Delta_P=self._Delta_P)
        metering_period_expr = float(metering_period_expr)
        peak_period_expr = 0.0
        offtake_peaks = None
        injection_peaks = None
        if self._involve_peaks:
            offtake_peak = np.max(net_consumption_meters_member_1)
            injection_peak = np.max(net_production_meters_member_1)
            peak_period_expr = float((
                offtake_peak*self._current_offtake_peak_cost
                + injection_peak*self._current_injection_peak_cost
            ) * (1.0/(self._Delta_prod)) * prorata)
            offtake_peaks = {member_1: offtake_peak}
            injection_peaks = {member_1: injection_peak}
        return metering_period_expr, peak_period_expr, offtake_peaks, injection_peaks
    
    def _special_two_members_case(self, consumption_meters, production_meters, buying_prices, selling_prices, tau_m=1, tau_p=1, detailed_solution=False):
        member_1 = self._members[0]
        member_2 = self._members[1]
        buying_price_member_1 = buying_prices[member_1]
        buying_price_member_2 = buying_prices[member_2]
        selling_price_member_1 = selling_prices[member_1]
        selling_price_member_2 = selling_prices[member_2]
        consumption_meters_member_1 = consumption_meters[member_1]
        consumption_meters_member_2 = consumption_meters[member_2]
        production_meters_member_1 = production_meters[member_1]
        production_meters_member_2 = production_meters[member_2]
        net_consumption_meters_member_1 = np.maximum(consumption_meters_member_1 - production_meters_member_1, 0.0)
        net_production_meters_member_1 = np.maximum(production_meters_member_1 - consumption_meters_member_1, 0.0)
        net_consumption_meters_member_2 = np.maximum(consumption_meters_member_2 - production_meters_member_2, 0.0)
        net_production_meters_member_2 = np.maximum(production_meters_member_2 - consumption_meters_member_2, 0.0)



        prorata = elapsed_timesteps_in_peak_period(0, 0, tau_m, tau_p, Delta_M=self._Delta_M, Delta_P=self._Delta_P)
        rec_import_member_1 = np.zeros_like(net_consumption_meters_member_1)
        rec_import_member_2 = np.zeros_like(net_consumption_meters_member_2)
        rec_export_member_1 = np.zeros_like(net_consumption_meters_member_1)
        rec_export_member_2 = np.zeros_like(net_consumption_meters_member_2)
        logical_selects = [
            np.logical_and(net_consumption_meters_member_1>0, net_production_meters_member_2>0),
            np.logical_and(net_production_meters_member_1>0, net_consumption_meters_member_2>0),
            np.logical_and(net_consumption_meters_member_2>0, net_production_meters_member_1>0),
            np.logical_and(net_production_meters_member_2>0, net_consumption_meters_member_1>0)
        ]
        rec_import_member_1[logical_selects[0]] = (
            np.minimum(
                net_consumption_meters_member_1[logical_selects[0]],
                net_production_meters_member_2[logical_selects[0]]
            )
        )
        rec_export_member_1[logical_selects[1]] = (
            np.minimum(
                net_production_meters_member_1[logical_selects[1]],
                net_consumption_meters_member_2[logical_selects[1]]
            )
        )
        rec_import_member_2[logical_selects[2]] = (
            np.minimum(
                net_consumption_meters_member_2[logical_selects[2]],
                net_production_meters_member_1[logical_selects[2]]
            )
        )
        rec_export_member_2[logical_selects[3]] = (
            np.minimum(
                net_production_meters_member_2[logical_selects[3]],
                net_consumption_meters_member_1[logical_selects[3]]
            )
        )
        grid_import_member_1 = net_consumption_meters_member_1 - rec_import_member_1
        grid_import_member_2 = net_consumption_meters_member_2 - rec_import_member_2
        grid_export_member_1 = net_production_meters_member_1 - rec_export_member_1
        grid_export_member_2 = net_production_meters_member_2 - rec_export_member_2
        


        metering_period_expr = (
            np.dot(grid_import_member_1, buying_price_member_1)
            + np.dot(grid_import_member_2, buying_price_member_2)
            - (np.dot(grid_export_member_1, selling_price_member_1)
            + np.dot(grid_export_member_2, selling_price_member_2))
        )
        metering_period_expr += (
            self._rec_import_fees*np.sum(rec_import_member_1 + rec_import_member_2)
            + self._rec_export_fees*np.sum(rec_export_member_1 + rec_export_member_2)
        )
        metering_period_expr = float(metering_period_expr)
        peak_period_expr = 0
        offtake_peaks = None
        injection_peaks = None
        if self._involve_peaks:
            offtake_peak_member_1 = np.max(grid_import_member_1)
            offtake_peak_member_2 = np.max(grid_import_member_2)
            injection_peak_member_1 = np.max(grid_export_member_1)
            injection_peak_member_2 = np.max(grid_export_member_2)
            peak_period_expr = float((
                offtake_peak_member_1*self._current_offtake_peak_cost
                + offtake_peak_member_2*self._current_offtake_peak_cost
                + injection_peak_member_1*self._current_injection_peak_cost
                + injection_peak_member_2*self._current_injection_peak_cost
            ) * (1.0/(self._Delta_prod)) * prorata)
            offtake_peaks = {
                member_1: offtake_peak_member_1,
                member_2: offtake_peak_member_2
            }
            injection_peaks = {
                member_1: injection_peak_member_1,
                member_2: injection_peak_member_2
            }

        if detailed_solution:
            rec_imports = {
                member_1: rec_import_member_1,
                member_2: rec_import_member_2
            }
            rec_exports = {
                member_1: rec_export_member_1,
                member_2: rec_export_member_2
            }
            grid_imports = {
                member_1: grid_import_member_1,
                member_2: grid_import_member_2
            }
            grid_exports = {
                member_1: grid_export_member_1,
                member_2: grid_export_member_2
            }
            return metering_period_expr, peak_period_expr, offtake_peaks, injection_peaks, rec_imports, rec_exports, grid_imports, grid_exports
        else:
            return metering_period_expr, peak_period_expr, offtake_peaks, injection_peaks
