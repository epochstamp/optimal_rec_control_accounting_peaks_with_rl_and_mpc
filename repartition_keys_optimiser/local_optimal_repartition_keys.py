from base.repartition_keys_optimiser import RepartitionKeysOptimiser
from typing import Any, List, Tuple, Dict
from env.peaks_utils import elapsed_timesteps_in_peak_period
from utils.utils import epsilonify, merge_dicts
import cvxpy as cp
import os
import uuid
import pickle
import warnings
import filelock
from time import time

repartition_keys_prob = dict()

class LocalOptimalRepartition(RepartitionKeysOptimiser):

    def __init__(self, id_prob = None, path_models = os.path.expanduser('~') + "/optim_models", lock_path_models = os.path.expanduser('~') + "/optim_models_lock", include_current_peaks=True, include_history_peaks=True, Delta_P=1):
        global repartition_keys_prob
        if id_prob is None:
            id_prob = str(uuid.uuid4())
            while id_prob in repartition_keys_prob:
                id_prob = str(uuid.uuid4())
        self._id_prob = id_prob
        self._path_models = path_models
        self._lock_path_models = lock_path_models
        self._include_current_peak = include_current_peaks
        self._include_history_peaks = include_history_peaks
        self._proratas = dict()


    def _max_offtake_peak_list(self, member, peak_states):
        lst = [0]
        if self._include_current_peak:
            lst += [peak_states[(member, "current_offtake_peak")]]
        if self._include_history_peaks:
            lst += peak_states[(member, "offtake_peaks")]
        return max(lst)

    def _max_injection_peak_list(self, member, peak_states):
        lst = [0]
        if self._include_current_peak:
            lst += [peak_states[(member, "current_injection_peak")]]
        if self._include_history_peaks:
            lst += peak_states[(member, "injection_peaks")]
        return max(lst)
        
    def optimise_repartition_keys(self, members: List[str], counter_states: Dict[str, int], meters_states: Dict[Tuple[str, str], int], exogenous_variables_prices: Dict[Tuple[str, str], float], surrogate: bool = False, Delta_C: float = 1, Delta_M: int = 1, Delta_P: int = 1, offtake_peak_cost: float = 0, injection_peak_cost: float = 0, peak_states: Dict[Tuple[str, str], float] = None):
        #First len_members variables are rec export keys
        #Second len_members variables are grid export keys
        #Third len_members variables are rec import keys
        #Fourth len_members variables are grid import keys
        #Fifth len_members variables are offtake peak cost
        #Sixth len_members variables are injection peak cost
        tau_m = counter_states["nb_timesteps_elapsed_current_metering_period"]
        tau_p = counter_states["peak_period_counter"]
        if (tau_m, tau_p) not in self._proratas:
            self._proratas[(tau_m, tau_p)] = elapsed_timesteps_in_peak_period(
                0, 0, tau_m, tau_p, Delta_M=Delta_M, Delta_P=Delta_P
            )
        prorata = self._proratas[(tau_m, tau_p)]
        d_parameters_values = {
            **meters_states,
            **exogenous_variables_prices
        }
        #Try to find corresponding pickle file
        pathfile_model = self._path_models + f"/{self._id_prob}.dat"
        if os.path.isfile(pathfile_model):
            with open(pathfile_model, "rb") as prob_file:
                try:
                    prob_data = pickle.load(prob_file)
                    repartition_keys_prob[self._id_prob] = prob_data
                except BaseException as e:
                    warnings.warn(f"File {pathfile_model} has issues being opened (details : {str(e)})")
        if self._id_prob not in repartition_keys_prob:
            nb_members = len(members)
            type_variables = ["rec_export", "grid_export", "rec_import", "grid_import", "offtake_peak", "injection_peak"]
            len_type_variables = len(type_variables)
            variables = cp.Variable((nb_members, len_type_variables))
            
            parameters = merge_dicts([
                {
                    (member, parameter_type): cp.Parameter(nonneg=True, value=d_parameters_values[(member, parameter_type)]) for member in members for parameter_type in ["buying_price", "selling_price", "electricity_consumption_metering_period_meter", "electricity_production_metering_period_meter"]
                },
                {
                    (member, "max_offtake_peak"): cp.Parameter(nonneg=True, value=self._max_offtake_peak_list(member, peak_states)) for member in members
                },
                {
                    (member, "max_injection_peak"): cp.Parameter(nonneg=True, value=self._max_injection_peak_list(member, peak_states)) for member in members for member in members
                }
            ])
            prorata_parameter = cp.Parameter(nonneg=True, value=prorata)
            
            
            d_variables = {
                member:dict() for member in members
            }
            for i, member in enumerate(members):
                for j, type_variable in enumerate(type_variables):
                    d_variables[member][type_variable] = variables[i][j]
            objective = cp.Minimize(
                (sum(
                    [d_variables[member]["grid_import"]*parameters[(member, "buying_price")] for member in members]
                ) +
                prorata_parameter*(sum(
                    [d_variables[member]["offtake_peak"]*(offtake_peak_cost) for member in members]
                )
                +
                sum(
                    [d_variables[member]["injection_peak"]*(injection_peak_cost) for member in members]
                )))
                -
                sum(
                    [d_variables[member]["grid_export"]*parameters[(member, "selling_price")] for member in members]
                )
                
            )
            constraints = (
                [
                    var >= 0 for var in variables
                ]+
                [
                    var <= 10000 for var in variables
                ]+
                [
                    sum([d_variables[member]["rec_import"] for member in members]) == sum([d_variables[member]["rec_export"] for member in members]) 
                ]+
                [
                    parameters[(member, "electricity_consumption_metering_period_meter")] == (d_variables[member]["rec_import"] + d_variables[member]["grid_import"]) for member in members
                ]+
                [
                    parameters[(member, "electricity_production_metering_period_meter")] == (d_variables[member]["rec_export"] + d_variables[member]["grid_export"]) for member in members
                ]+
                [
                    (d_variables[member]["offtake_peak"] >= offtake_peak) for member in members for offtake_peak in [parameters[(member, "max_offtake_peak")], d_variables[member]["grid_import"]/(Delta_M*Delta_C)]
                ]+
                [
                    (d_variables[member]["injection_peak"] >= injection_peak) for member in members for injection_peak in [parameters[(member, "max_injection_peak")], d_variables[member]["grid_export"]/(Delta_M*Delta_C)]
                ]
            )
            prob = cp.Problem(objective, constraints)
            repartition_keys_prob[self._id_prob] = (prob, parameters, d_variables, prorata_parameter)
            pathlock = self._lock_path_models + "/" + self._id_prob + ".lock"
            lock = filelock.FileLock(pathlock)
            with lock:
                if not os.path.isfile(pathfile_model):
                    with open(pathfile_model, "wb") as prob_file:
                        pickle.dump(repartition_keys_prob[self._id_prob], prob_file)
        else:
            prob, parameters, d_variables, prorata_parameter = repartition_keys_prob[self._id_prob]
            for parameter_tuple_key, parameter in parameters.items():
                member, parameter_key = parameter_tuple_key
                if parameter_key in ("max_offtake_peak", "max_injection_peak"):
                    parameter.value = (
                        self._max_offtake_peak_list(member, peak_states)
                        if parameter_key == "max_offtake_peak"
                        else self._max_injection_peak_list(member, peak_states)
                    )
                else:
                    parameter.value = (
                        d_parameters_values[(member, parameter_key)]
                    )
            prorata_parameter.value = prorata
        _ = prob.solve(solver="CPLEX")
        if abs(float(prob.value)) == float("inf"):
            raise BaseException(f"Repartition keys could not be computed, problem status: {prob.status}")
        else:
            repartition_keys = convert_import_export_variables_into_repartitions_keys(
                {
                    member:epsilonify(d_variables[member]["grid_export"].value)  for member in members
                },
                {
                    member:epsilonify(d_variables[member]["rec_import"].value)  for member in members
                },
                {
                    member:epsilonify(d_variables[member]["rec_export"].value)  for member in members
                }
            )
            #print(repartition_keys)
            return repartition_keys