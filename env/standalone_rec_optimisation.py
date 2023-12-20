import cvxpy as cp
import numpy as np
import click
import names
import random
from copy import deepcopy


def rec_decision_problem_optimisation(members, consumption_meters_states, production_meters_states, buying_prices, selling_prices, surrogate=False, current_offtake_peak_cost=0, current_injection_peak_cost=0, historical_offtake_peak_cost=0, historical_injection_peak_cost=0, Delta_C=1.0, Delta_M=1, Delta_P=1, Delta_P_prime=0, historical_offtake_peaks=dict(), historical_injection_peaks=dict(), detailed_output=False, apply_prorata_peaks=True, **kwargs_optimiser):
    """
        Solves the REC decision problem (without control)
        Last time step may not coincide with the end of a peak period or even a metering period

        Time discretisation:
        t in [0,1,...,T-1] where T = len(consumption_meters_states)
        Restriction : Each metering period contains exactly Delta_M time steps
        
        Arguments:
            members : List[str]
                List of members of the REC

            consumption_meters_states : Dict[str, List[float]]
                Mapping from members to consumption meters indexs at each time step t

            production_meters_states : Dict[str, List[float]]
                Mapping from members to production meters indexs at each time step t

            buying_prices : Dict[str, List[float]]
                Mapping from members to buying retail prices at each time step t
                /!\ Provide only retail prices related to each end of metering period 

            selling_prices : Dict[str, List[float]]
                Mapping from members to buying retail prices at each time step t
                /!\ Provide only retail prices related to each end of metering period

            surrogate: bool (default: False)
                Whether the optimisation problem is in surrogate (aka dense) mode

            current_offtake_peak_cost: float (default: 0.0):
                Cost coefficient for offtake peaks. Same for all members

            current_injection_peak_cost: float (default: 0.0):
                Cost coefficient for injection peaks. Same for all members

            historical_offtake_peak_cost: float (default: 0.0):
                Cost coefficient for offtake historical peaks up to Delta_P_prime. Only applies when Delta_P_prime > 0. Same for all members

            historical_injection_peak_cost: float (default: 0.0):
                Cost coefficient for injection historical peaks up to Delta_P_prime. Only applies when Delta_P_prime > 0. Same for all members

            Delta_C: float (default: 1.0):
                Duration between two time steps 
            
            Delta_M: int (default: 1):
                Number of time steps during a metering period

            Delta_P: int (default: 1):
                Number of metering periods during a peak period

            Delta_P_prime: int (default: 0)
                Number of peak periods behind to compute historical peaks

            historical_offtake_peaks: Dict[float] (default : {})
                List of historical offtake peaks per members (useful when considering several optimisation problems when Delta_P_prime > 0)
            
            historical_injection_peaks: Dict[float] (default : {})
                List of historical injection peaks per members (useful when considering several optimisation problems when Delta_P_prime > 0)

            detailed_output: bool (default: False)
                Whether detailed output is returned (see Returns section)

            apply_prorata_peaks: bool (default: True):
                Whether prorata peaks is applied in surrogate mode or in original mode when T is not the end of a peak period
                
            kwargs_optimiser: Dict (default: {})
                Arguments to pass to the solver

        Returns
            objective value if not detailed_output
            (float objective value, cost float sequence of size T, dict of rec import keys per member, dict of rec export keys per member)

        Decision variables : 
            rec_import[(member, t)]: Energy imported from REC for all members, t(imesteps)

            rec_export[(member, t)]: Energy exported to REC for all members, t(imesteps)

            // NB : In original case these decision variables are not even created when t is not the end of a metering period or the last time step

        Aliases:
            grid_import[(member, t)]: Energy imported from main grid for all members, t(imesteps) = consumption_meter_states[(member, t)] - rec_import[(member, t)]
            grid_import_avg_power[(member, t)]: Average power imported from main grid for all members, t(imesteps) = grid_import[(member, t)] / (Delta_M*Delta_C)
            
            grid_export[(member, t)]: Energy exported to main grid for all members, t(imesteps) = production_meter_states[(member, t)] - rec_export[(member, t)]
            grid_export_avg_power[(member, t)]: Energy exported to main grid for all members, t(imesteps) = grid_export[(member, t)] / (Delta_M*Delta_C)

            metering_period_counter_state[t] == t % Delta_M + 1
            (Example with Delta_M=4 and T = 24 : [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4])

            Case (t%Delta_M+1) % Delta_M != 0 | (t+1) // Delta_M % (Delta_P) != 0
                (In natural language : t is not the end of a metering nor of a peak period)
                peak_period_counter_state[t] == ((t+1) // Delta_M) % (Delta_P)
            Otherwise 
                peak_period_counter_state[t] == Delta_P
            (Example with Delta_M=4, Delta_P=3 and T = 24 : [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3])

        Auxiliary variables :
            current_offtake_peak[(member, t)]: This is the offtake peak (i.e., max average consumption power) of the current peak period
                                               at timestep t for all members, all t(imesteps). 
                                               It is computed by taking the maximum between the current average consumption power
                                               (grid_import[(member, t)] / (Delta_M*Delta_C))
                                               and the average consumption power registered at the end of previous metering periods
                                               of the current peak period at time step t

            historical_offtake_peak[(member, t)]: (Only when Delta_P_prime > 0)
                                               This is the maximum between the current offtake peak and the (Delta_P_prime) previous offtake peaks

            current_injection_peak[(member, t)]: Same as current_offtake_peak for injection
                                                (average production power is (grid_export[(member, t)] / (Delta_M*Delta_C)))

            historical_injection_peak[(member, t)]: (Only when Delta_P_prime > 0)
                                                 Same as historical_offtake_peak for injection

            metering_period_costs[t] : Instant cost related to metering period (see Cost Function Definitions section for computation details) at each timestep t
            
            peak_period_costs[t] : Instant cost related to peak period (see Cost Function Definitions section for computation details) at each timestep t
            // NB : In original case the peak variables are not even created when t is not the end of a peak period or the last time step


        Auxiliary functions:
            

            

            rho_m(t) = // Metering period cost (electricity bills after REC production reallocation)
                      Case (surrogate | t is the end of a metering period | t is the last time step):
                          Σ(member in members) grid_import[(member, t)] *buying_prices[(member, t)] - grid_export[(member, t)] * selling_prices[(member, t)]
                      Otherwise
                          0
                          
                      
            rho_p(t) = // Peak period cost (Off-take/injection peaks fees)
                    Case (surrogate | t is the end of a peak period | t is the last time step):
                        (current_offtake_peak_cost * Σ(member in members) current_offtake_peak[(member, t)]
                        + current_injection_peak_cost * Σ(member in members) current_injection_peak[(member, t)]) * current_peak_prorata(t) 
                    Otherwise
                        0
            
            (Only when Delta_P_prime > 0):
            rho_p'(t) = // Historical peak period cost (Historical off-take/injection peaks fees)
            Case (surrogate | t is the end of a peak period | t is the last time step):
                  historical_offtake_peak_cost * Σ(member in members) historical_offtake_peak[(member, t)]
                  + historical_injection_peak_cost * Σ(member in members) historical_injection_peak[(member, t)]
            Otherwise
                  0

            t is the end of metering period -> metering_period_counter_state[t] == Delta_M
            t is the end of peak period -> peak_period_counter_state[t] == Delta_P

            (Only for last time step t when it is not the end of a peak period)
            current_peak_prorata(t) = (number of time steps elapsed in current peak period) / (Delta_M*Delta_P)
            
            number of time steps elapsed in current peak period = peak_period_counter_state[t] * Delta_M + metering_period_counter_state[t]

        Cost Function Definitions

        
            

            Original case:
                metering_period_costs[t] = rho_m(t)
                peak_period_costs[t] = rho_p(t)
            Surrogate case:
                Case (t is first time step of the metering period)
                    metering_period_costs[t] = rho_m(t)
                Otherwise
                    metering_period_costs[t] = rho_m(t) - rho_m(t-1)

                Case (t is first time step of the peak period)
                    peak_period_costs[t] = rho_p(t)
                Otherwise
                    peak_period_costs[t] = rho_p(t) - rho_p(t-1)

        Objective function:
            Σ(t) metering_period_costs[t] + peak_period_costs[t]
                    
        Constraints 
            
            Case (surrogate or t is the end of a metering period or t is the last timestep)
                Σ(member in members) rec_imports[(member, t)] == Σ(member in members) rec_exports[(member, t)]

            
            Case (surrogate or t is the end of a metering period or t is the last timestep)
                rec_imports[(member, t)] <= consumption_meter_states[(member, t)] for all members
                rec_exports[(member, t)] <= production_meter_states[(member, t)] for all members
            Otherwise:
                rec_imports[(member, t)] == 0 //no costs at these timesteps anyway
                rec_exports[(member, t)] == 0


            Case (surrogate or t is the end of a peak period or t is the last timestep)
                current_offtake_peak[(member, t)] >= grid_import_avg_power[(member, t)]
                current_offtake_peak[(member, t)] >= grid_import_avg_power[(member, t')]
                current_injection_peak[(member, t)] >= grid_export_avg_power[(member, t)]
                current_injection_peak[(member, t)] >= grid_export_avg_power[(member, t')]
                where t'<t is the end of metering periods before time step t in the current peak period
            Otherwise:
                current_offtake_peak[(member, t)] == 0 //no costs at these timesteps anyway
                current_injection_peak[(member, t) == 0

            (Only when Delta_P_prime > 0)
            Case (surrogate or t is the end of a metering period or t is the last timestep)
                historical_offtake_peak[(member, t)] >= current_offtake_peak[(member, t)]
                historical_offtake_peak[(member, t)] >= current_injection_peak[(member, t')]
                where t'<t is the end of the (Delta_P_prime) previous peak periods at time step t

                historical_injection_peak[(member, t)] >= grid_export_avg_power[(member, t)]
                historical_injection_peak[(member, t)] >= grid_export_avg_power[(member, t')]
                where t'<t is the end of metering periods before time step t in the current peak period
            Otherwise:
                historical_offtake_peak[(member, t)] == 0 //no costs at these timesteps anyway
                historical_injection_peak[(member, t) == 0
            

        
        
    """
    T = len(consumption_meters_states[members[0]])
    range_T = list(range(T))

    metering_period_counter_state = [
        t % Delta_M + 1 for t in range_T
    ]
    nb_metering_periods = len([
        t for t in metering_period_counter_state if t == Delta_M
    ])
    """
        Case (t%Delta_M+1) % Delta_M != 0 | (t+1) // Delta_M % (Delta_P) != 0
        (In natural language : t is not the end of a metering nor of a peak period)
        peak_period_counter_state[t] == ((t+1) // Delta_M) % (Delta_P)
    """
    peak_period_counter_state = [
        ((t+1) // Delta_M) % (Delta_P) 
        if ((t%Delta_M+1) % Delta_M != 0 
             or (t+1) // Delta_M % (Delta_P) != 0) 
        else Delta_P 
        for t in range_T
    ]
    nb_peak_periods = len([
        t for t in peak_period_counter_state if t == Delta_P
    ])


    #Counters aliases
    def is_end_of_metering_period(t):
        return metering_period_counter_state[t] == Delta_M
    def is_end_of_peak_period(t):
        return peak_period_counter_state[t] == Delta_P
    def is_last_time_step(t):
        return t == T-1

    """
        If we are in original case we can keep only meters that
        are at the end of a metering period and the last meter
        if it does not belong to the end of a metering/peak period 
    """
    if not surrogate:
        old_T = T
        old_metering_period_counter_state = list(metering_period_counter_state)
        old_peak_period_counter_state = list(peak_period_counter_state)
        consumption_meters_states = {
            member:[consumption_meters_states[member][t] for t in range_T if is_end_of_metering_period(t) or is_last_time_step(t)] for member in members
        }
        production_meters_states = {
            member:[production_meters_states[member][t] for t in range_T if is_end_of_metering_period(t) or is_last_time_step(t)] for member in members
        }
        metering_period_counter_state = [
            (Delta_M if is_end_of_metering_period(t) else metering_period_counter_state[t]) for t in range_T if is_end_of_metering_period(t) or is_last_time_step(t)
        ]
        peak_period_counter_state = [
            (Delta_P if is_end_of_peak_period(t) else peak_period_counter_state[t]) for t in range_T if is_end_of_peak_period(t) or is_last_time_step(t)
        ]
        T = len(metering_period_counter_state)
        range_T = range(T)
        def is_last_time_step(t):
            return t == T-1
    else:
        nb_metering_periods_with_last_time_step = nb_metering_periods + (1 if T%Delta_M > 0 else 0)
        buying_prices = dict(buying_prices)
        selling_prices = dict(selling_prices)
        for member in members:
            if len(buying_prices[member]) != nb_metering_periods_with_last_time_step:
                raise BaseException(f"Buying price list length should be {nb_metering_periods_with_last_time_step}(nb of metering periods), current length for member {member}:{len(buying_prices[member])}")
            
            if len(selling_prices[member]) != nb_metering_periods_with_last_time_step:
                raise BaseException(f"Selling price list length should be {nb_metering_periods_with_last_time_step}(nb of metering periods), current length for member {member}:{len(selling_prices[member])}")
            buying_prices[member] = list(np.repeat(buying_prices[member], Delta_M))
            selling_prices[member] = list(np.repeat(selling_prices[member], Delta_M))
            
            buying_prices[member] = list(buying_prices[member][:T])
            selling_prices[member] = list(selling_prices[member][:T])

    #Parametrize inputs (useful for vectorization or model reuse)

    consumption_meters_states = {
        member:cp.Parameter(T, value=consumption_meters_states[member]) for member in members
    }
    production_meters_states = {
        member:cp.Parameter(T, value=production_meters_states[member]) for member in members
    }
    buying_prices = {
        member:cp.Parameter(T, value=buying_prices[member]) for member in members
    }
    selling_prices = {
        member:cp.Parameter(T, value=selling_prices[member]) for member in members
    }
    if Delta_P_prime > 0:
        for member in members:
            if (
                historical_offtake_peak_cost > 0 and
                historical_injection_peak_cost > 0 and
                member in historical_offtake_peaks and
                member in historical_injection_peaks and
                len(historical_offtake_peaks[member]) > 0 and
                len(historical_injection_peaks[member]) > 0 and
                len(historical_offtake_peaks[member]) != len(historical_injection_peaks[member])
            ):
                raise BaseException(
                    f"Historical injection and offtake peaks should be of same size if both provided not empty (historical offtake peaks length = {len(historical_offtake_peaks)}, historical injection peaks length = {len(historical_injection_peaks)})"
                )
            if historical_offtake_peak_cost > 0 and member in historical_offtake_peaks and len(historical_offtake_peaks[member]) > 0:
                historical_offtake_peaks[member] = cp.Parameter(Delta_P_prime, value=historical_offtake_peaks[member])
            if historical_injection_peak_cost > 0 and member in historical_injection_peaks and len(historical_injection_peaks[member]) > 0:
                historical_injection_peaks[member] = cp.Parameter(Delta_P_prime, value=historical_injection_peaks[member])
            

    """
        Variables
    """
    variables_metering_period_costs = cp.Variable(T)
    #No need to compute peaks if theses costs are at 0
    if (
        current_offtake_peak_cost > 0 
        or current_injection_peak_cost > 0
        or (
            Delta_P_prime > 0 and
            (
                historical_offtake_peak_cost > 0 or
                historical_injection_peak_cost > 0
            )
        )
    ):
        if surrogate:
            variables_peak_period_costs = cp.Variable(T)
        else:
            variables_peak_period_costs = cp.Variable(nb_peak_periods + (1 if peak_period_counter_state[-1] != Delta_P else 0))
    else:
        variables_peak_period_costs = None
    
    d_variables = {
        member:dict() for member in members
    }
    type_variables_metering_periods = ["rec_import", "rec_export"]
    type_variables_peak_periods = []
    if current_offtake_peak_cost > 0:
        type_variables_peak_periods += ["current_offtake_peak"]
    if current_injection_peak_cost > 0:
        type_variables_peak_periods += ["current_injection_peak"]

    if Delta_P_prime > 0 and historical_offtake_peak_cost > 0:
        type_variables_peak_periods += ["historical_offtake_peak"]
    if Delta_P_prime > 0 and historical_injection_peak_cost > 0:
        type_variables_peak_periods += ["historical_injection_peak"]

    if surrogate:
        type_variables_pairs = (
            [(type_variables_metering_periods + type_variables_peak_periods, T)]
        )
    else:

        type_variables_pairs = [(type_variables_metering_periods, nb_metering_periods + (1 if metering_period_counter_state[-1] != Delta_M else 0))]
        if variables_peak_period_costs is not None:
            type_variables_pairs += [(type_variables_peak_periods, nb_peak_periods + (1 if peak_period_counter_state[-1] != Delta_P else 0))]    
            


    for type_variable_lst, len_variable_lst in type_variables_pairs:
        for member in members:
            d_variables[member]["all"] = cp.Variable((len(type_variable_lst), len_variable_lst))
            for i, type_variable in enumerate(type_variable_lst):
                d_variables[member][type_variable] = d_variables[member]["all"][i]
    

    
    #Variable aliases
    def grid_import_sequence(member):
        consumption_meter = consumption_meters_states[member]
        rec_import = d_variables[member]["rec_import"]
        return consumption_meter - rec_import
    def grid_export_sequence(member):
        production_meter = production_meters_states[member]
        rec_export = d_variables[member]["rec_export"]
        return production_meter - rec_export

    grid_import_sequences = {
        member: grid_import_sequence(member) for member in members
    }
    grid_export_sequences = {
        member: grid_export_sequence(member) for member in members
    }
    if variables_peak_period_costs is not None:
        grid_import_avg_power_sequences = {
            member: grid_import_sequences[member]/(Delta_M*Delta_C) for member in members
        }
        grid_export_avg_power_sequences = {
            member: grid_export_sequences[member]/(Delta_M*Delta_C) for member in members
        }

    if surrogate:
        #Lists of time steps where t is end of metering/peak period
        end_of_metering_period_time_steps = [t for t in range_T if metering_period_counter_state[t] == Delta_M]
        end_of_peak_period_time_steps = [t for t in range_T if peak_period_counter_state[t] == Delta_P]

        #Lists of time steps where t is not end of metering nor peak period
        not_end_of_metering_period_time_steps = [t for t in range_T if metering_period_counter_state[t] != Delta_M]
        not_end_of_peak_period_time_steps = [t for t in range_T if peak_period_counter_state[t] != Delta_P]

        #Lists of time steps where t is start of metering or peak period
        start_of_metering_period_time_steps = [t for t in range_T if t == 0 or metering_period_counter_state[t-1] == Delta_M]
        start_of_peak_period_time_steps = [t for t in range_T if t == 0 or peak_period_counter_state[t-1] == Delta_P]

        #Lists of time steps where t is not start of metering nor peak period
        not_start_of_metering_period_time_steps = [t for t in range_T if t > 0 and metering_period_counter_state[t-1] != Delta_M]
        not_start_of_peak_period_time_steps = [t for t in range_T if t > 0 and peak_period_counter_state[t-1] != Delta_P]

        #Vectorisation purpose, get previous time steps of the two lists above
        left_shifted_not_start_of_metering_period_time_steps = [t-1 for t in not_start_of_metering_period_time_steps]
        left_shifted_not_start_of_peak_period_time_steps = [t-1 for t in not_start_of_peak_period_time_steps]


    # Objective and cost_function

    objective_expr = cp.sum(variables_metering_period_costs)
    if variables_peak_period_costs is not None:
        objective_expr += cp.sum(variables_peak_period_costs)
    objective = cp.Minimize(
        objective_expr
    )
    
    
    def rho_m(t):
        return sum(
            [grid_import_sequences[member][t] * buying_prices[member][t]
            - grid_export_sequences[member][t] * selling_prices[member][t] for member in members] 
        )

    def rho_m_sequence():
        return sum([cp.multiply(grid_import_sequences[member], buying_prices[member])
            - cp.multiply(grid_export_sequences[member], selling_prices[member]) for member in members])
    
    def number_of_time_steps_elapsed_in_current_peak_period(s_tau_m, s_tau_p):
        if surrogate:
            return (
                (s_tau_p * Delta_M) if s_tau_m == Delta_M
                else ((s_tau_p * Delta_M) + s_tau_m)
            )
        else:
            return (
                (s_tau_p * Delta_M) if s_tau_m == Delta_M
                else ((s_tau_p * Delta_M) + s_tau_m)
            )

    def prorata_rho_p_sequence():
        if surrogate:
            return [
                (1 if not apply_prorata_peaks or peak_period_counter_state[t] == Delta_P else
                 (
                    (peak_period_counter_state[t] * Delta_M)/(Delta_M*Delta_P) if metering_period_counter_state[t] == Delta_M
                    else ((peak_period_counter_state[t] * Delta_M) + metering_period_counter_state[t])/(Delta_M*Delta_P)
                 )
                ) for t in range_T
            ]
        else:
            proratas = [
                (1 if not apply_prorata_peaks or old_peak_period_counter_state[t] == Delta_P else
                 (
                    (old_peak_period_counter_state[t] * Delta_M)/(Delta_M*Delta_P) if old_metering_period_counter_state[t] == Delta_M
                    else ((old_peak_period_counter_state[t] * Delta_M) + old_metering_period_counter_state[t])/(Delta_M*Delta_P)
                 )
                ) for t in range(old_T)
            ]
            proratas = [prorata for t, prorata in enumerate(proratas) if old_peak_period_counter_state[t] == Delta_P or t == old_T-1]
            return proratas
    prorata_rho_p_seq = (
        cp.Parameter(len(peak_period_counter_state), value=prorata_rho_p_sequence())
        if variables_peak_period_costs is not None else None
    )
    def rho_p_sequence():
        rho_p_sec =  0
        if current_offtake_peak_cost > 0:
            current_offtake_peak_sequence = sum(
                [cp.multiply(d_variables[member]["current_offtake_peak"], prorata_rho_p_seq) for member in members]
            )

            rho_p_sec += current_offtake_peak_cost * current_offtake_peak_sequence
            
        if current_injection_peak_cost > 0:
            current_injection_peak_sequence = sum(
                [cp.multiply(d_variables[member]["current_injection_peak"], prorata_rho_p_seq) for member in members]
            )
            rho_p_sec += current_injection_peak_cost * current_injection_peak_sequence
        if Delta_P_prime > 0:
            if historical_offtake_peak_cost > 0:
                rho_p_sec += historical_offtake_peak_cost * sum([d_variables[member]["historical_offtake_peak"] for member in members])
            if historical_injection_peak_cost > 0:
                rho_p_sec += historical_injection_peak_cost * sum([d_variables[member]["historical_injection_peak"] for member in members])
        return rho_p_sec
        
    

    constraints = []
    #Constraints 1: Positivity of variables (except costs)
    constraints += [
        d_variables[member]["rec_import"] >= 0 for member in members
    ]
    constraints += [
        d_variables[member]["rec_export"] >= 0 for member in members
    ]

    if variables_peak_period_costs is not None:
        if current_offtake_peak_cost > 0:
            constraints += [
                d_variables[member]["current_offtake_peak"] >= 0 for member in members
            ]
        if current_injection_peak_cost > 0:
            constraints += [
                d_variables[member]["current_injection_peak"] >= 0 for member in members
            ]
        if Delta_P_prime > 0:
            if historical_offtake_peak_cost > 0:
                constraints += [
                    d_variables[member]["historical_offtake_peak"] >= 0 for member in members
                ]
            if historical_injection_peak_cost > 0:
                constraints += [
                    d_variables[member]["historical_injection_peak"] >= 0 for member in members
                ]
        

    #Constraints 2: Definition of costs functions
    rho_m_seq = rho_m_sequence()
    rho_p_seq = rho_p_sequence() if variables_peak_period_costs is not None else None
    if surrogate:
        constraints += (
            [
                variables_metering_period_costs[start_of_metering_period_time_steps] == rho_m_seq[start_of_metering_period_time_steps]
                
            ]+
            (
                [variables_peak_period_costs[start_of_peak_period_time_steps] == rho_p_seq[start_of_peak_period_time_steps]]
                if variables_peak_period_costs is not None else []
            )+
            (
                [
                    variables_metering_period_costs[not_start_of_metering_period_time_steps] == (
                        rho_m_seq[not_start_of_metering_period_time_steps]
                        - rho_m_seq[left_shifted_not_start_of_metering_period_time_steps]
                    )
                ] if not_start_of_metering_period_time_steps != [] else []
            )+
            (
                [
                    variables_peak_period_costs[not_start_of_peak_period_time_steps] == (
                        rho_p_seq[not_start_of_peak_period_time_steps]
                        - rho_p_seq[left_shifted_not_start_of_peak_period_time_steps]
                    )
                ] if variables_peak_period_costs is not None and not_start_of_peak_period_time_steps != [] else []
            )
        )
    else:
        constraints += (
            [
                variables_metering_period_costs == rho_m_seq     
            ] +
            (
                [
                    variables_peak_period_costs == rho_p_seq 
                ] if variables_peak_period_costs is not None else []
            )
        )

    #Constraint 3 : rec import/export must be within consumption/production meters
    constraints += (
        [
            d_variables[member]["rec_import"] <= consumption_meters_states[member] for member in members
        ]
        +
        [
            d_variables[member]["rec_export"] <= production_meters_states[member] for member in members
        ]
    )

    #Constraint 4 : rec import/export balancing
    sum_rec_import_per_timestep = sum(
        [d_variables[member]["rec_import"] for member in members]
    )
    sum_rec_export_per_timestep = sum(
        [d_variables[member]["rec_export"] for member in members]
    )

    constraints += [
        sum_rec_import_per_timestep == sum_rec_export_per_timestep
    ]
    #Constraint 5 : Injection/offtake peaks (the hell to vectorize)
    if variables_peak_period_costs is not None:
        if not surrogate:
            number_elapsed_peak_periods = 0
            number_elapsed_metering_periods = 0
            for tau_p in range(nb_peak_periods + (1 if old_T%(Delta_M*Delta_P) > 0 else 0)):
                constraints += (
                    ([
                        d_variables[member]["current_offtake_peak"][tau_p] 
                        >= grid_import_avg_power_sequences[member][tau_p*Delta_P:(tau_p+1)*Delta_P]
                        for member in members
                    ] if current_offtake_peak_cost > 0 else []) +
                    ([
                        d_variables[member]["current_injection_peak"][tau_p] 
                        >= grid_export_avg_power_sequences[member][tau_p*Delta_P:(tau_p+1)*Delta_P]
                        for member in members
                    ] if current_injection_peak_cost > 0 else [])
                )
                if Delta_P_prime > 0:
                    constraints += (
                        ([
                            d_variables[member]["historical_offtake_peak"][tau_p] 
                            >= d_variables[member]["current_offtake_peak"][max(tau_p-Delta_P_prime, 0):tau_p+1]
                            for member in members
                        ] if current_offtake_peak_cost > 0 else []) +
                        ([
                            d_variables[member]["historical_injection_peak"][tau_p] 
                            >= d_variables[member]["current_injection_peak"][max(tau_p-Delta_P_prime, 0):tau_p+1]
                            for member in members
                        ] if current_injection_peak_cost > 0 else [])
                    )
                    if (number_elapsed_peak_periods - Delta_P_prime < 0):
                        current_historical_peaks_length = Delta_P_prime - tau_p 
                        constraints += (
                            ([
                                d_variables[member]["historical_offtake_peak"][tau_p] 
                                >= historical_offtake_peaks[member][-current_historical_peaks_length:]
                                for member in members
                            ] if member in historical_offtake_peaks and len(historical_offtake_peaks[member]) > 0 and historical_offtake_peak_cost > 0 else []) +
                            ([
                                d_variables[member]["historical_injection_peak"][tau_p] 
                                >= historical_injection_peaks[member][-current_historical_peaks_length:]
                                for member in members
                            ] if member in historical_injection_peaks and len(historical_injection_peaks[member]) > 0 and historical_injection_peak_cost > 0 else [])
                        )
                
                if number_elapsed_metering_periods == Delta_P:
                    number_elapsed_metering_periods = 1
                    number_elapsed_peak_periods += 1
                else:
                    number_elapsed_metering_periods += 1
        else:
            #Douleur
            number_elapsed_peak_periods = 0
            number_elapsed_metering_periods = 0
            for t in range_T:
                constraints += (
                    ([
                        d_variables[member]["current_offtake_peak"][t] 
                        >= (
                            cp.hstack([grid_import_avg_power_sequences[member][end_of_metering_period_time_steps[number_elapsed_peak_periods * Delta_P:number_elapsed_peak_periods * Delta_P + number_elapsed_metering_periods + 1]], grid_import_avg_power_sequences[member][t]])
                            if end_of_metering_period_time_steps[number_elapsed_peak_periods * Delta_P:number_elapsed_peak_periods * Delta_P + number_elapsed_metering_periods + 1] != []
                            else grid_import_avg_power_sequences[member][t]
                        )
                        for member in members
                    ] if current_offtake_peak_cost > 0 else []) +
                    ([
                        d_variables[member]["current_injection_peak"][t] 
                        >= (
                            cp.hstack([grid_export_avg_power_sequences[member][end_of_metering_period_time_steps[number_elapsed_peak_periods * Delta_P:number_elapsed_peak_periods * Delta_P + number_elapsed_metering_periods + 1]], grid_export_avg_power_sequences[member][t]])
                            if end_of_metering_period_time_steps[number_elapsed_peak_periods * Delta_P:number_elapsed_peak_periods * Delta_P + number_elapsed_metering_periods + 1] != []
                            else grid_export_avg_power_sequences[member][t]
                        )
                        for member in members
                    ] if current_injection_peak_cost > 0 else [])
                )
                if Delta_P_prime > 0:
                    constraints += (
                        ([
                            d_variables[member]["historical_offtake_peak"][t] 
                            >= (
                                cp.hstack([d_variables[member]["current_offtake_peak"][end_of_peak_period_time_steps[max(number_elapsed_peak_periods-Delta_P_prime, 0):number_elapsed_peak_periods]],
                                d_variables[member]["current_offtake_peak"][t]])
                                if d_variables[member]["current_offtake_peak"][end_of_peak_period_time_steps[max(number_elapsed_peak_periods-Delta_P_prime, 0):number_elapsed_peak_periods]].size > 0
                                else d_variables[member]["current_offtake_peak"][t]
                            )
                            for member in members
                        ] if historical_offtake_peak_cost > 0 else []) +
                        ([
                            d_variables[member]["historical_injection_peak"][t] 
                            >= (
                                    cp.hstack([d_variables[member]["current_injection_peak"][end_of_peak_period_time_steps[max(number_elapsed_peak_periods-Delta_P_prime, 0):number_elapsed_peak_periods]],
                                    d_variables[member]["current_injection_peak"][t]])
                                    if d_variables[member]["current_injection_peak"][end_of_peak_period_time_steps[max(number_elapsed_peak_periods-Delta_P_prime, 0):number_elapsed_peak_periods]].size > 0
                                    else d_variables[member]["current_injection_peak"][t]
                                )
                            for member in members
                        ] if historical_injection_peak_cost > 0 else [])
                    )
                    if (number_elapsed_peak_periods - Delta_P_prime + 1 < 0):
                        current_historical_peaks_length = Delta_P_prime - number_elapsed_peak_periods + 1
                        constraints += (
                            ([
                                d_variables[member]["historical_offtake_peak"][t] 
                                >= historical_offtake_peaks[member][-current_historical_peaks_length:]
                                for member in members
                            ] if member in historical_offtake_peaks and len(historical_offtake_peaks[member]) > 0 and historical_offtake_peak_cost > 0 else []) +
                            ([
                                d_variables[member]["historical_injection_peak"][t] 
                                >= historical_injection_peaks[member][-current_historical_peaks_length:]
                                for member in members
                            ] if member in historical_injection_peaks and len(historical_injection_peaks[member]) > 0 and historical_injection_peak_cost > 0 else [])
                        )
                
                if peak_period_counter_state[t] == Delta_P:
                    number_elapsed_metering_periods = 0
                    number_elapsed_peak_periods += 1
                elif metering_period_counter_state[t] == Delta_M:
                    number_elapsed_metering_periods += 1
    prob = cp.Problem(objective, constraints)
    prob.solve(solver="CPLEX", cplex_params=kwargs_optimiser)
    if prob.value is None or abs(float(prob.value)) == float("inf"):
        raise BaseException(f"Repartition keys could not be computed for this model, problem status: {prob.status}")
    else:
        from pprint import pprint
        if not detailed_output:
            return prob.value
        else:
            
            if not surrogate:
                cost_sequence = sum(
                    [[0] * (Delta_M-1 if metering_period_counter_state[t] == Delta_M else (metering_period_counter_state[t] - 1)) + [variables_metering_period_costs[t].value] for t in range(T)],
                    start=[]
                )
            else:
                cost_sequence = variables_metering_period_costs.value
            if variables_peak_period_costs is not None:
                if not surrogate:
                    number_of_time_steps_elapsed_in_last_time_step_current_period = number_of_time_steps_elapsed_in_current_peak_period(
                        metering_period_counter_state[-1], peak_period_counter_state[-1]
                    ) - 1
                    cost_sequence_peaks = sum(
                        [[0] * ((Delta_M*Delta_P-1) if peak_period_counter_state[t] == Delta_P else (number_of_time_steps_elapsed_in_last_time_step_current_period)) + [variables_peak_period_costs[t].value] for t in range(len(variables_peak_period_costs.value))],
                        start=[]
                    )
                    cost_sequence = list(np.asarray(cost_sequence) + np.asarray(cost_sequence_peaks))
                else:
                    cost_sequence += variables_peak_period_costs.value
            return (
                prob.value,
                cost_sequence,
                {
                    member:d_variables[member]["rec_import"].value for member in members
                },
                {
                    member:d_variables[member]["rec_export"].value for member in members
                }
            )

def random_increasing_sequence(length, min_value=0, max_value=100):
    if length < 1 or min_value >= max_value:
        raise ValueError("Paramètres invalides.")

    if min_value <= 0:
        min_value = 1e-9

    increments = np.random.uniform(min_value, (max_value - min_value) / length, length - 1)
    random_values = np.concatenate(([min_value], increments))
    sequence = np.maximum.accumulate(np.cumsum(random_values))

    return sequence

def generate_rec_data(
    Delta_M=2, 
    N=1, 
    T=1
):
    members = set([names.get_first_name() for _ in range(N)])
    while len(members) != N:
        members.add(names.get_first_name())
    members = list(members)
    consumption_meters_states = {
        member:sum([
            list(random_increasing_sequence(Delta_M, np.random.uniform(0.2, 1), 2)) for _ in range(T//Delta_M)
        ], start=[]) for member in members
    }
    
    production_meters_states = {
        member:sum([
            list(random_increasing_sequence(Delta_M, np.random.uniform(0.2, 1), 2)) for _ in range(T//Delta_M)
        ], start=[]) for member in members
    }
    buying_prices = {
        member:[np.random.uniform(0.15, 0.3) for _ in range(T//Delta_M)] for member in members
    }
    selling_prices = {
        member:[np.random.uniform(0.15, 0.3) for _ in range(T//Delta_M)] for member in members
    }
    
    nb_remaining_time_steps = 0
    if T % Delta_M > 0:
        while (T - nb_remaining_time_steps) % Delta_M > 0:
            nb_remaining_time_steps += 1
            
    if nb_remaining_time_steps > 0:
        consumption_meters_states = {
            member:consumption_meters_states[member] + list(random_increasing_sequence(nb_remaining_time_steps, np.random.uniform(0.2, 1), 2)) for member in members
        }
        
        production_meters_states = {
            member:production_meters_states[member] + list(random_increasing_sequence(nb_remaining_time_steps, np.random.uniform(0.2, 1), 2)) for member in members for member in members
        }
        buying_prices = {
            member:buying_prices[member] + [np.random.uniform(0.15, 0.3)] for member in members
        }
        selling_prices = {
            member:selling_prices[member] + [np.random.uniform(0.15, 0.3)] for member in members
        }
    return members, consumption_meters_states, production_meters_states, buying_prices, selling_prices

@click.command()
@click.option('--Delta-C', "Delta_C", type=float, default=1.0, help='Duration between two time steps Delta_C.')
@click.option('--Delta-M', "Delta_M", type=int, default=2, help='Nb of timesteps in a metering period Delta_M.')
@click.option('--Delta-P', "Delta_P", type=int, default=1, help='Nb of metering period in a peak period Delta_P.')
@click.option('--Delta-P-prime', "Delta_P_prime", type=int, default=0, help='Number of peaks period for peak billing Delta_P_prime.')
@click.option('--N', "N", type=int, default=1, help='Number of members (names generated randomly).')
@click.option('--T', "T", type=int, default=1, help='Time horizon (leave default if None).')
@click.option('--surrogate/--original', "is_surrogate", is_flag=True, help='Choose between surrogate (aka dense) or original optimisation model')
@click.option('--disable-prorata-peaks/--enable-prorata-peaks', "disable_prorata_peaks", is_flag=True, help='Whether prorata is disabled for peaks')
@click.option('--random-seed', "random_seed", type=int, default=1, help='Set random seed.')
@click.option('--current-offtake-peak-cost', "current_offtake_peak_cost", type=float, default=1.0, help='Offtake peak cost')
@click.option('--current-injection-peak-cost', "current_injection_peak_cost", type=float, default=1.0, help='Injection peak cost')
@click.option('--historical-offtake-peak-cost', "historical_offtake_peak_cost", type=float, default=1.0, help='Historical offtake peak cost (only if Delta_P_prime > 0)')
@click.option('--historical-injection-peak-cost', "historical_injection_peak_cost", type=float, default=1.0, help='Historical injection peak cost (only if Delta_P_prime > 0)')
@click.option('--unit-test-mode', "unit_test_mode", is_flag=True, help='Whether to activate unit tests (all args will be ignored and be randomized except of random seed and number of unit tests)')
@click.option('--U', "n_unit_tests", type=int, default=1, help='Number of random unit tests (only if unit test mode activated).')
def test(Delta_C=1.0, 
         Delta_M=2, 
         Delta_P=1, 
         Delta_P_prime=0, 
         N=1, 
         T=1, 
         is_surrogate=False, 
         disable_prorata_peaks=False,
         random_seed=1, 
         current_offtake_peak_cost=1.0, 
         current_injection_peak_cost=1.0,
         historical_offtake_peak_cost=1.0,
         historical_injection_peak_cost=1.0,
         unit_test_mode=False,
         n_unit_tests=1):
    
    np.random.seed(random_seed)
    random.seed(random_seed)
    if not unit_test_mode:
        members, consumption_meters_states, production_meters_states, buying_prices, selling_prices = (
            generate_rec_data(Delta_M=Delta_M, N=N, T=T)
        )

        global_bill, cost_sequence, rec_import, _ = rec_decision_problem_optimisation(
            members,
            consumption_meters_states,
            production_meters_states,
            buying_prices,
            selling_prices,
            surrogate=is_surrogate,
            current_offtake_peak_cost=current_offtake_peak_cost,
            current_injection_peak_cost=current_injection_peak_cost,
            historical_offtake_peak_cost=historical_offtake_peak_cost,
            historical_injection_peak_cost=historical_injection_peak_cost,
            Delta_C=Delta_C,
            Delta_M=Delta_M,
            Delta_P=Delta_P,
            Delta_P_prime=Delta_P_prime,
            historical_injection_peaks=[],
            historical_offtake_peaks=[],
            detailed_output=True,
            apply_prorata_peaks=not disable_prorata_peaks
        )
        print(global_bill)
    else:
        results_mismatchs = []
        for _ in range(n_unit_tests):
            T = np.random.randint(1, 101)
            Delta_M = np.random.randint(1, max(2, np.floor(np.sqrt(T)) + 1))
            Delta_P = np.random.randint(1, max(2, T//Delta_M))
            Delta_C = float(np.round(np.random.uniform(0.25,1), 2)) 
            Delta_P_prime=np.random.randint(0, Delta_P+1)
            
            N=np.random.randint(1, 18)
            #is_surrogate=False,
            #disable_prorata_peaks=False
            #random_seed=1, 
            current_offtake_peak_cost=float(np.random.random()) 
            current_injection_peak_cost=float(np.random.random())
            historical_offtake_peak_cost=float(np.random.random())
            historical_injection_peak_cost=float(np.random.random())
            print("Test compare global bill match original and surrogate policies with/without peak prorata,")
            print(f"with {N} members, Delta_M={Delta_M} timesteps, Delta_P={Delta_P} metering periods, Delta_P_prime={Delta_P_prime} peak periods...")
            members, consumption_meters_states, production_meters_states, buying_prices, selling_prices = (
                generate_rec_data(Delta_M=Delta_M, N=N, T=T)
            )
            results = {
                (rec_formulation, with_prorata): None
                for with_prorata in ("with_prorata", "without_prorata")
                for rec_formulation in ("original", "surrogate")
            }
            results_keys = results.keys()
            for rec_formulation, with_prorata in results_keys:
                consumption_meters_states = {
                    member:deepcopy(consumption_meters_states[member]) for member in members
                }
                production_meters_states = {
                    member:deepcopy(production_meters_states[member]) for member in members
                }
                buying_prices = {
                    member:deepcopy(buying_prices[member]) for member in members
                }
                selling_prices = {
                    member:deepcopy(selling_prices[member]) for member in members
                }
                global_bill, cost_sequence, rec_import, _ = rec_decision_problem_optimisation(
                    members,
                    dict(consumption_meters_states),
                    dict(production_meters_states),
                    dict(buying_prices),
                    dict(selling_prices),
                    surrogate=rec_formulation == "surrogate",
                    current_offtake_peak_cost=current_offtake_peak_cost,
                    current_injection_peak_cost=current_injection_peak_cost,
                    historical_offtake_peak_cost=historical_offtake_peak_cost,
                    historical_injection_peak_cost=historical_injection_peak_cost,
                    Delta_C=Delta_C,
                    Delta_M=Delta_M,
                    Delta_P=Delta_P,
                    Delta_P_prime=Delta_P_prime,
                    historical_injection_peaks=[],
                    historical_offtake_peaks=[],
                    detailed_output=True,
                    apply_prorata_peaks=with_prorata == "with_prorata"
                )
                
                results[rec_formulation, with_prorata] = global_bill
            for with_prorata in ("with_prorata", "without_prorata"):
                if abs(results[("original", with_prorata)] - results[("surrogate", with_prorata)]) >= 1e-6:
                    results_mismatchs += [{
                        "results": {
                            "original": results[("original", with_prorata)],
                            "surrogate": results[("surrogate", with_prorata)]
                        },
                        "with_prorata": with_prorata,
                        "N": N,
                        "T": T,
                        "Delta_C":Delta_C,
                        "Delta_M":Delta_M,
                        "Delta_P":Delta_P,
                        "Delta_P_prime":Delta_P_prime,
                        "current_offtake_peak_cost":current_offtake_peak_cost,
                        "current_injection_peak_cost":current_injection_peak_cost,
                        "historical_offtake_peak_cost":historical_offtake_peak_cost,
                        "historical_injection_peak_cost":historical_injection_peak_cost
                    }]
                    print(f"Mismatch detected {with_prorata.replace('_', ' ')}, details at end of unit test")
                else:
                    print(f"Test {with_prorata.replace('_', ' ')} OK")
                print()
        if len(results_mismatchs) == 0:
            print("Every test went fine")
        else:
            print("Mismatch detected between original and surrogate policy")
            from pprint import pprint
            pprint(
                results_mismatchs
            )


if __name__=="__main__":
    test()