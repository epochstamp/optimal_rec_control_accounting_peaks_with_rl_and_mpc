from base.repartition_keys_optimiser import RepartitionKeysOptimiser
from typing import Any, List, Tuple, Dict

from utils.utils import merge_dicts

class GreedyPriceRepartition(RepartitionKeysOptimiser):

    def optimise_repartition_keys(self, members: List[str], metering_period_counter_state: int, peak_period_counter_state: int, consumption_meters_states: Dict[str, float], production_meters_states: Dict[str, float], buying_prices: Dict[str, float], selling_prices: Dict[str, float], Delta_C:float=1.0, Delta_M:int=1, Delta_P:int=1, Delta_P_prime:int=0, current_offtake_peak_cost:float=0, current_injection_peak_cost:float=0, historical_offtake_peak_cost:float=0, historical_injection_peak_cost:float=0, peak_states:Dict[Tuple[str, str], float]=None):
        
        total_available_production = sum([production_meters_states[member] for member in members])
        total_demand = sum([consumption_meters_states[member] for member in members])
        repartition_keys = merge_dicts([
            {
                (member, "rec_import"): 0
                for member in members
            },
            {
                (member, "rec_export"): 0
                for member in members
            }
        ])
        members_sorted_for_export_keys = sorted(
            members,
            key=lambda m: float(selling_prices[m])
        )
        members_sorted_for_import_keys = sorted(
            members,
            key=lambda m: -float(buying_prices[m])
        )
        
        phi = 0
        
        if total_demand > 0 and total_available_production > 0:
            for member in members_sorted_for_export_keys:
                if production_meters_states[member] > 0:
                    production_exported = min(total_demand, production_meters_states[member])
                    total_demand -= production_exported
                    phi += production_exported
                    repartition_keys[(member, "rec_export")] = production_exported
                    if total_demand <= 0:
                        break
            for member in members_sorted_for_import_keys:
                if consumption_meters_states[member] > 0:
                    consumption_imported = min(phi, consumption_meters_states[member])
                    repartition_keys[(member, "rec_import")] = consumption_imported
                    phi -= repartition_keys[(member, "rec_import")]
                    if phi <= 0:
                        break
            
            #members_sorted_for_export_keys = sorted(members, key=lambda m: exogenous_variables_prices[(m, "selling_price")][-1])
            
            #members_sorted_for_import_keys = sorted(members, key=lambda m: -exogenous_variables_prices[(m, "buying_price")][-1])
        return repartition_keys