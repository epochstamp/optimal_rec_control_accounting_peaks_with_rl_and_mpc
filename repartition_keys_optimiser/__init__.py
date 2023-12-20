from .greedy_price_repartition import GreedyPriceRepartition
from .no_repartition import NoRepartition
from .local_optimal_repartition_keys import LocalOptimalRepartition
#include_current_peaks=True, include_history_peaks=True
repartition_keys_optimisers = {
    "none": None,
    "no_repartition": (NoRepartition, dict()),
    "greedy_price_repartition": (GreedyPriceRepartition, dict()),
    "local_optimal_repartition_keys_full_peak_states": (LocalOptimalRepartition, dict()),
    "local_optimal_repartition_keys_only_current_peak_state": (LocalOptimalRepartition, {"include_history_peaks": False}),
    "local_optimal_repartition_keys_only_history_peak_state": (LocalOptimalRepartition, {"include_current_peaks": False}),
    "local_optimal_repartition_keys_no_peak_state": (LocalOptimalRepartition, {"include_current_peaks": False, "include_history_peaks": False}),
}

def create_repartition_keys_optimiser(id_optimiser: str, id_prob: str= None):
    if id_optimiser == "none":
        return None
    optimiser, optimiser_kwargs = repartition_keys_optimisers[id_optimiser]
    return optimiser(
        id_prob=id_prob,
        **optimiser_kwargs
    )