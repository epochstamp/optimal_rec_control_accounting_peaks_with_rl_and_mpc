from base import Policy, IneqType
from typing import Dict, Any, Callable, Tuple, List, Optional
from gym.spaces import Dict as DictSpace
from policies.max_self_consumption_rate import MaxSelfConsumptionRate
from utils.utils import merge_dicts

class GlobalMaxSelfConsumptionRate(MaxSelfConsumptionRate):

    def __init__(self,
                 members: List[str],
                 controllable_assets_state_space: DictSpace,
                 controllable_assets_action_space: DictSpace,
                 constraints_controllable_assets: Dict[str, Callable[[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]], Tuple[Any, float, IneqType]]],
                 consumption_function: Dict[str, Callable[[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float], Optional[Dict[Tuple[str, str], float]]], float]],
                 production_function: Dict[str, Callable[[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float], Optional[Dict[Tuple[str, str], float]]], float]],
                 exogenous_space: DictSpace,
                 members_with_controllable_assets=[],
                 Delta_M=1):
        super().__init__(
            members,
            controllable_assets_state_space,
            controllable_assets_action_space,
            constraints_controllable_assets,
            consumption_function,
            production_function,
            exogenous_space,
            sum_self_consumption=True,
            members_with_controllable_assets=members_with_controllable_assets,
            Delta_M=Delta_M
        )