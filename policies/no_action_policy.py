from base import IneqType
from typing import Dict, Any, Callable, Tuple, List, Optional
from gym.spaces import Dict as DictSpace
from policies.simple_policy import SimplePolicy
from utils.utils import merge_dicts

class NoActionPolicy(SimplePolicy):

    def __init__(self,
                 members: List[str],
                 controllable_assets_state_space: DictSpace,
                 controllable_assets_action_space: DictSpace,
                 constraints_controllable_assets: Dict[str, Callable[[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]], Tuple[Any, float, IneqType]]],
                 consumption_function: Dict[str, Callable[[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float], Optional[Dict[Tuple[str, str], float]]], float]],
                 production_function: Dict[str, Callable[[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float], Optional[Dict[Tuple[str, str], float]]], float]],
                 exogenous_space: DictSpace,
                 Delta_M=1,
                 **kwargs):
        super().__init__(
            members,
            controllable_assets_state_space,
            controllable_assets_action_space,
            constraints_controllable_assets,
            consumption_function,
            production_function,
            exogenous_space,
            Delta_M=Delta_M,
            **kwargs
        )

    def _action(self, state: Dict[str, Any], exogenous_variable_members: Dict[Tuple[str, str], List[float]], exogenous_prices: Dict[Tuple[str, str], List[float]]):
        return {
            k:0.0 for k in self._controllable_assets_action_space.keys()
        }