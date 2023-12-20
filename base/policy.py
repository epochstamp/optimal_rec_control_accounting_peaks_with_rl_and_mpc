from abc import ABCMeta, abstractmethod
from typing import Any, List, Tuple, Dict, Callable, Optional
from base.types import IneqType
from gym.spaces import Dict as DictSpace
from utils.utils import merge_dicts,  epsilonify, roundify
import numpy as np

class Policy(object, metaclass=ABCMeta):

    def __init__(self,
                 members: List[str],
                 controllable_assets_state_space: DictSpace,
                 controllable_assets_action_space: DictSpace,
                 constraints_controllable_assets: Dict[str, Tuple[Callable[[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]], Tuple[Any, float, IneqType]]]],
                 consumption_function: Dict[str, Callable[[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float], Optional[Dict[Tuple[str, str], float]]], float]],
                 production_function: Dict[str, Callable[[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float], Optional[Dict[Tuple[str, str], float]]], float]],
                 members_with_controllable_assets = []
    ):
        self._members = members
        self._controllable_assets_state_space = controllable_assets_state_space
        self._controllable_assets_action_space = controllable_assets_action_space
        self._constraints_controllable_assets = constraints_controllable_assets
        self._consumption_function = consumption_function
        self._production_function = production_function
        self._controllable_assets_state_space_keys = list(controllable_assets_state_space.keys())
        self._controllable_assets_action_space_keys = list(controllable_assets_action_space.keys())
        self._members_with_controllable_assets = members_with_controllable_assets

    def action(self, state: Dict[str, Any], exogenous_variable_members: Dict[Tuple[str, str], List[float]], exogenous_prices: Dict[Tuple[str, str], List[float]]):
        action = self._action(state, exogenous_variable_members, exogenous_prices)
        new_action = {
            action_key:epsilonify(action_value, epsilon=1e-8) for action_key, action_value in action.items()
        }
        return new_action

    @abstractmethod
    def _action(self, state: Dict[str, Any], exogenous_variable_members: Dict[Tuple[str, str], List[float]], exogenous_prices: Dict[Tuple[str, str], List[float]]):
        """
            Returns action given a state
        """
        raise NotImplementedError()
    


    def reset(self):
        pass
        