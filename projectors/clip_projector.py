from typing import Any, Dict, List, Tuple, Callable
from base import Projector
from env.rec_env import RecEnv
from policies.model_predictive_control_policy import ModelPredictiveControlPolicy
from utils.utils import epsilonify, merge_dicts, softmax, normalize_0_1
import numpy as np

class Clip_Projector(Projector):

    def __init__(
        self,
        rec_env,
        custom_controllable_assets_action_clipping: Callable[[Dict[str, Any], Dict[Tuple[str, str], List[float]], Dict[Any, float]], Dict[Any, float]]
    ):
        
        super().__init__(
            rec_env
        )
        self._custom_controllable_assets_action_clipping = custom_controllable_assets_action_clipping
        


    def project_action(self, state: Dict[str, Any], exogenous_sequences: Dict[Tuple[str, str], List[float]], action: Dict[Any, float]) -> Dict[Any, float]:
        controllable_actions = self._custom_controllable_assets_action_clipping(state, exogenous_sequences, action)
        return controllable_actions

    def project_type(self):
        return "Projection via custom clipping for controllable action"
