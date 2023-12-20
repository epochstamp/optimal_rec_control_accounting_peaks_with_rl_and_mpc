from .model_predictive_control_policy import ModelPredictiveControlPolicy
from .local_max_self_consumption_rate import LocalMaxSelfConsumptionRate
from .global_max_self_consumption_rate import GlobalMaxSelfConsumptionRate
from .no_action_policy import NoActionPolicy

simple_policies = {
    "local_max_self_consumption_rate": LocalMaxSelfConsumptionRate,
    "global_max_self_consumption_rate": GlobalMaxSelfConsumptionRate,
    "no_action": NoActionPolicy
}