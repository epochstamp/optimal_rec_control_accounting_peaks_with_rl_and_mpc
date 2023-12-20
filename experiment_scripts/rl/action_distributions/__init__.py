from ray.rllib.models import ModelCatalog

from experiment_scripts.rl.action_distributions.torch_beta_shifted_squash import TorchBetaShiftedSquash
from .torch_squashed_softplus_diag_gaussian import TorchSquashedSoftplusedDiagGaussian
from ray.rllib.models.torch.torch_action_dist import TorchDiagGaussian, TorchBeta
from .torch_beta_custom import TorchBetaCustom
from ray.rllib.models.torch.torch_distributions import TorchDiagGaussian as NewTorchDiagGaussian
from ..action_distributions.torch_diag_clipped_gaussian import TorchDiagClippedGaussian

action_distribution_zoo = {
    "default": None,
    "new_torch_diag_gaussian": NewTorchDiagGaussian,
    "torch_squashed_softplus_diag_gaussian": TorchSquashedSoftplusedDiagGaussian,
    "torch_diag_gaussian": TorchDiagGaussian,
    "torch_beta": TorchBeta,
    "torch_beta_custom": TorchBetaCustom,
    "torch_beta_shifted_squash": TorchBetaShiftedSquash,
    "torch_diag_clipped_gaussian": TorchDiagClippedGaussian
}
for action_dist_name, action_dist_class in action_distribution_zoo.items():
    if action_dist_class is not None:
        ModelCatalog.register_custom_action_dist(action_dist_name, action_dist_class)