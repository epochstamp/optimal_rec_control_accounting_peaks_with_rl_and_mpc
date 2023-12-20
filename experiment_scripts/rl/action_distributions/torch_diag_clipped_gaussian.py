import functools
import gymnasium as gym
from math import log
import numpy as np
import tree  # pip install dm_tree
from typing import Optional

from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override, DeveloperAPI, ExperimentalAPI
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.numpy import SMALL_NUMBER, MIN_LOG_NN_OUTPUT, MAX_LOG_NN_OUTPUT
from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space
from ray.rllib.utils.typing import TensorType, List, Union, Tuple, ModelConfigDict
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
import torch
from ray.rllib.models.torch.torch_distributions import TorchDiagGaussian



@DeveloperAPI
class TorchDiagClippedGaussian(TorchDiagGaussian):
    """Wrapper class for PyTorch Normal distribution."""

    def __init__(
        self,
        loc: Union[float, torch.Tensor],
        scale: Optional[Union[float, torch.Tensor]],
        low: float=-1.0,
        high: float =1.0
    ):
        super().__init__(loc=loc, scale=scale)
        self._low = low
        self._high = high

    def logp(self, value: TensorType) -> TensorType:
        log_probas = super(TorchDiagGaussian, self).logp(value)
        self._low = torch.ones_like(log_probas)*self._low
        self._high = torch.ones_like(log_probas)*self._high
        if torch.any(value >= self._high):
            log_probas[value >= self._high] = torch.log(1-self._dist.cdf(self._high)[value >= self._high])
        if torch.any(value <= self._low):
            log_probas[value <= self._low] = torch.log(self._dist.cdf(self._low)[value <= self._low])
        return log_probas.sum(-1)

    @classmethod
    def from_logits(cls, logits: TensorType, **kwargs) -> "TorchDiagGaussian":
        loc, log_std = logits.chunk(2, dim=-1)
        scale = log_std.exp() + 1e-6
        return TorchDiagClippedGaussian(loc=loc, scale=scale)

    def sample(
        self,
        *,
        sample_shape=torch.Size(),
    ) -> Union[TensorType, Tuple[TensorType, TensorType]]:
        sample = super().sample(sample_shape=sample_shape)
        sample = torch.clamp(sample, self._low, self._high)
        return sample

    def rsample(
        self,
        *,
        sample_shape=torch.Size(),
    ) -> Union[TensorType, Tuple[TensorType, TensorType]]:
        sample = super().rsample(sample_shape=sample_shape)
        sample = torch.clamp(sample, self._low, self._high)
        return sample
    """
        
    @override(TorchDistributionWrapper)
    def sample(self) -> TensorType:
        sample = super().sample()
        sample = torch.clamp(sample, self.low, self.high)
        if self.zero_action_dim:
            return torch.squeeze(sample, dim=-1)
        return sample

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        self.last_sample = torch.clamp(self.dist.mean, self.low, self.high)
        return self.last_sample

    @override(TorchDistributionWrapper)
    def logp(self, actions: TensorType) -> TensorType:
        log_probas = super().logp(actions)
        if torch.any(actions >= self.high):
            log_probas[actions >= self.high] = torch.log(1-self.dist.cdf(self.high)[actions >= self.high])
        if torch.any(actions <= self.low):
            log_probas[actions <= self.low] = torch.log(self.dist.cdf(self.low)[actions <= self.low])
        return log_probas.sum(-1)

    @override(TorchDistributionWrapper)
    def entropy(self) -> TensorType:
        return super().entropy().sum(-1)

    @override(TorchDistributionWrapper)
    def kl(self, other: ActionDistribution) -> TensorType:
        return super().kl(other).sum(-1)

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(
        action_space: gym.Space, model_config: ModelConfigDict
    ) -> Union[int, np.ndarray]:
        return np.prod(action_space.shape, dtype=np.int32) * 2µ
    """