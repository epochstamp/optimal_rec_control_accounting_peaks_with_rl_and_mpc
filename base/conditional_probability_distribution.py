

from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Any, List, Union, Tuple


class ConditionalProbabilityDistribution(object, metaclass=ABCMeta):
    """Batch of distributions of data."""

    @abstractmethod
    def support(self) -> Tuple[float, float]:
        """
            Support of the distribution

            Returns:
        """
        raise NotImplementedError()

    @abstractmethod
    def sample(self, c: List = [], length=1) -> Union[List, Any]:
        """Sample from distributions.

        """
        raise NotImplementedError()

    @abstractmethod
    def prob(self, x: List, c: List = []) -> float:
        """Compute p(x).

        """
        raise NotImplementedError()

    @abstractmethod
    def log_prob(self, x, c: List = []) -> float:
        """Compute log p(x).
        """
        raise NotImplementedError()

    @abstractproperty
    def most_probable(self, c: List = []):
        """Most probable data points.
        """
        raise NotImplementedError()

    def sample_with_log_prob(self, c: List = [], length=1):
        """Do `sample` and `log_prob` at the same time.

        This can be more efficient than calling `sample` and `log_prob`
        separately.
        """
        y = self.sample(c, length=length)
        return y, self.log_prob(y, c=c)
