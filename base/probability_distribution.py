from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Any, List, Union, Tuple

class ProbabilityDistribution(object, metaclass=ABCMeta):

    @abstractmethod
    def support(self) -> Tuple[float, float]:
        """
            Support of the distribution

            Returns:
        """
        raise NotImplementedError()

    @abstractmethod
    def sample(self, length=1) -> Union[float, int, List, Any]:
        """Sample from distributions.

        """
        raise NotImplementedError()

    @abstractmethod
    def prob(self, x: Union[List, int, float]) -> float:
        """Compute p(x).

        """
        raise NotImplementedError()

    @abstractmethod
    def log_prob(self, x: Union[List, int, float]) -> float:
        """Compute log p(x).
        """
        raise NotImplementedError()

    @abstractproperty
    def most_probable(self) -> Union[List, int, float]:
        """Most probable data points.
        """
        raise NotImplementedError()

    def sample_with_log_prob(self, length=1) -> Tuple[Union[List, int, float], float]:
        """Do `sample` and `log_prob` at the same time.

        This can be more efficient than calling `sample` and `log_prob`
        separately.
        """
        y = self.sample(length=length)
        return y, self.log_prob(y)
