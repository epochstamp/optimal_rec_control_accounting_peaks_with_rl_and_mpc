

from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Any, List, Union


class ConditionalDistribution(object, metaclass=ABCMeta):
    """Batch of distributions of data."""

    @abstractmethod
    def _sample_single(self, c: List = [], length=1) -> Union[List, Any]:
        """Sample from distributions.

        Returns:
            chainer.Variable
        """
        raise NotImplementedError()

    @abstractmethod
    def _prob_single(self, x, c: List = []) -> float:
        """Compute p(x).

        Returns:
            chainer.Variable
        """
        raise NotImplementedError()

    @abstractmethod
    def _log_prob_single(self, x, c: List = []) -> float:
        """Compute p(x).

        Returns:
            chainer.Variable
        """
        raise NotImplementedError()

    def sample(self, c: List = [], length=1) -> Union[List, Any]:
        """Sample from distributions.

        Returns:
            chainer.Variable
        """
        if length == 1:
            return self._sample_single(c)
        l = []
        for i in range(length):
            x = l.append(self._sample_single(c + l))
            l.append(x)
        return l


    def prob(self, x: List, c: List = []) -> float:
        """Compute p(x).

        Returns:
            chainer.Variable
        """
        l = c + x
        len_l = len(l)
        for i in range(len_l):
            j = len_l - i
        raise NotImplementedError()

    @abstractmethod
    def log_prob(self, x, c: List = []) -> float:
        """Compute log p(x).

        Returns:
            chainer.Variable
        """
        raise NotImplementedError()

    @abstractproperty
    def most_probable(self, c: List = []):
        """Most probable data points.

        Returns:
            chainer.Variable
        """
        raise NotImplementedError()

    def sample_with_log_prob(self, c: List = [], length=1):
        """Do `sample` and `log_prob` at the same time.

        This can be more efficient than calling `sample` and `log_prob`
        separately.

        Returns:
            chainer.Variable: Samples.
            chainer.Variable: Log probability of the samples.
        """
        y = self.sample(c, length=length)
        return y, self.log_prob(y, c=c)
