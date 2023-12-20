from abc import ABCMeta, abstractmethod
from typing import Any, List, Tuple, Dict

class Projector(object, metaclass=ABCMeta):

    def __init__(
        self,
        rec_env
    ):
        self._rec_env = rec_env

    @abstractmethod
    def project_action(self, state: Dict[str, Any], exogenous_sequences: Dict[Tuple[str, str], List[float]], action: Dict[Any, float]) -> Dict[Any, float]:
        raise NotImplementedError()

    @abstractmethod
    def project_type(self):
        raise NotImplementedError()