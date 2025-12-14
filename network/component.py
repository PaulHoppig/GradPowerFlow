from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Component(ABC):
    """
    Abstract base class for anything that contributes to the network equations.
    """
    name: str

    @abstractmethod
    def stamp_ybus(self, ybus):
        """
        Return an updated Y-bus matrix with this component stamped in.
        Must be implemented in a JAX-friendly way (functional updates).
        """
        raise NotImplementedError
