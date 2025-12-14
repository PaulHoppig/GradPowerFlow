from __future__ import annotations

from dataclasses import dataclass

from .component import Component


@dataclass(frozen=True)
class Edge(Component):
    """
    Abstract base class for two-terminal components connecting two buses.
    """
    from_bus: int
    to_bus: int
