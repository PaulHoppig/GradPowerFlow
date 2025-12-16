from __future__ import annotations
from dataclasses import dataclass



@dataclass(frozen=True)
class Edge():

    from_bus: int
    to_bus: int
