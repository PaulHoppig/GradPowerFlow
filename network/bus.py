from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BusType(str, Enum):
    PQ = "PQ"       # P and Q specified
    PV = "PV"       # P and |V| specified
    SLACK = "SLACK" # |V| and angle specified (reference)


@dataclass(frozen=True)
class Bus:
    """
    Represents one network node (positive sequence, balanced operation).

    Sign convention (typical for power flow):
    - P_spec > 0 means net injection (generation > load)
    - P_spec < 0 means net consumption (load > generation)
    Same for Q_spec.
    """
    bus_id: int
    name: str = ""
    bus_type: BusType = BusType.PQ

    # Specifications (net injection). Used depending on bus_type.
    P_spec: float = 0.0
    Q_spec: float = 0.0

    # Voltage magnitude specification (used for PV and SLACK)
    V_spec: float = 1.0

    # Voltage angle specification (used for SLACK)
    theta_spec: float = 0.0
