from __future__ import annotations

from dataclasses import dataclass

from typing import List, Sequence

import jax.numpy as jnp

from network.bus import Bus, BusType

from network.edge import Edge


#@dataclass(frozen=True)
#class NetworkModel:
