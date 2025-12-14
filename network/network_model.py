from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import jax.numpy as jnp

from network.bus import Bus, BusType

from network.edge import Edge


@dataclass(frozen=True)
class NetworkModel:
    """
    Graph-based network model (positive sequence). Builds the power flow residual.

    State x:
      x = [theta_0..theta_{N-1}, V_0..V_{N-1}]
    where:
      theta in radians
      V in p.u. voltage magnitude

    Residual r:
      r has length 2N and is ordered by bus_id:
        r[2*i]   = equation 1 at bus i
        r[2*i+1] = equation 2 at bus i

      For PQ bus:
        rP = P_calc - P_spec
        rQ = Q_calc - Q_spec

      For PV bus:
        rP = P_calc - P_spec
        rV = V_mag - V_spec

      For SLACK bus:
        rV = V_mag - V_spec
        rT = theta - theta_spec
    """
    buses: Sequence[Bus]
    edges: Sequence[Edge]

    def n_buses(self) -> int:
        return len(self.buses)

    def build_ybus(self):
        n = self.n_buses()
        ybus = jnp.zeros((n, n), dtype=jnp.complex128)
        for e in self.edges:
            ybus = e.stamp_ybus(ybus)
        return ybus

    def residual(self, x: jnp.ndarray) -> jnp.ndarray:
        n = self.n_buses()
        if x.shape[0] != 2 * n:
            raise ValueError(f"x must have length 2*N (N={n}); got {x.shape[0]}")

        theta = x[:n]
        vmag = x[n:]

        # Complex bus voltages (positive sequence)
        V = vmag * jnp.exp(1j * theta)

        # Build Ybus and compute currents
        Y = self.build_ybus()
        I = Y @ V

        # Complex power injection at each bus: S = V * conj(I)
        S = V * jnp.conj(I)
        P_calc = jnp.real(S)
        Q_calc = jnp.imag(S)

        # Build residual per bus depending on bus type
        res = []
        # Assumption: bus.bus_id corresponds to its index in x/Ybus
        for b in self.buses:
            i = b.bus_id
            if b.bus_type == BusType.PQ:
                res.append(P_calc[i] - b.P_spec)
                res.append(Q_calc[i] - b.Q_spec)
            elif b.bus_type == BusType.PV:
                res.append(P_calc[i] - b.P_spec)
                res.append(vmag[i] - b.V_spec)
            elif b.bus_type == BusType.SLACK:
                res.append(vmag[i] - b.V_spec)
                res.append(theta[i] - b.theta_spec)
            else:
                raise ValueError(f"Unknown BusType: {b.bus_type}")

        return jnp.asarray(res, dtype=jnp.float64)
