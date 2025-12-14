from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from .edge import Edge


@dataclass(frozen=True)
class Line(Edge):
    """
    Two-terminal transmission line in positive sequence.

    Model: nominal PI (lumped) with:
      - series impedance: Z = r + jx
      - total shunt susceptance: b_shunt (Siemens, imaginary part)
        applied as j*b_shunt/2 at each line end

    Parameters:
      r_ohm, x_ohm: series impedance components
      b_shunt_siemens: total shunt susceptance B (not B/2!)
    """
    r_ohm: float
    x_ohm: float
    b_shunt_siemens: float = 0.0

    def stamp_ybus(self, ybus):
        """
        Stamp this PI line into Ybus.

        For a line between i and k:
          y_series = 1 / (r + jx)
          y_sh_end = j*b/2

        Y_ii += y_series + y_sh_end
        Y_kk += y_series + y_sh_end
        Y_ik -= y_series
        Y_ki -= y_series
        """
        i = self.from_bus
        k = self.to_bus

        z = self.r_ohm + 1j * self.x_ohm
        y_series = 1.0 / z

        # total shunt susceptance b_shunt_siemens -> split half to each end
        y_sh_end = 0.5j * self.b_shunt_siemens

        ybus = ybus.at[i, i].add(y_series + y_sh_end)
        ybus = ybus.at[k, k].add(y_series + y_sh_end)
        ybus = ybus.at[i, k].add(-y_series)
        ybus = ybus.at[k, i].add(-y_series)
        return ybus
