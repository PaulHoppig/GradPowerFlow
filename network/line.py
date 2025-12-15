from __future__ import annotations
from dataclasses import dataclass
from network.edge import Edge


@dataclass(frozen=True)
class Line(Edge):

    r_ohm: float
    x_ohm: float
    b_shunt_siemens: float = 0.0

    def stamp_ybus(self, ybus):

        i = self.from_bus
        j = self.to_bus

        z = self.r_ohm + 1j * self.x_ohm
        y_series = 1.0 / z

        # total shunt susceptance b_shunt_siemens -> split half to each end
        y_sh_end = 0.5j * self.b_shunt_siemens

        ybus = ybus.at[i, i].add(y_series + y_sh_end)
        ybus = ybus.at[j, j].add(y_series + y_sh_end)
        ybus = ybus.at[i, j].add(-y_series)
        ybus = ybus.at[j, i].add(-y_series)
        return ybus
