# wls_scipy.py
from __future__ import annotations
from typing import Dict, Any, Optional, List
import io
import re
import contextlib

import numpy as np
from scipy.optimize import least_squares


def run_wls_scipy(
    theta_meas: np.ndarray,
    security: np.ndarray,
    method: str = "trf",
    jac: str = "2-point",
    max_nfev: Optional[int] = None,
) -> Dict[str, Any]:
    """
    WLS-ähnlich via SciPy least_squares, wobei *dieselben* security-Werte verwendet werden.
    SciPy nutzt x_scale zur internen Skalierung:
      x_scale = 1/sqrt(security)

    Für die Loss-Historie parsen wir die von verbose=2 ausgegebene Cost-Spalte.
    SciPy: cost = 0.5 * ||r||^2  -> wir geben loss = ||r||^2 = 2*cost aus.

    Returns:
      dict mit loss_hist (L = r^2) über Iterationen.
    """
    theta_meas = np.asarray(theta_meas, dtype=float).reshape(-1)
    security = np.asarray(security, dtype=float).reshape(-1)

    if theta_meas.shape != (4,) or security.shape != (4,):
        raise ValueError("theta_meas und security müssen shape (4,) haben: [U1, U2, R, I].")

    def residual(x: np.ndarray) -> np.ndarray:
        u1, u2, r, i = x
        return np.array([u2 - u1 - r * i], dtype=float)

    x_scale = 1.0 / np.sqrt(security)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = least_squares(
            residual,
            theta_meas,
            jac=jac,
            method=method,
            x_scale=x_scale,
            max_nfev=max_nfev,
            verbose=2,  # notwendig für Parsing
        )
    out = buf.getvalue()

    # Parse "Cost" column from verbose=2 table lines: iter, nfev, cost, ...
    # Robust float: supports scientific + decimal
    pattern = re.compile(
        r"^\s*(\d+)\s+(\d+)\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)\s+",
        re.IGNORECASE,
    )

    costs: List[float] = []
    for line in out.splitlines():
        m = pattern.match(line)
        if m:
            costs.append(float(m.group(3)))

    if not costs:
        raise RuntimeError(
            "Konnte keine Iterationskosten aus SciPy verbose=2 Output parsen. "
            "Bitte SciPy-Version/Outputformat prüfen."
        )

    cost_hist = np.array(costs, dtype=float)
    loss_hist = 2.0 * cost_hist  # L = ||r||^2

    return {
        "method": "scipy_least_squares_wls_xscale",
        "x_scale": x_scale,
        "jac": jac,
        "method_scipy": method,
        "max_nfev": max_nfev,
        "result": result,
        "loss_hist": loss_hist,
        "cost_hist": cost_hist,
        "scipy_verbose_output": out,
    }
