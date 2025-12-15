# plot_compare.py
from __future__ import annotations
from typing import Dict, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt

from plot_grad_scale import run_gd_grad_scale
from plot_loss_scale import run_gd_loss_scale
from plot_wls import run_wls_scipy


def make_loss_comparison_plot(
    theta_meas: np.ndarray,
    security: np.ndarray,
    gd_lr: float = 0.4,
    gd_num_iters: int = 12,
    log_y: bool = True,
    figsize: Tuple[float, float] = (8.0, 4.8),
) -> Tuple[plt.Figure, plt.Axes, Dict[str, Any]]:
    """
    Erstellt ein Diagramm Loss vs. Iterationsschritt für:
      - GD Gradient Scaling
      - GD Loss Scaling (Reparametrisierung)
      - SciPy least_squares (WLS via x_scale)

    Gibt (fig, ax, results) zurück; plt.show() wird NICHT automatisch aufgerufen.
    """
    theta_meas = np.asarray(theta_meas, dtype=float)
    security = np.asarray(security, dtype=float)

    res_gd_grad = run_gd_grad_scale(theta_meas=theta_meas, security=security, lr=gd_lr, num_iters=gd_num_iters)
    res_gd_loss = run_gd_loss_scale(theta_meas=theta_meas, security=security, lr=gd_lr, num_iters=gd_num_iters)
    res_wls = run_wls_scipy(theta_meas=theta_meas, security=security)

    loss_grad = res_gd_grad["loss_hist"]
    loss_loss = res_gd_loss["loss_hist"]
    loss_wls = res_wls["loss_hist"]

    x_grad = np.arange(loss_grad.shape[0])
    x_loss = np.arange(loss_loss.shape[0])
    x_wls = np.arange(loss_wls.shape[0])

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x_grad, loss_grad, marker="o", label="GD: Gradient scaling")
    ax.plot(x_loss, loss_loss, marker="o", label="GD: Loss scaling")
    ax.plot(x_wls, loss_wls, marker="o", label="SciPy: least_squares")

    ax.set_title("Vergleich der Optimierungsverfahren: Loss über Iterationen (GD vs. WLS)")
    ax.set_xlabel("Iteration steps")
    ax.set_ylabel("Loss  (r^2)")
    ax.grid(True)

    if log_y:
        ax.set_yscale("log")

    ax.legend()

    results = {
        "gd_grad_scale": res_gd_grad,
        "gd_loss_scale": res_gd_loss,
        "wls_scipy": res_wls,
    }
    return fig, ax, results


# Optional: Beispielwerte (nur ausführen, wenn du dieses Skript direkt startest)
if __name__ == "__main__":
    theta_meas = np.array([230.0, 242.0, 5.0, 2.0], dtype=float)
    security = np.array([1e3, 0.001, 1e3, 1e3], dtype=float)

    fig, ax, results = make_loss_comparison_plot(
        theta_meas=theta_meas,
        security=security,
        gd_lr=0.4,
        gd_num_iters=12,
        log_y=True,
    )
    plt.show()
