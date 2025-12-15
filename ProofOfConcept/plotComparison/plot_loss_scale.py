# gd_loss_scale.py
from __future__ import annotations
from typing import Dict, Any, Optional

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def run_gd_loss_scale(
    theta_meas: np.ndarray,
    security: np.ndarray,
    lr: float = 0.4,
    num_iters: int = 10,
    phi_init: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Loss-internes Skalieren über Reparametrisierung (ohne Zusatzterm/MAP):
      theta(phi) = theta_meas + update_weights * phi
      L(phi) = r(theta(phi))^2

    Damit sind die Gewichte multiplikativ an die Parametervariation gebunden.

    Returns:
      dict mit loss_hist (L = r^2) und theta_hist, jeweils Länge num_iters+1.
    """
    theta_meas = jnp.asarray(theta_meas, dtype=jnp.float64)
    security = jnp.asarray(security, dtype=jnp.float64)

    def residual(theta: jnp.ndarray) -> jnp.ndarray:
        U1, U2, R, I = theta
        return (U2 - U1) - R * I

    base_weights = 1.0 / security
    update_weights = base_weights / jnp.max(base_weights)

    def theta_from_phi(phi: jnp.ndarray) -> jnp.ndarray:
        return theta_meas + update_weights * phi

    def loss_phi(phi: jnp.ndarray) -> jnp.ndarray:
        r = residual(theta_from_phi(phi))
        return r ** 2

    grad_loss_phi = jax.grad(loss_phi)

    phi = jnp.zeros_like(theta_meas) if phi_init is None else jnp.asarray(phi_init, dtype=jnp.float64)

    theta_hist = []
    loss_hist = []

    for _ in range(num_iters):
        theta = theta_from_phi(phi)
        theta_hist.append(theta)
        loss_hist.append(loss_phi(phi))

        g_phi = grad_loss_phi(phi)
        phi = phi - lr * g_phi

    # finaler Zustand
    theta = theta_from_phi(phi)
    theta_hist.append(theta)
    loss_hist.append(loss_phi(phi))

    return {
        "method": "gd_loss_scale_reparam",
        "lr": float(lr),
        "num_iters": int(num_iters),
        "update_weights": np.asarray(update_weights),
        "theta_hist": np.asarray(jnp.stack(theta_hist)),
        "loss_hist": np.asarray(jnp.stack(loss_hist)),
        "phi_final": np.asarray(phi),
    }
