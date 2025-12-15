# gd_grad_scale.py
from __future__ import annotations
from typing import Dict, Any

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def run_gd_grad_scale(
    theta_meas: np.ndarray,
    security: np.ndarray,
    lr: float = 0.4,
    num_iters: int = 10,
) -> Dict[str, Any]:
    """
    Gradient Descent mit nachträglicher Gradient-Skalierung:
      theta <- theta - lr * (grad(L(theta)) * update_weights)

    update_weights werden aus security abgeleitet:
      update_weights = (1/security) / max(1/security)
      -> unsicher (kleine security) bewegt sich mehr.

    Returns:
      dict mit loss_hist (L = r^2) und theta_hist, jeweils Länge num_iters+1.
    """
    theta_meas = jnp.asarray(theta_meas, dtype=jnp.float64)
    security = jnp.asarray(security, dtype=jnp.float64)

    def residual(theta: jnp.ndarray) -> jnp.ndarray:
        U1, U2, R, I = theta
        return (U2 - U1) - R * I

    def loss(theta: jnp.ndarray) -> jnp.ndarray:
        r = residual(theta)
        return r ** 2

    base_weights = 1.0 / security
    update_weights = base_weights / jnp.max(base_weights)

    grad_loss = jax.grad(loss)

    theta = theta_meas
    theta_hist = []
    loss_hist = []

    # inkl. Startwert
    for _ in range(num_iters):
        theta_hist.append(theta)
        loss_hist.append(loss(theta))

        g = grad_loss(theta)
        theta = theta - lr * (g * update_weights)

    # finaler Zustand
    theta_hist.append(theta)
    loss_hist.append(loss(theta))

    return {
        "method": "gd_grad_scale",
        "lr": float(lr),
        "num_iters": int(num_iters),
        "update_weights": np.asarray(update_weights),
        "theta_hist": np.asarray(jnp.stack(theta_hist)),
        "loss_hist": np.asarray(jnp.stack(loss_hist)),
    }
