import numpy as np
from scipy.optimize import least_squares

# ---------------------------
# residual function
# ---------------------------
def residual(x):
    u1, u2, r, i = x
    return np.array([u2 - u1 - r * i])

# ---------------------------
# initial values
# ---------------------------
x0 = np.array([
    230.0,  # u1 [V]
    242.0,  # u2 [V]
    5.0,    # r  [Ohm]
    2.0     # i  [A]
])

# ---------------------------
# parameter weights
# lowest confidence on u2
# ---------------------------
weights = np.array([
    1.0,   # u1
    0.001,   # u2 (lowest weight)
    1.0,   # r
    1.0    # i
])

# SciPy applies scaling internally
x_scale = 1.0 / np.sqrt(weights)

# ---------------------------
# optimization
# ---------------------------
result = least_squares(
    residual,
    x0,
    jac="2-point",        # automatic Jacobian
    method="trf",
    x_scale=x_scale,
    verbose=2             # prints iteration steps
)

# ---------------------------
# outputs
# ---------------------------
print("\nFinal solution:")
print(f"u1 = {result.x[0]:.6f}")
print(f"u2 = {result.x[1]:.6f}")
print(f"r  = {result.x[2]:.6f}")
print(f"i  = {result.x[3]:.6f}")

print("\nFinal residual:")
print(result.fun)

print("\nFinal Jacobian (computed by SciPy):")
print(result.jac)
