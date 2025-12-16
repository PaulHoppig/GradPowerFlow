import jax
import jax.numpy as jnp

# Wahre physikalische Werte
U1_true = 230.0
R_true  = 5.0
I_true  = 2.0
U2_true = U1_true + R_true * I_true  # = 240 V

theta_true = jnp.array([U1_true, U2_true, R_true, I_true])

# Messwerte: U2 ist fehlerhaft (242V statt 240V)
theta_meas = jnp.array([230.0, 242.0, 5.0, 2.0])

# Vertrauen in die Messwerte
security = jnp.array([
    1e3,     # U1 sicher
    0.001,   # U2 unsicher
    1e3,     # R  sicher
    1e3,     # I  sicher
])

# Unsicher -> darf sich mehr bewegen
base_weights = 1.0 / security
update_weights = base_weights / base_weights.max()

print("Wahre Parameter:       ", theta_true)
print("Messwerte (Start):     ", theta_meas)
print("Security (Vertrauen):  ", security)
print("Update-Gewichte:       ", update_weights)
print()

def residual(theta):
    U1, U2, R, I = theta
    return (U2 - U1) - R * I

#  Reparametrisierung: theta = theta_meas + w * phi
def theta_from_phi(phi):
    return theta_meas + update_weights * phi

def loss_phi(phi):
    r = residual(theta_from_phi(phi))
    return r**2

grad_loss_phi = jax.grad(loss_phi)

def gradientDescent():
    # Start: phi=0 => theta = theta_meas
    phi = jnp.zeros_like(theta_meas)

    lr = 0.4
    step = 0

    theta0 = theta_from_phi(phi)
    print("Startzustand:")
    print("  theta        =", theta0)
    print("  residual     =", float(residual(theta0)))
    print("  loss         =", float(loss_phi(phi)))
    print()

    while True:
        g_phi = grad_loss_phi(phi)

        # Gradient Descent
        phi = phi - lr * g_phi
        step += 1

        theta = theta_from_phi(phi)

        print(f"\n\nGradient-Descent Schritt: {step}")
        print("Gradienten dL/dphi:      ", g_phi)
        print("phi:                     ", phi)
        print("Neue Parameter theta:    ", theta)
        print("Residual r(theta):       ", float(residual(theta)))
        print("Loss L(phi):             ", float(loss_phi(phi)))
        print("Abweichung zu true:      ", theta - theta_true)
        print("Abweichung zu Messwerten:", theta - theta_meas)

        user_input = input("\nWeiteren Schritt ausführen? (Enter = Ja / q = Beenden): ")
        if user_input.lower() == "q":
            break

    print("\nOptimierung beendet.")
    print("Wahre Parameter:      ", theta_true)
    print("Messwerte (Start):    ", theta_meas)
    print("Optimierte Parameter: ", theta)
    print("Abweichung zu true:   ", theta - theta_true)
    print("Abweichung zu meas:   ", theta - theta_meas)

gradientDescent()
