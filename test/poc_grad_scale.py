import jax
import jax.numpy as jnp

#Vergleich skalieren Gradienten und Loss gewichten
#Vergleich WLS-Solver
#Schauen wie es nach n Iterationen aussieht --> Graphen programmieren y: Loss, x: Iterationen
jax.config.update("jax_enable_x64", True)

# -------------------------------------------------
# 1. Wahre Werte & Messwerte
# -------------------------------------------------

# Wahre physikalische Werte (konsistent):
U1_true = 230.0
R_true  = 5.0
I_true  = 2.0
U2_true = U1_true + R_true * I_true  # = 240 V

theta_true = jnp.array([U1_true, U2_true, R_true, I_true])

# Messwerte: nur U2 ist "kaputt" (242 statt 240)
theta_meas = jnp.array([230.0, 242.0, 5.0, 2.0])  # [U1, U2, R, I]

#Vertrsuen in die Messwerte
security = jnp.array([
    1e6,   # U1
    0.000001,   # U2 (unsicher)
    1e6,   # R
    1e6,   # I
])

# Unsichere Parameter sollen größere Schritte machen:
base_weights = 1.0 / security           # unsicher -> großer Wert
update_weights = base_weights / base_weights.max()  # normieren auf [0, 1]

print("Wahre Parameter:       ", theta_true)
print("Messwerte (Start):    ", theta_meas)
print("Security (Vertrauen): ", security)
print("Update-Gewichte:      ", update_weights)
print()

def residual(theta):
    """
    theta = [U1, U2, R, I]
    r = (U2 - U1) - R * I           #Abweiung * Weights, dann Residual mit in Loss nehmen
    """
    U1, U2, R, I = theta
    return (U2 - U1) - R * I

def loss(theta):          #Änderung --> Gewichte normieren, statt Gradienten Skalieren
    #r = weights*residual(theta)
    r = residual(theta)

    return r**2


grad_loss = jax.grad(loss)

def main():
    theta = theta_meas.copy()
    lr = 0.05
    step = 0

    print("Startzustand:")
    print("  theta        =", theta)
    print("  residual     =", float(residual(theta)))
    print("  loss         =", float(loss(theta)))
    print()

    while True:
        # rohe Gradienten
        g = grad_loss(theta)

        # skaliere Gradienten mit update_weights:
        # U2 bekommt den größten Schritt
        scaled_g = g * update_weights

        # Gradient-Descent-Update
        theta = theta - lr * scaled_g
        step += 1

        print("\n----------------------------------------")
        print(f"Gradient-Descent Schritt {step}")
        print("----------------------------------------")
        print("Gradienten (roh):        ", g)
        print("Gradienten (skaliert):   ", scaled_g)
        print("Neue Parameter theta:    ", theta)
        print("Residual r(theta):       ", float(residual(theta)))
        print("Loss L(theta):           ", float(loss(theta)))
        print("Abweichung zu true:      ", theta - theta_true)
        print("Abweichung zu Messwerten:", theta - theta_meas)

        user_input = input("\nWeiteren Schritt ausführen? (Enter = Ja / q = Beenden): ")
        if user_input.lower() == "q":
            break

    print("\n================================================")
    print("Optimierung beendet.")
    print("Wahre Parameter:      ", theta_true)
    print("Messwerte (Start):    ", theta_meas)
    print("Optimierte Parameter: ", theta)
    print("Abweichung zu true:   ", theta - theta_true)
    print("Abweichung zu meas:   ", theta - theta_meas)
    print("================================================")


if __name__ == "__main__":
    main()
