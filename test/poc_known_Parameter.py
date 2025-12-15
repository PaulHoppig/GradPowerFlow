import jax

#Gegebene Werte:
U1 = 230.0
R  = 5.0
I  = 2.0

# Unsicherer Parameter
U2_init = 242.0   # 2 V zu hoch

# Residuum
def residual(U2):
    return (U2 - U1) - R * I

# Loss quadriert Residuum
def loss(U2):
    r = residual(U2)
    return r**2



#Erster Schritt Gradient Descent mit Lernrate = 0.1
learning_rate = 0.2
U2_current = U2_init

print("Gegebene (sichere) Werte:")
print(f"  U1 = {U1} V")
print(f"  R  = {R} Ohm")
print(f"  I  = {I} A")

print("\nUnsicherer Startwert:")
print(f"  U2_init = {U2_init} V")
print(f"  Residuum r(U2_init) = {residual(U2_init):.6f}")
print(f"  Loss(U2_init) = {loss(U2_init):.6f}")



#Hinzufügen weiterer Gradient Descent-Schritte auf Abfrage
step = 0
while True:

    g = jax.grad(loss)(U2_current) # Gradient der Loss Funktion nach U2
    U2_current = U2_current - learning_rate * g #Optimieren von U2

    step += 1

    print("\n")
    print(f"Gradient-Descent Schritt {step}:")
    print(f"Gradient dLoss/dU2 = {float(g):.6f}")
    print(f"Neuer U2 = {float(U2_current):.6f} V")
    print(f"Residual = {float(residual(U2_current)):.6f}")
    print(f"Loss     = {float(loss(U2_current)):.6f}")

    # Benutzerabfrage
    user_input = input("\nWeitere Schritt ausführen? (Beliebige Taste = Ja / q = Beenden): ")
    if user_input.lower() == "q":
        U2_correct = U1 + R * I  # 240 V
        print("\n\n\nKorrekter Wert für U2 wäre:", U2_correct)
        print(f"Optimierter Wert für U2 nach {step} Schritten: {U2_current:.6f}")
        print(f"Abweichung = {U2_current - U2_correct:.6f}")
        break

