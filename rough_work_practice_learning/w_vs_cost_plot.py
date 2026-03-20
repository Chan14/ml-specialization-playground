import matplotlib.pyplot as plt
import numpy as np

# One training example
x = 2
y = 4

# Try a range of weights
w_vals = np.linspace(-5, 5, 100)
print(w_vals)
b = 0
costs = []

# Compute cost for each w
for w in w_vals:
    y_hat = w * x + b
    cost = (y_hat - y) ** 2  # MSE for one data point = squared error
    costs.append(cost)

# Plot
plt.plot(w_vals, costs, label="Cost vs Weight")
plt.xlabel("Weight (w)")
plt.ylabel("Cost (MSE)")
plt.title("Cost Function for One Point (x=2, y=4)")
plt.grid(True)
plt.axvline(x=2, color="red", linestyle="--", label="Ideal w = 2")
plt.legend()
plt.show()
