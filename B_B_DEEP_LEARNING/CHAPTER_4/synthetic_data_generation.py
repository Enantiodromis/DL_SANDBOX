import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

# Generate 10,000 data points for x uniformly between -10 and 10
x = np.random.uniform(-10, 10, 10000)

# Define the function y with a non-linear relationship
noise = np.random.normal(0, 2, x.shape)  # Gaussian noise with mean 0 and std deviation 2
y = 2.5 * x - 0.3 * x**2 + 5 * np.sin(0.5 * x) + noise

data = pd.DataFrame({'x': x, 'y': y})
data.to_csv("synthetic_dataset.csv", index=False)

# Plot and save the dataset
plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.3, s=1)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Synthetic Dataset")
plt.savefig("synthetic_data.png")
plt.show()
