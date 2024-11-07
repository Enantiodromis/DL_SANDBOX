import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize

synthetic_data = pd.read_csv("synthetic_dataset.csv")
x = synthetic_data["x"]
y = synthetic_data["y"]

# Defining a list of weights
length_of_x = len(x)

weights = [0.0] * length_of_x

def guassian_basis_function(x, mu, s):
    gbf = np.exp(-((x - mu) ** 2) / (2 * s ** 2))

    return gbf

def sigmoid_basis_function(x, mu, s):
    sbf = 1 / (1 + np.exp(-(x - mu) / s))

    return sbf

# Define parameters for the basis functions (mu, s)
mu_values = np.linspace(-10, 10, 5)  # example mu values for the Gaussian/Sigmoid centers
s_values = np.linspace(0.5, 2, 5)  # example s values for the spread

# Creating the design matrix
def create_design_matrix(x, basis_function, mu_values, s_values):
    x = np.array(x)[:, np.newaxis, np.newaxis]  # Reshape x to be (n, 1, 1) for broadcasting
    mu_values = np.array(mu_values)[np.newaxis, :, np.newaxis]  # Shape (1, M, 1)
    s_values = np.array(s_values)[np.newaxis, np.newaxis, :]  # Shape (1, 1, K)

    # Apply the basis function to each (x, mu, s) combination
    phi = basis_function(x, mu_values, s_values)

    # Reshape phi to (len(x), len(mu_values) * len(s_values))
    phi = phi.reshape(len(x), -1)
    return phi

# Create the design matrix for Gaussian basis functions
phi_gaussian = create_design_matrix(x, guassian_basis_function, mu_values, s_values)

# Optionally, you could repeat for sigmoid basis function:
phi_sigmoid = create_design_matrix(x, sigmoid_basis_function, mu_values, s_values)

# Closed form solution to W_ml
def optimise_weights(design_matrix, y, regularization=100):
    phi_T = np.transpose(design_matrix)
    # Add regularization to avoid singular matrix issues
    inv_phi_T_phi = np.linalg.inv(phi_T @ design_matrix + regularization * np.identity(design_matrix.shape[1]))
    phi_T_y = phi_T @ y
    w_ml = inv_phi_T_phi @ phi_T_y
    
    return w_ml

weights = optimise_weights(phi_gaussian, y)
y_pred = np.dot(phi_gaussian, weights)

# Sort the x values to ensure the plot is smooth
sorted_indices = np.argsort(x)  # Get sorted indices of x
x_sorted = x[sorted_indices]
y_sorted_pred = y_pred[sorted_indices]  # Sort the predicted y values accordingly

# Plot the data and the fitted line
plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.3, s=1, label="Data")
plt.plot(x_sorted, y_sorted_pred, color="red", label="Line of best fit", linewidth=4)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Fit")
plt.legend()
plt.savefig("basis_function_fit_to_data.png")
plt.show()