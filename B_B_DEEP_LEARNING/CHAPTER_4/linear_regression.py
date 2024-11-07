import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Loading in data
synthetic_data = pd.read_csv("synthetic_dataset.csv")
x = synthetic_data["x"]
y = synthetic_data["y"]

X = np.hstack([np.ones((len(x), 1)), x.to_numpy().reshape(-1, 1)])
xt = np.transpose(X)
inverse_xtx = np.linalg.inv(xt @ X)
xty = xt @ y
w = inverse_xtx @ xty

y_pred = X @ w

plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.3, s=1, label="Data")
plt.plot(x, y_pred, color="red", label="Linear Fit")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression Fit")
plt.savefig("linear_fit_to_data.png")
plt.legend()
plt.show()
