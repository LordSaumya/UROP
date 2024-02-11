import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
from scipy.optimize import curve_fit

# Load the data
d: pd.DataFrame = pd.read_pickle("pulsar_residuals.pkl")

# Assign a theta value to each first_diff (theta = 2 * pi * (x - min) / (max - min))
d['theta'] = 2 * np.pi * (d['first_diff'] - d['first_diff'].min()) / (d['first_diff'].max() - d['first_diff'].min())

# Fit function with torch linear regression model

# Convert to tensor
index: torch.Tensor = torch.tensor(d.index.values, dtype=torch.float32)
theta: torch.Tensor = torch.tensor(d['theta'].values, dtype=torch.float32)

# Define the model
class FitFunc(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.A = torch.nn.Parameter(torch.tensor(d['theta'].max() - d['theta'].min()))
        self.B = torch.nn.Parameter(torch.tensor(1.0))
        self.C = torch.nn.Parameter(torch.tensor(d['theta'].mean()))

    def forward(self, x):
        return self.A * torch.cos(self.B * x) + self.C

model: FitFunc = FitFunc()

# Define the loss function and the optimiser
loss_fn: torch.nn.MSELoss = torch.nn.MSELoss()
optimiser: Adam = Adam(model.parameters(), lr=0.01)

# Train the model
for t in range(1000):
    y_pred = model(index)
    loss = loss_fn(y_pred, theta)
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()


# Fit function with scipy curve_fit

def fit_func(x: np.ndarray, A: float, B: float, C: float) -> np.ndarray:
    return A * np.cos(B * x) + C

initial_guess: np.ndarray = np.array([d['theta'].max() - d['theta'].min(), 1.0, d['theta'].mean()])

popt, pcov = curve_fit(fit_func, d.index, d['theta'], p0=initial_guess)

# Plot the data
plt.scatter(d.index, d['theta'], label='Data')
plt.plot(d.index, model(index).detach().numpy(), label='Fit (torch linear regression)', color='red')
plt.plot(d.index, fit_func(d.index, *popt), label='Fit (curve_fit)', color='green')
plt.xlabel('Index')
plt.ylabel('Theta')
plt.legend()
plt.show()