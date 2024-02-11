import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
from scipy.optimize import curve_fit

# Load the data
d: pd.DataFrame = pd.read_pickle("pulsar_residuals.pkl")

# Assign a theta value to each index (theta = 2 * pi * (x - min) / (max - min))
d['theta'] = 2 * np.pi * d.index/d.index.max()
d['first_diff'] = d['first_diff'] - d['first_diff'].mean()

# Fit function with torch linear regression model

# Convert to tensor
first_diff: torch.Tensor = torch.tensor(d['first_diff'].values, dtype=torch.float32)
theta: torch.Tensor = torch.tensor(d['theta'].values, dtype=torch.float32)

# Define the model
class FitFunc(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.A = torch.nn.Parameter(torch.tensor(d['first_diff'].max() - d['first_diff'].min()))
        self.B = torch.nn.Parameter(torch.tensor(1.0))
        self.C = torch.nn.Parameter(torch.tensor(d['first_diff'].mean()))

    def forward(self, x):
        return self.A * torch.cos(self.B + x) + self.C

model: FitFunc = FitFunc()

# Define the loss function and the optimiser
loss_fn: torch.nn.MSELoss = torch.nn.MSELoss()
optimiser: Adam = Adam(model.parameters(), lr=0.01)

# Train the model
for t in range(1000):
    y_pred = model(theta)
    loss = loss_fn(y_pred, first_diff)
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()


# Fit function with scipy curve_fit

def fit_func(x: np.ndarray, A: float, B: float, C: float) -> np.ndarray:
    return A * np.cos(B + x) + C

initial_guess: np.ndarray = np.array([d['first_diff'].max() - d['first_diff'].min(), 1.0, d['first_diff'].mean()])

popt, pcov = curve_fit(fit_func, d['theta'], d['first_diff'], p0=initial_guess)

# Plot the data (x axis from 0 to 2 * pi)
plt.scatter(d['theta'], d['first_diff'], label='Data')
plt.plot(d['theta'], fit_func(d['theta'], *popt), label='Scipy Fit')
plt.plot(d['theta'], model(theta).detach().numpy(), label='Torch Fit')
plt.xlabel('Theta')
plt.ylabel('First Difference')
plt.xticks(ticks=np.linspace(0, 2 * np.pi, 5), labels=['0', 'pi/2', 'pi', '3pi/2', '2pi'])
plt.legend()
plt.show()