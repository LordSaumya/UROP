import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
from scipy.optimize import curve_fit

# Load the data
d: pd.DataFrame = pd.read_pickle("pulsar_residuals.pkl")
d2: pd.DataFrame = pd.read_pickle("pulsar_residuals_v2.pkl")
d['residuals'] = d2['residuals']

# Assign a theta value to each TOA (theta = 2pi * (x - min) / (max - min))
d['theta'] = 2 * np.pi * (d['TOA'] - d['TOA'].min()) / (d['TOA'].max() - d['TOA'].min())

# Fit function with torch linear regression model

# # Convert to tensor
# theta: torch.Tensor = torch.tensor(d['theta'].values, dtype=torch.float32)
# residuals: torch.Tensor = torch.tensor(d['residuals'].values, dtype=torch.float32)

# # Define the model
# class FitFunc(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.A = torch.nn.Parameter(torch.tensor(d['residuals'].max() - d['residuals'].min()))
#         self.B = torch.nn.Parameter(torch.tensor(1.0))
#         self.C = torch.nn.Parameter(torch.tensor(d['residuals'].mean()))

#     def forward(self, x):
#         return self.A * torch.cos(self.B + x) + self.C
    
# model: FitFunc = FitFunc()

# # Define the loss function and the optimiser
# loss_fn: torch.nn.MSELoss = torch.nn.MSELoss()
# optimiser: Adam = Adam(model.parameters(), lr=0.01)

# # Train the model
# for t in range(1000):
#     y_pred = model(theta)
#     loss = loss_fn(y_pred, residuals)
#     optimiser.zero_grad()
#     loss.backward()
#     optimiser.step()


# Fit function with scipy curve_fit

def fit_func(x: np.ndarray, A: float, B: float, C: float) -> np.ndarray:
    return A * np.cos(B + x) + C

initial_guess: np.ndarray = np.array([d['residuals'].max() - d['residuals'].min(), 1.0, d['residuals'].mean()])
popt, pcov = curve_fit(fit_func, d['theta'], d['residuals'], p0=initial_guess)

# # Plot the data
# plt.scatter(d['theta'], d['residuals'], label='Data')
# plt.plot(d['theta'], model(theta).detach().numpy(), label='Torch Fit', color='red')
# plt.plot(d['theta'], fit_func(d['theta'], *popt), label='Scipy Fit', color='green')
# plt.xlabel('Theta')
# plt.ylabel('Residuals')
# plt.legend()
# plt.show()

# # Print parameters
# print("Torch Fit Parameters: A = {}, B = {}, C = {}".format(model.A.item(), model.B.item(), model.C.item()))
# print("Scipy Fit Parameters: A = {}, B = {}, C = {}".format(popt[0], popt[1], popt[2]))

# Calculate circular motion residuals
circResiduals: np.ndarray = d['residuals'] - fit_func(d['theta'], *popt)
d['circResiduals'] = circResiduals

# # Plot the circular motion residuals
# plt.scatter(d['theta'], d['circResiduals'])
# plt.xlabel('Theta')
# plt.ylabel('Circular Motion Residuals')
# plt.show()


# Fit circResiduals with theta
def fit_func(x: np.ndarray, A: float, B: float, C: float, D: float) -> np.ndarray:
    return A * np.cos(B + C*x) + D

initial_guess: np.ndarray = np.array([d['circResiduals'].max() - d['circResiduals'].min(), 1.0, d['theta'].max(), d['circResiduals'].mean()])
popt, pcov = curve_fit(fit_func, d['theta'], d['circResiduals'])

# Plot the data
plt.scatter(d['theta'], d['circResiduals'], label='Data')
plt.plot(d['theta'], fit_func(d['theta'], *popt), color='green', label='Scipy Fit: {} * cos({} + {}x) + {}'.format(popt[0], popt[1], popt[2], popt[3]))
plt.xlabel('Theta')
plt.ylabel('Circular Motion Residuals')
plt.legend()
plt.show()

# Save data to csv
d.to_csv("pulsar_ucm_residuals.csv")