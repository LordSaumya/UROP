## Imports
import numpy as np
import pandas as pd
import datetime as dt
from re import match, split
from astropy import units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy.optimize import curve_fit
import torch

## DATA PREPROCESSING

## Read in data from pickle file
df = pd.read_pickle("lunar_anomalies.pkl")

# Extract month from 'Datetime'
df["Month"] = df["Datetime"].dt.month

# Extract year from 'Datetime'
df["Year"] = df["Datetime"].dt.year

# Extract day from 'Datetime'
df["Day"] = df["Datetime"].dt.day

# For circular motion, the geocentric distance is constant at about 384,400 km / 149,597,870.7 ~ 0.00256955528 AU
df["residual"] = (df["delta"] - 0.00256955528) * 384400 # Residual in KM

cycle_num = 3
df = df[df["lunar_cycle"] == cycle_num]

# Scale anomaly residual and add Gaussian noise
df["anomaly_residual"] = df["anomaly_residual"] * 10 + np.random.normal(0, 0.1, len(df))

### Models

# Random train-test split
train_size = 0.9
train, test = train_test_split(df, train_size=train_size)
train_inputs = torch.tensor(train[["RA", "DEC"]].values, dtype=torch.float32)
train_labels = torch.tensor(train["anomaly_residual"].values, dtype=torch.float32).view(-1, 1)
test_inputs = torch.tensor(test[["RA", "DEC"]].values, dtype=torch.float32)
test_labels = torch.tensor(test["anomaly_residual"].values, dtype=torch.float32).view(-1, 1)
criterion = torch.nn.MSELoss()

# Model 1: 3-MLP
class MLP_3(torch.nn.Module):
    def __init__(self):
        super(MLP_3, self).__init__()
        self.fc1 = torch.nn.Linear(2, 16)
        self.fc2 = torch.nn.Linear(16, 32)
        self.fc3 = torch.nn.Linear(32, 1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model_3MLP = MLP_3()
optimiser = torch.optim.Adam(model_3MLP.parameters(), lr=0.001)

# Training
epochs = 1000
print("Training 3-MLP model...")
for epoch in range(epochs):
    model_3MLP.train()
    optimiser.zero_grad()
    outputs = model_3MLP(train_inputs)
    loss = criterion(outputs, train_labels)
    loss.backward()
    optimiser.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch} - Loss: {loss.item()}")
print("Training complete.")

# Model 2: 5-MLP
class MLP_5(torch.nn.Module):
    def __init__(self):
        super(MLP_5, self).__init__()
        self.fc1 = torch.nn.Linear(2, 16)
        self.fc2 = torch.nn.Linear(16, 32)
        self.fc3 = torch.nn.Linear(32, 64)
        self.fc4 = torch.nn.Linear(64, 32)
        self.fc5 = torch.nn.Linear(32, 1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.relu(self.fc4(x))
        x = self.fc5(x)
        return x

model_5MLP = MLP_5()
optimiser = torch.optim.Adam(model_5MLP.parameters(), lr=0.001)

# Training
print("Training 5-MLP model...")
for epoch in range(epochs):
    model_5MLP.train()
    optimiser.zero_grad()
    outputs = model_5MLP(train_inputs)
    loss = criterion(outputs, train_labels)
    loss.backward()
    optimiser.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch} - Loss: {loss.item()}")
print("Training complete.")

# Model 3: 5-MLP with batch normalisation
class MLP_5_BN(torch.nn.Module):
    def __init__(self):
        super(MLP_5_BN, self).__init__()
        self.fc1 = torch.nn.Linear(2, 16)
        self.fc2 = torch.nn.Linear(16, 32)
        self.fc3 = torch.nn.Linear(32, 64)
        self.fc4 = torch.nn.Linear(64, 32)
        self.fc5 = torch.nn.Linear(32, 1)
        self.bn1 = torch.nn.BatchNorm1d(16)
        self.bn2 = torch.nn.BatchNorm1d(32)
        self.bn3 = torch.nn.BatchNorm1d(64)
        self.bn4 = torch.nn.BatchNorm1d(32)
    def forward(self, x):
        x = self.bn1(torch.nn.functional.relu(self.fc1(x)))
        x = self.bn2(torch.nn.functional.relu(self.fc2(x)))
        x = self.bn3(torch.nn.functional.relu(self.fc3(x)))
        x = self.bn4(torch.nn.functional.relu(self.fc4(x)))
        x = self.fc5(x)
        return x

model_5MLP_BN = MLP_5_BN()
optimiser = torch.optim.Adam(model_5MLP_BN.parameters(), lr=0.001)

# Training
print("Training 5-MLP with batch normalisation model...")
for epoch in range(epochs):
    model_5MLP_BN.train()
    optimiser.zero_grad()
    outputs = model_5MLP_BN(train_inputs)
    loss = criterion(outputs, train_labels)
    loss.backward()
    optimiser.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch} - Loss: {loss.item()}")
print("Training complete.")

# Model 4: Single sine wave
def sine_wave(x, a, b):
    return a * np.sin(b * x)
popt, pcov = curve_fit(sine_wave, train["mean_anomaly"], train["anomaly_residual"])
train["sine_residual"] = sine_wave(train["mean_anomaly"], *popt)
test["sine_residual"] = sine_wave(test["mean_anomaly"], *popt)

# Model 5: Multiple sine waves
def multi_sine_wave(x, a, b, c):
    return a * np.sin(x) + b * np.sin(2 * x) + c * np.sin(3 * x)
popt2, pcov2 = curve_fit(multi_sine_wave, train["mean_anomaly"], train["anomaly_residual"])
train["multi_sine_residual"] = multi_sine_wave(train["mean_anomaly"], *popt2)
test["multi_sine_residual"] = multi_sine_wave(test["mean_anomaly"], *popt2)

## Testing
model_3MLP.eval()
test_outputs_3MLP = model_3MLP(test_inputs)
test_loss_3MLP = criterion(test_outputs_3MLP, test_labels)
print(f"3-MLP Test Loss: {test_loss_3MLP.item()}")

model_5MLP.eval()
test_outputs_5MLP = model_5MLP(test_inputs)
test_loss_5MLP = criterion(test_outputs_5MLP, test_labels)
print(f"5-MLP Test Loss: {test_loss_5MLP.item()}")

model_5MLP_BN.eval()
test_outputs_5MLP_BN_DO = model_5MLP_BN(test_inputs)
test_loss_5MLP_BN_DO = criterion(test_outputs_5MLP_BN_DO, test_labels)
print(f"5-MLP with batch normalisation Test Loss: {test_loss_5MLP_BN_DO.item()}")

test_outputs_sine = sine_wave(test["mean_anomaly"], *popt)
test_loss_sine = np.mean((test_outputs_sine - test["anomaly_residual"])**2)
print(f"Single Sine Wave Test Loss: {test_loss_sine}")

test_outputs_multi_sine = multi_sine_wave(test["mean_anomaly"], *popt2)
test_loss_multi_sine = np.mean((test_outputs_multi_sine - test["anomaly_residual"])**2)
print(f"Multiple Sine Waves Test Loss: {test_loss_multi_sine}")

# Print equations for sines
print(f"Single Sine Wave: y = {popt[0]:.5f} * sin({popt[1]:.5f} * x)")
print(f"Multiple Sine Waves: y = {popt2[0]:.5f} * sin(x) + {popt2[1]:.5f} * sin(2x) + {popt2[2]:.5f} * sin(3x)")

## Graphs
inputs = torch.tensor(df[["RA", "DEC"]].values, dtype=torch.float32)
plt.figure(figsize=(10, 5))
plt.scatter(df["Datetime"], df["anomaly_residual"], s=1, label="Data")
plt.scatter(df["Datetime"], model_3MLP(inputs).detach().numpy(), s=1, label="3-MLP")
plt.scatter(df["Datetime"], model_5MLP(inputs).detach().numpy(), s=1, label="5-MLP")
plt.scatter(df["Datetime"], model_5MLP_BN(inputs).detach().numpy(), s=1, label="5-MLP with BN")
plt.scatter(df["Datetime"], sine_wave(df["mean_anomaly"], *popt), s=1, label="Single Sine Wave")
plt.scatter(df["Datetime"], multi_sine_wave(df["mean_anomaly"], *popt2), s=1, label="Multiple Sine Waves")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d-%m"))
plt.gcf().autofmt_xdate()
plt.xlabel("Datetime")
plt.ylabel("Anomaly Residual")
plt.title(f"Anomaly Residuals vs Datetime - Lunar Cycle {cycle_num}")
plt.legend()
plt.show()


    
