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
df = pd.read_pickle("horizons_results.pkl")

# Extract month from 'Datetime'
df["Month"] = df["Datetime"].dt.month

# Extract year from 'Datetime'
df["Year"] = df["Datetime"].dt.year

# Extract day from 'Datetime'
df["Day"] = df["Datetime"].dt.day

df["time_from_start"] = (df["Datetime"] - df["Datetime"][0]).dt.total_seconds()

# For circular motion, the geocentric distance is constant at about 384,400 km / 149,597,870.7 ~ 0.00256955528 AU
df["residual"] = (df["delta"] - 0.00256955528) * 384400 # Residual in KM

dates_of_maxima = [dt.datetime(2024, 1, 1, 15), dt.datetime(2024, 1, 29, 8), dt.datetime(2024, 2, 25, 15), dt.datetime(2024, 3, 23, 16), dt.datetime(2024, 4, 20, 2), dt.datetime(2024, 5, 17, 19), dt.datetime(2024, 6, 14, 14), dt.datetime(2024, 7, 12, 8), dt.datetime(2024, 8, 9, 2), dt.datetime(2024, 9, 5, 15), dt.datetime(2024, 10, 2, 20), dt.datetime(2024, 10, 29, 23), dt.datetime(2024, 11, 26, 12), dt.datetime(2024, 12, 24, 7)]

# Assign a lunar cycle number to all entries in DF based on maxima
for i in range(0, len(dates_of_maxima) - 1):
    df.loc[(df['Datetime'] >= dates_of_maxima[i]) & (df['Datetime'] < dates_of_maxima[i + 1]), 'lunar_cycle'] = i + 1

# Replace all NaN values with 0
df['lunar_cycle'] = df['lunar_cycle'].fillna(0)

lunar_dfs = [0] * (len(dates_of_maxima) - 1)
for cycle in range(0, len(dates_of_maxima) - 1):
    lunar_dfs[cycle] = df[df['lunar_cycle'] == cycle]

# Pad the dataframes with zeros to make them all the same length for RNN
max_length = max([len(lunar_df) for lunar_df in lunar_dfs])
for lunar_df in lunar_dfs:
    lunar_df = pd.concat([lunar_df, pd.DataFrame(np.zeros((max_length - len(lunar_df), len(lunar_df.columns))))])

### MODELS

# Random train-test split
train_size = 0.9
train, test = train_test_split(df, train_size=train_size)
train_inputs = torch.tensor(train[["RA", "DEC"]].values, dtype=torch.float32)
train_labels = torch.tensor(train["residual"].values, dtype=torch.float32).view(-1, 1)
test_inputs = torch.tensor(test[["RA", "DEC"]].values, dtype=torch.float32)
test_labels = torch.tensor(test["residual"].values, dtype=torch.float32).view(-1, 1)
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

# Train 3-MLP
model_3 = MLP_3()
optimiser = torch.optim.Adam(model_3.parameters(), lr=0.001)
for epoch in range(1000):
    optimiser.zero_grad()
    outputs = model_3(train_inputs)
    loss = criterion(outputs, train_labels)
    loss.backward()
    optimiser.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

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

# Train 5-MLP
model_5 = MLP_5()
optimiser = torch.optim.Adam(model_5.parameters(), lr=0.001)
for epoch in range(1000):
    optimiser.zero_grad()
    outputs = model_5(train_inputs)
    loss = criterion(outputs, train_labels)
    loss.backward()
    optimiser.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Model 3: 5-MLP with batch normalisation and dropout
class MLP_5_BN_DO(torch.nn.Module):
    def __init__(self):
        super(MLP_5_BN_DO, self).__init__()
        self.fc1 = torch.nn.Linear(2, 16)
        self.fc2 = torch.nn.Linear(16, 32)
        self.fc3 = torch.nn.Linear(32, 64)
        self.fc4 = torch.nn.Linear(64, 32)
        self.fc5 = torch.nn.Linear(32, 1)
        self.bn1 = torch.nn.BatchNorm1d(16)
        self.bn2 = torch.nn.BatchNorm1d(32)
        self.bn3 = torch.nn.BatchNorm1d(64)
        self.bn4 = torch.nn.BatchNorm1d(32)
        self.dropout = torch.nn.Dropout(0.1)
    def forward(self, x):
        x = self.dropout(torch.nn.functional.relu(self.bn1(self.fc1(x))))
        x = self.dropout(torch.nn.functional.relu(self.bn2(self.fc2(x))))
        x = self.dropout(torch.nn.functional.relu(self.bn3(self.fc3(x))))
        x = self.dropout(torch.nn.functional.relu(self.bn4(self.fc4(x))))
        x = self.fc5(x)
        return x
    
# Train 5-MLP with batch normalisation and dropout
model_5_BN_DO = MLP_5_BN_DO()
optimiser = torch.optim.Adam(model_5_BN_DO.parameters(), lr=0.001)
for epoch in range(1000):
    optimiser.zero_grad()
    outputs = model_5_BN_DO(train_inputs)
    loss = criterion(outputs, train_labels)
    loss.backward()
    optimiser.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Test 3-MLP
model_3.eval()
test_outputs = model_3(test_inputs)
test_loss = criterion(test_outputs, test_labels)
print(f"3-MLP - Test Loss: {test_loss.item()}")

# Test 5-MLP
model_5.eval()
test_outputs = model_5(test_inputs)
test_loss = criterion(test_outputs, test_labels)
print(f"5-MLP - Test Loss: {test_loss.item()}")

# Test 5-MLP with batch normalisation and dropout
model_5_BN_DO.eval()
test_outputs = model_5_BN_DO(test_inputs)
test_loss = criterion(test_outputs, test_labels)
print(f"5-MLP-BN-DO - Test Loss: {test_loss.item()}")

# Plot models vs data
inputs = torch.tensor(df[["RA", "DEC"]].values, dtype=torch.float32)
plt.figure(figsize=(10, 5))
plt.scatter(df["Datetime"], df["residual"], s=1, label="Data")
plt.scatter(df["Datetime"], model_3(inputs).detach().numpy(), s=1, label="3-MLP")
plt.scatter(df["Datetime"], model_5(inputs).detach().numpy(), s=1, label="5-MLP")
plt.scatter(df["Datetime"], model_5_BN_DO(inputs).detach().numpy(), s=1, label="5-MLP with BN and DO")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%Y"))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gcf().autofmt_xdate()
plt.xlabel("Datetime")
plt.ylabel("Residual (KM)")
plt.title("Residual vs Datetime")
plt.legend()
plt.show()

