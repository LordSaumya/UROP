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
from scipy.optimize import curve_fit

## Read in data from pickle file
df = pd.read_pickle("horizons_results.pkl")

# Extract month from 'Datetime'
df["Month"] = df["Datetime"].dt.month

# Extract year from 'Datetime'
df["Year"] = df["Datetime"].dt.year

# Extract day from 'Datetime'
df["Day"] = df["Datetime"].dt.day

Lon = df["Lon"]
Lat = df["Lat"]

# Plot 1: Delta vs Lat
plt.figure(figsize=(10, 5))
plt.scatter(Lat, df["delta"], s=1)
plt.xlabel("Latitude (deg)")
plt.ylabel("Delta (AU)")
plt.title("Delta vs Latitude")
plt.show()

# Plot 2: Delta vs Lon
plt.figure(figsize=(10, 5))
plt.scatter(Lon, df["delta"], s=1)
plt.xlabel("Longitude (deg)")
plt.ylabel("Delta (AU)")
plt.title("Delta vs Longitude")
plt.show()

## PCA
pca = PCA(n_components=1)
pca.fit(np.array([df["Lat"], df["Lon"]]).T)
projected = pca.transform(np.array([df["Lat"], df["Lon"]]).T)
df["PC"] = projected

# Plot 3: Delta vs PCA
plt.figure(figsize=(10, 5))
plt.scatter(df["PC"], df["delta"], s=1)
plt.xlabel("PC (Lat, Lon)")
plt.ylabel("Delta (AU)")
plt.title("Delta vs PC (Lat, Lon)")
plt.show()

dates_of_maxima = [dt.datetime(2024, 1, 1, 15), dt.datetime(2024, 1, 29, 8), dt.datetime(2024, 2, 25, 15), dt.datetime(2024, 3, 23, 16), dt.datetime(2024, 4, 20, 2), dt.datetime(2024, 5, 17, 19), dt.datetime(2024, 6, 14, 14), dt.datetime(2024, 7, 12, 8), dt.datetime(2024, 8, 9, 2), dt.datetime(2024, 9, 5, 15), dt.datetime(2024, 10, 2, 20), dt.datetime(2024, 10, 29, 23), dt.datetime(2024, 11, 26, 12), dt.datetime(2024, 12, 24, 7)]

# Assign a lunar cycle number to all entries in DF based on maxima
for i in range(0, len(dates_of_maxima) - 1):
    df.loc[(df['Datetime'] >= dates_of_maxima[i]) & (df['Datetime'] < dates_of_maxima[i + 1]), 'lunar_cycle'] = i + 1

# Replace all NaN values with 0
df['lunar_cycle'] = df['lunar_cycle'].fillna(0)

lunar_dfs = {}
for cycle in range(0, 14):
    lunar_dfs[cycle] = df[df['lunar_cycle'] == cycle]

# Save cycle 4 as csv
lunar_dfs[4].to_csv('horizons_results_ec_pca_cycle_4.csv')