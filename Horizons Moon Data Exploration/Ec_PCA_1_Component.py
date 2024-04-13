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
print(pca.explained_variance_ratio_)

# Plot 3: Delta vs PCA
plt.figure(figsize=(10, 5))
plt.scatter(df["PC"], df["delta"], s=1)
plt.xlabel("PC (Lat, Lon)")
plt.ylabel("Delta (AU)")
plt.title("Delta vs PC (Lat, Lon)")
plt.show()