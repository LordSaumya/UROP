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

## Eccentricity
r_apo = [0] * len(lunar_dfs)
r_per = [0] * len(lunar_dfs)
ecc = [0] * len(lunar_dfs)
for i in range(0, len(lunar_dfs)):
    r_apo[i] = max(lunar_dfs[i]["delta"])
    r_per[i] = min(lunar_dfs[i]["delta"])
    ecc[i] = (r_apo[i] - r_per[i]) / (r_apo[i] + r_per[i])

## Eccentric, Mean, and True Anomalies
semi_major_axis = [0] * len(lunar_dfs)
error_vals = []
for i in range(0, len(lunar_dfs)):
    semi_major_axis[i] = (r_apo[i] + r_per[i]) / 2
    lunar_df = lunar_dfs[i].copy()
    lunar_df.loc[:, "eccentric_anomaly"] = np.arccos(np.clip((1 - lunar_df["delta"] / semi_major_axis[i]) / ecc[i], -1, 1))
    lunar_df.loc[:, "mean_anomaly"] = lunar_df["eccentric_anomaly"] - ecc[i] * np.sin(lunar_df["eccentric_anomaly"])
    lunar_df.loc[:, "true_anomaly"] = 2 * np.arctan(np.sqrt((1 + ecc[i]) / (1 - ecc[i])) * np.tan(lunar_df["eccentric_anomaly"] / 2))
    lunar_dfs[i] = lunar_df

## Residuals
for i in range(0, len(lunar_dfs)):
    lunar_dfs[i].loc[:, "anomaly_residual"] = lunar_dfs[i]["true_anomaly"] - lunar_dfs[i]["mean_anomaly"]

## Graph residuals
fig, ax = plt.subplots()
for i in range(1, len(lunar_dfs)):
    ax.plot(lunar_dfs[i]["Datetime"], lunar_dfs[i]["anomaly_residual"], label="Lunar Cycle " + str(i + 1))
ax.set_xlabel("Datetime")
ax.set_ylabel("Anomaly Residual")
ax.set_title("Anomaly Residuals")
ax.legend()
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.show()

## Transform data for AI Feynman
df = pd.concat(lunar_dfs)
df["sin_1_mean_anomaly"] = np.sin(df["mean_anomaly"])
df["sin_2_mean_anomaly"] = np.sin(2 * df["mean_anomaly"])
df["sin_3_mean_anomaly"] = np.sin(3 * df["mean_anomaly"])
df["scaled_anomaly_residual"] = df["anomaly_residual"] * 10
df = df[["mean_anomaly", "sin_1_mean_anomaly", "sin_2_mean_anomaly", "sin_3_mean_anomaly", "scaled_anomaly_residual", "lunar_cycle"]]

## Save results to csv
df.to_csv("lunar_anomalies.csv")