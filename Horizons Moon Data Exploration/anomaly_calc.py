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

## Find all perigees
delta_diff = df["delta"].diff()
perigees = (delta_diff.shift(-1) > 0) & (delta_diff < 0)
perigees = df[perigees]

last_perigee = lambda datetime: perigees["Datetime"][perigees["Datetime"] < datetime].max()
df["time_since_perigee"] = (df["Datetime"] - df["Datetime"].apply(last_perigee)).dt.total_seconds()

## Find all apogees
apogees = (delta_diff.shift(-1) < 0) & (delta_diff > 0)
apogees = df[apogees]

# Anomalous maxima
# dates_of_maxima = [dt.datetime(2024, 1, 1, 15), dt.datetime(2024, 1, 29, 8), dt.datetime(2024, 2, 25, 15), dt.datetime(2024, 3, 23, 16), dt.datetime(2024, 4, 20, 2), dt.datetime(2024, 5, 17, 19), dt.datetime(2024, 6, 14, 14), dt.datetime(2024, 7, 12, 8), dt.datetime(2024, 8, 9, 2), dt.datetime(2024, 9, 5, 15), dt.datetime(2024, 10, 2, 20), dt.datetime(2024, 10, 29, 23), dt.datetime(2024, 11, 26, 12), dt.datetime(2024, 12, 24, 7)]

# Synodic maxima (full moons):
dates_of_maxima = [dt.datetime(2024, 1, 25, 17), dt.datetime(2024, 2, 29, 8), dt.datetime(2024, 3, 25, 7), dt.datetime(2024, 4, 23, 23), dt.datetime(2024, 5, 23, 13), dt.datetime(2024, 6, 22, 1), dt.datetime(2024, 7, 21, 10), dt.datetime(2024, 8, 19, 18), dt.datetime(2024, 9, 18, 2), dt.datetime(2024, 10, 17, 11), dt.datetime(2024, 11, 15, 9), dt.datetime(2024, 12, 15, 9)]

# Assign a lunar cycle number to all entries in DF based on maxima
for i in range(0, len(dates_of_maxima) - 1):
    df.loc[(df['Datetime'] >= dates_of_maxima[i]) & (df['Datetime'] < dates_of_maxima[i + 1]), 'lunar_cycle'] = i + 1

df.loc[df['Datetime'] >= dates_of_maxima[-1], 'lunar_cycle'] = len(dates_of_maxima)

# Replace all NaN values with 0
df['lunar_cycle'] = df['lunar_cycle'].fillna(0)

lunar_dfs = [0] * (len(dates_of_maxima))
for cycle in range(0, len(dates_of_maxima)):
    lunar_dfs[cycle] = df[df['lunar_cycle'] == cycle]

lunar_dfs.pop(0)

## Eccentricity
r_apo = [0] * len(lunar_dfs)
r_per = [0] * len(lunar_dfs)
ecc = [0] * len(lunar_dfs)
for i in range(0, len(lunar_dfs)):
    r_apo[i] = apogees["delta"][apogees["Datetime"] > lunar_dfs[i]["Datetime"].iloc[0]].iloc[0]
    r_per[i] = perigees["delta"][perigees["Datetime"] > lunar_dfs[i]["Datetime"].iloc[0]].iloc[0]
    ecc[i] = (r_apo[i] - r_per[i]) / (r_apo[i] + r_per[i])

print(ecc)

## Eccentric, Mean, and True Anomalies
semi_major_axis = [0] * len(lunar_dfs)
semi_minor_axis = [0] * len(lunar_dfs)

sign = lambda x: (1, -1)[x < 0]

# Function to iteratively solve Kepler's equation to find the eccentric anomaly given M and e
def kepler(M, e):
    E = M
    while True:
        E_new = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        if abs(E - E_new) < 1e-5:
            break
        E = E_new
    return E

for i in range(0, len(lunar_dfs)):
    semi_major_axis[i] = (r_apo[i] + r_per[i]) / 2
    semi_minor_axis[i] = semi_major_axis[i] * np.sqrt(1 - ecc[i] ** 2)
    lunar_df = lunar_dfs[i].copy()
    lunar_df.reset_index(inplace=True)

    # Mean anomaly
    t = lunar_df["time_since_perigee"]
    T = 27.32 * 24 * 60 * 60
    lunar_df.loc[:, "mean_anomaly"] = 2 * np.pi * t / T

    # True anomaly
    E = lunar_df["mean_anomaly"].apply(kepler, args=(ecc[i],))
    beta = ecc[i]/(1 + np.sqrt(1 - ecc[i]**2))
    lunar_df.loc[:, "true_anomaly"] = E + 2 * np.arctan(beta * np.sin(E)/(1 - beta * np.cos(E)))
    # lunar_df.loc[:, "true_anomaly"] = 2 * np.arctan2(np.sqrt(1 + ecc[i]) * np.sin(E / 2), np.sqrt(1 - ecc[i]) * np.cos(E / 2))

    # n = 2*np.pi/(27.32 * 24 * 60 * 60)
    # t = lunar_df["time_from_start"] - lunar_df["time_from_start"].iloc[0]
    # tau = lunar_df["Datetime"].iloc[lunar_df["delta"].idxmin()]
    # t_tau_diff = (lunar_df["Datetime"] - tau).dt.total_seconds()
    # lunar_df.loc[:, "mean_anomaly"] = n * t_tau_diff
    # lunar_df.loc[:, "true_anomaly"] =  lunar_df["mean_anomaly"].apply(sign) * np.arccos((-(ecc[i])**2 * semi_major_axis[i] + semi_major_axis[i] - lunar_df["delta"])/(ecc[i] * lunar_df["delta"]))
    # lunar_df.loc[:, "eccentric_anomaly"] = np.arccos((1 - lunar_df["delta"] / semi_major_axis[i]) / ecc[i])
    # lunar_df.loc[:, "mean_anomaly"] = lunar_df["eccentric_anomaly"] - ecc[i] * np.sin(lunar_df["eccentric_anomaly"])
    # lunar_df.loc[:, "true_anomaly"] = 2 * np.arctan(np.sqrt((1 + ecc[i]) / (1 - ecc[i])) * np.tan(lunar_df["eccentric_anomaly"] / 2))
    lunar_df.dropna(inplace=True)
    lunar_df.reset_index(drop=True, inplace=True)
    lunar_dfs[i] = lunar_df

## Residuals
for i in range(0, len(lunar_dfs)):
    lunar_dfs[i].loc[:, "anomaly_residual"] = lunar_dfs[i]["true_anomaly"] - lunar_dfs[i]["mean_anomaly"]

## Graph distance vs time coloured by lunar cycle
fig, ax = plt.subplots()
for i in range(0, len(lunar_dfs)):
    ax.plot(lunar_dfs[i]["Datetime"], lunar_dfs[i]["delta"], label="Lunar Cycle " + str(i + 1))
ax.plot(perigees["Datetime"], perigees["delta"], 'ro', label="Perigee")
ax.set_xlabel("Datetime")
ax.set_ylabel("Distance (AU)")
ax.set_title("Distance vs Time")
ax.legend()

## Graph true anomaly
fig, ax = plt.subplots()
for i in range(0, len(lunar_dfs)):
    ax.plot(lunar_dfs[i]["Datetime"], lunar_dfs[i]["true_anomaly"], label="Lunar Cycle " + str(i + 1))
ax.set_xlabel("Datetime")
ax.set_ylabel("True Anomaly")
ax.set_title("True Anomalies")
ax.legend()
ax.xaxis.set_major_locator(mdates.MonthLocator())

## Graph mean anomaly
fig, ax = plt.subplots()
for i in range(0, len(lunar_dfs)):
    ax.plot(lunar_dfs[i]["Datetime"], lunar_dfs[i]["mean_anomaly"], label="Lunar Cycle " + str(i + 1))
ax.set_xlabel("Datetime")
ax.set_ylabel("Mean Anomaly")
ax.set_title("Mean Anomalies")
ax.legend()

## Graph residuals against mean anomaly
fig, ax = plt.subplots()
for i in range(0, len(lunar_dfs)):
    print(lunar_dfs[i].head(30))
    ax.scatter(lunar_dfs[i]["mean_anomaly"], lunar_dfs[i]["anomaly_residual"], label="Synodic Cycle " + str(i))
ax.set_xticks(np.arange(0, 2*np.pi + 0.01, np.pi/4), ['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$', r'$\frac{5\pi}{4}$', r'$\frac{3\pi}{2}$', r'$\frac{7\pi}{4}$', r'$2\pi$'])
ax.set_xlabel("Mean Anomaly (radians)")
ax.set_ylabel("Anomaly Residual (radians)")
ax.set_title("Lunar Anomaly Residual vs Mean Anomaly")
ax.legend()

# Gather all data into a single dataframe
df = pd.concat(lunar_dfs)
df["sin_1_mean_anomaly"] = np.sin(df["mean_anomaly"])
df["sin_2_mean_anomaly"] = np.sin(2 * df["mean_anomaly"])
df["sin_3_mean_anomaly"] = np.sin(3 * df["mean_anomaly"])
df["scaled_anomaly_residual"] = df["anomaly_residual"] * 10
df["sec_ord_SAR"] = df["scaled_anomaly_residual"] - 1.146627312067 * df["sin_1_mean_anomaly"]
df = df[["mean_anomaly", "sin_1_mean_anomaly", "sin_2_mean_anomaly", "sin_3_mean_anomaly", "anomaly_residual", "scaled_anomaly_residual", "lunar_cycle", "sec_ord_SAR"]]

## Save results to csv
df.to_csv("lunar_anomalies.csv")

plt.show()