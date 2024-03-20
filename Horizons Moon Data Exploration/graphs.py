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

## Plot

# Extract month from 'Datetime'
df["Month"] = df["Datetime"].dt.month

# Extract year from 'Datetime'
df["Year"] = df["Datetime"].dt.year

# Extract day from 'Datetime'
df["Day"] = df["Datetime"].dt.day

RA = df['RA'] * np.pi / 180
DEC = df['DEC'] * np.pi / 180

# # Plot 1: Sky Motion vs. Time (months on x-axis)
# plt.figure(1)
# plt.plot(df["Datetime"], df["Sky_motion"])
# plt.xlabel("Month")
# plt.ylabel("Sky Motion (deg/day)")
# plt.title("Sky Motion vs. Time")
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
# plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
# plt.show()

# # Plot 2: Sky Motion (Position Angle) vs. Time (months on x-axis)
# plt.figure(2)
# plt.plot(df["Datetime"], df["Sky_mot_PA"])
# plt.xlabel("Month")
# plt.ylabel("Sky Motion (Position Angle) in degrees")
# plt.title("Sky Motion (Position Angle) vs. Time")
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
# plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
# plt.show()

# # Plot 3: Apparent Magnitude vs. Time (months on x-axis)
# plt.figure(3)
# plt.plot(df["Datetime"], df["APmag"])
# plt.xlabel("Month")
# plt.ylabel("Apparent Magnitude")
# plt.title("Apparent Magnitude vs. Time")
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
# plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
# plt.show()

# # Plot 4: Log of Surface Brightness vs. Time (months on x-axis)
# plt.figure(4)
# plt.plot(df["Datetime"], np.log10(df["S-brt"].astype(float)))
# plt.xlabel("Month")
# plt.ylabel("Log of Surface Brightness")
# plt.title("Log of Surface Brightness vs. Time")
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
# plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
# plt.show()

# # Plot 5: Aitoff projection of coordinates
# plt.figure(5)
# plt.subplot(111, projection="aitoff")
# plt.grid(True)
# plt.scatter(RA, DEC, s=1)
# plt.title("Aitoff Projection")
# plt.show()

# # Plot 6: 3D plot of coordinates vs time
# plt.figure(6)
# ax = plt.axes(projection="3d")
# df['date_delta'] = (df['Datetime'] - df['Datetime'].min())  / np.timedelta64(1,'D')
# ax.scatter3D(df["date_delta"], RA, DEC)
# ax.set_xlabel("Time (in days since 1st January 2024)")
# ax.set_ylabel("Right Ascension (rad)")
# ax.set_zlabel("Declination (rad)")
# plt.show()

# # Plot 7: Polar plot of coordinates vs time
# plt.figure(7)
# ax = plt.axes(projection="polar")
# ax.scatter(RA, DEC)
# ax.set_xlabel("Right Ascension (rad)")
# ax.set_ylabel("Declination (rad)")
# plt.show()

# # Plot 8: Monthly 2D plot of coordinates vs time
# # Split DataFrame by month
# monthly_dfs = {}
# for month in df["Month"].unique():
#     monthly_dfs[month] = df[df["Month"] == month]
# # Create the plot
# fig, ax = plt.subplots()

# # Iterate through each monthly DataFrame
# for month, df in monthly_dfs.items():
#     ra = np.array(df["RA"])
#     dec = np.array(df["DEC"])
#     ax.scatter(ra, dec, label=f"Month {month}", s=10) 
# plt.title("Right Ascension vs Declination (Colour-Coded by Month)")
# plt.xlabel("Right Ascension (Rad)")
# plt.ylabel("Declination (Rad)")
# plt.grid(True)
# plt.legend()
# plt.show()

# # Plot 9: Delta vs Time (months on x-axis)
# plt.figure(9)
# plt.plot(df["Datetime"], df["delta"])
# plt.xlabel("Month")
# plt.ylabel("Delta (AU)")
# plt.title("Delta vs. Time")
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
# plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
# plt.show()

# # Plot 10: Delta vs RA
# plt.figure(10)
# plt.scatter(RA , df["delta"])
# plt.xlabel("Right Ascension (rad)")
# plt.ylabel("Delta (AU)")
# plt.title("Delta vs. Right Ascension")
# plt.show()

# # Plot 11: Delta vs DEC
# plt.figure(11)
# plt.scatter(DEC , df["delta"])
# plt.xlabel("Declination (rad)")
# plt.ylabel("Delta (AU)")
# plt.title("Delta vs. Declination")
# plt.show()

# # Plot 12: Delta vs DEC and RA
# fig = plt.figure(12)
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(RA, DEC, df["delta"])
# ax.set_xlabel("Right Ascension (rad)")
# ax.set_ylabel("Declination (rad)")
# ax.set_zlabel("Delta (AU)")
# plt.show()

# # Plot 13: Delta vs DEC and RA (3D polar)
# fig = plt.figure(13)
# ax = fig.add_subplot(projection='polar')
# ax.scatter(RA, DEC, df["delta"])
# ax.set_xlabel("Right Ascension (rad)")
# ax.set_ylabel("Declination (rad)")
# plt.show()

# # Plot 14: Delta vs sin(DEC) and sin(RA)
# fig = plt.figure(12)
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(np.sin(RA), np.sin(DEC), df["delta"])
# ax.set_xlabel("Sine of Right Ascension")
# ax.set_ylabel("Sine of Declination")
# ax.set_zlabel("Delta (AU)")
# plt.show()

# Plot 15: Delta vs PCA of DEC and RA
## Calculate PCA using sklearn
pca = PCA(n_components=1)
pca.fit(np.array([RA, DEC]).T)
## Project data onto PCA
projected: np.ndarray = pca.transform(np.array([RA, DEC]).T)
df["pca"] = projected

# # Plot
# fig = plt.figure(15)
# ax = fig.add_subplot(111)
# ax.scatter(projected, df["delta"])
# ax.set_xlabel("PCA of Right Ascension and Declination")
# ax.set_ylabel("Delta (AU)")
# plt.show()

# # Curve fit sine to PCA vs delta
# def sine(x, A, B, C, D):
#     return A * np.sin(B * x + C) + D

# popt, pcov = curve_fit(sine, projected.flatten(), df["delta"])

# # Plot 16: Delta vs PCA of DEC and RA with sine fit
# fig = plt.figure(16)
# ax = fig.add_subplot(111)
# ax.scatter(projected, df["delta"], label="Data")
# ax.plot(projected, sine(projected, *popt), color="red", label="Scipy curve fit: {:.5f} * sin({:.5f} * x + {:.5f}) + {:.5f}".format(*popt))
# ax.set_xlabel("PCA of Right Ascension and Declination")
# ax.set_ylabel("Delta (gigametres)")
# plt.legend()
# plt.show()

# # Plot 17: Delta vs PCA of DEC and RA (polar plot with radius = delta and angle = PCA)
# fig = plt.figure(17)
# ax = fig.add_subplot(projection='polar', label="polar")
# ax.scatter(projected, df["delta"])
# ax.set_xlabel("PCA of Right Ascension and Declination")
# ax.set_ylim(0, 0.003)
# ax.set_ylabel("Delta (AU)")
# plt.show()

# Curve fit (-a*x^2 + b)^0.5 + c to PCA vs delta
def fit_func(x, a, b, c):
    return (-a * x ** 2 + b) ** 0.5 + c

popt, pcov = curve_fit(fit_func, projected.flatten(), df["delta"], bounds=([0, 0, -10000], [1000, 1000000, 10000]))

# # Plot 18: Plot of Delta vs PCA of DEC and RA with curve fit
# fig = plt.figure(18)
# ax = fig.add_subplot(111)
# ax.scatter(projected, df["delta"], label="Data")
# ax.plot(projected, fit_func(projected, *popt), color="red", label="Scipy curve fit: (-{:.5f} * x^2 + {:.5f})^0.5 + {:.5f}".format(*popt))
# ax.set_xlabel("PCA of Right Ascension and Declination")
# ax.set_ylabel("Delta (AU)")
# plt.legend()
# plt.show()

df['residuals'] = df['delta'] - fit_func(projected.flatten(), *popt)

# # Plot 19: Residuals vs PCA of DEC and RA (individual graphs for each month)
# for month in df["Month"].unique():
#     plt.scatter(df[df["Month"] == month]["pca"], df[df["Month"] == month]["delta"], label=f"Month {month}", s=10)
#     plt.xlabel(f"PCA of Right Ascension and Declination for {month}")
#     plt.ylabel("Delta")
#     plt.title("Residuals vs PCA of DEC and RA")
#     plt.legend()
#     plt.show()

# # Per day max of delta
# daily_max = df.groupby([df['Day'], df['Month']])['delta'].max().reset_index()
# daily_max["date"] = daily_max.apply(lambda row: dt.datetime(2024, int(row["Month"]), int(row["Day"])), axis=1)
# daily_max.sort_values(by="date", inplace=True)

# # Find all maximum days
# maxima = pd.DataFrame(columns=daily_max.columns)
# for i in range(14, len(daily_max) - 14):
#     window = daily_max[i - 14: i + 15]
#     if window["delta"].iloc[14] == window["delta"].max():
#         maxima.loc[len(maxima)] = window.iloc[14]

# max_first_idx = daily_max[0:14]["delta"].idxmax()
# max_last_idx = daily_max[-14:]["delta"].idxmax()
# maxima.loc[len(maxima)] = daily_max.loc[max_first_idx]
# maxima.loc[len(maxima)] = daily_max.loc[max_last_idx]

# # For each maximum date, find the time at which the maximum occurs
# max_times = pd.DataFrame(columns=["Datetime"])
# for i in range(len(maxima)):
#     max_date = maxima.iloc[i]["date"]
#     max_day = df[(df["Day"] == max_date.day) * (df["Month"] == max_date.month)]
#     max_time = max_day[max_day["delta"] == max_day["delta"].max()]["Datetime"].iloc[0]
#     max_times.loc[len(max_times)] = max_time

dates_of_maxima = [dt.datetime(2024, 1, 1, 15), dt.datetime(2024, 1, 29, 8), dt.datetime(2024, 2, 25, 15), dt.datetime(2024, 3, 23, 16), dt.datetime(2024, 4, 20, 2), dt.datetime(2024, 5, 17, 19), dt.datetime(2024, 6, 14, 14), dt.datetime(2024, 7, 12, 8), dt.datetime(2024, 8, 9, 2), dt.datetime(2024, 9, 5, 15), dt.datetime(2024, 10, 2, 20), dt.datetime(2024, 10, 29, 23), dt.datetime(2024, 11, 26, 12), dt.datetime(2024, 12, 24, 7)]

# Assign a lunar cycle number to all entries in DF based on maxima
for i in range(0, len(dates_of_maxima) - 1):
    df.loc[(df['Datetime'] >= dates_of_maxima[i]) & (df['Datetime'] < dates_of_maxima[i + 1]), 'lunar_cycle'] = i + 1

# Replace all NaN values with 0
df['lunar_cycle'] = df['lunar_cycle'].fillna(0)

# Divide df into separate dataframes based on lunar cycle
lunar_dfs = {}
for cycle in range(1, 12):
    lunar_dfs[cycle] = df[df['lunar_cycle'] == cycle]

# # Plot 20: Residuals vs PCA of DEC and RA (superimposed graphs)
for cycle, df in lunar_dfs.items():
    plt.scatter(df["pca"], df["residuals"], label=f"Lunar Cycle {cycle}", s=10)
plt.xlabel("PCA of Right Ascension and Declination")
plt.ylabel("Residuals (AU)")
plt.title("Residuals vs PCA of DEC and RA")
plt.legend()
plt.show()

# Plot 21: Delta vs PCA of DEC and RA (superimposed graphs)
for cycle, df in lunar_dfs.items():
    plt.scatter(df["pca"], df["delta"], label=f"Lunar Cycle {cycle}", s=10)
plt.xlabel("PCA of Right Ascension and Declination")
plt.ylabel("Delta (AU)")
plt.title("Delta vs PCA of DEC and RA")
plt.legend()
plt.show()