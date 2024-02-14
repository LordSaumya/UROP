## Imports
import numpy as np
import pandas as pd
import datetime as dt
from re import match, split
from astropy import units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

## Read in data from pickle file
df = pd.read_pickle("horizons_results.pkl")

## Plot

# Extract month from 'Datetime'
df["Month"] = df["Datetime"].dt.month

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
# RA_rad = df["Coord"].apply(lambda x: x.ra.wrap_at(180 * u.deg).radian)
# DEC_rad = df["Coord"].apply(lambda x: x.dec.radian)
# plt.scatter(RA_rad, DEC_rad, s=1)
# plt.title("Aitoff Projection")
# plt.show()

# # Plot 6: 3D plot of coordinates vs time
# plt.figure(6)
# ax = plt.axes(projection="3d")
# df['date_delta'] = (df['Datetime'] - df['Datetime'].min())  / np.timedelta64(1,'D')
# ax.scatter3D(df["date_delta"], df["Coord"].apply(lambda x: x.ra.wrap_at(180 * u.deg).radian), df["Coord"].apply(lambda x: x.dec.radian))
# ax.set_xlabel("Time (in days since 1st January 2024)")
# ax.set_ylabel("Right Ascension (rad)")
# ax.set_zlabel("Declination (rad)")
# plt.show()

# # Plot 7: Polar plot of coordinates vs time
# plt.figure(7)
# ax = plt.axes(projection="polar")
# ax.scatter(
#     df["Coord"].apply(lambda x: x.ra.wrap_at(180 * u.deg).radian),
#     df["Coord"].apply(lambda x: x.dec.radian),
# )
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
#     ra = np.array([coord.ra.radian for coord in df["Coord"]])
#     dec = np.array([coord.dec.radian for coord in df["Coord"]])
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

# RA = df['RA'] * np.pi / 180
# DEC = df['DEC'] * np.pi / 180

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