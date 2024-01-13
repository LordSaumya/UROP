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
df['Month'] = df['Datetime'].dt.month

# Plot 1: Sky Motion vs. Time (months on x-axis)
plt.figure(1)
plt.plot(df["Datetime"], df["Sky_motion"])
plt.xlabel("Month")
plt.ylabel("Sky Motion (deg/day)")
plt.title("Sky Motion vs. Time")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y')) 
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.show()

# Plot 2: Sky Motion (Position Angle) vs. Time (months on x-axis)
plt.figure(2)
plt.plot(df["Datetime"], df["Sky_mot_PA"])
plt.xlabel("Month")
plt.ylabel("Sky Motion (Position Angle) in degrees")
plt.title("Sky Motion (Position Angle) vs. Time")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.show()

# Plot 3: Apparent Magnitude vs. Time (months on x-axis)
plt.figure(3)
plt.plot(df["Datetime"], df["APmag"])
plt.xlabel("Month")
plt.ylabel("Apparent Magnitude")
plt.title("Apparent Magnitude vs. Time")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.show()

# Plot 4: Log of Surface Brightness vs. Time (months on x-axis)
plt.figure(4)
plt.plot(df["Datetime"], np.log10(df["S-brt"].astype(float)))
plt.xlabel("Month")
plt.ylabel("Log of Surface Brightness")
plt.title("Log of Surface Brightness vs. Time")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.show()


# Plot 5: Aitoff projection of coordinates
plt.figure(5)
plt.subplot(111, projection="aitoff")
plt.grid(True)
RA_rad = df["Coord"].apply(lambda x: x.ra.wrap_at(180 * u.deg).radian)
DEC_rad = df["Coord"].apply(lambda x: x.dec.radian)
plt.scatter(RA_rad, DEC_rad, s=1)
plt.title("Aitoff Projection")
plt.show()

# Plot 6: 3D plot of coordinates vs time
plt.figure(6)
ax = plt.axes(projection="3d")
df['date_delta'] = (df['Datetime'] - df['Datetime'].min())  / np.timedelta64(1,'D')
ax.scatter3D(df["date_delta"], df["Coord"].apply(lambda x: x.ra.wrap_at(180 * u.deg).radian), df["Coord"].apply(lambda x: x.dec.radian))
ax.set_xlabel("Time (in days since 1st January 2024)")
ax.set_ylabel("Right Ascension (rad)")
ax.set_zlabel("Declination (rad)")
plt.show()
