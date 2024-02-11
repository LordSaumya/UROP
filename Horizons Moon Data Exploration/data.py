## Imports
import numpy as np
import pandas as pd
import datetime as dt
from re import match
from astropy import units as u
from astropy.coordinates import SkyCoord

## Read in data from txt file
data = open("horizons_results.txt", "r").readlines()
data = data[61:52704] #end = 52704

## Make dataframe
header_list = [
    ["Datetime", "RA", "DEC", "APmag", "S-brt", "delta", "deldot", "S-O-T /r", "S-T-O", "Sky_motion", "Sky_mot_PA", "RelVel-ANG", "Lun_Sky_Brt", "sky_SNR"],
]

df = pd.DataFrame(columns=header_list[0])
for line in data:
    line = line.split()
    date = dt.datetime.strptime(line[0] + " " + line[1], "%Y-%b-%d %H:%M") # convert to datetime object
    if not (match("[0-9]", line[2])):
        line.pop(2) # remove * and other chars after datetime
    RA = float(line[2]) * 15 + float(line[3]) / 4 + float(line[4]) / 240 # convert to decimal degrees
    DEC = float(line[5]) + float(line[6]) / 60 + float(line[7]) / 3600 # convert to decimal degrees
    Coord = SkyCoord(ra = RA, dec = DEC, unit="deg") # convert to SkyCoord object
    APmag = float(line[8])
    S_brt = line[9]
    delta = float(line[10])
    deldot = float(line[11])
    S_O_T = line[12] + " " + line[13]
    S_T_O = float(line[14])
    Sky_motion = float(line[15])
    Sky_mot_PA = float(line[16])
    RelVel_ANG = float(line[17])
    Lun_Sky_Brt = line[18]
    sky_SNR = line[19]
    line = [date, RA, DEC, APmag, S_brt, delta, deldot, S_O_T, S_T_O, Sky_motion, Sky_mot_PA, RelVel_ANG, Lun_Sky_Brt, sky_SNR]
    df.loc[len(df)] = line

## Add column for time_from_start
df["time_from_start"] = (df["Datetime"] - df["Datetime"][0]).dt.total_seconds()

## Sample 1000 entries from dataframe
df_sample = df.sample(1000)

## Save dataframe sample as csv file
df_sample.to_csv("horizons_results_sample.csv")

## Save dataframe as pickle file
df.to_pickle("horizons_results.pkl") # save dataframe as pickle file