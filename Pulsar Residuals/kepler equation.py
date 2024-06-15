import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None # default='warn'
from scipy.optimize import curve_fit

C = 299792458  # Speed of light in m/s

# Load the data
df: pd.DataFrame = pd.read_pickle("pulsar_residuals.pkl")

# Values from ATNF
pb: float = 0.10225156248 * 86400  # orbital period in seconds
p0: dict[str, float] = {'a': 0.0226993785996239, 'b': 2.77346077007, 'MJD': 53156.0}
p_dot: dict[str, float] = {'a': 1.76e-18, 'b': 8.92e-16}

# Calculate spin periods for current epoch
p: dict[str, float] = {'MJD': 55990.2879049469477}
p['a'] = p0['a'] + p_dot['a'] * (p['MJD'] - p0['MJD'])
p['b'] = p0['b'] + p_dot['b'] * (p['MJD'] - p0['MJD'])

# Calculate the number of spins between TOAs [USE DF['spin_a'] FOR CALCULATIONS]
df['spin_a'] = df['first_diff'] / p['a']
df['spin_b'] = df['first_diff'] / p['b']

# Generate range
def generate_range(spin_num: float, p: float, first_diff: float) -> np.ndarray:
    MAX_CYCLES: float = np.floor(spin_num)
    MAX_DIST: float = first_diff * C * 0.015
    spin_residual_range: list[float] = []
    for num_cycles in range(int(MAX_CYCLES), -1, -1):
        spin_residual: float = (first_diff - num_cycles * p) * C
        if abs(spin_residual) >= MAX_DIST:
            break
        spin_residual_range.append(spin_residual)
    return np.array(spin_residual_range)

# Generate the range for spin_a and spin_b
df["a_dist_range"] = pd.Series(dtype='object')
for i in range(len(df)):
    df.at[i, 'a_dist_range'] = generate_range(df.at[i, 'spin_a'], p['a'], df.at[i, 'first_diff'])
df['a_rel_vel_range'] = (df['a_dist_range'] / df['first_diff'])/C

df["b_dist_range"] = pd.Series(dtype='object')
for i in range(len(df)):
    df.at[i, 'b_dist_range'] = generate_range(df.at[i, 'spin_b'], p['b'], df.at[i, 'first_diff'])
df['b_rel_vel_range'] = (df['b_dist_range'] / df['first_diff'])/C

df['a_dist_first'] = df['a_dist_range'].apply(lambda x: x[0])


# Correct the breaks

# def break_finder(dvec, dt):
#     derivs = dvec/dt
#     breaks = []
#     for i in range(len(dvec)):
#         if derivs[i] < 7e3: # tune this hyperparameter until the breaks can be seen
#             breaks.append(i)
#     return derivs, breaks
# derivs, breaks = break_finder(df['a_dist_first'], df['first_diff'])

breaks = [24, 88, 122, 161, 231, 278, 297]

delta = max(df['a_dist_first']) - min(df['a_dist_first'])
deltas = [-1, 0, 1, 2, 1, 0, 1]
df["a_dist_first_corrected"] = df["a_dist_first"].copy()
for i in range(0, len(breaks) - 1):
    df["a_dist_first_corrected"].iloc[breaks[i]:breaks[i+1]] += deltas[i] * delta
df["a_dist_first_corrected"].iloc[breaks[-1]:] += deltas[-1] * delta

# Centre the data
df["a_dist_first_corrected"] = df["a_dist_first_corrected"] - df["a_dist_first_corrected"].mean()

# Remove outliers
df['a_dist_first_corrected'] = df['a_dist_first_corrected'].apply(lambda x: x if -1e7 < x < 9e6 else np.nan)
df.dropna(subset=['a_dist_first_corrected'], inplace=True)
df.reset_index(drop=True, inplace=True)

outliers_to_remove = [223, 226, 289, 269, 270, 253, 254, 290, 221, 227]
df.drop(outliers_to_remove, inplace=True)
df.reset_index(drop=True, inplace=True)

# def outlier_finder(dvec, dt):
#     derivs = dvec/dt
#     breaks = []
#     for i in range(len(dvec)):
#         if abs(derivs[i]) > 5e4: # tune this hyperparameter until the breaks can be seen
#             breaks.append(i)
#     return derivs, breaks
# derivs, outliers = outlier_finder(df['a_dist_first_corrected'].diff(), df['first_diff'])
# outliers = [x for x in outliers if x not in outliers_to_remove]
# print(outliers)

plt.figure()
plt.scatter(df["TOA"], df["a_dist_first"], label = "Uncorrected")
plt.scatter(df["TOA"], df["a_dist_first_corrected"], color='red', label = "Corrected")
plt.xlabel("Time of Arrival (sec)")
plt.ylabel("Distance (m)")
plt.title("Distance vs. Time of Arrival")
plt.legend()
plt.show()

print(df)
df[['TOA', 'a_dist_first_corrected']].to_csv('kepler_equation.csv', index=False)