import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

print("Spin periods for current epoch:", p)

# Calculate the number of spins between TOAs [USE DF['spin_a'] FOR CALCULATIONS]
df['spin_a'] = df['first_diff'] / p['a']
df['spin_b'] = df['first_diff'] / p['b']

# Calculate the residuals
df['orbital_residuals'] = df['TOA'] - pb

# Plot the residuals
plt.plot(df['first_diff'], df['orbital_residuals'])
plt.xlabel("First Difference (sec)")
plt.ylabel("Residuals (sec)")
plt.title("Orbital Residuals")
plt.show()

# Plot the number of spins between TOAs
plt.plot(df['TOA'], df['spin_a'], label="Spin period a")
plt.xlabel("TOA (sec)")
plt.ylabel("Number of spins")
plt.title("Number of spins between TOAs")
plt.legend()
plt.show()

# Plot first diff
plt.plot(df['TOA'], df['first_diff'])
plt.xlabel("TOA (sec)")
plt.ylabel("First Difference (sec)")
plt.title("Difference between consecutive TOAs")
plt.show()