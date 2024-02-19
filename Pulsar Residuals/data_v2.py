import numpy as np
import pandas as pd

# Load the data
d: np.ndarray = np.loadtxt("residuals.csv", delimiter=',', dtype='float')

# Convert to df
df: pd.DataFrame = pd.DataFrame(data = {"residuals": d})
df.to_pickle("pulsar_residuals.pkl")
print(df)

# Save data as pickle
df.to_pickle("pulsar_residuals_v2.pkl")