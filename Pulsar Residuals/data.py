import numpy as np
import pandas as pd

# Load the data
d: np.ndarray = np.loadtxt("t120304_065403_toa_scr.tim", dtype='str')

first_TOA: float = d[0, 3].astype(float) * 86400
TOA: np.ndarray = d[:, 3].astype(float) * 86400 - first_TOA
first_diff: np.ndarray = np.diff(TOA)

# Convert to df
df: pd.DataFrame = pd.DataFrame(data={'TOA': TOA[0:-1], 'first_diff': first_diff})

# Save data as pickle
df.to_pickle("pulsar_residuals.pkl")