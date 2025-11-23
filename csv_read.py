import numpy as np
import pandas as pd
import math, os
from glob import glob
from sequence import make_sequences
site_id_map = {
    "ludhiana": 0,
    "lucknow": 1,
    "kolkata": 2,
    "jorhat": 3,
    "thanjavur": 4,
}

site_id_test = {
    "visakhapatnam": 5
}
all_X, all_y, all_site = [], [], []

for file in glob("data/*.csv"):
    name = os.path.basename(file).split(".")[0].lower()
    if name not in site_id_map: continue

    df = pd.read_csv(file)
    df = df.iloc[:22633]
    df = df.reset_index(drop=True)
    X, y = make_sequences(df)
    site_id = np.full((len(y),1), site_id_map[name])
    all_X.append(X)
    all_y.append(y)
    all_site.append(site_id)

X = np.concatenate(all_X, axis=0)
y = np.concatenate(all_y, axis=0)
site_ids = np.concatenate(all_site, axis=0)

np.save("X.npy", X)
np.save("y.npy", y)
np.save("site_ids.npy", site_ids)
print("Saved:", X.shape, y.shape, site_ids.shape)

test_X, test_y, test_site = [], [], []

for file in glob("data/*.csv"):
    name = os.path.basename(file).split(".")[0].lower()
    if name not in site_id_test: continue

    df = pd.read_csv(file)
    df = df.iloc[:22633]
    df = df.reset_index(drop=True)
    X, y = make_sequences(df)
    site_id = np.full((len(y),1), site_id_test[name])
    test_X.append(X)
    test_y.append(y)
    test_site.append(site_id)

X = np.concatenate(test_X, axis=0)
y = np.concatenate(test_y, axis=0)
site_ids = np.concatenate(test_site, axis=0)

np.save("X_test.npy", X)
np.save("y_test.npy", y)
np.save("site_ids_test.npy", site_ids)
print("Saved:", X.shape, y.shape, site_ids.shape)