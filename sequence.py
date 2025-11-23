from helper_func import hybrid_label
import pandas as pd
import numpy as np

def make_sequences(df, seq_len=48):
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    df = df.rename(columns={
        "temperature_2m (°C)": "temperature_2m",
        "relative_humidity_2m (%)": "relative_humidity_2m",
        "precipitation (mm)": "precipitation",
        "wind_speed_10m (km/h)": "wind_speed_10m",
        "et0_fao_evapotranspiration (mm)": "et0_fao_evapotranspiration",
        "soil_moisture_0_to_7cm (m³/m³)": "soil_moisture",
        "shortwave_radiation_instant (W/m²)": "shortwave_radiation"
    })

    cols_to_fix = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "et0_fao_evapotranspiration",
    "soil_moisture",
    "shortwave_radiation"
    ]

    for c in cols_to_fix:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=cols_to_fix)

    X, y = [], []
    for i in range(seq_len, len(df)):
        window = df.iloc[i-seq_len:i].copy()
        label = hybrid_label(window)
        features = window[[
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
            "shortwave_radiation",
            "et0_fao_evapotranspiration",
            "soil_moisture"
        ]].values
        X.append(features)
        y.append(label)

    return np.array(X), np.array(y)