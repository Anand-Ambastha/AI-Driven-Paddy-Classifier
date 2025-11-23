import numpy as np
import pandas as pd
import math, os


def vpd_kpa(temp_c, rh):
    es = 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))
    ea = es * (rh / 100.0)
    return es - ea

def jagadish_score(window):
    T_max = window["temperature_2m"].max()
    RH_min = window["relative_humidity_2m"].min()
    P48    = window["precipitation"].sum()
    VPD48  = vpd_kpa(window["temperature_2m"], window["relative_humidity_2m"]).mean()

    
    daily = window.resample("1D", on="time")["temperature_2m"].max()
    consec_hot_days = np.sum(daily > 33)

    score = 0
    if T_max > 35: score += 2
    elif T_max > 33: score += 1
    if RH_min < 50: score += 1
    if P48 < 2: score += 1
    if consec_hot_days >= 2: score += 1
    if VPD48 > 2.5: score += 1
    return score

def hybrid_label(window):
    score = jagadish_score(window)
    et0_48 = window["et0_fao_evapotranspiration"].sum()
    p48    = window["precipitation"].sum()
    deficit = et0_48 - p48

    if score >= 4 or deficit >= 40:   return 2  
    elif score >= 2 or deficit >= 15: return 1  
    else:                              return 0  
