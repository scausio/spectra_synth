#!/usr/bin/env python3
import numpy as np
import os
import json
import xarray as xr

def convert_to_native(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_native(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(item) for item in obj]
    return obj

# -------------------------------------------------
# USER SETTINGS
# -------------------------------------------------
spectra_minmax_file = "../data/SS_2026/spcs_grid/EF_minmax_global.txt"
stats_minmax_file   = "../data/SS_2026/stats_grid/stats_minmax_global.txt"

depth_file = "/work/cmcc/ww3_cst-dev/work/ML/data/cmems_mod_med_wav_anfc_4.2km_static_1718290291661.nc"   # <-- EDIT
depth_var  = "deptho"                                     # <-- EDIT

outdir = "../data/SS_2026/scalers"
os.makedirs(outdir, exist_ok=True)

# Scaling method:
# "minmax"        -> [0, 1]
# "minmax_-1_1"   -> [-1, 1]
# "standard"      -> (x - mean) / std
# "log_minmax"    -> log(x) then [0, 1]
SCALING_METHOD = "minmax"

EPS = 1e-12

# -------------------------------------------------
# SCALER FACTORY
# -------------------------------------------------
def build_scaler(vmin, vmax, method):
    if vmax - vmin < EPS:
        raise ValueError(f"Constant variable detected (min=max={vmin})")

    if method == "minmax":
        return {
            "type": "minmax",
            "min": vmin,
            "max": vmax,
            "scale": 1.0 / (vmax - vmin),
            "offset": -vmin / (vmax - vmin),
        }

    if method == "minmax_-1_1":
        return {
            "type": "minmax_-1_1",
            "min": vmin,
            "max": vmax,
            "scale": 2.0 / (vmax - vmin),
            "offset": -1.0 - 2.0 * vmin / (vmax - vmin),
        }

    if method == "standard":
        mean = 0.5 * (vmin + vmax)
        std  = 0.5 * (vmax - vmin)
        return {
            "type": "standard",
            "mean": mean,
            "std": std,
            "scale": 1.0 / std,
            "offset": -mean / std,
        }

    if method == "log_minmax":
        if vmin <= 0:
            raise ValueError("log scaling requires strictly positive values")
        lvmin, lvmax = np.log(vmin), np.log(vmax)
        return {
            "type": "log_minmax",
            "min": lvmin,
            "max": lvmax,
            "scale": 1.0 / (lvmax - lvmin),
            "offset": -lvmin / (lvmax - lvmin),
        }

    raise ValueError(f"Unknown scaling method: {method}")

# -------------------------------------------------
# UNIFIED SCALER DICTIONARY
# -------------------------------------------------
scalers = {}

# -------------------------------------------------
# 1️⃣ SPECTRA (EF)
# -------------------------------------------------
with open(spectra_minmax_file, "r") as f:
    EF_min = float(f.readline())
    EF_max = float(f.readline())

scalers["EF"] = build_scaler(EF_min, EF_max, SCALING_METHOD)

# -------------------------------------------------
# 2️⃣ STATISTICS VARIABLES
# -------------------------------------------------
with open(stats_minmax_file, "r") as f:
    for line in f:
        if not line.strip():
            continue
        var, vmin, vmax = line.split()
        scalers[var] = build_scaler(float(vmin), float(vmax), SCALING_METHOD)

# -------------------------------------------------
# 3️⃣ DEPTH (DPT)
# -------------------------------------------------
ds_dpt = xr.open_dataset(depth_file)

if depth_var not in ds_dpt:
    raise KeyError(f"Variable {depth_var} not found in {depth_file}")

DPT = ds_dpt[depth_var].values

# Remove land / invalid values if needed
DPT = np.where(np.isfinite(DPT), DPT, np.nan)

DPT_min = np.nanmin(DPT)
DPT_max = np.nanmax(DPT)

scalers["DPT"] = build_scaler(DPT_min, DPT_max, SCALING_METHOD)

# -------------------------------------------------
# SAVE UNIFIED SCALERS
# -------------------------------------------------
json_file = os.path.join(outdir, f"scalers_unified_{SCALING_METHOD}.json")
scalers_native = convert_to_native(scalers)
with open(json_file, 'w') as f:
    json.dump(scalers_native, f, indent=2)


npz_file = os.path.join(outdir, f"scalers_unified_{SCALING_METHOD}.npz")
np.savez(
    npz_file,
    **{
        var: np.array([cfg["scale"], cfg["offset"]], dtype=np.float64)
        for var, cfg in scalers.items()
    }
)

print("Unified scalers written:")
print(" ", json_file)
print(" ", npz_file)

