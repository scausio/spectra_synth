#!/usr/bin/env python3
import xarray as xr
from glob import glob
from natsort import natsorted
import os
import numpy as np
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

# -------------------------------------------------
# LOGGING
# -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# -------------------------------------------------
# USER SETTINGS
# -------------------------------------------------
stats_base = (
    "/data/inputs/METOCEAN/historical/model/ocean/CMS/MedSea/HCMR/"
    "analysis/1h/{year}/{month}/"
    "{year}{month}*_h-HCMR--WAVE-MEDWAM4-MEDATL-*_an00-sv09.00.nc"
)

spectra_dir = "../data/SS_2026/spcs_grid"
outdir = "../data/SS_2026/stats_grid"
os.makedirs(outdir, exist_ok=True)

years = [2024, 2025]
max_workers = 32  # MUST match LSF -n

# Variables to DROP (keep everything else)
DROP_VARS = ["VSDX", "VSDY", "VMXL", "VCMX"]

# -------------------------------------------------
# PREPROCESSING
# -------------------------------------------------
def preproc_MP(ds):
    for v in DROP_VARS:
        if v in ds:
            ds = ds.drop(v)
    return ds

# -------------------------------------------------
# LOAD REFERENCE GRID FROM SPECTRA
# -------------------------------------------------
spectra_files = natsorted(glob(os.path.join(spectra_dir, "*.zarr")))
if not spectra_files:
    raise RuntimeError("No spectra Zarr files found")

with xr.open_zarr(spectra_files[0]) as ref:
    ref_lon = ref.longitude.values
    ref_lat = ref.latitude.values

logging.info("Reference grid loaded")

# -------------------------------------------------
# GLOBAL MIN/MAX CONTAINER
# -------------------------------------------------
GLOBAL_MINMAX = {}

# -------------------------------------------------
# PROCESS SINGLE DAILY FILE
# -------------------------------------------------
def process_stats_file(stat_file):
    """
    Process ONE statistics NetCDF file:
    - reduce variables
    - interpolate to spectra grid
    - align time to spectra
    - write daily Zarr
    - return per-variable min/max
    """
    try:
        fname = os.path.basename(stat_file)
        day = fname[:8]  # YYYYMMDD
        year = day[:4]
        month = day[4:6]

        logging.info(f"Processing {fname}")

        # Corresponding spectra file (daily)
        spectra_match = glob(os.path.join(spectra_dir, f"{day}_reduced*space.zarr"))
        if not spectra_match:
            logging.warning(f"No spectra for {day}, skipping")
            return None

        ds_spec = xr.open_zarr(spectra_match[0])
        spec_time = ds_spec.time.values

        ds = xr.open_dataset(stat_file)
        ds = preproc_MP(ds)

        # Spatial reduction
        ds = ds.sel(
            longitude=ref_lon,
            latitude=ref_lat,
            method="nearest",
        )

        # Time alignment
        ds = ds.sel(time=spec_time, method="nearest")

        # Compute local min/max
        local_minmax = {}
        for v in ds.data_vars:
            arr = ds[v].values
            if np.issubdtype(arr.dtype, np.number):
                local_minmax[v] = (
                    np.nanmin(arr),
                    np.nanmax(arr),
                )

        # Write Zarr
        out_zarr = os.path.join(outdir, f"{day}_stats.zarr")
        ds.to_zarr(out_zarr, mode="w")

        return local_minmax

    except Exception as e:
        logging.error(f"Failed {stat_file}: {e}")
        return None

# -------------------------------------------------
# COLLECT ALL STATISTICS FILES
# -------------------------------------------------
all_stat_files = []
for y in years:
    for m in range(1, 13):
        mstr = f"{m:02d}"
        all_stat_files.extend(
            natsorted(glob(stats_base.format(year=y, month=mstr)))
        )

logging.info(f"Found {len(all_stat_files)} statistics files")

# -------------------------------------------------
# PARALLEL EXECUTION
# -------------------------------------------------
with ProcessPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(process_stats_file, f): f for f in all_stat_files}

    for future in as_completed(futures):
        res = future.result()
        if res is None:
            continue

        for v, (vmin, vmax) in res.items():
            if v not in GLOBAL_MINMAX:
                GLOBAL_MINMAX[v] = [vmin, vmax]
            else:
                GLOBAL_MINMAX[v][0] = min(GLOBAL_MINMAX[v][0], vmin)
                GLOBAL_MINMAX[v][1] = max(GLOBAL_MINMAX[v][1], vmax)

# -------------------------------------------------
# SAVE GLOBAL MIN/MAX
# -------------------------------------------------
minmax_file = os.path.join(outdir, "stats_minmax_global.txt")
with open(minmax_file, "w") as f:
    for v, (vmin, vmax) in GLOBAL_MINMAX.items():
        f.write(f"{v} {vmin} {vmax}\n")

logging.info(f"Global statistics min/max saved to {minmax_file}")

