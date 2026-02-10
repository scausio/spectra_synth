#!/usr/bin/env python3
import xarray as xr
from glob import glob
from natsort import natsorted
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# -----------------------------
# USER SETTINGS
# -----------------------------
base = '/data/inputs/METOCEAN/historical/model/ocean/HCMR/WAM_spectra/analysis/1h/{year}/{month}/{year}{month}*_h-HCMR--WAVESPEC-MEDWAM4-MEDATL-*_an-sv09.00.nc'
outdir = '../data/SS_2026/spcs_grid'
os.makedirs(outdir, exist_ok=True)

xstep = 10
ystep = 10
tstep = 2

years = [ 2024, 2025]
max_workers = 32  # Set this to the number of cores you requested in LSF

# -----------------------------
# FUNCTION TO PROCESS SINGLE FILE
# -----------------------------
def process_file(f):
    day = os.path.basename(f)[:8]
    EF_min_local = np.inf
    EF_max_local = -np.inf

    try:
        ds = xr.open_dataset(f)
        ds = ds.isel(latitude=slice(0, None, ystep),
                      longitude=slice(0, None, xstep),
                      time=slice(0, None, tstep))
    except Exception as e:
        print(f"{f} not valid: {e}")
        return EF_min_local, EF_max_local  # skip this file

    ds = ds.dropna(dim="time", subset=["EFTH"], how="all")
    if "EFTH" not in ds:
        return EF_min_local, EF_max_local

    ds['EF'] = (('time','kt_flat','latitude','longitude'),
                ds['EFTH'].values.reshape(len(ds.time), -1, len(ds.latitude), len(ds.longitude)))

    # Track local min/max
    EF_min_local = np.nanmin(ds['EF'].values)
    EF_max_local = np.nanmax(ds['EF'].values)

    freq = ds.frequency
    dires = ds.direction
    ds = ds.drop('EFTH')
    ds['frequency'] = freq
    ds['direction'] = dires

    # Save reduced dataset
    out_file = os.path.join(outdir, f"{day}_reduced{xstep}space.zarr")
    ds.to_zarr(out_file, mode='w')

    return EF_min_local, EF_max_local

# -----------------------------
# GATHER ALL FILES
# -----------------------------
all_files = []
for year in years:
    for month in range(1, 13):
        month_str = f"{month:02d}"
        files = natsorted(glob(base.format(year=year, month=month_str)))
        all_files.extend(files)

print(f"Found {len(all_files)} files to process.")

# -----------------------------
# PARALLEL PROCESSING
# -----------------------------
EF_global_min = np.inf
EF_global_max = -np.inf

with ProcessPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(process_file, f): f for f in all_files}

    for future in as_completed(futures):
        f = futures[future]
        try:
            file_min, file_max = future.result()
            EF_global_min = min(EF_global_min, file_min)
            EF_global_max = max(EF_global_max, file_max)
        except Exception as e:
            print(f"Error processing {f}: {e}")

# -----------------------------
# SAVE GLOBAL MIN/MAX
# -----------------------------
minmax_file = os.path.join(outdir, "EF_minmax_global.txt")
np.savetxt(minmax_file, np.array([EF_global_min, EF_global_max]))
print(f"Global EF min/max saved to {minmax_file}: {EF_global_min}, {EF_global_max}")

