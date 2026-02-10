import os.path
import xarray as xr
import numpy as np
import random
import torch
from glob import glob
import os
from torch.utils.data import DataLoader, Dataset
import json

def build_file_pairs(stats_dir, spc_dir,fname="*2025*.zarr"):
    stats_files = sorted(glob(os.path.join(stats_dir, fname)))
    spc_files   = sorted(glob(os.path.join(spc_dir, fname)))

    stats_map = {os.path.basename(f)[:8]: f for f in stats_files}
    spc_map   = {os.path.basename(f)[:8]: f for f in spc_files}

    common_days = sorted(set(stats_map) & set(spc_map))

    pairs = [(stats_map[d], spc_map[d]) for d in common_days]
    return pairs

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False


class CreateDataset(Dataset):
    def __init__(self, file_pairs, reader_fn, config):
        """
        file_pairs : list of (stats_file, spectra_file)
        reader_fn  : Reader class
        """
        self.file_pairs = file_pairs
        self.reader_fn = reader_fn
        self.config = config

        # cache per file_id
        self.cache = {}

        # global index: (file_id, time_index)
        self.index = []

        for i, (fs, fp) in enumerate(self.file_pairs):
            reader = self.reader_fn(fs, fp, self.config)
            X, Y = reader.process_stats()   # X:(T,9) Y:(T,1,24,32)

            self.cache[i] = (
                torch.from_numpy(X).float(),
                torch.from_numpy(Y).float()
            )

            T = X.shape[0]
            self.index.extend([(i, t) for t in range(T)])

        print(f"Dataset FAST: {len(self.index)} samples cached")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_id, t = self.index[idx]
        X, Y = self.cache[file_id]
        return X[t], Y[t]


class Reader():
    def __init__(self,stats_path, spcs_path,config):
        self.config=config
        self.spcs_path=spcs_path
        self.stats_path=stats_path
        self.depth = config['depth']
        self.scaling_coeffs()
        self.wind=config['wind']
        self.spectra()
        self.step=config['decimate_input']
        self.stats()


    def spectra(self):
        #print(f'Opening wave spectra file')
        ds_spc = xr.open_zarr(self.spcs_path)
        self.time = ds_spc.time.values
        self.freqs = ds_spc.frequency.values
        self.dires = ds_spc.direction.values

        self.kappa_bins = len(self.freqs)
        self.theta_bins = len(self.dires)

        ds_spc=ds_spc.drop('frequency')
        ds_spc=ds_spc.drop('direction')
        ds_spc=ds_spc.reset_coords()

        efth=ds_spc['EF'].transpose('latitude','longitude', 'time','kt_flat').values
        #print ("EF shape", efth.shape)

        efth=efth.reshape(-1,self.kappa_bins * self.theta_bins)

        self.EF =  self.scale('EF',efth)
       # print(f"EF shape:", self.EF.shape)

    def stats(self):
        step=self.step
        #print (f'Opening wave stats file')
        ds_stat = xr.open_zarr(self.stats_path).drop_duplicates(dim='time').sel(time=self.time, method='nearest')
        ds_stat=ds_stat.transpose('latitude', 'longitude', 'time')
        #print (f'Decimation factor of : {step} used')

        SWH_WW = self.scale('VHM0_WW',ds_stat['VHM0_WW'].values.flatten())
        self.idx = np.isfinite(SWH_WW)
        self.SWH_WW=SWH_WW[self.idx][::step]# significant wave height
        self.WPP_WW = self.scale('VTM01_WW', ds_stat['VTM01_WW'].values.flatten())[self.idx][::step]# peak wave period
        self.MWP_WW =  self.scale('VMDR_WW',ds_stat['VMDR_WW'].values.flatten())[self.idx][::step] # mean wave period

        self.SWH_SW1 =  self.scale('VHM0_SW1',ds_stat['VHM0_SW1'].values.flatten())[self.idx][::step]
        self.WPP_SW1 =  self.scale('VTM01_SW1',ds_stat['VTM01_SW1'].values.flatten())[self.idx][::step]# peak wave period
        self.MWP_SW1 =  self.scale('VMDR_SW1',ds_stat['VMDR_SW1'].values.flatten())[self.idx][::step] # mean wave period

        self.SWH_SW2 =  self.scale('VHM0_SW2',ds_stat['VHM0_SW2'].values.flatten())[self.idx][::step]
        self.WPP_SW2 =  self.scale('VTM01_SW2',ds_stat['VTM01_SW2'].values.flatten())[self.idx][::step]# peak wave period
        self.MWP_SW2 =  self.scale('VMDR_SW2',ds_stat['VMDR_SW2'].values.flatten())[self.idx][::step] # mean wave period
       # self.SWH =  self.scale('VHM0',ds_stat['VHM0'].values.flatten()[self.idx][::step]  # significant wave height

        # PLEASE IF ADD INTO THE TRAINING IMPLEMEMT THE SCALERS
        #self.LON = np.tile(ds_stat['longitude'].values, (len(ds_stat['latitude'])*len(ds_stat['time']))).flatten()[self.idx][::step]
        #self.LAT = np.tile(ds_stat['latitude'].values, (len(ds_stat['longitude']) * len(ds_stat['time']))).flatten()[self.idx][::step]

        # if self.wind:
        #     self.u10=ds_stat['U10M'].values.flatten() [self.idx][::step]
        #     self.v10 = ds_stat['V10M'].values.flatten()[self.idx][::step]

        self.timelen=len(ds_stat['time'].values)
        self.EF=self.EF[self.idx][::step]

        #print('EF', np.all(np.isfinite(self.EF)))
        dpt=xr.open_dataset(self.depth)
        self.DPT=self.scale('DPT',np.tile(dpt.deptho.values.flatten(),self.timelen))[self.idx][::step]
        #print ('dpt',self.DPT.shape)
        #print ('wpp',self.WPP_WW.shape)

    def scaling_coeffs(self):
        with open(self.config['scaler'], "r") as f:
            self.scaler= json.load(f)

    def scale(self, key, values):

        # keys = ['EF', 'VHM0', 'VTPK', 'VTM10', 'VTM02',
        #         'SWH_SW2', 'WPP_SW2', 'MWP_SW2', 'DPT', 'DUMMY']  
        s = self.scaler[key]['scale']
        o = self.scaler[key]['offset']
        return values * s +o


    def process_stats(self):
        coords = self.config['add_coords']
        wind = self.config['wind']

        self.x_in = np.vstack([
        self.SWH_WW, self.WPP_WW, self.MWP_WW,
        self.SWH_SW1, self.WPP_SW1, self.MWP_SW1,
        self.SWH_SW2, self.WPP_SW2, self.MWP_SW2
        ]).T

        nt_samples = self.EF.shape[0]
        self.EF = self.EF.reshape(nt_samples, 1, self.kappa_bins,self.theta_bins)

        return self.x_in, self.EF








