import matplotlib.pyplot as plt
import os
import numpy as np
from yamaguchi import JONSWAP
from shaper import Ds_Conv
from utils import fixBCdir
import torch
from natsort import natsorted
from glob import glob


def main():
    os.makedirs(outdir, exist_ok=True)
    # Stats (input) file
    stats_path = os.path.join(base, 'wave_stats_short.nc')

    # Spectra (target) file
    spc_path = os.path.join(base, 'wave_spectra_short.nc')

    ds = Reader(spc_path, stats_path, "FFNN0",mp=mp)

    # Define frequencies and spectra dimensions (k_bins, theta_bins) = (32, 24)
    theta_bins = ds.theta_bins
    k_bins = ds.kappa_bins
    freqs = ds.freqs

    ml_ds = Ds_Conv(ds, reshape_size=(k_bins, theta_bins),mp=mp,visualize=True,coords=coords)
    # Test dataset
    model_name=natsorted(glob(os.path.join(pt_dir,  f'best_model_epoch_*.pt')))[-1]
    model = torch.load(model_name, map_location=torch.device('cpu'))
    with torch.no_grad():
        X_test_scaled = ml_ds.X_test_scaled
        print(X_test_scaled.shape)
        # Y_pred = ml_ds.scaler_Y.inverse_transform(model(X_test_scaled).cpu()[:,0].reshape([-1,theta_bins*k_bins]))
        Y_pred = ml_ds.scaler_Y.inverse_transform(model(X_test_scaled).cpu().reshape([-1, theta_bins * k_bins]))
        Y_test = ml_ds.Y_test
        X_test = ml_ds.X_test

    for i in range(60):
        print(f'plotting {i}')
        fig, axs = plt.subplots(ncols=3, figsize=(17, 5))
        vmin = 0
        vmax = np.nanmax(Y_test[i])

        im1 = axs[0].imshow(Y_test[i].reshape(k_bins, theta_bins), aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
        axs[0].set_title('True Wave Spectrum')
        axs[0].set_xlabel('Θ')
        axs[0].set_ylabel('κ')

        axs[1].imshow(Y_pred[i].reshape(k_bins, theta_bins), aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
        axs[1].set_title('Predicted Wave Spectrum')
        axs[1].set_xlabel('Θ')
        axs[1].set_ylabel('κ')
        hs = X_test[i, 0]
        tp = X_test[i, 1]
        dir = X_test[i, 2]
        # print((X_test[i,:4]))
        nonDimSpec, dimSpec = JONSWAP(hs, tp, fixBCdir(dir), theta_bins, freqs).main()

        axs[2].imshow(dimSpec.T[:, ::-1], aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
        tp = str(np.round(tp, 1))[:4]
        hs = str(np.round(hs, 1))[:3]
        axs[2].set_title(f'Yamaguchi approx. Wave Spectrum. Hs:{hs}, Tp:{tp}, Dir:{int(dir)}')
        axs[2].set_xlabel('Θ')
        axs[2].set_ylabel('κ')
        cbar = fig.colorbar(im1, orientation='vertical')
        cbar.set_label('E m2Hz-1deg-1')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'{os.path.basename(model_name)}_{i}.png'))
        plt.close()

    train_loss = np.load(os.path.join(pt_dir,  f'train_loss.npy'))
    val_loss = np.load(os.path.join(pt_dir,  f'val_loss.npy'))
    plt.plot(train_loss, label='train loss')
    plt.plot(val_loss, label='validation loss')
    plt.savefig(os.path.join(outdir, f'loss.png'))

# base = './data'
# base='/work/cmcc/ww3_cst-dev/work/ML/data/SSMEDdef_dataset'
mp=True
coords=False
if mp:
    base = '/Users/scausio/CMCC Dropbox/Salvatore Causio/PycharmData/ML/MED/MP'
else:
    base = '/Users/scausio/CMCC Dropbox/Salvatore Causio/PycharmData/ML/MED/'

#pt_dir='/Users/scausio/Dropbox (CMCC)/PycharmData/ML/outputSparseYam'
pt_dir='output_5j_MP_juno'
outdir = os.path.join(pt_dir,'pics')
#model_name='best_model_epoch_*'

# model_name='ckpt_epoch_90'
#pt_dir='/Users/scausio/Dropbox (CMCC)/PycharmData/ML/longCoordsPi/'

main()


