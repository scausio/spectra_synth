from datetime import datetime
import numpy as np
import logging
import torch
import os
def fixBCdir(deg):

    if isinstance(deg, np.ndarray):

        m_low = deg <= 180
        m_hi = deg > 180

        deg[m_low] = np.abs(deg[m_low] - 180)
        deg[m_hi] = 360 - (np.abs(deg[m_hi] - 180))

    else:
        if deg <= 180:
            deg = np.abs(deg - 180)
        else:
            deg = 360 - (np.abs(deg - 180))
    return deg
def getFreqs(minFreq, nFreq, ratio):
    freqs = []
    for f in range(nFreq):
        freqs.append(minFreq)
        minFreq = minFreq * ratio
    return np.array(freqs)

def getDirs(nDir):
    return np.arange(0, 360, (360 / nDir))
def init():
    start_time = datetime.now()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')


    np.random.seed(42)
    return start_time,logging


def getDevice():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    return device

def setWorkingDirs(outdir):

    checkpoint_dir = outdir
    os.makedirs(outdir, exist_ok=True)

    os.makedirs(checkpoint_dir, exist_ok=True)

    return {'checkpoint':checkpoint_dir}
