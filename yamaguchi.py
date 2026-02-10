import numpy as np


class JONSWAP():
    def __init__(self,hs,tp,dir,ndir,freqs,gamma=3.3):
        self.hs= hs
        self.tp=tp
        self.dir=dir
        self.ndir=ndir
        self.freqs=freqs
        self.gamma=gamma


    def jsEnergy(self):
        mo = []
        for f in self.freqs:
            freqPeak = 1. / self.tp  # fpk
            freqPeak4 = freqPeak ** 4  # fpk4
            alpha = ((self.hs ** 2) * freqPeak4) / ((0.06533 * ((self.gamma ** 0.8015) + 0.13467)) * 16)
            f4 = f ** 4
            f5 = f ** 5
            cpshap = 1.25 * freqPeak4 / f4
            if cpshap > 10:
                ra = 0
            else:
                ra = (alpha / f5) * np.exp(-cpshap)
            if f <= freqPeak:
                sigma = 0.07
            else:
                sigma = 0.09
            apshap = 0.5 * ((f - freqPeak) / (sigma * freqPeak)) ** 2
            if apshap > 10:
                syf = 1.
            else:
                ppshap = np.exp(-apshap)
                syf = self.hs ** ppshap
            specDens = syf * ra / (2 * np.pi * f)
            mo.append(specDens)
        return np.array(mo)

    def gammLn(self,val):
        coefs=[76.18009173,-86.50532033,24.01409822, -1.231739516,.120858003e-2,-.536382e-5]
        stp=2.50662827465
        fpf=4.5
        tmp=(val+0.5+fpf)*np.log(val+fpf)-(val+fpf)
        ser=1
        for coef in coefs:
            ser=ser+coef/(val)
            val=+1
        return tmp+(np.log(stp*ser))

    def gammaF(self,val):
        return np.exp(np.clip(self.gammLn(val),-30,30))


    def main(self):
        energySpectra = self.jsEnergy()
        self.directions = np.deg2rad(np.arange(0, 360, (360 / self.ndir)))#[np.pi * i / 180 for i in np.arange(0, 360, (360 / self.ndir))]
        dirSpread = 360 / self.ndir
        rad = np.deg2rad(self.dir)

        if dirSpread < 12:
            ctot = (2. ** dirSpread) * (self.gammaF(0.5 * dirSpread + 1.)) ** 2 / (np.pi * self.gammaF(dirSpread + 1.))
        else:
            ctot = np.sqrt(0.5 * dirSpread / np.pi) / (1. - 0.25 / dirSpread)
        dirSpectra = []
        for d in self.directions:
            acos = np.cos(d - rad)

            if acos > 0:
                cdir = ctot * np.clip((acos ** dirSpread), 1.e-10, None)
            else:
                cdir = 0

            for spectrum in energySpectra:
                dirSpectra.append(cdir * spectrum)

        return energySpectra, np.array(dirSpectra).reshape(self.ndir, len(self.freqs))


