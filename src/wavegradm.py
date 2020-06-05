"""
Water, vegetation, and grazing dynamics in water-controlled
environments

Main models

"""

# Packages ----------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


# Classes -----------------------------------------------------------

class Soil:

    def __init__(self, s=None, rain=None, dt=None, **kwargs):
        self.p = kwargs
        self.s = s
        self.rain = rain
        self.dt = dt

    def evapotranspiration(self):
        """
         Compute the Evapotranspiration (ET) based on soil moisture.

         :return: the daily evapotransporation in the same unit of
         rainfall. Ex.: cm/day.
         """
        s = self.s
        p = self.p
        # print(p['emax'])
        if s < p['sw']:
            et = p['ew'] * (s - p['sh']) / (p['sw'] - p['sh'])
        elif p['sw'] < s <= p['sstar']:
            et = p['ew'] + (p['emax'] - p['ew']) * (s - p['sw']) / (p['sstar'] - p['sw'])
        else:
            et = p['emax']
        return et

    def leakage(self):
        """
        Compute the drainage based on soil moisture.

        :return: the daily drainage in the same unit of
        rainfall. Ex.: cm/day.
        """
        s = self.s
        p = self.p
        lk = p['ks'] * s ** (2 * p['b'] + 3)
        return lk

    def runoff(self):
        s = self.s
        if s > 1.0:
            q = s - 1
            s = 1
        else:
            s = s
            q = 0
        return [s, q]

    def assimilation(self):
        p = self.p
        s = self.s
        if s <= p['sw']:
            asm = 0
        elif s > p['sstar']:
            asm = 1
        else:
            asm = (s - p['sw']) / (p['sstar'] - p['sw'])
        return asm

    def swbs(self):
        """
        Runs the soil water balance model for each component and
        returns a dictionary with daily simulation.

        :return: the water balance components (s, ET, Lk, and Q)
        at daily scale.
        """
        p = self.p
        if self.dt is None:
            dt = 1 / 48
        else:
            dt = self.dt
        nr = int(np.round(1 / dt, 0))
        swsc = p['n'] * p['zr']
        sr = []
        et = []
        lk = []
        qr = []
        for i in np.arange(nr):
            if i == 0:
                if self.rain is None:
                    self.rain = 0
                else:
                    self.rain = self.rain
                rain = self.rain
            else:
                rain = 0
            self.s = self.s + rain / swsc
            self.s, q = Soil.runoff(self)
            q = q * swsc
            etr = (Soil.evapotranspiration(self) / swsc) * dt
            lkr = (Soil.leakage(self) / swsc) * dt
            self.s = self.s - (etr + lkr)
            # print(s)
            sr.append(self.s)
            et.append(etr * swsc)
            lk.append(lkr * swsc)
            qr.append(q)
            self.s = sr[-1]
        self.s = np.mean(sr)
        rswb = dict(sr=sr[-1], rain=self.rain, s=np.mean(sr),
                    ET=np.sum(et), Lk=np.sum(lk), Q=np.sum(qr),
                    Asm=Soil.assimilation(self))
        return rswb


class Vegetation:

    def __init__(self, A=None, rain=None, btr=None, **kwargs):
        self.A = A
        self.rain = rain
        self.p = kwargs
        print(self.p)
        self.btr = btr
        if self.btr is None:
            self.btr = 1
        else:
            self.btr = self.btr
        self.dt = 1
        print(self.p)

    def biomassNDVI(self):
        ndvi = (self.btr / self.p['kappa']) + self.p['nmin']
        return ndvi

    def etmaxrate(self):
        p = self.p
        ndvi = Vegetation.biomassNDVI(self)
        nc = 2 * (ndvi - p['nmin'])
        nx = p['nmax'] - p['nmin']
        emax = p['emaxb'] * (1 + (p['eta'] * ((nc / nx) - 1)))
        return emax

    def canopyinterception(self):
        p = self.p
        ndvi = Vegetation.biomassNDVI(self)
        rain = self.rain
        laimax = 0.447 * np.exp(1.9363 * 1)
        lai0 = 0.447 * np.exp(1.9363 * ndvi / p['nmax'])
        alpha_f = lai0 / laimax
        Rstar = 0.5
        if rain < Rstar:
            ic = alpha_f * rain
            return ic
        else:
            ic = alpha_f * (0.1 * (rain - Rstar) + Rstar)
            return ic

    def throughfall(self):
        rain = self.rain
        return rain - Vegetation.canopyinterception(self)

    def fodder(self):
        p = self.p
        btr = self.btr
        bmax = p['kappa'] * (p['nmax'] - p['nmin'])
        thinf = p['thn'] * bmax
        thsup = p['thx'] * bmax
        if btr <= thinf:
            return 0
        elif btr > thsup:
            return 1
        else:
            return (btr - thinf) / (thsup - thinf)

    def biomass(self):
        p = self.p
        bmax = p['kappa'] * (p['nmax'] - p['nmin'])
        btr = self.btr
        dt = 1
        dvb = p['ka'] * self.A * (bmax - btr) - p['kr'] * btr
        vb = btr + dvb * dt
        return vb


class Cattle:

    def __init__(self, C0=None, nc=1, fodder=None, **kwargs):
        self.C = C0
        self.fodder = fodder
        self.nc = nc
        self.p = kwargs
        pass

    def cattle(self):
        p = self.p
        C = self.C
        nc = self.nc
        fodder = self.fodder
        weight = fodder * p['kg'] * C * (1 - C / (nc * p['cmax'])) - p['kd'] * C
        return weight


class Wvgd:

    def __init__(self, s0=None, C0=0, nc=1, rain=None, dt=None, btr=1, **kwargs):
        self.p = kwargs
        self.s = s0
        self.C = C0
        self.nc = nc
        self.rain = rain
        self.dt = dt
        self.btr = btr
        self.A = None
        self.fodder = None

    def wavegrad(self):
        # p = self.p
        # Initial value of s0 if None is provided
        if self.s is None:
            self.s = (0.75 * self.p['sh'] + 1.25 * self.p['sw']) / 2
        else:
            self.s = self.s
        rains = self.rain
        nr = len(rains)
        s_out = np.zeros(nr)
        cint_out = np.zeros(nr)
        et_out = np.zeros(nr)
        lk_out = np.zeros(nr)
        q_out = np.zeros(nr)
        db_out = np.zeros(nr)
        dc_out = np.zeros(nr)
        nrr = np.arange(nr)
        dayC0 = 120
        for i in tqdm(nrr):
            # Effects vegetation on ETmax
            self.p['emax'] = Vegetation.etmaxrate(self)
            # p = self.p
            # Effects vegetation on Canopy Interception
            self.rain = rains[i]
            cint_out[i] = Vegetation.canopyinterception(self)
            self.rain = self.rain - cint_out[i]
            swbr = Soil.swbs(self)
            self.s = swbr['sr']

            # Vegetation and aninal growth
            self.fodder = Vegetation.fodder(self)
            if i < dayC0:
                dc_out[i] = 0
            else:
                dc_out[i] = self.C + Cattle.cattle(self) * 1.0
                self.C = dc_out[i]
            self.A = Soil.assimilation(self)
            db_out[i] = Vegetation.biomass(self) - self.p['fC'] * dc_out[i] * self.fodder
            self.btr = db_out[i]

            # Prepare outputs
            cint_out = np.round(cint_out, 4)
            s_out[i] = np.round(swbr['s'], 4)
            et_out[i] = np.round(swbr['ET'], 4)
            lk_out[i] = np.round(swbr['Lk'], 4)
            q_out[i] = np.round(swbr['Q'], 4)
            db_out = np.round(db_out, 2)
            dc_out = np.round(dc_out, 2)
        out = pd.DataFrame({'Rain': rains, 'Cint': cint_out, 's': s_out,
                            'ET': et_out, 'Lk': lk_out, 'Q': q_out,
                            'VegBiomass': db_out, 'AnimalWeight': dc_out})

        return out


class Plots:

    def __init__(self, df=None):
        self.df = df

    def plotres(self):
        df = self.df
        days = np.arange(len(df.Rain))
        fig = plt.figure(figsize=(16, 8))

        ax1 = fig.add_subplot(321)
        ax1.bar(days, df.Rain)
        ax1.set_ylabel('Rain (cm)')
        
        ax2 = fig.add_subplot(322)
        ax2.plot(df.s)
        ax2.set_ylabel('Soil moisture')
        
        ax3 = fig.add_subplot(323)
        ax3.plot(df.VegBiomass)
        ax3.set_ylabel('Biomass (kg)')

        ax4 = fig.add_subplot(324)
        ax4.plot(df.ET)
        ax4.set_ylabel('ET (cm/day)')

        ax5 = fig.add_subplot(325)
        ax5.plot(df.AnimalWeight)
        ax5.set_xlabel('Days')
        ax5.set_ylabel('Cows (kg)')

        ax6 = fig.add_subplot(326)
        ax6.plot(df.Lk)
        ax6.set_xlabel('Days')
        ax6.set_ylabel('Leakage (cm/day)')

        plt.tight_layout()
        plt.savefig('results/Figure_simulation.pdf')
