"""
Water, vegetation, and grazing dynamics in water-controlled
environments

Main models

"""

# Packages ----------------------------------------------------------
import numpy as np
import pandas as pd
from tqdm import tqdm


# Classes -----------------------------------------------------------

class SoilWB:

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
        lk = p['ks'] * s ** (2 * p['phi'] + 3)
        return lk

    def swbdt(self):
        """
        Runs the soil water balance model for each component and
        returns a dictionary with daily simulation.

        :return: the water balance components (s, ET, Lk, and Q)
        at daily scale.
        """
        s = self.s
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
            # q = 0
            if i == 0:
                if self.rain is None:
                    self.rain = 0
                else:
                    self.rain = self.rain
                rain = self.rain
            else:
                rain = 0

            s_in = s + rain / swsc
            if s_in > 1.0:
                q = (s_in - 1.0) * swsc
                s_in = 1.0
            else:
                q = 0
            etr = (SoilWB.evapotranspiration(self) / swsc) * dt
            lkr = (SoilWB.leakage(self) / swsc) * dt
            s = s_in - (etr + lkr)
            # print(s)
            sr.append(s)
            et.append(etr * swsc)
            lk.append(lkr * swsc)
            qr.append(q)
            self.s = sr[-1]
        rswb = dict(sr=sr[-1], rain=self.rain, s=np.mean(sr), ET=np.sum(et), Lk=np.sum(lk), Q=np.sum(qr))
        return rswb

    def swbday(self):
        """
        Runs the soil water balance model for a rainfall series and
        compute the water balance component (s, ET, Lk, and Q) and
        the soil penetration resistance.

        :return: water balance components and soil penetration
        resistance for the same amount of days in the rainfall series.
        """
        p = self.p
        if self.s is None:
            self.s = (0.75 * p['sh'] + 1.25 * p['sw']) / 2
        else:
            self.s = self.s
        rains = self.rain
        nr = len(rains)
        s_out = np.zeros(nr)
        et_out = np.zeros(nr)
        lk_out = np.zeros(nr)
        q_out = np.zeros(nr)
        nrr = np.arange(nr)
        for i in tqdm(nrr):
            self.rain = rains[i]
            swbr = SoilWB.swbdt(self)
            self.s = swbr['sr']
            s_out[i] = np.round(swbr['s'], 4)
            et_out[i] = np.round(swbr['ET'], 4)
            lk_out[i] = np.round(swbr['Lk'], 4)
            q_out[i] = np.round(swbr['Q'], 4)
        out = pd.DataFrame({'Rain': self.rain, 's': s_out, 'ET': et_out,
                            'Lk': lk_out, 'Q': q_out})
        return out


class SwbVg(object):
    def __init__(self, rain, s, ndvi, dt, **kwargs):
        self.rain = rain
        self.s = s
        self.ndvi = ndvi
        self.dt = dt
        self.p = kwargs
        # self.res = SwbVg.run(self)

    def etmaxrate(self):
        p = self.p
        ndvi = self.ndvi
        nc = 2 * (ndvi - p['Nmin'])
        nx = p['Nmax'] - p['Nmin']
        emax = p['emax0'] * (1 + (p['eta'] * ((nc / nx) - 1)))
        return emax

    def canopyinterception(self):
        p = self.p
        ndvi = self.ndvi
        rain = self.rain
        laimax = 0.447 * np.exp(1.9363 * 1)
        lai0 = 0.447 * np.exp(1.9363 * ndvi / p['Nmax'])
        alpha_f = lai0 / laimax
        # alpha_f =1
        Rstar = 0.5
        if rain < Rstar:
            ic = alpha_f * rain
            return ic
        else:
            ic = alpha_f * (0.1 * (rain - Rstar) + Rstar)
            return ic

    def throughfall(self):
        rain = self.rain
        return rain - SwbVg.canopyinterception(self)

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