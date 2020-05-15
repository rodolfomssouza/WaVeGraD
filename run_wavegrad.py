"""
Water Vegetation Grazing Dynamics modeling

Rodolfo Souza et al. (2020)

"""

# Packages ----------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
from src import wavegradm as wvg

# Data and parameters -----------------------------------------------
# All parameters
ap = {'sh': 0.10,
      'sw': 0.13,
      'sstar': 0.25,
      'ks': 205,
      'b': 4.38,
      'n': 0.48,
      'zr': 40,
      'emax': 0.5,
      'emaxb': 0.5,
      'ew': 0.05,
      'nmax': 0.95,
      'nmin': 0.22,
      'ka': 0.061,
      'kr': 0.011,
      'eta': 0.50,
      'kappa': 4387,
      'thn': 0.10,
      'thx': 0.50,
      'cmax': 450,
      'kg': 0.014,
      'kd': 0.0035,
      'fC': 0.03}

# Rainfall data
drain = pd.read_csv('data/Data_rainfall.csv')
rainfall = drain.rain.values

# Run the model -----------------------------------------------------
simulation = wvg.Wvgd(C0=120, rain=rainfall, **ap).wavegrad()
# print(simulation)

simulation.to_csv('results/WaVeGraD_simulation.csv')
wvg.Plots(simulation).plotres()